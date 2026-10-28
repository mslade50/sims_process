"""Build leakage-safe pre-event shot-derived mean factors for market backtests.

This module is research-only.  It turns the existing R1-frozen shot feature
library into one row per target event/player and refits the incremental APP
coefficient before every event date.  A target event can therefore never
influence its own coefficient, and concurrent events share the same prior-only
fit.

The primary factor is approach cross-track RMS over the trailing 16 completed
rounds.  Its coefficient is fit against future R1/R2 adjusted APP SG while
controlling for the player's event-safe trailing APP mean and standard
deviation.  The emitted delta is field-neutral and is intended to move both
APP mean and total pre-tournament mean by the same SG/round amount.

The archived distance x lie study can produce a genuine player/event panel,
but only aggregate evaluation results were persisted.  ``standardize_distance_lie_panel``
accepts a future explicit player/event export and deliberately refuses an
aggregate result; this script does not reconstruct that expensive panel by
default or pretend the aggregate JSON is player-level data.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SIM_PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
DEFAULT_FEATURES = (
    DEFAULT_SIM_PREP / "shot_bridge" / "shot_feature_library.parquet"
)
DEFAULT_DB = DEFAULT_SIM_PREP / "work" / "dg_historical.db"
DEFAULT_OUTPUT = (
    REPO / "permanent_data" / "research" / "shot_mean_factor_panel.parquet"
)

FACTOR_NAME = "app_cross_track_rms_trail16"
FEATURE_COLUMN = "app_dir_rms_cross_trail16"
BASE_COLUMNS = ("prior_app_mean50", "prior_app_sd50")
OUTPUT_COLUMNS = (
    "event_key",
    "year",
    "event_id",
    "start",
    "dg_id",
    "player_name",
    "factor_name",
    "factor_raw",
    "factor_centered",
    "factor_delta_sg_per_round",
)
DISTANCE_LIE_VALUE_COLUMN = "phys__exact8_full_k400"


def _normal_name(value: object) -> str:
    return str(value).strip().lower()


def canonical_event_key(year: int, event_id: int) -> str:
    """Return the canonical DataGolf/R2 PGA event key."""
    return f"pga:R{int(year):04d}{int(event_id):03d}"


def load_pre_event_cross_track_features(path: Path) -> pd.DataFrame:
    """Load one R1-frozen cross-track snapshot per event/player."""
    columns = [
        "dg_id",
        "player_name",
        "event_id",
        "year",
        "round_num",
        "round_date",
        FEATURE_COLUMN,
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["round_num"].eq(1)].copy()
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="raise").astype(int)
    frame["event_id"] = pd.to_numeric(
        frame["event_id"], errors="raise"
    ).astype(int)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["start"] = pd.to_datetime(
        frame["round_date"], errors="raise"
    ).dt.normalize()
    frame["player_name"] = frame["player_name"].map(_normal_name)
    frame["event_key"] = [
        canonical_event_key(year, event_id)
        for year, event_id in frame[["year", "event_id"]].itertuples(
            index=False, name=None
        )
    ]
    key = ["year", "event_id", "dg_id"]
    if frame.duplicated(key).any():
        duplicate = frame.loc[frame.duplicated(key, keep=False), key].head()
        raise ValueError(
            "R1 shot feature snapshot has duplicate event/player rows: "
            f"{duplicate.to_dict(orient='records')}"
        )
    return frame[
        [
            "event_key",
            "year",
            "event_id",
            "start",
            "dg_id",
            "player_name",
            FEATURE_COLUMN,
        ]
    ].sort_values(["start", "event_key", "dg_id"], kind="mergesort")


def load_app_rounds(db_path: Path, min_year: int | None = None) -> pd.DataFrame:
    """Read the minimal historical APP columns from DataGolf SQLite."""
    where = "WHERE tour='pga'"
    parameters: tuple[Any, ...] = ()
    if min_year is not None:
        where += " AND year>=?"
        parameters = (int(min_year),)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rounds = pd.read_sql_query(
            "SELECT dg_id, player_name, event_id, year, round_num, "
            f"round_date, sg_app_adj FROM player_rounds {where}",
            connection,
            params=parameters,
        )
    rounds["round_date"] = pd.to_datetime(
        rounds["round_date"], errors="coerce"
    ).dt.normalize()
    return rounds.dropna(
        subset=["dg_id", "event_id", "year", "round_num", "round_date"]
    )


def build_event_safe_app_history(rounds: pd.DataFrame) -> pd.DataFrame:
    """Build target and prior APP summaries without current-event leakage.

    Events sharing a start date are snapshotted together.  None of their
    outcomes enter another same-date event's prior history.
    """
    required = {
        "dg_id",
        "player_name",
        "event_id",
        "year",
        "round_num",
        "round_date",
        "sg_app_adj",
    }
    missing = sorted(required - set(rounds.columns))
    if missing:
        raise ValueError(f"APP round history is missing columns: {missing}")

    data = rounds.copy()
    data["dg_id"] = pd.to_numeric(data["dg_id"], errors="coerce")
    data["event_id"] = pd.to_numeric(data["event_id"], errors="coerce")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["round_num"] = pd.to_numeric(data["round_num"], errors="coerce")
    data["sg_app_adj"] = pd.to_numeric(data["sg_app_adj"], errors="coerce")
    data["round_date"] = pd.to_datetime(
        data["round_date"], errors="coerce"
    ).dt.normalize()
    data = data.dropna(
        subset=["dg_id", "event_id", "year", "round_num", "round_date"]
    )
    data[["dg_id", "event_id", "year", "round_num"]] = data[
        ["dg_id", "event_id", "year", "round_num"]
    ].astype(int)

    event_rows: list[dict[str, Any]] = []
    event_keys = ["dg_id", "year", "event_id"]
    for keys, event in data.groupby(event_keys, sort=False):
        by_round = (
            event.groupby("round_num", as_index=False)
            .agg(
                round_date=("round_date", "min"),
                sg_app_adj=("sg_app_adj", "mean"),
            )
            .sort_values("round_num")
        )
        first_two = by_round[by_round["round_num"].isin([1, 2])]
        complete_first_two = set(first_two["round_num"].tolist()) == {1, 2}
        event_rows.append(
            {
                "dg_id": int(keys[0]),
                "year": int(keys[1]),
                "event_id": int(keys[2]),
                "player_name": _normal_name(event["player_name"].dropna().iloc[-1])
                if event["player_name"].notna().any()
                else "",
                "start": by_round["round_date"].min(),
                "target_app": float(first_two["sg_app_adj"].mean())
                if complete_first_two and first_two["sg_app_adj"].notna().all()
                else np.nan,
                "round_values": by_round["sg_app_adj"].dropna().astype(float).tolist(),
            }
        )

    if not event_rows:
        return pd.DataFrame(
            columns=[
                "dg_id",
                "year",
                "event_id",
                "player_name",
                "start",
                "target_app",
                *BASE_COLUMNS,
            ]
        )

    events = pd.DataFrame(event_rows).sort_values(
        ["dg_id", "start", "event_id"], kind="mergesort"
    )
    output: list[dict[str, Any]] = []
    for _, player in events.groupby("dg_id", sort=False):
        history: deque[float] = deque(maxlen=50)
        for _, same_date in player.groupby("start", sort=True):
            values = np.asarray(history, dtype=float)
            prior_mean = float(values.mean()) if len(values) >= 8 else np.nan
            prior_sd = float(values.std(ddof=1)) if len(values) >= 12 else np.nan
            for row in same_date.to_dict(orient="records"):
                row["prior_app_mean50"] = prior_mean
                row["prior_app_sd50"] = prior_sd
                output.append(row)
            for round_values in same_date["round_values"]:
                history.extend(round_values)

    return pd.DataFrame(output).drop(columns="round_values")


def assemble_cross_track_training_panel(
    features: pd.DataFrame, app_history: pd.DataFrame
) -> pd.DataFrame:
    """Attach event-safe targets/baselines to the R1 shot snapshots."""
    keys = ["dg_id", "year", "event_id"]
    history_columns = [*keys, "target_app", *BASE_COLUMNS]
    if app_history.duplicated(keys).any():
        raise ValueError("APP history has duplicate event/player rows")
    panel = features.merge(
        app_history[history_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    return panel.sort_values(["start", "event_key", "dg_id"], kind="mergesort")


def _fit_cross_track_coefficient(
    train: pd.DataFrame, ridge_alpha: float
) -> dict[str, Any]:
    """Fit the incremental raw-unit feature coefficient on a prior-only set."""
    data = train.dropna(subset=["target_app"]).copy()
    data["_fit_feature"] = data[FEATURE_COLUMN] - data.groupby(
        "event_key", sort=False
    )[FEATURE_COLUMN].transform("median")
    columns = [*BASE_COLUMNS, "_fit_feature"]
    values = data[columns].replace([np.inf, -np.inf], np.nan)
    medians = values.median()
    if not np.isfinite(medians["_fit_feature"]):
        return {"status": "no_feature_history", "fit_rows": len(data)}
    values = values.fillna(medians)
    means = values.mean()
    stds = values.std().replace(0.0, 1.0).fillna(1.0)
    if not np.isfinite(stds["_fit_feature"]) or stds["_fit_feature"] <= 0:
        return {"status": "no_feature_variance", "fit_rows": len(data)}
    design = ((values - means) / stds).to_numpy(float)
    model = Ridge(alpha=float(ridge_alpha))
    model.fit(design, data["target_app"].to_numpy(float))
    raw_coefficient = float(model.coef_[-1] / stds["_fit_feature"])
    return {
        "status": "fitted",
        "fit_rows": int(len(data)),
        "fit_events": int(data["event_key"].nunique()),
        "fit_feature_rows": int(data[FEATURE_COLUMN].notna().sum()),
        "fit_feature_coverage": float(data[FEATURE_COLUMN].notna().mean()),
        "coefficient": raw_coefficient,
        "standardized_coefficient": float(model.coef_[-1]),
        "fit_feature_median": float(medians["_fit_feature"]),
        "fit_feature_mean": float(means["_fit_feature"]),
        "fit_feature_std": float(stds["_fit_feature"]),
    }


def build_cross_track_factor_panel(
    panel: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    ridge_alpha: float = 10.0,
    min_fit_rows: int = 500,
    min_fit_events: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create walk-forward factors for target events in ``[start, end]``."""
    required = {
        "event_key",
        "year",
        "event_id",
        "start",
        "dg_id",
        "player_name",
        FEATURE_COLUMN,
        "target_app",
        *BASE_COLUMNS,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Cross-track training panel is missing columns: {missing}")

    data = panel.copy()
    data["start"] = pd.to_datetime(data["start"], errors="raise").dt.normalize()
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if first > last:
        raise ValueError("start must be on or before end")
    target = data[data["start"].between(first, last, inclusive="both")].copy()
    if target.empty:
        raise ValueError(f"No shot feature rows between {first.date()} and {last.date()}")

    outputs: list[pd.DataFrame] = []
    fit_audit: list[dict[str, Any]] = []
    for target_start, date_rows in target.groupby("start", sort=True):
        train = data[data["start"].lt(target_start)].copy()
        usable = train.dropna(subset=["target_app"])
        fit_events = int(usable["event_key"].nunique())
        fit_max_start = usable["start"].max() if len(usable) else pd.NaT
        if len(usable) < min_fit_rows or fit_events < min_fit_events:
            fit: dict[str, Any] = {
                "status": "insufficient_history",
                "fit_rows": int(len(usable)),
                "fit_events": fit_events,
            }
        else:
            fit = _fit_cross_track_coefficient(usable, ridge_alpha)
            fit.setdefault("fit_events", fit_events)

        coefficient = float(fit.get("coefficient", np.nan))
        for event_key, event in date_rows.groupby("event_key", sort=False):
            event = event.copy()
            field_mean = float(event[FEATURE_COLUMN].mean())
            event["factor_name"] = FACTOR_NAME
            event["factor_raw"] = event[FEATURE_COLUMN]
            event["factor_available"] = event[FEATURE_COLUMN].notna()
            event["factor_field_mean"] = field_mean
            event["factor_centered"] = (
                event[FEATURE_COLUMN] - field_mean
            ).fillna(0.0)
            event["factor_delta_sg_per_round"] = (
                coefficient * event["factor_centered"]
                if np.isfinite(coefficient)
                else np.nan
            )
            event["coefficient_sg_per_unit"] = coefficient
            event["coefficient_fit_status"] = fit["status"]
            event["coefficient_fit_rows"] = int(fit.get("fit_rows", 0))
            event["coefficient_fit_events"] = int(fit.get("fit_events", 0))
            event["coefficient_fit_feature_rows"] = int(
                fit.get("fit_feature_rows", 0)
            )
            event["coefficient_fit_feature_coverage"] = float(
                fit.get("fit_feature_coverage", np.nan)
            )
            event["coefficient_fit_max_start"] = fit_max_start
            event["coefficient_fit_strictly_prior"] = bool(
                pd.isna(fit_max_start) or fit_max_start < target_start
            )
            event["delta_application"] = "app_mean_and_total_mean"
            outputs.append(event)
            fit_audit.append(
                {
                    "event_key": event_key,
                    "start": target_start,
                    "fit_status": fit["status"],
                    "fit_rows": int(fit.get("fit_rows", 0)),
                    "fit_events": int(fit.get("fit_events", 0)),
                    "fit_max_start": fit_max_start,
                    "coefficient_sg_per_unit": coefficient,
                    "field_rows": int(len(event)),
                    "feature_rows": int(event[FEATURE_COLUMN].notna().sum()),
                }
            )

    result = pd.concat(outputs, ignore_index=True)
    result = result.drop(columns=[FEATURE_COLUMN, "target_app", *BASE_COLUMNS])
    result = result.sort_values(
        ["start", "event_key", "player_name"], kind="mergesort"
    ).reset_index(drop=True)
    leakage_violations = int((~result["coefficient_fit_strictly_prior"]).sum())
    if leakage_violations:
        raise AssertionError(
            f"Detected {leakage_violations} rows with same/future coefficient data"
        )

    fitted = result["coefficient_fit_status"].eq("fitted")
    available = result["factor_available"]
    deltas = result.loc[fitted & available, "factor_delta_sg_per_round"].dropna()
    event_delta_sums = result.loc[fitted].groupby("event_key")[
        "factor_delta_sg_per_round"
    ].sum(min_count=1)
    audit: dict[str, Any] = {
        "contract": "leakage_safe_shot_mean_factor_panel_v1",
        "factor_name": FACTOR_NAME,
        "target_definition": "mean future R1/R2 adjusted APP SG per round",
        "coefficient_model": "ridge_incremental_on_prior_app_mean50_and_sd50",
        "ridge_alpha": float(ridge_alpha),
        "fit_event_rule": "fit rows require event start strictly before target start",
        "concurrent_event_rule": "same-start events share an earlier-only fit",
        "training_feature_centering": "within-event median",
        "deployment_feature_centering": "within-event available-player mean",
        "missing_feature_policy": "neutral zero centered factor and zero delta when fit exists",
        "target_start": first.date().isoformat(),
        "target_end": last.date().isoformat(),
        "source_panel_start_min": data["start"].min().date().isoformat(),
        "source_panel_start_max": data["start"].max().date().isoformat(),
        "observed_target_start_min": result["start"].min().date().isoformat(),
        "observed_target_start_max": result["start"].max().date().isoformat(),
        "rows": int(len(result)),
        "events": int(result["event_key"].nunique()),
        "players": int(result["dg_id"].nunique()),
        "events_by_year": {
            str(int(key)): int(value)
            for key, value in result.groupby("year")["event_key"].nunique().items()
        },
        "feature_rows": int(available.sum()),
        "feature_coverage": float(available.mean()),
        "fitted_rows": int(fitted.sum()),
        "fitted_events": int(result.loc[fitted, "event_key"].nunique()),
        "fit_status_counts": {
            str(key): int(value)
            for key, value in result.groupby("coefficient_fit_status")[
                "event_key"
            ].nunique().items()
        },
        "coefficient_min": float(result.loc[fitted, "coefficient_sg_per_unit"].min())
        if fitted.any()
        else None,
        "coefficient_max": float(result.loc[fitted, "coefficient_sg_per_unit"].max())
        if fitted.any()
        else None,
        "delta_min_sg_per_round": float(deltas.min()) if len(deltas) else None,
        "delta_max_sg_per_round": float(deltas.max()) if len(deltas) else None,
        "max_abs_event_delta_sum": float(event_delta_sums.abs().max())
        if len(event_delta_sums)
        else None,
        "leakage_violations": leakage_violations,
        "distance_lie_status": (
            "not_built: aggregate archived JSON is not a player/event factor panel; "
            "use standardize_distance_lie_panel on an explicit build_panel export"
        ),
        "event_fits": fit_audit,
    }
    return result, audit


def standardize_distance_lie_panel(
    frame: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    value_column: str = DISTANCE_LIE_VALUE_COLUMN,
) -> pd.DataFrame:
    """Standardize a genuine archived distance x lie player/event export.

    ``frame`` must be the row-level output of
    ``approach_lie_course_fit_extended.build_panel``.  Aggregate JSON results
    intentionally fail this contract.
    """
    required = {"event_key", "source_player_id", "player_name", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Distance x lie input must be a genuine player/event panel; "
            f"missing columns: {missing}"
        )
    roster_required = {
        "event_key",
        "year",
        "event_id",
        "start",
        "dg_id",
        "player_name",
    }
    roster_missing = sorted(roster_required - set(roster.columns))
    if roster_missing:
        raise ValueError(f"Roster is missing columns: {roster_missing}")

    factor = frame[["event_key", "player_name", value_column]].copy()
    factor["_name"] = factor["player_name"].map(_normal_name)
    factor = factor.drop(columns="player_name")
    if factor.duplicated(["event_key", "_name"]).any():
        raise ValueError("Distance x lie panel has duplicate event/player rows")
    base = roster[list(roster_required)].drop_duplicates(
        ["event_key", "dg_id"]
    ).copy()
    base["_name"] = base["player_name"].map(_normal_name)
    output = base.merge(
        factor,
        on=["event_key", "_name"],
        how="left",
        validate="one_to_one",
    ).drop(columns="_name")
    output["factor_name"] = "app_distance_lie_physical_course_fit_exact8_k400"
    output["factor_raw"] = pd.to_numeric(output[value_column], errors="coerce")
    output["factor_available"] = output["factor_raw"].notna()
    output["factor_centered"] = (
        output["factor_raw"]
        - output.groupby("event_key")["factor_raw"].transform("mean")
    ).fillna(0.0)
    # The archived value is already expressed as a direct SG/round course-fit
    # adjustment; do not refit it here or borrow an aggregate OOS coefficient.
    output["factor_delta_sg_per_round"] = output["factor_centered"]
    output["coefficient_sg_per_unit"] = 1.0
    output["coefficient_fit_status"] = "direct_archived_formula"
    output["coefficient_fit_strictly_prior"] = True
    output["delta_application"] = "app_mean_and_total_mean"
    return output.drop(columns=value_column).sort_values(
        ["start", "event_key", "player_name"], kind="mergesort"
    )


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("Output suffix must be .csv, .parquet, or .pq")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--start", required=True, help="First target start, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last target start, YYYY-MM-DD")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-fit-rows", type=int, default=500)
    parser.add_argument("--min-fit-events", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()

    features = load_pre_event_cross_track_features(args.features)
    earliest_feature_year = int(features["year"].min())
    history = build_event_safe_app_history(
        load_app_rounds(args.db, min_year=earliest_feature_year - 3)
    )
    training = assemble_cross_track_training_panel(features, history)
    output, audit = build_cross_track_factor_panel(
        training,
        start=args.start,
        end=args.end,
        ridge_alpha=args.ridge_alpha,
        min_fit_rows=args.min_fit_rows,
        min_fit_events=args.min_fit_events,
    )
    audit_path = args.audit or args.output.with_suffix(".audit.json")
    _write_frame(output, args.output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, default=_json_default), encoding="utf-8"
    )
    print(
        f"Wrote {len(output):,} player-events across "
        f"{output['event_key'].nunique():,} events to {args.output}"
    )
    print(
        f"Feature coverage {audit['feature_coverage']:.1%}; "
        f"leakage violations {audit['leakage_violations']}"
    )
    print(f"Audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

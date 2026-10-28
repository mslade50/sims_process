"""Build frozen pre-event shot-process features for the current BMW field.

This is a shadow-only research export.  It reads the August R2 shot archive,
uses only completed events before the BMW start date, and writes a separate
CSV that no production script currently consumes.

Two feature families are emitted:

* per-category, span-50 EMA pooled within-round shot variance multiplied by
  the player's expected category shot count per round; and
* the existing approach cross-track RMS definition over the player's last 16
  completed rounds, weighted by qualifying directional shots.

The values are tournament-frozen: all BMW rounds should receive the same
pre-event snapshot.  The variance columns are estimates for one round, not
four-round tournament variances.
"""

from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_DG_DB = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"
)
DEFAULT_SIM_PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
DEFAULT_FIELD = REPO / "field_updates.csv"
DEFAULT_DISTS = REPO / "dashboard_data" / "this_week_dists_v2.csv"
DEFAULT_PREDICTIONS = REPO / "final_predictions_bmw.csv"
DEFAULT_OUTPUT = REPO / "bmw_shot_features_shadow.csv"
DEFAULT_META = REPO / "bmw_shot_features_shadow.meta.json"

CATEGORIES = ("ott", "app", "arg", "putt")
SPAN = 50
APP_WINDOW_ROUNDS = 16
APP_MIN_WEIGHT = 40
VARIANCE_SHRINK_K = 50.0
VARIANCE_BLEND_WEIGHT = 0.50
SHOT_KEYS = ["event_key", "source_player_id", "round_no"]

# Frozen 2026-08-19 after the leakage-safe discovery/confirmation decision.
# Reproduce or audit with --fit-app-coefficient-only; normal weekly builds do
# not refit it as new rows arrive.
APP_CROSS_BETA_SG_PER_YD = -0.09819528139411673
APP_CROSS_STANDARDIZED_BETA_SG = -0.07614618280763796
APP_CROSS_TRAINING_CENTERED_MEAN_YD = 0.059659661201762575
APP_CROSS_TRAINING_CENTERED_STD_YD = 0.7754566383084899
APP_CROSS_FIT_ROWS = 25_846
APP_CROSS_FIT_EVENTS = 235


def _norm_name(value: object) -> str:
    return str(value).strip().lower()


def _load_name_replacements(path: Path) -> dict[str, str]:
    """Read the literal mapping without importing sim_inputs side effects."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "name_replacements"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        return {
            _norm_name(key): _norm_name(replacement)
            for key, replacement in value.items()
        }
    raise ValueError(f"No literal name_replacements mapping found in {path}")


def _event_identity(series: pd.Series) -> pd.DataFrame:
    parsed = series.astype(str).str.extract(r"^[^:]+:R(\d{4})(\d+)$")
    parsed.columns = ["season_from_key", "event_id_from_key"]
    for column in parsed:
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce").astype(
            "Int64"
        )
    return parsed


def fit_app_cross_coefficient(sim_prep: Path) -> dict[str, object]:
    """Refit the confirmed incremental APP coefficient for freeze/verification."""
    from sklearn.linear_model import Ridge

    from shot_feature_predictive_study import MatrixBuilder, _center_within_event
    from shot_process_category_study import (
        add_event_safe_history,
        load_feature_snapshot,
        load_rounds,
    )

    feature = "app_dir_rms_cross_trail16"
    target = "target_app"
    baseline = ["prior_app_mean50", "prior_app_sd50"]
    features = load_feature_snapshot(sim_prep)
    history = add_event_safe_history(load_rounds(sim_prep))
    panel = history.merge(
        features,
        on=["dg_id", "event_id", "year"],
        how="inner",
        validate="one_to_one",
    )
    data = _center_within_event(panel, [feature]).dropna(subset=[target]).copy()
    builder = MatrixBuilder([*baseline, feature])
    design = builder.fit_transform(data)
    model = Ridge(alpha=10.0)
    model.fit(design, data[target].to_numpy(float))
    feature_index = len(baseline)
    standardized_coefficient = float(model.coef_[feature_index])
    feature_scale = float(builder.stds[feature])
    raw_coefficient = standardized_coefficient / feature_scale
    return {
        "fit_rows": len(data),
        "fit_events": int(data[["year", "event_id"]].drop_duplicates().shape[0]),
        "fit_year_min": int(data["year"].min()),
        "fit_year_max": int(data["year"].max()),
        "feature_coverage": float(data[feature].notna().mean()),
        "ridge_alpha": 10.0,
        "feature_definition": feature,
        "target_definition": "mean future R1/R2 adjusted approach SG per round",
        "baseline_columns": baseline,
        "historical_centering": "subtract event median before fit",
        "feature_imputation_median_centered_yd": float(builder.medians[feature]),
        "feature_training_mean_centered_yd": float(builder.means[feature]),
        "feature_training_std_centered_yd": feature_scale,
        "standardized_coefficient_sg": standardized_coefficient,
        "raw_coefficient_sg_per_cross_track_yd": raw_coefficient,
        "model_intercept_sg": float(model.intercept_),
    }


def load_field(path: Path) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    field = pd.read_csv(path, low_memory=False)
    required = {
        "player_name", "player_num", "dg_id", "event_id", "date_start",
        "event_name",
    }
    missing = sorted(required - set(field.columns))
    if missing:
        raise ValueError(f"Field file is missing columns: {missing}")
    event_ids = pd.to_numeric(field["event_id"], errors="raise").unique()
    starts = pd.to_datetime(field["date_start"], errors="raise").dt.normalize().unique()
    if len(event_ids) != 1 or len(starts) != 1:
        raise ValueError("Field file must contain exactly one event and start date")
    if not field["event_name"].astype(str).str.contains("BMW", case=False).all():
        raise ValueError("Refusing to build a BMW file from a non-BMW roster")
    out = field[["player_name", "player_num", "dg_id"]].copy()
    out["source_player_name"] = out["player_name"].map(_norm_name)
    replacements = _load_name_replacements(REPO / "sim_inputs.py")
    out["player_name"] = out["source_player_name"].replace(replacements)
    out["source_player_id"] = (
        pd.to_numeric(out["player_num"], errors="raise").astype("Int64").astype(str)
    )
    out["dg_id"] = pd.to_numeric(out["dg_id"], errors="raise").astype(int)
    if out["source_player_id"].duplicated().any():
        raise ValueError("BMW source_player_id values are not unique")
    return (
        out.drop(columns="player_num").sort_values("player_name").reset_index(drop=True),
        pd.Timestamp(starts[0]),
        int(event_ids[0]),
    )


def load_distribution_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"player_name", "category_clean", "mean", "std"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Distribution file is missing columns: {missing}")
    frame["player_name"] = frame["player_name"].map(_norm_name)
    replacements = _load_name_replacements(REPO / "sim_inputs.py")
    frame["player_name"] = frame["player_name"].replace(replacements)
    frame["category"] = frame["category_clean"].str.removeprefix("sg_")
    frame = frame[frame["category"].isin(CATEGORIES)].copy()
    if frame.duplicated(["player_name", "category"]).any():
        raise ValueError("Distribution baseline has duplicate player/category rows")
    wide = frame.pivot(
        index="player_name", columns="category", values=["mean", "std"]
    )
    wide.columns = [
        f"{category}_dist_{metric}_raw" for metric, category in wide.columns
    ]
    wide = wide.reset_index()
    for category in CATEGORIES:
        wide[f"{category}_dist_var_raw"] = wide[
            f"{category}_dist_std_raw"
        ] ** 2
    return wide


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if not {"player_name", "my_pred"}.issubset(frame.columns):
        raise ValueError("Prediction file must contain player_name and my_pred")
    frame["player_name"] = frame["player_name"].map(_norm_name)
    replacements = _load_name_replacements(REPO / "sim_inputs.py")
    frame["player_name"] = frame["player_name"].replace(replacements)
    if frame["player_name"].duplicated().any():
        raise ValueError("Prediction file has duplicate player names")
    return frame[["player_name", "my_pred"]].rename(
        columns={"my_pred": "my_pred_raw"}
    )


def add_calibrated_shadow_columns(
    output: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Median-scale shot variances, blend 50/50, and add frozen APP delta."""
    out = output.copy()
    category_diagnostics: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        baseline = f"{category}_dist_var_raw"
        shot = f"{category}_shot_indep_var50_shrunk"
        out[baseline] = pd.to_numeric(out[baseline], errors="raise")
        out[shot] = pd.to_numeric(out[shot], errors="raise")
        baseline_median = float(out[baseline].median())
        shot_median = float(out[shot].median())
        if not np.isfinite(shot_median) or shot_median <= 0:
            raise ValueError(f"Invalid BMW shot-variance median for {category}")
        scale = baseline_median / shot_median
        scaled = f"{category}_shot_indep_var50_shrunk_median_scaled"
        blended_var = f"{category}_shadow_var_blend50"
        blended_std = f"{category}_shadow_std_blend50"
        out[f"{category}_shot_variance_median_scale"] = scale
        out[scaled] = out[shot] * scale
        out[blended_var] = (
            (1.0 - VARIANCE_BLEND_WEIGHT) * out[baseline]
            + VARIANCE_BLEND_WEIGHT * out[scaled]
        )
        out[blended_std] = np.sqrt(out[blended_var].clip(lower=0.0))
        category_diagnostics[category] = {
            "baseline_variance_field_median": baseline_median,
            "shot_variance_field_median_before_scale": shot_median,
            "shot_variance_median_scale": scale,
            "shot_variance_field_median_after_scale": float(
                out[scaled].median()
            ),
            "baseline_std_min": float(out[f"{category}_dist_std_raw"].min()),
            "baseline_std_max": float(out[f"{category}_dist_std_raw"].max()),
            "shadow_blend50_std_min": float(out[blended_std].min()),
            "shadow_blend50_std_max": float(out[blended_std].max()),
        }

    base_vars = [f"{category}_dist_var_raw" for category in CATEGORIES]
    shadow_vars = [f"{category}_shadow_var_blend50" for category in CATEGORIES]
    out["baseline_indep_total_var_raw"] = out[base_vars].sum(axis=1)
    out["baseline_indep_total_std_raw"] = np.sqrt(
        out["baseline_indep_total_var_raw"]
    )
    out["shadow_indep_total_var_blend50"] = out[shadow_vars].sum(axis=1)
    out["shadow_indep_total_std_blend50"] = np.sqrt(
        out["shadow_indep_total_var_blend50"]
    )

    field_mean = float(out["app_dir_rms_cross_trail16"].mean())
    out["app_cross_field_mean_yd"] = field_mean
    out["app_cross_centered_yd"] = (
        out["app_dir_rms_cross_trail16"] - field_mean
    )
    out["app_cross_beta_sg_per_yd"] = APP_CROSS_BETA_SG_PER_YD
    out["app_cross_delta_sg_per_round"] = (
        APP_CROSS_BETA_SG_PER_YD * out["app_cross_centered_yd"]
    )
    # Move the total target and APP mean together.  The gap between category
    # sum and my_pred is unchanged, so new_sim's category recentering cannot
    # spread this APP-only delta across OTT/ARG/PUTT.
    out["my_pred_app_cross_shadow"] = (
        out["my_pred_raw"] + out["app_cross_delta_sg_per_round"]
    )
    out["app_mu_app_cross_shadow"] = (
        out["app_dist_mean_raw"] + out["app_cross_delta_sg_per_round"]
    )
    out["app_cross_recenter_gap_check"] = (
        out["my_pred_app_cross_shadow"] - out["my_pred_raw"]
        - (out["app_mu_app_cross_shadow"] - out["app_dist_mean_raw"])
    )

    deltas = out["app_cross_delta_sg_per_round"]
    best = out.loc[deltas.idxmax(), ["player_name", "app_dir_rms_cross_trail16"]]
    worst = out.loc[deltas.idxmin(), ["player_name", "app_dir_rms_cross_trail16"]]
    diagnostics: dict[str, object] = {
        "variance_blend_weight_on_shot_feature": VARIANCE_BLEND_WEIGHT,
        "variance_categories": category_diagnostics,
        "baseline_independent_total_std_min": float(
            out["baseline_indep_total_std_raw"].min()
        ),
        "baseline_independent_total_std_max": float(
            out["baseline_indep_total_std_raw"].max()
        ),
        "shadow_independent_total_std_min": float(
            out["shadow_indep_total_std_blend50"].min()
        ),
        "shadow_independent_total_std_max": float(
            out["shadow_indep_total_std_blend50"].max()
        ),
        "app_cross_fit": {
            "frozen_at": "2026-08-19",
            "fit_rows": APP_CROSS_FIT_ROWS,
            "fit_events": APP_CROSS_FIT_EVENTS,
            "fit_years": "2020-2026",
            "ridge_alpha": 10.0,
            "historical_centering": "within-event median",
            "training_centered_mean_yd": APP_CROSS_TRAINING_CENTERED_MEAN_YD,
            "training_centered_std_yd": APP_CROSS_TRAINING_CENTERED_STD_YD,
            "standardized_beta_sg": APP_CROSS_STANDARDIZED_BETA_SG,
            "raw_beta_sg_per_yd": APP_CROSS_BETA_SG_PER_YD,
            "deployment_centering": "BMW field mean",
            "bmw_field_mean_cross_track_yd": field_mean,
            "bmw_delta_mean_sg_per_round": float(deltas.mean()),
            "bmw_delta_min_sg_per_round": float(deltas.min()),
            "bmw_delta_max_sg_per_round": float(deltas.max()),
            "bmw_delta_sum_sg_per_round": float(deltas.sum()),
            "best_consistency_player": str(best["player_name"]),
            "best_consistency_cross_track_yd": float(
                best["app_dir_rms_cross_trail16"]
            ),
            "worst_consistency_player": str(worst["player_name"]),
            "worst_consistency_cross_track_yd": float(
                worst["app_dir_rms_cross_trail16"]
            ),
            "recenter_gap_max_abs": float(
                out["app_cross_recenter_gap_check"].abs().max()
            ),
        },
    }
    return out, diagnostics


def load_event_dates(db_path: Path) -> pd.DataFrame:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        dates = pd.read_sql_query(
            "SELECT year, event_id, MIN(round_date) AS event_date "
            "FROM player_rounds "
            "WHERE tour='pga' AND round_date IS NOT NULL "
            "GROUP BY year, event_id",
            connection,
        )
    dates["year"] = pd.to_numeric(dates["year"], errors="coerce").astype("Int64")
    dates["event_id"] = pd.to_numeric(
        dates["event_id"], errors="coerce"
    ).astype("Int64")
    dates["event_date"] = pd.to_datetime(
        dates["event_date"], errors="coerce"
    ).dt.normalize()
    return dates.dropna().drop_duplicates(["year", "event_id"])


def add_event_dates(frame: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = _event_identity(out["event_key"])
    out["year_for_date"] = parsed["season_from_key"]
    out["event_id_for_date"] = parsed["event_id_from_key"]
    out = out.merge(
        dates,
        left_on=["year_for_date", "event_id_for_date"],
        right_on=["year", "event_id"],
        how="left",
        validate="many_to_one",
    )
    return out.drop(
        columns=["year_for_date", "event_id_for_date", "year", "event_id"]
    )


def _read_field_rows(
    path: Path, columns: list[str], source_ids: list[str]
) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("source_player_id", "in", source_ids)],
        )
    except (TypeError, ValueError, NotImplementedError):
        frame = pd.read_parquet(path, columns=columns)
        frame["source_player_id"] = frame["source_player_id"].astype(str)
        frame = frame[frame["source_player_id"].isin(source_ids)]
    frame["source_player_id"] = frame["source_player_id"].astype(str)
    return frame


def load_round_spine(
    export: Path,
    source_ids: list[str],
    dates: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    pieces = []
    raw_rows = 0
    for year in range(2020, cutoff.year + 1):
        path = export / f"pga_{year}_player_rounds.parquet"
        frame = _read_field_rows(
            path,
            [
                "event_key", "source_player_id", "round_no", "player_name",
                "complete",
            ],
            source_ids,
        )
        raw_rows += len(frame)
        pieces.append(frame[frame["complete"].eq(1)].copy())
    rounds = pd.concat(pieces, ignore_index=True)
    rounds = add_event_dates(rounds, dates)
    missing_date_rows = int(rounds["event_date"].isna().sum())
    rounds = rounds[
        rounds["event_date"].notna() & rounds["event_date"].lt(cutoff)
    ].copy()
    rounds = rounds.drop_duplicates(SHOT_KEYS)
    return rounds, {
        "field_round_rows_read": raw_rows,
        "complete_prior_rounds": len(rounds),
        "complete_rounds_missing_event_date": missing_date_rows,
    }


def load_shot_round_stats(
    export: Path,
    source_ids: list[str],
    dates: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    pieces = []
    valid_shots = 0
    for year in range(2020, cutoff.year + 1):
        frame = _read_field_rows(
            export / f"pga_{year}_shot_sg.parquet",
            [
                *SHOT_KEYS, "sg_category", "sg_raw", "sg_valid",
                "complete_hole",
            ],
            source_ids,
        )
        frame = frame[
            frame["sg_valid"].eq(1)
            & frame["complete_hole"].eq(1)
            & frame["sg_category"].isin(CATEGORIES)
            & frame["sg_raw"].notna()
        ].copy()
        valid_shots += len(frame)
        frame["sg2"] = frame["sg_raw"] ** 2
        agg = (
            frame.groupby([*SHOT_KEYS, "sg_category"], sort=False)
            .agg(
                shot_n=("sg_raw", "size"),
                shot_sum=("sg_raw", "sum"),
                shot_sumsq=("sg2", "sum"),
            )
            .reset_index()
        )
        denom = agg["shot_n"] - 1.0
        agg["shot_var"] = (
            agg["shot_sumsq"] - agg["shot_sum"] ** 2 / agg["shot_n"]
        ) / denom
        agg.loc[agg["shot_n"].lt(2), "shot_var"] = np.nan
        pieces.append(agg)
    stats = pd.concat(pieces, ignore_index=True)
    stats = add_event_dates(stats, dates)
    stats = stats[
        stats["event_date"].notna() & stats["event_date"].lt(cutoff)
    ].copy()
    return stats, valid_shots


def build_dispersion_features(
    stats: pd.DataFrame, field: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    beta = 1.0 - 2.0 / (SPAN + 1.0)
    rows: list[dict[str, object]] = []
    for (player, category), group in stats.groupby(
        ["source_player_id", "sg_category"], sort=False
    ):
        group = group.dropna(subset=["shot_var"]).sort_values(
            ["event_date", "event_key", "round_no"], kind="mergesort"
        )
        n_rounds = len(group)
        if not n_rounds:
            continue
        ages = np.arange(n_rounds - 1, -1, -1, dtype=float)
        weights = beta**ages
        shot_n = group["shot_n"].to_numpy(float)
        degrees = shot_n - 1.0
        shot_var = group["shot_var"].to_numpy(float)
        var_num = float(np.sum(weights * degrees * shot_var))
        var_den = float(np.sum(weights * degrees))
        pooled = var_num / var_den if var_den > 0 else np.nan
        round_w = float(np.sum(weights))
        exp_n = float(np.sum(weights * shot_n) / round_w)
        round_neff = round_w**2 / float(np.sum(weights**2))
        history_neff = var_den**2 / float(np.sum(weights**2 * degrees))
        rows.append(
            {
                "source_player_id": player,
                "category": category,
                "shot_pooled_var50_raw": pooled,
                "shot_exp_n50": exp_n,
                "shot_indep_var50_raw": pooled * exp_n,
                "shot_history_neff50": history_neff,
                "shot_round_neff50": round_neff,
                "shot_history_rounds": n_rounds,
                "shot_history_n": int(group["shot_n"].sum()),
                "shot_history_last_date": group["event_date"].max(),
                "shot_history_last_event": group.loc[
                    group["event_date"].idxmax(), "event_key"
                ],
            }
        )
    long = pd.DataFrame(rows)

    # Shrink only the per-shot variance, then restore the player's expected
    # category shot count.  The BMW-field median is robust to outliers and is
    # used solely as a low-history fallback in this shadow output.
    category_priors = (
        long.groupby("category")["shot_pooled_var50_raw"].median().to_dict()
    )
    long["shot_variance_prior"] = long["category"].map(category_priors)
    long["shot_variance_reliability"] = (
        long["shot_history_neff50"]
        / (long["shot_history_neff50"] + VARIANCE_SHRINK_K)
    )
    long["shot_pooled_var50_shrunk"] = (
        long["shot_variance_reliability"] * long["shot_pooled_var50_raw"]
        + (1.0 - long["shot_variance_reliability"])
        * long["shot_variance_prior"]
    )
    long["shot_indep_var50_shrunk"] = (
        long["shot_pooled_var50_shrunk"] * long["shot_exp_n50"]
    )

    id_frame = pd.MultiIndex.from_product(
        [field["source_player_id"], CATEGORIES],
        names=["source_player_id", "category"],
    ).to_frame(index=False)
    long = id_frame.merge(
        long, on=["source_player_id", "category"], how="left"
    )
    value_columns = [
        "shot_pooled_var50_raw", "shot_pooled_var50_shrunk",
        "shot_exp_n50", "shot_indep_var50_raw", "shot_indep_var50_shrunk",
        "shot_history_neff50", "shot_round_neff50", "shot_history_rounds",
        "shot_history_n", "shot_variance_reliability",
        "shot_history_last_date", "shot_history_last_event",
    ]
    wide = long.pivot(
        index="source_player_id", columns="category", values=value_columns
    )
    wide.columns = [f"{category}_{metric}" for metric, category in wide.columns]
    wide = wide.reset_index()
    return wide, {str(key): float(value) for key, value in category_priors.items()}


def load_approach_directional_rounds(
    export: Path,
    source_ids: list[str],
    dates: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    pieces = []
    qualifying_shots = 0
    for year in range(2020, cutoff.year + 1):
        sg = _read_field_rows(
            export / f"pga_{year}_shot_sg.parquet",
            [
                *SHOT_KEYS, "hole_no", "shot_key", "sg_category", "sg_valid",
                "complete_hole", "start_lie", "start_distance_yd",
            ],
            source_ids,
        )
        sg = sg[
            sg["sg_valid"].eq(1)
            & sg["complete_hole"].eq(1)
            & sg["sg_category"].eq("app")
            & sg["start_lie"].astype(str).str.lower().isin(["tee", "fairway"])
            & pd.to_numeric(sg["start_distance_yd"], errors="coerce").between(
                100.0, 200.0
            )
        ].copy()
        shots = _read_field_rows(
            export / f"pga_{year}_shots.parquet",
            [
                "shot_key", "source_player_id", "event_type", "ball_holed",
                "start_x", "start_y", "end_x", "end_y", "course_id",
            ],
            source_ids,
        )
        shots = shots[
            shots["event_type"].eq("STROKE")
            & pd.to_numeric(shots["ball_holed"], errors="coerce").fillna(0).eq(0)
        ].copy()
        merged = sg.merge(
            shots.drop(columns="source_player_id"),
            on="shot_key",
            how="inner",
            validate="one_to_one",
        )
        geometry = pd.read_parquet(
            export / f"pga_{year}_hole_geometry.parquet",
            columns=[
                "event_key", "course_id", "round_no", "hole_no", "pin_x",
                "pin_y",
            ],
        ).drop_duplicates(["event_key", "course_id", "round_no", "hole_no"])
        merged = merged.merge(
            geometry,
            on=["event_key", "course_id", "round_no", "hole_no"],
            how="inner",
            validate="many_to_one",
        )
        coord_cols = ["start_x", "start_y", "end_x", "end_y", "pin_x", "pin_y"]
        merged = merged.dropna(subset=coord_cols)
        merged = merged[(merged["pin_x"].ne(0)) | (merged["pin_y"].ne(0))].copy()
        dx = merged["pin_x"].to_numpy(float) - merged["start_x"].to_numpy(float)
        dy = merged["pin_y"].to_numpy(float) - merged["start_y"].to_numpy(float)
        norm = np.hypot(dx, dy)
        keep = np.isfinite(norm) & (norm > 0)
        merged = merged.loc[keep].copy()
        ux = dx[keep] / norm[keep]
        uy = dy[keep] / norm[keep]
        ex = merged["end_x"].to_numpy(float) - merged["pin_x"].to_numpy(float)
        ey = merged["end_y"].to_numpy(float) - merged["pin_y"].to_numpy(float)
        merged["cross_yd"] = (ex * uy - ey * ux) / 3.0
        merged["cross_sq"] = merged["cross_yd"] ** 2
        qualifying_shots += len(merged)
        by_round = (
            merged.groupby(SHOT_KEYS, sort=False)
            .agg(
                app_dir_cross_sq_sum=("cross_sq", "sum"),
                app_dir_n=("cross_sq", "size"),
            )
            .reset_index()
        )
        by_round["app_dir_rms_cross"] = np.sqrt(
            by_round["app_dir_cross_sq_sum"] / by_round["app_dir_n"]
        )
        by_round.loc[
            by_round["app_dir_n"].lt(5), "app_dir_rms_cross"
        ] = np.nan
        pieces.append(by_round)
    rounds = pd.concat(pieces, ignore_index=True)
    rounds = add_event_dates(rounds, dates)
    rounds = rounds[
        rounds["event_date"].notna() & rounds["event_date"].lt(cutoff)
    ].copy()
    return rounds, qualifying_shots


def build_approach_feature(
    round_spine: pd.DataFrame,
    directional: pd.DataFrame,
    field: pd.DataFrame,
) -> pd.DataFrame:
    spine = round_spine.merge(
        directional[
            [*SHOT_KEYS, "app_dir_rms_cross", "app_dir_n"]
        ],
        on=SHOT_KEYS,
        how="left",
        validate="one_to_one",
    ).sort_values(
        ["source_player_id", "event_date", "event_key", "round_no"],
        kind="mergesort",
    )
    rows = []
    for player in field["source_player_id"]:
        history = spine[spine["source_player_id"].eq(player)].tail(
            APP_WINDOW_ROUNDS
        )
        valid = history["app_dir_rms_cross"].notna()
        weights = history.loc[valid, "app_dir_n"].to_numpy(float)
        values = history.loc[valid, "app_dir_rms_cross"].to_numpy(float)
        total_weight = float(weights.sum())
        feature = (
            float(np.sum(weights * values) / total_weight)
            if total_weight >= APP_MIN_WEIGHT
            else np.nan
        )
        rows.append(
            {
                "source_player_id": player,
                "app_dir_rms_cross_trail16": feature,
                "app_dir_shots_trail16": total_weight,
                "app_dir_valid_rounds_trail16": int(valid.sum()),
                "app_history_rounds_window": len(history),
                "app_history_last_date": (
                    history["event_date"].max() if len(history) else pd.NaT
                ),
                "app_history_last_event": (
                    history.iloc[-1]["event_key"] if len(history) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_new(path: Path, writer, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {path}; pass --allow-overwrite to refresh it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--dg-db", type=Path, default=DEFAULT_DG_DB)
    parser.add_argument("--sim-prep", type=Path, default=DEFAULT_SIM_PREP)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--dists", type=Path, default=DEFAULT_DISTS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--fit-app-coefficient-only", action="store_true")
    args = parser.parse_args()

    if args.fit_app_coefficient_only:
        print(json.dumps(fit_app_cross_coefficient(args.sim_prep), indent=2))
        return 0

    field, cutoff, event_id = load_field(args.field)
    source_ids = field["source_player_id"].tolist()
    dates = load_event_dates(args.dg_db)
    round_spine, round_diagnostics = load_round_spine(
        args.export, source_ids, dates, cutoff
    )
    shot_stats, valid_shots = load_shot_round_stats(
        args.export, source_ids, dates, cutoff
    )
    dispersion, priors = build_dispersion_features(shot_stats, field)
    directional, directional_shots = load_approach_directional_rounds(
        args.export, source_ids, dates, cutoff
    )
    approach = build_approach_feature(round_spine, directional, field)
    distributions = load_distribution_baseline(args.dists)
    predictions = load_predictions(args.predictions)

    output = (
        field.merge(dispersion, on="source_player_id", how="left", validate="one_to_one")
        .merge(approach, on="source_player_id", how="left", validate="one_to_one")
        .merge(distributions, on="player_name", how="left", validate="one_to_one")
        .merge(predictions, on="player_name", how="left", validate="one_to_one")
        .sort_values("player_name")
        .reset_index(drop=True)
    )
    baseline_required = [
        *[f"{category}_dist_mean_raw" for category in CATEGORIES],
        *[f"{category}_dist_std_raw" for category in CATEGORIES],
        "my_pred_raw",
    ]
    if output[baseline_required].isna().any().any():
        missing_players = output.loc[
            output[baseline_required].isna().any(axis=1), "player_name"
        ].tolist()
        raise ValueError(
            f"BMW distribution/prediction baseline is incomplete for {missing_players}"
        )
    output, calibration_diagnostics = add_calibrated_shadow_columns(output)
    output.insert(0, "event_id", event_id)
    output.insert(1, "event_start", cutoff.date().isoformat())
    output.insert(2, "feature_scope", "pre_event_frozen_per_round")
    output.insert(3, "feature_status", "shadow_only")

    variance_complete = output[
        [f"{category}_shot_indep_var50_shrunk" for category in CATEGORIES]
    ].notna().all(axis=1)
    approach_complete = output["app_dir_rms_cross_trail16"].notna()
    manifest = json.loads((args.export / "manifest.json").read_text())
    metadata = {
        "status": "shadow_only",
        "event": "BMW Championship",
        "event_id": event_id,
        "event_start": cutoff.date().isoformat(),
        "feature_scope": "one-round estimates frozen before R1 for R1-R4",
        "field_path": str(args.field.resolve()),
        "distribution_baseline_path": str(args.dists.resolve()),
        "distribution_baseline_mtime": pd.Timestamp(
            args.dists.stat().st_mtime, unit="s", tz="UTC"
        ).isoformat(),
        "prediction_baseline_path": str(args.predictions.resolve()),
        "prediction_baseline_mtime": pd.Timestamp(
            args.predictions.stat().st_mtime, unit="s", tz="UTC"
        ).isoformat(),
        "r2_export_path": str(args.export.resolve()),
        "r2_export_generated_at": manifest.get("generated_at"),
        "dg_event_date_path": str(args.dg_db.resolve()),
        "players": len(output),
        "players_all_four_variance_categories": int(variance_complete.sum()),
        "players_app_cross_track": int(approach_complete.sum()),
        "players_both_feature_families": int(
            (variance_complete & approach_complete).sum()
        ),
        "valid_field_shots_read": valid_shots,
        "qualifying_directional_app_shots": directional_shots,
        "max_prior_event_date": (
            round_spine["event_date"].max().date().isoformat()
            if len(round_spine)
            else None
        ),
        "variance_span_rounds": SPAN,
        "variance_shrink_effective_df_k": VARIANCE_SHRINK_K,
        "variance_field_median_priors": priors,
        "app_window_completed_rounds": APP_WINDOW_ROUNDS,
        "app_min_directional_shots": APP_MIN_WEIGHT,
        **calibration_diagnostics,
        **round_diagnostics,
    }

    _write_new(
        args.output,
        lambda path: output.to_csv(path, index=False),
        args.allow_overwrite,
    )
    _write_new(
        args.meta,
        lambda path: path.write_text(json.dumps(metadata, indent=2) + "\n"),
        args.allow_overwrite,
    )
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {args.meta.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

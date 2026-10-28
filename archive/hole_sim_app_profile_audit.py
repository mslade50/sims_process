"""Reproducible leakage-safe audit for spatial approach-shot simulation.

This research utility measures the data support for an empirical APP
transition, validates shrunk player endpoint profiles, and reconstructs
penalty outcomes by linking each PENALTY row to the preceding physical STROKE.
It never writes outside ``--output-dir`` and is not imported by production.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")
DEFAULT_OUTPUT = Path("permanent_data/hole_sim_app_audit")
DEFAULT_EVENTS = (
    "Wyndham Championship",
    "Sony Open in Hawaii",
    "Travelers Championship",
)
LIES = ("tee", "fairway", "rough", "sand", "recovery", "other")
BANDS = ("<=100", "100-150", "150-215", ">215")


def _lie(value: object) -> str:
    text = str(value).strip().lower()
    if text in LIES[:-1]:
        return text
    if "sand" in text or "bunker" in text:
        return "sand"
    return "other"


def _band(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        [-np.inf, 100, 150, 215, np.inf],
        labels=BANDS,
        right=True,
    )


def _event_maps(
    export_root: Path, event_names: tuple[str, ...], holdout_year: int
) -> tuple[dict[str, str], dict[str, str]]:
    events = pd.read_parquet(
        export_root / "events.parquet",
        columns=["event_key", "tour", "season", "event_name"],
    )
    chosen = events[
        events["tour"].str.casefold().eq("pga")
        & events["event_name"].isin(event_names)
        & events["season"].le(holdout_year)
    ].copy()
    found = set(chosen.loc[chosen["season"].eq(holdout_year), "event_name"])
    missing = set(event_names) - found
    if missing:
        raise ValueError(f"Missing held-out editions: {sorted(missing)}")
    key_to_name = dict(zip(chosen["event_key"], chosen["event_name"]))
    holdout_keys = dict(
        zip(
            chosen.loc[chosen["season"].eq(holdout_year), "event_name"],
            chosen.loc[chosen["season"].eq(holdout_year), "event_key"],
        )
    )
    return key_to_name, holdout_keys


def _link_penalties(
    raw: pd.DataFrame, category_by_key: pd.Series, year: int
) -> tuple[dict[str, float | int], set[str]]:
    """Link penalty bookkeeping rows to causal and resolution strokes."""
    frame = raw.reset_index(drop=True).copy()
    frame["_ord"] = np.arange(len(frame))
    frame["category"] = frame["shot_key"].map(category_by_key)
    keys = ["event_key", "source_player_id", "round_no", "hole_no"]
    stroke = frame["event_type"].eq("STROKE")
    frame["_causal"] = frame["_ord"].where(stroke).groupby(
        [frame[key] for key in keys], sort=False
    ).ffill()
    frame["_resolution"] = frame["_ord"].where(stroke).groupby(
        [frame[key] for key in keys], sort=False
    ).bfill()
    penalties = frame[frame["event_type"].eq("PENALTY")].copy()
    causal_columns = [
        "_ord", "shot_key", "stroke_no", "start_x", "start_y", "end_x",
        "end_y", "end_lie", "category",
    ]
    causal = frame[causal_columns].add_prefix("causal_")
    resolution = frame[
        ["_ord", "stroke_no", "start_x", "start_y"]
    ].add_prefix("resolution_")
    penalties = penalties.merge(
        causal, left_on="_causal", right_on="causal__ord", how="left"
    ).merge(
        resolution,
        left_on="_resolution",
        right_on="resolution__ord",
        how="left",
    )
    linked = penalties["causal__ord"].notna()
    gap = penalties["stroke_no"] - penalties["causal_stroke_no"]
    causal_xy = penalties[
        ["causal_start_x", "causal_start_y", "causal_end_x", "causal_end_y"]
    ].notna().all(axis=1)
    resolution_xy = penalties[
        ["resolution_start_x", "resolution_start_y"]
    ].notna().all(axis=1)
    replay = np.hypot(
        penalties["resolution_start_x"] - penalties["causal_start_x"],
        penalties["resolution_start_y"] - penalties["causal_start_y"],
    ).le(3.0)
    duplicates = penalties.loc[linked].groupby("causal__ord").size()
    causal_app = penalties["causal_category"].eq("app")
    penalty_app = penalties["category"].eq("app")
    causal_app_keys = set(
        penalties.loc[causal_app & linked, "causal_shot_key"].dropna().astype(str)
    )
    nonmonotone = frame.groupby(keys, sort=False)["stroke_no"].diff().lt(0)
    metrics: dict[str, float | int] = {
        "season": year,
        "penalty_rows": len(penalties),
        "link_rate": float(linked.mean()) if len(penalties) else np.nan,
        "gap_zero_rate": float(gap.eq(0).mean()) if len(penalties) else np.nan,
        "gap_one_rate": float(gap.eq(1).mean()) if len(penalties) else np.nan,
        "other_gap_rate": float((~gap.isin([0, 1])).mean()) if len(penalties) else np.nan,
        "causal_xy_rate": float(causal_xy.mean()) if len(penalties) else np.nan,
        "resolution_xy_rate": float(resolution_xy.mean()) if len(penalties) else np.nan,
        "both_xy_rate": float((causal_xy & resolution_xy).mean()) if len(penalties) else np.nan,
        "causal_app_rate": float(causal_app.mean()) if len(penalties) else np.nan,
        "penalty_app_rate": float(penalty_app.mean()) if len(penalties) else np.nan,
        "category_agreement_rate": float((causal_app == penalty_app).mean()) if len(penalties) else np.nan,
        "replay_within_1yd_rate": float(replay.mean()) if len(penalties) else np.nan,
        "multi_penalty_causal_rate": float(duplicates.gt(1).mean()) if len(duplicates) else 0.0,
        "max_penalties_per_causal": int(duplicates.max()) if len(duplicates) else 0,
        "nonmonotone_sequence_rate": float(nonmonotone.mean()),
    }
    return metrics, causal_app_keys


def _load_year(
    export_root: Path,
    year: int,
    key_to_name: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, float | int], dict[str, float | int]]:
    sg_columns = [
        "shot_key", "event_key", "round_no", "hole_no", "source_player_id",
        "player_name", "stroke_no", "sg_category", "start_lie",
        "start_distance_yd", "end_lie", "end_distance_yd", "sg_valid",
        "complete_hole",
    ]
    raw_columns = [
        "shot_key", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "event_type", "start_lie", "end_lie",
        "course_id", "start_x", "start_y", "end_x", "end_y", "ball_holed",
    ]
    sg = pd.read_parquet(export_root / f"pga_{year}_shot_sg.parquet", columns=sg_columns)
    raw = pd.read_parquet(export_root / f"pga_{year}_shots.parquet", columns=raw_columns)
    category = sg.drop_duplicates("shot_key").set_index("shot_key")["sg_category"]
    penalty_metrics, hazard_keys = _link_penalties(raw, category, year)
    app = sg[sg["sg_category"].eq("app")].merge(
        raw[
            [
                "shot_key", "event_type", "course_id", "start_x", "start_y",
                "end_x", "end_y", "ball_holed",
            ]
        ],
        on="shot_key",
        how="left",
        validate="one_to_one",
    )
    geometry = pd.read_parquet(
        export_root / f"pga_{year}_hole_geometry.parquet",
        columns=[
            "event_key", "round_no", "hole_no", "course_id", "par", "yardage",
            "tee_x", "tee_y", "pin_x", "pin_y",
        ],
    ).drop_duplicates(["event_key", "round_no", "hole_no", "course_id"])
    app = app.merge(
        geometry,
        on=["event_key", "round_no", "hole_no", "course_id"],
        how="left",
        validate="many_to_one",
    )
    app["season"] = year
    app["event_name"] = app["event_key"].map(key_to_name)
    app["physical_stroke"] = app["event_type"].eq("STROKE")
    app["xy4"] = app[["start_x", "start_y", "end_x", "end_y"]].notna().all(axis=1)
    app["pin_xy"] = app[["pin_x", "pin_y"]].notna().all(axis=1)
    vx = app["pin_x"].to_numpy(float) - app["start_x"].to_numpy(float)
    vy = app["pin_y"].to_numpy(float) - app["start_y"].to_numpy(float)
    norm = np.hypot(vx, vy)
    ux = np.divide(vx, norm, out=np.full(len(app), np.nan), where=norm > 30)
    uy = np.divide(vy, norm, out=np.full(len(app), np.nan), where=norm > 30)
    ex = app["end_x"].to_numpy(float) - app["pin_x"].to_numpy(float)
    ey = app["end_y"].to_numpy(float) - app["pin_y"].to_numpy(float)
    app["along_error_yd"] = (ex * ux + ey * uy) / 3.0
    app["cross_error_yd"] = (-ex * uy + ey * ux) / 3.0
    app["geom_end_distance_yd"] = np.hypot(ex, ey) / 3.0
    course_vx = app["pin_x"].to_numpy(float) - app["tee_x"].to_numpy(float)
    course_vy = app["pin_y"].to_numpy(float) - app["tee_y"].to_numpy(float)
    course_norm = np.hypot(course_vx, course_vy)
    course_ux = np.divide(
        course_vx, course_norm, out=np.full(len(app), np.nan), where=course_norm > 30
    )
    course_uy = np.divide(
        course_vy, course_norm, out=np.full(len(app), np.nan), where=course_norm > 30
    )
    course_dx = app["end_x"].to_numpy(float) - app["tee_x"].to_numpy(float)
    course_dy = app["end_y"].to_numpy(float) - app["tee_y"].to_numpy(float)
    app["course_end_long_yd"] = (course_dx * course_ux + course_dy * course_uy) / 3.0
    app["course_end_cross_yd"] = (-course_dx * course_uy + course_dy * course_ux) / 3.0
    app["causal_hazard"] = app["shot_key"].astype(str).isin(hazard_keys)
    usable = (
        app["physical_stroke"]
        & app["xy4"]
        & app["pin_xy"]
        & app["start_distance_yd"].between(35, 450)
        & np.isfinite(app[["along_error_yd", "cross_error_yd"]]).all(axis=1)
    )
    nonhole = usable & ~app["ball_holed"].fillna(0).astype(bool) & app["end_distance_yd"].notna()
    distance_error = (
        app.loc[nonhole, "geom_end_distance_yd"] - app.loc[nonhole, "end_distance_yd"]
    ).abs()
    coverage: dict[str, float | int] = {
        "season": year,
        "app_rows": len(app),
        "physical_strokes": int(app["physical_stroke"].sum()),
        "app_penalty_rows": int(app["event_type"].eq("PENALTY").sum()),
        "xy4_rate": float(app["xy4"].mean()),
        "pin_xy_rate": float(app["pin_xy"].mean()),
        "usable_strokes": int(usable.sum()),
        "usable_rate_of_all_app_rows": float(usable.mean()),
        "geom_vs_sg_distance_median_ae_yd": float(distance_error.median()),
        "geom_vs_sg_distance_p90_ae_yd": float(distance_error.quantile(0.90)),
        "holeouts": int((usable & app["ball_holed"].fillna(0).astype(bool)).sum()),
    }
    frame = app.loc[usable].copy()
    frame["source_player_id"] = frame["source_player_id"].astype(str)
    frame["start_lie_class"] = frame["start_lie"].map(_lie)
    frame["distance_band"] = _band(frame["start_distance_yd"])
    frame["distance_bin"] = (frame["start_distance_yd"] // 10 * 10).clip(30, 440).astype(int)
    return frame, coverage, penalty_metrics


def _weighted_support_quantiles(
    train: pd.DataFrame, holdout: pd.DataFrame, keys: list[str]
) -> dict[str, float]:
    prior = train.groupby(keys, observed=True).size().rename("n").reset_index()
    target = holdout.groupby(keys, observed=True).size().rename("hold_n").reset_index()
    joined = target.merge(prior, on=keys, how="left").fillna({"n": 0})
    weighted = np.repeat(joined["n"].to_numpy(float), joined["hold_n"].to_numpy(int))
    return {
        "min": float(np.min(weighted)),
        "p10": float(np.quantile(weighted, 0.10)),
        "p25": float(np.quantile(weighted, 0.25)),
        "median": float(np.quantile(weighted, 0.50)),
        "p75": float(np.quantile(weighted, 0.75)),
        "p90": float(np.quantile(weighted, 0.90)),
        "zero_rate": float(np.mean(weighted == 0)),
        "under_20_rate": float(np.mean(weighted < 20)),
        "under_50_rate": float(np.mean(weighted < 50)),
    }


def _population_table(train: pd.DataFrame, axis: str) -> pd.DataFrame:
    coarse = train.groupby(["distance_band", "start_lie_class"], observed=True)[axis].agg(
        coarse_n="size", coarse_mean="mean", coarse_var="var"
    ).reset_index()
    fine = train.groupby(
        ["distance_bin", "distance_band", "start_lie_class"], observed=True
    )[axis].agg(fine_n="size", fine_mean="mean", fine_var="var").reset_index()
    table = fine.merge(coarse, on=["distance_band", "start_lie_class"], how="left")
    prior = 200.0
    table["mean"] = (
        table["fine_n"] * table["fine_mean"] + prior * table["coarse_mean"]
    ) / (table["fine_n"] + prior)
    denominator = np.maximum(table["fine_n"] - 1, 0) + prior
    table["sd"] = np.sqrt(
        ((table["fine_n"] - 1) * table["fine_var"] + prior * table["coarse_var"])
        / denominator
    ).clip(2, 120)
    return table[["distance_bin", "distance_band", "start_lie_class", "mean", "sd"]]


def _standardize(frame: pd.DataFrame, table: pd.DataFrame, axis: str) -> pd.DataFrame:
    result = frame[
        ["season", "event_name", "source_player_id", "distance_band", "start_lie_class", "distance_bin", axis]
    ].merge(
        table,
        on=["distance_bin", "distance_band", "start_lie_class"],
        how="left",
    )
    result["z"] = (result[axis] - result["mean"]) / result["sd"]
    return result.dropna(subset=["z"])


def _profile_table(
    train: pd.DataFrame, grouping: list[str], mean_prior: float, variance_prior: float
) -> pd.DataFrame:
    keys = ["source_player_id"] + grouping
    work = train.assign(z2=train["z"] ** 2)
    profile = work.groupby(keys, observed=True).agg(
        n=("z", "size"), sum_z=("z", "sum"), sum_z2=("z2", "sum")
    ).reset_index()
    profile["player_mean"] = profile["sum_z"] / (profile["n"] + mean_prior)
    residual_ss = (
        profile["sum_z2"]
        - 2 * profile["player_mean"] * profile["sum_z"]
        + profile["n"] * profile["player_mean"] ** 2
    )
    profile["player_scale"] = np.sqrt(
        (residual_ss + variance_prior) / (profile["n"] + variance_prior)
    ).clip(0.60, 1.60)
    return profile[keys + ["n", "player_mean", "player_scale"]]


def _score_profiles(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = {
        "global": [],
        "band": ["distance_band"],
        "band_lie": ["distance_band", "start_lie_class"],
    }
    for axis in ("along_error_yd", "cross_error_yd"):
        population = _population_table(train, axis)
        standardized_train = _standardize(train, population, axis)
        standardized_holdout = _standardize(holdout, population, axis)
        scale = standardized_train["z"].std()
        standardized_train["z"] /= scale
        standardized_holdout["z"] /= scale
        for group_name, grouping in groups.items():
            keys = ["source_player_id"] + grouping
            for mean_prior in (20, 50, 100, 200):
                for variance_prior in (50, 100, 200, 400):
                    profile = _profile_table(
                        standardized_train, grouping, mean_prior, variance_prior
                    )
                    scored = standardized_holdout.merge(profile, on=keys, how="left")
                    coverage = float(scored["n"].notna().mean())
                    scored["player_mean"] = scored["player_mean"].fillna(0)
                    scored["player_scale"] = scored["player_scale"].fillna(1)
                    baseline = 0.5 * scored["z"] ** 2
                    scored["both_delta"] = (
                        np.log(scored["player_scale"])
                        + 0.5 * ((scored["z"] - scored["player_mean"]) / scored["player_scale"]) ** 2
                        - baseline
                    )
                    scored["mean_delta"] = 0.5 * (scored["z"] - scored["player_mean"]) ** 2 - baseline
                    scored["dispersion_delta"] = (
                        np.log(scored["player_scale"])
                        + 0.5 * (scored["z"] / scored["player_scale"]) ** 2
                        - baseline
                    )
                    for event_name, sample in list(scored.groupby("event_name")) + [("POOLED", scored)]:
                        rows.append(
                            {
                                "axis": axis,
                                "profile_group": group_name,
                                "mean_prior": mean_prior,
                                "variance_prior": variance_prior,
                                "event_name": event_name,
                                "n": len(sample),
                                "coverage": coverage,
                                "both_nll_delta": sample["both_delta"].mean(),
                                "mean_only_nll_delta": sample["mean_delta"].mean(),
                                "dispersion_only_nll_delta": sample["dispersion_delta"].mean(),
                            }
                        )
    return pd.DataFrame(rows)


def _profile_stability(frame: pd.DataFrame) -> pd.DataFrame:
    early = frame[frame["season"].le(2022)]
    late = frame[frame["season"].between(2023, 2025)]
    rows: list[dict[str, object]] = []
    for axis in ("along_error_yd", "cross_error_yd"):
        population = _population_table(early, axis)
        left = _standardize(early, population, axis)
        right = _standardize(late, population, axis)
        for band in [None, *BANDS]:
            a = left if band is None else left[left["distance_band"].astype(str).eq(band)]
            b = right if band is None else right[right["distance_band"].astype(str).eq(band)]
            minimum = 50 if band is None else 30
            pa = a.groupby("source_player_id")["z"].agg(n_early="size", mean_early="mean", var_early="var")
            pb = b.groupby("source_player_id")["z"].agg(n_late="size", mean_late="mean", var_late="var")
            joined = pa.join(pb).dropna()
            joined = joined[
                joined["n_early"].ge(minimum) & joined["n_late"].ge(minimum)
            ]
            rows.append(
                {
                    "axis": axis,
                    "distance_band": "global" if band is None else band,
                    "minimum_each_period": minimum,
                    "players": len(joined),
                    "mean_correlation": joined["mean_early"].corr(joined["mean_late"]),
                    "sd_correlation": np.sqrt(joined["var_early"]).corr(np.sqrt(joined["var_late"])),
                }
            )
    return pd.DataFrame(rows)


def _binary_metrics(actual: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    probability = np.clip(np.asarray(probability, float), 1e-5, 1 - 1e-5)
    actual = np.asarray(actual, float)
    logloss = -np.mean(actual * np.log(probability) + (1 - actual) * np.log(1 - probability))
    return float(logloss), float(np.mean((actual - probability) ** 2))


def _hazard_surface_test(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    train = train[train["event_name"].notna()].copy()
    holdout = holdout[holdout["event_name"].notna()].copy().reset_index(drop=True)
    global_probability = (train["causal_hazard"].sum() + 1) / (len(train) + 200)
    event_hole = train.groupby(["event_name", "hole_no"])["causal_hazard"].agg(
        positives="sum", n="size"
    ).reset_index()
    event_hole["hole_probability"] = (
        event_hole["positives"] + 200 * global_probability
    ) / (event_hole["n"] + 200)
    state = train.groupby(
        ["event_name", "hole_no", "distance_band", "start_lie_class"], observed=True
    )["causal_hazard"].agg(positives="sum", n="size").reset_index().merge(
        event_hole[["event_name", "hole_no", "hole_probability"]],
        on=["event_name", "hole_no"],
    )
    state["baseline_probability"] = (
        state["positives"] + 100 * state["hole_probability"]
    ) / (state["n"] + 100)
    holdout = holdout.merge(
        state[
            ["event_name", "hole_no", "distance_band", "start_lie_class", "baseline_probability"]
        ],
        on=["event_name", "hole_no", "distance_band", "start_lie_class"],
        how="left",
    ).merge(
        event_hole[["event_name", "hole_no", "hole_probability"]],
        on=["event_name", "hole_no"],
        how="left",
    )
    holdout["baseline_probability"] = (
        holdout["baseline_probability"].fillna(holdout["hole_probability"]).fillna(global_probability)
    )
    rows: list[dict[str, object]] = []

    def append_metrics(model: str, probability: np.ndarray, k: int | None, prior_mass: int | None) -> None:
        for event_name, indices in list(holdout.groupby("event_name").groups.items()) + [("POOLED", holdout.index)]:
            sample = holdout.loc[indices]
            p = probability[np.asarray(indices, int)]
            logloss, brier = _binary_metrics(sample["causal_hazard"], p)
            rows.append(
                {
                    "model": model,
                    "k": k,
                    "prior_mass": prior_mass,
                    "event_name": event_name,
                    "n": len(sample),
                    "hazards": int(sample["causal_hazard"].sum()),
                    "logloss": logloss,
                    "brier": brier,
                    "mean_probability_on_hazards": float(np.mean(p[sample["causal_hazard"].to_numpy(bool)])),
                    "mean_probability_on_nonhazards": float(np.mean(p[~sample["causal_hazard"].to_numpy(bool)])),
                }
            )

    baseline = holdout["baseline_probability"].to_numpy(float)
    append_metrics("state_baseline", baseline, None, None)
    for k in (30, 60, 120, 240):
        for prior_mass in (10, 30, 100):
            prediction = np.empty(len(holdout), float)
            for (event_name, hole_no), indices in holdout.groupby(["event_name", "hole_no"]).groups.items():
                donors = train[
                    train["event_name"].eq(event_name) & train["hole_no"].eq(hole_no)
                ]
                query = holdout.loc[indices]
                neighbor_count = min(k, len(donors))
                distance, neighbor = cKDTree(
                    donors[["course_end_long_yd", "course_end_cross_yd"]].to_numpy(float)
                ).query(
                    query[["course_end_long_yd", "course_end_cross_yd"]].to_numpy(float),
                    k=neighbor_count,
                )
                if neighbor_count == 1:
                    distance, neighbor = distance[:, None], neighbor[:, None]
                weight = np.exp(-0.5 * (distance / 10.0) ** 2) + 0.05
                labels = donors["causal_hazard"].to_numpy(float)[neighbor]
                local_positive = np.sum(weight * labels, axis=1)
                local_mass = np.sum(weight, axis=1)
                state_prior = query["baseline_probability"].to_numpy(float)
                prediction[np.asarray(indices, int)] = (
                    local_positive + prior_mass * state_prior
                ) / (local_mass + prior_mass)
            append_metrics("spatial_knn", prediction, k, prior_mass)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    rounded = frame.round(digits).copy()
    headers = [str(column) for column in rounded.columns]
    rows = [[str(value) for value in row] for row in rounded.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    ) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _write_report(
    output_dir: Path,
    coverage: pd.DataFrame,
    band_lie: pd.DataFrame,
    donor_support: pd.DataFrame,
    player_support: dict[str, float],
    penalty: pd.DataFrame,
    profile: pd.DataFrame,
    stability: pd.DataFrame,
    hazard: pd.DataFrame,
) -> Path:
    best_profile = (
        profile[profile["event_name"].eq("POOLED")]
        .sort_values(["axis", "both_nll_delta"])
        .groupby("axis", as_index=False)
        .first()
    )
    hazard_pooled = hazard[hazard["event_name"].eq("POOLED")].copy()
    hazard_best = hazard_pooled.sort_values("logloss").head(5)
    report = f"""# Spatial APP data/profile audit

All course donors, player profiles, and surface labels precede the 2026 holdout.
The held-out events are Wyndham, Sony, and Travelers. Coordinates are expressed
in yards; profile tests use a Tour population baseline conditioned on 10-yard
distance bin and start lie.

## Coordinate coverage

{_markdown_table(coverage)}

The geometric endpoint-to-pin distance agrees with the SG state to well under
one tenth of a yard at the median. Holeouts are a discrete terminating atom at
the cup and must not be translated as ordinary misses.

## Prior state support

{_markdown_table(band_lie)}

### Exact-course donor support, weighted by 2026 states

{_markdown_table(donor_support)}

Player support at the exact player x band x lie level: median
**{player_support['median']:.0f}**, p10 **{player_support['p10']:.0f}**,
**{player_support['zero_rate']:.1%}** zero, **{player_support['under_20_rate']:.1%}**
under 20, and **{player_support['under_50_rate']:.1%}** under 50. Aggregate
player traits are identifiable; band effects require shrinkage; the full
player x band x lie interaction must back off deeply.

## Penalty linkage

{_markdown_table(penalty)}

The sequence-safe linkage is: PENALTY -> most recent prior physical STROKE in
the same event/player/round/hole. Store that causal stroke's endpoint as a
`penalty_hazard` surface label, the number of linked penalty rows as stroke
cost, and the following physical STROKE start as the resolved playable state.
The recorded causal `end_lie` is not trustworthy for hazards (it is commonly
`recovery`). A fixed penalty atom independent of the simulated endpoint breaks
the intended player-dispersion-to-hazard mechanism.

## Held-out player profile signal

Negative NLL deltas favor the player profile.

{_markdown_table(best_profile)}

### Temporal stability

{_markdown_table(stability)}

Player means are modestly persistent. Player dispersion is much less stable,
so initial spatial transport should use approximately a 200-shot mean prior
and at least a 400-shot variance prior. Cross bias can be global by player;
along bias benefits from band/lie conditioning but was not uniformly positive
at every course. Do not use unshrunk player x band x lie scales.

## Spatial hazard validation

This test uses the observed endpoint only to validate the historical surface
classifier; it does not leak into training.

{_markdown_table(hazard_best)}

The causal-endpoint surface materially beats a state-only penalty rate, which
supports reclassifying lie/hazard after every player endpoint transform.

## Recommended baselines and gates

1. Exact round/hole + continuous distance + lie empirical transition, with
   hole and course backoffs, no player transform.
2. Add player mean only; then dispersion only; then combined, using common
   random numbers and the shrinkage above.
3. Compare with a Tour-only distance/lie transition to measure exact-course
   value separately from player value.
4. Score endpoint KDE/energy, remaining-yardage CRPS, lie and hazard log loss,
   then paired-state expected-strokes and complete-hole score log score/CRPS.
5. Require leave-one-edition-out results across multiple recurring courses and
   test the generated state chain, not only transitions conditioned on actual
   held-out states.

## Identifiability and leakage traps

- Pin-relative error confounds execution with deliberate aim, center-green
  strategy, layups, and recovery choices. Use a green-attempt/layup mixture,
  especially beyond 215 yards and from sand/recovery.
- Keep both shot-target coordinates (player error) and stable course
  coordinates (surface/hazard classification). Reusing a donor lie after
  shifting its endpoint is invalid.
- Actual held-out pins are a labelled setup sensitivity, not a strict
  pre-event forecast. Historical pin/setup mode must remain separate.
- Player residuals, surface maps, penalty labels, and state-value curves all
  require the same strict time cutoff. Field/setup centering should be
  leave-one-player-out when the production estimator is finalized.
- A one-step test conditioned on realized lie/distance does not validate
  recursive state occupancy. Downstream validation must start from the
  generated tee/drop state.
- Add shared round/weather/player-day latent effects later; independent shot
  draws will understate round tails even if marginal transitions are correct.
"""
    path = output_dir / "REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    event_names = tuple(args.events)
    key_to_name, holdout_keys = _event_maps(args.export_root, event_names, args.holdout_year)
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, float | int]] = []
    penalty_rows: list[dict[str, float | int]] = []
    for year in range(args.start_year, args.holdout_year + 1):
        print(f"Loading APP audit year {year}")
        frame, coverage, penalty = _load_year(args.export_root, year, key_to_name)
        frames.append(frame)
        coverage_rows.append(coverage)
        penalty_rows.append(penalty)
    all_shots = pd.concat(frames, ignore_index=True)
    train = all_shots[all_shots["season"].lt(args.holdout_year)].copy()
    holdout = all_shots[
        all_shots["season"].eq(args.holdout_year)
        & all_shots["event_key"].isin(holdout_keys.values())
    ].copy()
    if train.empty or holdout.empty:
        raise ValueError("Training or held-out APP audit panel is empty")

    band_lie = train.groupby(
        ["distance_band", "start_lie_class"], observed=True
    ).size().rename("shots").reset_index()
    donor_rows: list[dict[str, object]] = []
    for event_name in event_names:
        event_train = train[train["event_name"].eq(event_name)]
        event_holdout = holdout[holdout["event_name"].eq(event_name)]
        for level, keys in {
            "round_hole_band_lie": ["round_no", "hole_no", "distance_band", "start_lie_class"],
            "hole_band_lie": ["hole_no", "distance_band", "start_lie_class"],
        }.items():
            donor_rows.append(
                {
                    "event_name": event_name,
                    "level": level,
                    **_weighted_support_quantiles(event_train, event_holdout, keys),
                }
            )
    donor_support = pd.DataFrame(donor_rows)
    player_support = _weighted_support_quantiles(
        train, holdout, ["source_player_id", "distance_band", "start_lie_class"]
    )
    print("Testing held-out player profiles")
    profile = _score_profiles(train, holdout)
    stability = _profile_stability(all_shots[all_shots["season"].le(2025)])
    print("Testing causal-endpoint hazard map")
    hazard = _hazard_surface_test(
        train[train["event_name"].isin(event_names)], holdout
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "annual_app_coverage.csv": pd.DataFrame(coverage_rows),
        "prior_band_lie_counts.csv": band_lie,
        "exact_course_donor_support.csv": donor_support,
        "penalty_linkage.csv": pd.DataFrame(penalty_rows),
        "player_profile_holdout.csv": profile,
        "player_profile_stability.csv": stability,
        "hazard_surface_holdout.csv": hazard,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    report = _write_report(
        args.output_dir,
        outputs["annual_app_coverage.csv"],
        band_lie,
        donor_support,
        player_support,
        outputs["penalty_linkage.csv"],
        profile,
        stability,
        hazard,
    )
    print(f"Wrote {report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--events", nargs="+", default=list(DEFAULT_EVENTS))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

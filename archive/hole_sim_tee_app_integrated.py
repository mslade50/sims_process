"""Strict-prior integrated tee -> approach transition research prototype.

This script joins the geometric tee model in ``hole_sim_tee_transition.py``
to a second, empirical approach transition.  The important experimental rule
is that the approach transition is conditioned on the *simulated* tee endpoint,
lie, and remaining distance.  The held-out tee result is retained only as an
audit column and never enters the prediction path.

Four paired arms are evaluated on a held-out recurring-course edition:

    population      population tee, population approach
    player_tee      player tee, population approach
    player_app      population tee, player approach
    both            player tee, player approach

The approach model mixes same-round/hole, same-hole, same-par, and full-course
donors from earlier editions.  Player approach profiles are shrunk pre-holdout
SG residual distributions; they tilt otherwise identical conditional donor
sets.  A pre-holdout state-value table supplies a deliberately labelled
partial-hole proxy.  This is a research harness, not a production dependency.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import hole_sim_tee_transition as tee


DEFAULT_EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")
DEFAULT_OUTPUT = Path("permanent_data/hole_sim_tee_app_integrated")
DEFAULT_EVENT = "Wyndham Championship"
ARMS = ("population", "player_tee", "player_app", "both")
PLAYABLE_LIES = ("fairway", "rough", "sand", "recovery")
METRICS = (
    "tee_endpoint_nll",
    "tee_endpoint_error_yd",
    "tee_remaining_crps",
    "tee_lie_logloss",
    "tee_lie_brier",
    "endpoint_nll",
    "endpoint_error_yd",
    "remaining_crps",
    "lie_logloss",
    "lie_brier",
)


def _lie(value: object) -> str:
    return tee._normalize_lie(value)


def _start_lie(value: object) -> str:
    value = _lie(value)
    if value in PLAYABLE_LIES:
        return value
    # The two-transition prototype does not yet model a drop location.  A
    # generated hazard/unknown tee state therefore uses recovery donors and is
    # explicitly counted as a backoff rather than silently discarded.
    return "recovery"


def _sample_categorical(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(len(probabilities))
    return np.sum(draws[:, None] > cumulative, axis=1).clip(0, probabilities.shape[1] - 1)


def _uniform_crps(samples: np.ndarray, observed: float) -> float:
    weights = np.full(len(samples), 1.0 / len(samples))
    return tee._crps(np.asarray(samples, float), weights, float(observed))


def _read_event_year(
    export_root: Path, year: int, event_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_columns = [
        "event_key", "season", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "event_type", "start_lie", "end_lie",
        "shot_distance_yd", "distance_remaining_yd", "start_x", "start_y",
        "end_x", "end_y", "ball_holed", "course_id", "shot_key",
    ]
    sg_columns = [
        "event_key", "season", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "sg_category", "start_lie",
        "start_distance_yd", "end_lie", "end_distance_yd",
        "expected_strokes_end", "sg_raw", "sg_valid", "complete_hole", "shot_key",
    ]
    raw = pd.read_parquet(
        export_root / f"pga_{year}_shots.parquet",
        columns=raw_columns,
        filters=[("event_key", "==", event_key)],
    )
    sg = pd.read_parquet(
        export_root / f"pga_{year}_shot_sg.parquet",
        columns=sg_columns,
        filters=[("event_key", "==", event_key)],
    )
    geometry = tee._read_year_geometry(export_root, year, event_key)
    return raw, sg, geometry


def _approach_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = ["start_x", "start_y", "end_x", "end_y", "pin_x", "pin_y"]
    frame = frame.copy()
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric)
    sx = frame["start_x"].to_numpy(float)
    sy = frame["start_y"].to_numpy(float)
    ex = frame["end_x"].to_numpy(float)
    ey = frame["end_y"].to_numpy(float)
    px = frame["pin_x"].to_numpy(float)
    py = frame["pin_y"].to_numpy(float)
    vx, vy = px - sx, py - sy
    norm = np.hypot(vx, vy)
    okay = norm > 30.0
    frame = frame.loc[okay].copy()
    vx, vy, norm = vx[okay], vy[okay], norm[okay]
    ux, uy = vx / norm, vy / norm
    end_pin_x = ex[okay] - px[okay]
    end_pin_y = ey[okay] - py[okay]
    frame["geom_start_distance_yd"] = norm / 3.0
    frame["error_along_yd"] = (end_pin_x * ux + end_pin_y * uy) / 3.0
    frame["error_cross_yd"] = (-end_pin_x * uy + end_pin_y * ux) / 3.0
    frame["geom_end_distance_yd"] = np.hypot(
        frame["error_along_yd"], frame["error_cross_yd"]
    )
    return frame


def build_course_approaches(
    export_root: Path,
    event_keys: dict[int, str],
    holdout_year: int,
    course_start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return strict-prior APP donors plus held-out raw/SG/geometry."""
    parts: list[pd.DataFrame] = []
    target: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    for year, event_key in sorted(event_keys.items()):
        if year < course_start_year:
            continue
        raw, sg, geometry = _read_event_year(export_root, year, event_key)
        if year == holdout_year:
            target = raw, sg, geometry
            continue
        app_sg = sg[
            sg["sg_category"].eq("app")
            & sg["sg_valid"].fillna(0).eq(1)
            & sg["start_distance_yd"].between(45, 350)
        ][["shot_key", "sg_raw", "expected_strokes_end", "complete_hole"]]
        strokes = raw[
            raw["event_type"].eq("STROKE")
            & ~raw["start_lie"].astype(str).str.casefold().isin(["tee", "green"])
        ].merge(app_sg, on="shot_key", how="inner", validate="one_to_one")
        keys = ["event_key", "round_no", "hole_no", "course_id"]
        strokes = strokes.merge(
            geometry.drop_duplicates(keys)[
                keys + ["par", "tee_x", "tee_y", "pin_x", "pin_y"]
            ],
            on=keys,
            how="inner",
            validate="many_to_one",
        )
        strokes = _approach_geometry(strokes)
        strokes["start_lie_class"] = strokes["start_lie"].map(_start_lie)
        strokes["end_lie_class"] = strokes["end_lie"].map(_lie)
        strokes["season"] = year
        parts.append(
            strokes[
                strokes["geom_start_distance_yd"].between(45, 350)
                & strokes["geom_end_distance_yd"].between(0, 260)
                & strokes["error_along_yd"].between(-180, 180)
                & strokes["error_cross_yd"].between(-180, 180)
                & strokes["sg_raw"].between(-4, 2)
            ].copy()
        )
    if not parts or target is None:
        raise ValueError("Course approach history or held-out edition is missing")
    return pd.concat(parts, ignore_index=True), target[0], target[1], target[2]


def load_tour_app_training(
    export_root: Path, start_year: int, holdout_year: int
) -> pd.DataFrame:
    columns = [
        "event_key", "season", "source_player_id", "player_name", "round_no",
        "hole_no", "sg_category", "start_lie", "start_distance_yd", "end_lie",
        "end_distance_yd", "expected_strokes_end", "sg_raw", "sg_valid", "shot_key",
    ]
    parts: list[pd.DataFrame] = []
    for year in range(start_year, holdout_year):
        print(f"Loading APP profiles/value states: {year}")
        frame = pd.read_parquet(
            export_root / f"pga_{year}_shot_sg.parquet",
            columns=columns,
            filters=[("sg_category", "==", "app"), ("sg_valid", "==", 1)],
        )
        frame = frame[
            frame["start_distance_yd"].between(45, 350)
            & frame["end_distance_yd"].between(0, 260)
            & frame["sg_raw"].between(-4, 2)
        ].copy()
        frame["source_player_id"] = frame["source_player_id"].astype(str)
        frame["start_lie_class"] = frame["start_lie"].map(_start_lie)
        frame["end_lie_class"] = frame["end_lie"].map(_lie)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def fit_app_profiles(
    training: pd.DataFrame,
    mean_prior: float = 100.0,
    variance_prior: float = 250.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit player APP SG location/scale after removing conditional field means."""
    frame = training.copy()
    frame["distance_bucket"] = (frame["start_distance_yd"] / 25).round().astype(int)
    cell = [
        "event_key", "round_no", "hole_no", "start_lie_class", "distance_bucket"
    ]
    frame["cell_mean"] = frame.groupby(cell, observed=True)["sg_raw"].transform("mean")
    frame["sg_resid"] = frame["sg_raw"] - frame["cell_mean"]
    player = frame.groupby("source_player_id", observed=True)
    player_mean = player["sg_resid"].transform("mean")
    frame["scale_resid"] = frame["sg_resid"] - player_mean
    global_variance = float(np.mean(frame["scale_resid"].to_numpy(float) ** 2))
    profiles = player.agg(
        player_name=("player_name", "last"),
        n_app=("shot_key", "size"),
        sg_resid_sum=("sg_resid", "sum"),
        sg_resid_ss=("scale_resid", lambda x: float(np.sum(np.asarray(x, float) ** 2))),
    ).reset_index()
    n = profiles["n_app"].to_numpy(float)
    profiles["app_sg_edge"] = (
        profiles["sg_resid_sum"] / (n + mean_prior)
    ).clip(-0.45, 0.45)
    profiles["app_sg_scale"] = np.sqrt(
        (profiles["sg_resid_ss"] + variance_prior * global_variance)
        / ((n + variance_prior) * global_variance)
    ).clip(0.72, 1.38)
    profiles["source_player_id"] = profiles["source_player_id"].astype(str)
    return profiles, {
        "global_sg_sd": math.sqrt(global_variance),
        "mean_prior": mean_prior,
        "variance_prior": variance_prior,
    }


class StateValueModel:
    """Pre-holdout interpolation of expected strokes remaining by lie/distance."""

    def __init__(self, training: pd.DataFrame):
        frame = training.dropna(
            subset=["end_distance_yd", "expected_strokes_end", "end_lie_class"]
        ).copy()
        frame = frame[
            frame["end_distance_yd"].between(0, 260)
            & frame["expected_strokes_end"].between(0, 8)
        ]
        frame["distance_bin"] = (frame["end_distance_yd"] / 2).round() * 2
        self.curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for lie_name, group in frame.groupby("end_lie_class", observed=True):
            curve = group.groupby("distance_bin", as_index=False)["expected_strokes_end"].mean()
            self.curves[str(lie_name)] = (
                curve["distance_bin"].to_numpy(float),
                curve["expected_strokes_end"].to_numpy(float),
            )
        curve = frame.groupby("distance_bin", as_index=False)["expected_strokes_end"].mean()
        self.fallback = (
            curve["distance_bin"].to_numpy(float),
            curve["expected_strokes_end"].to_numpy(float),
        )

    def predict(self, lies: Iterable[object], distances: np.ndarray) -> np.ndarray:
        lies_array = np.asarray([_lie(value) for value in lies], dtype=object)
        distances = np.asarray(distances, float).clip(0, 260)
        result = np.empty(len(distances), dtype=float)
        for lie_name in np.unique(lies_array):
            mask = lies_array == lie_name
            x, y = self.curves.get(str(lie_name), self.fallback)
            result[mask] = np.interp(distances[mask], x, y, left=y[0], right=y[-1])
        return result


class ApproachEndpointSurface:
    """Prior-course lie map in a common physical-hole coordinate frame."""

    def __init__(self, transitions: pd.DataFrame):
        frame = transitions.dropna(
            subset=["common_end_long_yd", "common_end_lat_yd", "end_lie_class"]
        ).copy()
        self.exact: dict[tuple[int, int], tuple[tee.SurfaceMap, int]] = {}
        self.hole: dict[int, tee.SurfaceMap] = {}

        def make_map(group: pd.DataFrame) -> tee.SurfaceMap:
            surface_frame = group.rename(
                columns={
                    "common_end_long_yd": "long_yd",
                    "common_end_lat_yd": "lat_yd",
                    "end_lie_class": "lie_class",
                }
            )
            return tee.SurfaceMap(surface_frame[["long_yd", "lat_yd", "lie_class"]])

        for key, group in frame.groupby(["round_no", "hole_no"], observed=True):
            if len(group) >= 30:
                self.exact[(int(key[0]), int(key[1]))] = (make_map(group), len(group))
        for hole_no, group in frame.groupby("hole_no", observed=True):
            self.hole[int(hole_no)] = make_map(group)

    def probabilities(
        self, round_no: int, hole_no: int, endpoints: np.ndarray
    ) -> np.ndarray:
        hole_map = self.hole[hole_no]
        hole_probability = hole_map.probabilities(endpoints)
        exact = self.exact.get((round_no, hole_no))
        if exact is None:
            return hole_probability
        exact_map, n_exact = exact
        exact_probability = exact_map.probabilities(endpoints)
        weight = n_exact / (n_exact + 100.0)
        return weight * exact_probability + (1.0 - weight) * hole_probability


@dataclass(frozen=True)
class AppDistribution:
    indices: np.ndarray
    weights: np.ndarray
    source: str


class HierarchicalApproachModel:
    def __init__(self, transitions: pd.DataFrame):
        self.frame = transitions.reset_index(drop=True)
        self.distance = self.frame["geom_start_distance_yd"].to_numpy(float)
        self.sg = self.frame["sg_raw"].to_numpy(float)
        self.error_along = self.frame["error_along_yd"].to_numpy(float)
        self.error_cross = self.frame["error_cross_yd"].to_numpy(float)
        self.end_lie = self.frame["end_lie_class"].astype(str).to_numpy()
        self.holed = self.frame["ball_holed"].fillna(0).eq(1).to_numpy(bool)
        self.setup: dict[tuple[int, int, str], np.ndarray] = {}
        self.hole: dict[tuple[int, str], np.ndarray] = {}
        self.par: dict[tuple[int, str], np.ndarray] = {}
        self.lie: dict[str, np.ndarray] = {}
        for key, group in self.frame.groupby(
            ["round_no", "hole_no", "start_lie_class"], observed=True
        ):
            self.setup[(int(key[0]), int(key[1]), str(key[2]))] = group.index.to_numpy(int)
        for key, group in self.frame.groupby(["hole_no", "start_lie_class"], observed=True):
            self.hole[(int(key[0]), str(key[1]))] = group.index.to_numpy(int)
        for key, group in self.frame.groupby(["par", "start_lie_class"], observed=True):
            self.par[(int(key[0]), str(key[1]))] = group.index.to_numpy(int)
        for key, group in self.frame.groupby("start_lie_class", observed=True):
            self.lie[str(key)] = group.index.to_numpy(int)
        self.cache: dict[tuple[int, int, int, str, int], AppDistribution] = {}
        self.backoff_counts: dict[str, int] = {}

    def _nearest(self, indices: np.ndarray, distance: float, cap: int) -> np.ndarray:
        if len(indices) <= cap:
            return indices
        delta = np.abs(self.distance[indices] - distance)
        chosen = np.argpartition(delta, cap - 1)[:cap]
        return indices[chosen]

    def distribution(
        self, round_no: int, hole_no: int, par: int, start_lie: str, distance: float
    ) -> AppDistribution:
        query_lie = _start_lie(start_lie)
        distance_bin = int(round(float(distance) / 5.0) * 5)
        key = (round_no, hole_no, par, query_lie, distance_bin)
        if key in self.cache:
            return self.cache[key]
        setup = self.setup.get((round_no, hole_no, query_lie), np.empty(0, int))
        hole = self.hole.get((hole_no, query_lie), np.empty(0, int))
        par_pool = self.par.get((par, query_lie), np.empty(0, int))
        lie_pool = self.lie.get(query_lie, np.empty(0, int))
        if not len(lie_pool):
            query_lie = "rough"
            setup = self.setup.get((round_no, hole_no, query_lie), np.empty(0, int))
            hole = self.hole.get((hole_no, query_lie), np.empty(0, int))
            par_pool = self.par.get((par, query_lie), np.empty(0, int))
            lie_pool = self.lie[query_lie]

        alpha = len(setup) / (len(setup) + 60.0)
        beta = len(hole) / (len(hole) + 250.0)
        gamma = len(par_pool) / (len(par_pool) + 1200.0)
        component_spec = [
            (setup, alpha, 18.0, 500, "setup"),
            (hole, (1 - alpha) * beta, 23.0, 800, "hole"),
            (par_pool, (1 - alpha) * (1 - beta) * gamma, 28.0, 1000, "par"),
            (lie_pool, (1 - alpha) * (1 - beta) * (1 - gamma), 35.0, 1200, "course"),
        ]
        all_indices: list[np.ndarray] = []
        all_mass: list[np.ndarray] = []
        labels: list[str] = []
        for indices, mass, bandwidth, cap, label in component_spec:
            if not len(indices) or mass <= 0:
                continue
            indices = self._nearest(indices, distance_bin, cap)
            kernel = np.exp(-0.5 * ((self.distance[indices] - distance_bin) / bandwidth) ** 2)
            kernel += 0.002
            kernel = mass * kernel / kernel.sum()
            all_indices.append(indices)
            all_mass.append(kernel)
            labels.append(f"{label}={mass:.2f}")
        indices = np.concatenate(all_indices)
        mass = np.concatenate(all_mass)
        unique, inverse = np.unique(indices, return_inverse=True)
        weights = np.zeros(len(unique), float)
        np.add.at(weights, inverse, mass)
        weights /= weights.sum()
        result = AppDistribution(unique, weights, ",".join(labels))
        self.cache[key] = result
        return result

    def player_weights(
        self, distribution: AppDistribution, profile: pd.Series, mode: str = "combined"
    ) -> np.ndarray:
        weights = distribution.weights
        sg = self.sg[distribution.indices]
        mean = float(np.sum(weights * sg))
        variance = float(np.sum(weights * (sg - mean) ** 2))
        sd = math.sqrt(max(variance, 0.04))
        z = (sg - mean) / sd
        shift = (
            float(profile["app_sg_edge"]) / sd if mode in {"combined", "mean_only"} else 0.0
        )
        scale = float(profile["app_sg_scale"]) if mode in {"combined", "scale_only"} else 1.0
        # Importance ratio from N(0,1) to N(shift, scale).  It changes the
        # paired empirical outcomes rather than fabricating new endpoint/lie
        # combinations.  Clipping keeps sparse conditional pools stable.
        log_ratio = -math.log(scale) - 0.5 * ((z - shift) / scale) ** 2 + 0.5 * z**2
        log_ratio = np.clip(log_ratio, -4.5, 4.5)
        adjusted = weights * np.exp(log_ratio)
        return adjusted / adjusted.sum()


def _fit_rigid_geometry(
    source_geometry: pd.DataFrame, forecast: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    paired = source_geometry.merge(
        forecast, on=["round_no", "hole_no"], how="inner", validate="one_to_one"
    )
    source_points = np.vstack(
        [
            paired[["tee_x", "tee_y"]].to_numpy(float),
            paired[["pin_x", "pin_y"]].to_numpy(float),
        ]
    )
    target_points = np.vstack(
        [
            paired[["forecast_tee_x", "forecast_tee_y"]].to_numpy(float),
            paired[["forecast_pin_x", "forecast_pin_y"]].to_numpy(float),
        ]
    )
    finite = np.isfinite(source_points).all(axis=1) & np.isfinite(target_points).all(axis=1)
    source_points, target_points = source_points[finite], target_points[finite]
    if len(source_points) < 60:
        raise ValueError(f"Too few rigid-alignment geometry anchors: {len(source_points)}")
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    left = source_points - source_center
    right = target_points - target_center
    u, _, vt = np.linalg.svd(left.T @ right)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    fitted = left @ rotation + target_center
    rmse = float(np.sqrt(np.mean(np.sum((fitted - target_points) ** 2, axis=1))) / 3.0)
    return source_center, target_center, rotation, rmse


def _apply_rigid(
    x: np.ndarray,
    y: np.ndarray,
    transform: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    source_center, target_center, rotation, _ = transform
    points = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    return (points - source_center) @ rotation + target_center


def _setup_forecast(
    course_drives: pd.DataFrame, target_drives: pd.DataFrame, setup_mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["round_no", "hole_no"]
    if setup_mode == "actual":
        forecast = (
            target_drives.groupby(keys, as_index=False)
            .agg(
                forecast_tee_x=("start_x", "median"),
                forecast_tee_y=("start_y", "median"),
                forecast_pin_x=("pin_x", "median"),
                forecast_pin_y=("pin_y", "median"),
            )
        )
        return course_drives, target_drives, forecast
    # Use the most recent strictly prior edition as the coordinate reference.
    # This avoids taking medians across possibly different ShotLink origins.
    reference = course_drives[course_drives["season"].eq(course_drives["season"].max())]
    forecast = (
        reference.groupby(keys, as_index=False)
        .agg(
            forecast_tee_x=("start_x", "median"),
            forecast_tee_y=("start_y", "median"),
            forecast_pin_x=("pin_x", "median"),
            forecast_pin_y=("pin_y", "median"),
        )
    )

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for _, event in frame.groupby("event_key", observed=True):
            geometry = (
                event.groupby(keys, as_index=False)
                .agg(
                    tee_x=("start_x", "median"), tee_y=("start_y", "median"),
                    pin_x=("pin_x", "median"), pin_y=("pin_y", "median"),
                )
            )
            rigid = _fit_rigid_geometry(geometry, forecast)
            aligned = _apply_rigid(event["end_x"], event["end_y"], rigid)
            event = event.copy()
            event["aligned_end_x"] = aligned[:, 0]
            event["aligned_end_y"] = aligned[:, 1]
            event["geometry_alignment_rmse_yd"] = rigid[3]
            parts.append(event)
        result = pd.concat(parts, ignore_index=True).merge(
            forecast, on=keys, how="inner", validate="many_to_one"
        )
        vx = result["forecast_pin_x"].to_numpy(float) - result["forecast_tee_x"].to_numpy(float)
        vy = result["forecast_pin_y"].to_numpy(float) - result["forecast_tee_y"].to_numpy(float)
        norm = np.hypot(vx, vy)
        ux, uy = vx / norm, vy / norm
        dx = result["aligned_end_x"].to_numpy(float) - result["forecast_tee_x"].to_numpy(float)
        dy = result["aligned_end_y"].to_numpy(float) - result["forecast_tee_y"].to_numpy(float)
        result["long_yd"] = (dx * ux + dy * uy) / 3.0
        result["lat_yd"] = (-dx * uy + dy * ux) / 3.0
        result["pin_long_yd"] = norm / 3.0
        result["geom_remaining_yd"] = np.hypot(
            result["pin_long_yd"] - result["long_yd"], result["lat_yd"]
        )
        return result

    return transform(course_drives), transform(target_drives), forecast


def build_target_cases(
    target_drives: pd.DataFrame,
    target_raw: pd.DataFrame,
    target_sg: pd.DataFrame,
    target_geometry: pd.DataFrame,
    forecast: pd.DataFrame,
    setup_mode: str,
    max_player_rounds: int,
    seed: int,
) -> pd.DataFrame:
    app = target_sg[
        target_sg["stroke_no"].eq(2)
        & target_sg["sg_category"].eq("app")
        & target_sg["sg_valid"].fillna(0).eq(1)
        & target_sg["start_distance_yd"].between(45, 350)
    ][["shot_key", "complete_hole"]]
    second = target_raw[
        target_raw["stroke_no"].eq(2) & target_raw["event_type"].eq("STROKE")
    ].merge(app, on="shot_key", how="inner", validate="one_to_one")
    second = second.rename(
        columns={
            "start_x": "second_start_x", "start_y": "second_start_y",
            "end_x": "second_end_x", "end_y": "second_end_y",
            "start_lie": "second_start_lie", "end_lie": "second_end_lie",
            "shot_key": "second_shot_key",
        }
    )
    join = ["event_key", "source_player_id", "round_no", "hole_no"]
    columns = join + [
        "second_start_x", "second_start_y", "second_end_x", "second_end_y",
        "second_start_lie", "second_end_lie", "second_shot_key", "complete_hole",
    ]
    cases = target_drives.merge(second[columns], on=join, how="inner", validate="one_to_one")
    if setup_mode == "historical":
        geometry = target_geometry.groupby(
            ["round_no", "hole_no"], as_index=False
        )[["tee_x", "tee_y", "pin_x", "pin_y"]].first()
        rigid = _fit_rigid_geometry(geometry, forecast)
        aligned_start = _apply_rigid(
            cases["second_start_x"], cases["second_start_y"], rigid
        )
        aligned_end = _apply_rigid(cases["second_end_x"], cases["second_end_y"], rigid)
        cases["second_start_x_for_score"] = aligned_start[:, 0]
        cases["second_start_y_for_score"] = aligned_start[:, 1]
        cases["second_end_x_for_score"] = aligned_end[:, 0]
        cases["second_end_y_for_score"] = aligned_end[:, 1]
    else:
        cases["second_start_x_for_score"] = cases["second_start_x"]
        cases["second_start_y_for_score"] = cases["second_start_y"]
        cases["second_end_x_for_score"] = cases["second_end_x"]
        cases["second_end_y_for_score"] = cases["second_end_y"]
    forecast_columns = [
        "forecast_tee_x", "forecast_tee_y", "forecast_pin_x", "forecast_pin_y"
    ]
    if not all(column in cases.columns for column in forecast_columns):
        cases = cases.merge(
            forecast, on=["round_no", "hole_no"], how="inner", validate="many_to_one"
        )
    numeric = [
        "second_start_x", "second_start_y", "second_end_x", "second_end_y",
        "forecast_tee_x", "forecast_tee_y", "forecast_pin_x", "forecast_pin_y",
    ]
    for column in numeric:
        cases[column] = pd.to_numeric(cases[column], errors="coerce")
    cases = cases.dropna(subset=numeric)
    vx = cases["forecast_pin_x"].to_numpy(float) - cases["forecast_tee_x"].to_numpy(float)
    vy = cases["forecast_pin_y"].to_numpy(float) - cases["forecast_tee_y"].to_numpy(float)
    norm = np.hypot(vx, vy)
    ux, uy = vx / norm, vy / norm
    dx = cases["second_end_x_for_score"].to_numpy(float) - cases["forecast_tee_x"].to_numpy(float)
    dy = cases["second_end_y_for_score"].to_numpy(float) - cases["forecast_tee_y"].to_numpy(float)
    cases["actual_second_long_yd"] = (dx * ux + dy * uy) / 3.0
    cases["actual_second_lat_yd"] = (-dx * uy + dy * ux) / 3.0
    cases["forecast_pin_long_yd"] = norm / 3.0
    cases["actual_second_remaining_yd"] = np.hypot(
        cases["forecast_pin_long_yd"] - cases["actual_second_long_yd"],
        cases["actual_second_lat_yd"],
    )
    cases["actual_second_lie"] = cases["second_end_lie"].map(_lie)
    # Continuity is used only to define a clean evaluable target, not as a model
    # input.  It excludes penalty/drop sequences between strokes one and two.
    continuity_ft = np.hypot(
        cases["end_x"] - cases["second_start_x"],
        cases["end_y"] - cases["second_start_y"],
    )
    cases["actual_tee_to_second_continuity_ft"] = continuity_ft
    cases = cases[
        continuity_ft.le(5)
        & cases["actual_second_remaining_yd"].between(0, 260)
        & cases["complete_hole"].fillna(0).eq(1)
    ].copy()

    holed = target_raw[
        target_raw["ball_holed"].fillna(0).eq(1)
        & target_raw["event_type"].eq("STROKE")
    ]
    scores = holed.groupby(join, as_index=False)["stroke_no"].max().rename(
        columns={"stroke_no": "actual_hole_score"}
    )
    cases = cases.merge(scores, on=join, how="inner", validate="one_to_one")
    if max_player_rounds > 0:
        player_rounds = cases[["source_player_id", "round_no"]].drop_duplicates()
        if len(player_rounds) > max_player_rounds:
            chosen = player_rounds.sample(max_player_rounds, random_state=seed)
            cases = cases.merge(chosen, on=["source_player_id", "round_no"], how="inner")
    return cases.sort_values(["source_player_id", "round_no", "hole_no"]).reset_index(drop=True)


def add_common_course_frame(
    transitions: pd.DataFrame, forecast: pd.DataFrame
) -> pd.DataFrame:
    """Map historical endpoints into the target's physical tee/pin frame.

    Absolute ShotLink coordinates are stable at a physical course.  The frame
    is either the known target setup or a strictly prior-edition setup forecast,
    depending on ``--setup-mode``.  No held-out shot endpoint enters the map.
    """
    # Fit one proper rigid transform per historical edition from its full
    # tee/pin geometry to the target forecast geometry.  This tolerates archive
    # coordinate-origin and orientation changes without pretending that raw XY
    # values are directly comparable across seasons.
    transformed_parts: list[pd.DataFrame] = []
    for event_key, event in transitions.groupby("event_key", observed=True):
        source_geometry = event.groupby(
            ["round_no", "hole_no"], as_index=False
        )[["tee_x", "tee_y", "pin_x", "pin_y"]].first()
        paired = source_geometry.merge(
            forecast, on=["round_no", "hole_no"], how="inner", validate="one_to_one"
        )
        source_points = np.vstack(
            [
                paired[["tee_x", "tee_y"]].to_numpy(float),
                paired[["pin_x", "pin_y"]].to_numpy(float),
            ]
        )
        target_points = np.vstack(
            [
                paired[["forecast_tee_x", "forecast_tee_y"]].to_numpy(float),
                paired[["forecast_pin_x", "forecast_pin_y"]].to_numpy(float),
            ]
        )
        finite = np.isfinite(source_points).all(axis=1) & np.isfinite(target_points).all(axis=1)
        source_points, target_points = source_points[finite], target_points[finite]
        if len(source_points) < 60:
            raise ValueError(f"Too few geometry anchors for {event_key}: {len(source_points)}")
        source_center = source_points.mean(axis=0)
        target_center = target_points.mean(axis=0)
        left = source_points - source_center
        right = target_points - target_center
        u, _, vt = np.linalg.svd(left.T @ right)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        aligned_anchors = left @ rotation + target_center
        alignment_rmse = float(
            np.sqrt(np.mean(np.sum((aligned_anchors - target_points) ** 2, axis=1))) / 3.0
        )
        endpoints = event[["end_x", "end_y"]].to_numpy(float)
        aligned = (endpoints - source_center) @ rotation + target_center
        event = event.copy()
        event["aligned_end_x"] = aligned[:, 0]
        event["aligned_end_y"] = aligned[:, 1]
        event["geometry_alignment_rmse_yd"] = alignment_rmse
        transformed_parts.append(event)
    result = pd.concat(transformed_parts, ignore_index=True).merge(
        forecast, on=["round_no", "hole_no"], how="inner", validate="many_to_one"
    )
    vx = result["forecast_pin_x"].to_numpy(float) - result["forecast_tee_x"].to_numpy(float)
    vy = result["forecast_pin_y"].to_numpy(float) - result["forecast_tee_y"].to_numpy(float)
    norm = np.hypot(vx, vy)
    ux, uy = vx / norm, vy / norm
    dx = result["aligned_end_x"].to_numpy(float) - result["forecast_tee_x"].to_numpy(float)
    dy = result["aligned_end_y"].to_numpy(float) - result["forecast_tee_y"].to_numpy(float)
    result["common_end_long_yd"] = (dx * ux + dy * uy) / 3.0
    result["common_end_lat_yd"] = (-dx * uy + dy * ux) / 3.0
    return result[
        result["common_end_long_yd"].between(-100, 800)
        & result["common_end_lat_yd"].between(-250, 250)
    ].reset_index(drop=True)


def _simulate_arm(
    record: object,
    distribution: tee.Distribution,
    surface: tee.SurfaceMap,
    tee_profile: pd.Series,
    app_profile: pd.Series,
    app_model: HierarchicalApproachModel,
    app_surface: ApproachEndpointSurface,
    value_model: StateValueModel,
    player_tee: bool,
    player_app: bool,
    app_profile_mode: str,
    simulations: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    points = distribution.points
    if player_tee:
        points = tee.adjust_distribution(points, distribution.weights, tee_profile)
    chosen = rng.choice(
        len(points), size=simulations, replace=True, p=distribution.weights
    )
    starts = points[chosen]
    lie_probabilities = surface.probabilities(starts)
    lie_codes = _sample_categorical(lie_probabilities, rng)
    start_lies = np.asarray(tee.LIE_CLASSES, dtype=object)[lie_codes]
    pin = np.array([float(record.forecast_pin_long_yd), 0.0])
    vector = pin[None, :] - starts
    distances = np.linalg.norm(vector, axis=1)
    unsupported = ~np.isin(start_lies, PLAYABLE_LIES)
    out_of_range = (distances < 45) | (distances > 350)

    donor_index = np.empty(simulations, dtype=int)
    # Bin/cache is part of the predictor, so all grouping uses simulated state.
    state_keys: dict[tuple[str, int], list[int]] = {}
    for index, (lie_name, distance) in enumerate(zip(start_lies, distances)):
        key = (str(lie_name), int(round(float(distance) / 5) * 5))
        state_keys.setdefault(key, []).append(index)
    for (lie_name, distance_bin), positions in state_keys.items():
        donor_distribution = app_model.distribution(
            int(record.round_no), int(record.hole_no), int(record.par), lie_name, distance_bin
        )
        weights = (
            app_model.player_weights(donor_distribution, app_profile, app_profile_mode)
            if player_app
            else donor_distribution.weights
        )
        donor_index[np.asarray(positions, int)] = rng.choice(
            donor_distribution.indices,
            size=len(positions),
            replace=True,
            p=weights,
        )

    safe_distance = np.maximum(distances, 1e-6)
    unit = vector / safe_distance[:, None]
    perp = np.column_stack([-unit[:, 1], unit[:, 0]])
    endpoints = (
        pin[None, :]
        + unit * app_model.error_along[donor_index, None]
        + perp * app_model.error_cross[donor_index, None]
    )
    # Rotating a donor's miss vector changes its physical landing point, so the
    # donor lie cannot simply follow it.  Re-sample ordinary endpoint lies from
    # prior-course physical surfaces in the common target frame. True holeout
    # atoms stay intact. Rare water/other donor labels are preserved only as an
    # unsupported-outcome audit; they are not a penalty-incidence model.
    holeout_atom = app_model.holed[donor_index]
    endpoints[holeout_atom] = pin
    endpoint_lie_probabilities = app_surface.probabilities(
        int(record.round_no), int(record.hole_no), endpoints
    )
    endpoint_lie_codes = _sample_categorical(endpoint_lie_probabilities, rng)
    end_lies = np.asarray(tee.LIE_CLASSES, dtype=object)[endpoint_lie_codes]
    unsupported_outcome_atom = np.isin(
        app_model.end_lie[donor_index], ["water", "other"]
    )
    end_lies[unsupported_outcome_atom] = app_model.end_lie[donor_index][
        unsupported_outcome_atom
    ]
    end_lies[holeout_atom] = "green"
    # Preserve the conditional probabilities for low-noise proper lie scoring;
    # the sampled labels remain the state input to the value proxy.
    for special_index in np.flatnonzero(unsupported_outcome_atom):
        endpoint_lie_probabilities[special_index] = 0.0
        code = tee.LIE_CLASSES.index(str(end_lies[special_index]))
        endpoint_lie_probabilities[special_index, code] = 1.0
    if np.any(holeout_atom):
        endpoint_lie_probabilities[holeout_atom] = 0.0
        endpoint_lie_probabilities[holeout_atom, tee.LIE_CLASSES.index("green")] = 1.0
    final_remaining = np.linalg.norm(endpoints - pin[None, :], axis=1)
    remaining_value = value_model.predict(end_lies, final_remaining)
    remaining_value[holeout_atom] = 0.0
    partial_value = 2.0 + remaining_value
    return {
        "tee_starts": starts,
        "tee_start_lies": start_lies,
        "tee_lie_probabilities": lie_probabilities,
        "tee_remaining": distances,
        "endpoints": endpoints,
        "remaining": final_remaining,
        "end_lies": end_lies,
        "end_lie_probabilities": endpoint_lie_probabilities,
        "partial_value": partial_value,
        # No explicit drop/penalty-cost transition exists yet. Hazard outcomes
        # remain in endpoint/remaining/lie scores but are excluded from the
        # partial-hole decision proxies so they cannot receive a false value.
        "decision_proxy_valid": ~np.isin(end_lies, ["water", "other"]),
        "unsupported_tee_lie_rate": float(np.mean(unsupported)),
        "out_of_app_range_rate": float(np.mean(out_of_range)),
        "holeout_atom_rate": float(np.mean(holeout_atom)),
        "unsupported_outcome_atom_rate": float(np.mean(unsupported_outcome_atom)),
        "final_hazard_lie_rate": float(np.mean(np.isin(end_lies, ["water", "other"]))),
    }


def evaluate(
    cases: pd.DataFrame,
    course_drives: pd.DataFrame,
    tee_profiles: pd.DataFrame,
    app_profiles: pd.DataFrame,
    app_model: HierarchicalApproachModel,
    app_surface: ApproachEndpointSurface,
    value_model: StateValueModel,
    simulations: int,
    seed: int,
    app_profile_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tee_lookup = tee_profiles.set_index("source_player_id")
    app_lookup = app_profiles.set_index("source_player_id")
    neutral_tee = pd.Series(
        {"distance_ratio": 1.0, "lateral_bias_yd": 0.0, "long_scale": 1.0, "lat_scale": 1.0}
    )
    neutral_app = pd.Series({"app_sg_edge": 0.0, "app_sg_scale": 1.0})
    distributions: dict[tuple[int, int, int], tee.Distribution] = {}
    surfaces: dict[int, tee.SurfaceMap] = {}
    rows: list[dict[str, object]] = []

    for number, record in enumerate(cases.itertuples(index=False), start=1):
        if number % 250 == 0:
            print(f"Scoring integrated case {number:,}/{len(cases):,}")
        setup_key = (int(record.round_no), int(record.hole_no), int(record.par))
        if setup_key not in distributions:
            distributions[setup_key] = tee.build_distribution(course_drives, *setup_key)
        if int(record.hole_no) not in surfaces:
            surfaces[int(record.hole_no)] = tee.SurfaceMap(
                course_drives[course_drives["hole_no"].eq(record.hole_no)]
            )
        player_id = str(record.source_player_id)
        tee_profile = tee_lookup.loc[player_id] if player_id in tee_lookup.index else neutral_tee
        app_profile = app_lookup.loc[player_id] if player_id in app_lookup.index else neutral_app
        if isinstance(tee_profile, pd.DataFrame):
            tee_profile = tee_profile.iloc[-1]
        if isinstance(app_profile, pd.DataFrame):
            app_profile = app_profile.iloc[-1]
        simulations_by_arm: dict[str, dict[str, np.ndarray | float]] = {}
        for arm in ARMS:
            # Common random numbers: every arm begins from the same empirical
            # tee index and uniforms for this case.  Population/player-APP and
            # player-tee/both consequently differ because of the model, not
            # avoidable Monte Carlo jitter.
            arm_rng = np.random.default_rng(np.random.SeedSequence([seed, number]))
            simulations_by_arm[arm] = _simulate_arm(
                record,
                distributions[setup_key],
                surfaces[int(record.hole_no)],
                tee_profile,
                app_profile,
                app_model,
                app_surface,
                value_model,
                player_tee=arm in {"player_tee", "both"},
                player_app=arm in {"player_app", "both"},
                app_profile_mode=app_profile_mode,
                simulations=simulations,
                rng=arm_rng,
            )

        actual_endpoint = np.array(
            [record.actual_second_long_yd, record.actual_second_lat_yd], float
        )
        actual_tee_endpoint = np.array([record.long_yd, record.lat_yd], float)
        actual_tee_lie = str(record.lie_class)
        actual_tee_lie_index = tee.LIE_CLASSES.index(actual_tee_lie)
        actual_tee_onehot = np.zeros(len(tee.LIE_CLASSES))
        actual_tee_onehot[actual_tee_lie_index] = 1.0
        actual_lie = str(record.actual_second_lie)
        actual_lie_index = tee.LIE_CLASSES.index(actual_lie)
        actual_onehot = np.zeros(len(tee.LIE_CLASSES))
        actual_onehot[actual_lie_index] = 1.0
        actual_state_value = 2.0 + value_model.predict(
            [actual_lie], np.array([record.actual_second_remaining_yd])
        )[0]
        row: dict[str, object] = {
            "event_key": record.event_key,
            "source_player_id": player_id,
            "player_name": record.player_name,
            "round_no": int(record.round_no),
            "hole_no": int(record.hole_no),
            "par": int(record.par),
            "has_tee_profile": player_id in tee_lookup.index,
            "has_app_profile": player_id in app_lookup.index,
            "app_profile_n": int(app_profile.get("n_app", 0)),
            "app_sg_edge": float(app_profile["app_sg_edge"]),
            "app_sg_scale": float(app_profile["app_sg_scale"]),
            "actual_tee_long_yd_audit_only": float(record.long_yd),
            "actual_tee_lat_yd_audit_only": float(record.lat_yd),
            "actual_second_long_yd": float(record.actual_second_long_yd),
            "actual_second_lat_yd": float(record.actual_second_lat_yd),
            "actual_second_remaining_yd": float(record.actual_second_remaining_yd),
            "actual_second_lie": actual_lie,
            "actual_partial_state_value": float(actual_state_value),
            "actual_hole_score": int(record.actual_hole_score),
        }
        # Anchoring strips only the incremental player-APP mean.  It preserves
        # the tee-generated state mean: both is anchored to player_tee, while
        # player_app is anchored to population.
        anchors = {
            "population": "population",
            "player_tee": "player_tee",
            "player_app": "population",
            "both": "player_tee",
        }
        for arm, simulated in simulations_by_arm.items():
            endpoints = np.asarray(simulated["endpoints"], float)
            tee_starts = np.asarray(simulated["tee_starts"], float)
            tee_remaining = np.asarray(simulated["tee_remaining"], float)
            remaining = np.asarray(simulated["remaining"], float)
            end_lies = np.asarray(simulated["end_lies"], object)
            partial_value = np.asarray(simulated["partial_value"], float)
            decision_valid = np.asarray(simulated["decision_proxy_valid"], bool)
            reference = np.asarray(
                simulations_by_arm[anchors[arm]]["partial_value"], float
            )
            reference_valid = np.asarray(
                simulations_by_arm[anchors[arm]]["decision_proxy_valid"], bool
            )
            scored_partial_value = partial_value[decision_valid]
            scored_reference = reference[reference_valid]
            anchored_value = (
                scored_partial_value
                - scored_partial_value.mean()
                + scored_reference.mean()
            )
            lie_probs = np.asarray(simulated["end_lie_probabilities"], float).mean(axis=0)
            # A tiny floor keeps truly unseen classes finite without adding the
            # substantial Monte Carlo noise of sampled-count smoothing.
            lie_probs = np.maximum(lie_probs, 1e-6)
            lie_probs /= lie_probs.sum()
            endpoint_mean = endpoints.mean(axis=0)
            weights = np.full(simulations, 1.0 / simulations)
            tee_lie_probs = np.asarray(
                simulated["tee_lie_probabilities"], float
            ).mean(axis=0)
            tee_lie_probs = np.maximum(tee_lie_probs, 1e-6)
            tee_lie_probs /= tee_lie_probs.sum()
            row[f"{arm}_tee_endpoint_nll"] = tee._kde_nll(
                tee_starts, weights, actual_tee_endpoint
            )
            row[f"{arm}_tee_endpoint_error_yd"] = float(
                np.linalg.norm(tee_starts.mean(axis=0) - actual_tee_endpoint)
            )
            row[f"{arm}_tee_remaining_crps"] = _uniform_crps(
                tee_remaining, record.geom_remaining_yd
            )
            row[f"{arm}_tee_lie_logloss"] = -math.log(
                max(tee_lie_probs[actual_tee_lie_index], 1e-12)
            )
            row[f"{arm}_tee_lie_brier"] = float(
                np.mean((tee_lie_probs - actual_tee_onehot) ** 2)
            )
            row[f"{arm}_endpoint_nll"] = tee._kde_nll(endpoints, weights, actual_endpoint)
            row[f"{arm}_endpoint_error_yd"] = float(np.linalg.norm(endpoint_mean - actual_endpoint))
            row[f"{arm}_remaining_crps"] = _uniform_crps(
                remaining, record.actual_second_remaining_yd
            )
            row[f"{arm}_lie_logloss"] = -math.log(max(lie_probs[actual_lie_index], 1e-12))
            row[f"{arm}_lie_brier"] = float(np.mean((lie_probs - actual_onehot) ** 2))
            row[f"{arm}_partial_value_crps"] = _uniform_crps(
                scored_partial_value, actual_state_value
            )
            row[f"{arm}_partial_value_crps_mean_anchored"] = _uniform_crps(
                anchored_value, actual_state_value
            )
            row[f"{arm}_hole_proxy_crps"] = _uniform_crps(
                scored_partial_value, record.actual_hole_score
            )
            row[f"{arm}_hole_proxy_crps_mean_anchored"] = _uniform_crps(
                anchored_value, record.actual_hole_score
            )
            row[f"{arm}_pred_partial_value_mean"] = float(scored_partial_value.mean())
            row[f"{arm}_pred_partial_value_sd"] = float(scored_partial_value.std())
            row[f"{arm}_unsupported_tee_lie_rate"] = simulated["unsupported_tee_lie_rate"]
            row[f"{arm}_out_of_app_range_rate"] = simulated["out_of_app_range_rate"]
            row[f"{arm}_holeout_atom_rate"] = simulated["holeout_atom_rate"]
            row[f"{arm}_unsupported_outcome_atom_rate"] = simulated[
                "unsupported_outcome_atom_rate"
            ]
            row[f"{arm}_final_hazard_lie_rate"] = simulated["final_hazard_lie_rate"]
        rows.append(row)
    scores = pd.DataFrame(rows)
    return scores, summarize(scores)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clusters = scores["source_player_id"].astype(str) + ":R" + scores["round_no"].astype(str)
    for arm in ARMS:
        for metric in METRICS:
            column = f"{arm}_{metric}"
            baseline = f"population_{metric}"
            base_value = float(scores[baseline].mean())
            value = float(scores[column].mean())
            delta_row = scores[column] - scores[baseline]
            cluster_delta = delta_row.groupby(clusters).mean()
            se = (
                float(cluster_delta.std(ddof=1) / math.sqrt(len(cluster_delta)))
                if len(cluster_delta) > 1 else float("nan")
            )
            delta = value - base_value
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "n": len(scores),
                    "player_round_clusters": clusters.nunique(),
                    "population": base_value,
                    "arm_value": value,
                    "delta_vs_population": delta,
                    "cluster_se": se,
                    "ci_low": delta - 1.96 * se,
                    "ci_high": delta + 1.96 * se,
                    "lower_is_better": True,
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, rule]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(
    args: argparse.Namespace,
    course_drives: pd.DataFrame,
    course_approaches: pd.DataFrame,
    cases: pd.DataFrame,
    tee_profiles: pd.DataFrame,
    app_profiles: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: pd.DataFrame,
    app_globals: dict[str, float],
) -> str:
    headline = metrics[
        metrics["arm"].ne("population")
        & metrics["metric"].isin(
            ["endpoint_nll", "remaining_crps", "lie_logloss"]
        )
    ][["arm", "metric", "population", "arm_value", "delta_vs_population", "ci_low", "ci_high"]]
    tee_diagnostic = metrics[
        metrics["arm"].isin(["player_tee", "both"])
        & metrics["metric"].isin(
            ["tee_endpoint_nll", "tee_remaining_crps", "tee_lie_logloss"]
        )
    ][["arm", "metric", "population", "arm_value", "delta_vs_population", "ci_low", "ci_high"]]
    support_rows = []
    for arm in ARMS:
        support_rows.append(
            {
                "arm": arm,
                "unsupported_lie_rate": scores[f"{arm}_unsupported_tee_lie_rate"].mean(),
                "out_of_APP_range_rate": scores[f"{arm}_out_of_app_range_rate"].mean(),
                "holeout_atom_rate": scores[f"{arm}_holeout_atom_rate"].mean(),
                "unsupported_outcome_atom_rate": scores[
                    f"{arm}_unsupported_outcome_atom_rate"
                ].mean(),
                "final_hazard_lie_rate": scores[f"{arm}_final_hazard_lie_rate"].mean(),
            }
        )
    support = pd.DataFrame(support_rows)
    profile_coverage = scores[["has_tee_profile", "has_app_profile"]].mean()
    proxy_rows = []
    for arm in ARMS:
        proxy_rows.append(
            {
                "arm": arm,
                "conditional_partial_value_CRPS": scores[f"{arm}_partial_value_crps"].mean(),
                "conditional_hole_proxy_CRPS": scores[f"{arm}_hole_proxy_crps"].mean(),
            }
        )
    proxy_diagnostic = pd.DataFrame(proxy_rows)
    return f"""# Integrated tee -> approach holdout

## Scope

- Event: **{args.event_name}**, holdout: **{args.holdout_year}**
- Setup mode: **{args.setup_mode}**
- Player APP reweighting ablation: **{args.app_profile_mode}**
- Evaluation rows: **{len(cases):,}** across **{scores.groupby(['source_player_id', 'round_no']).ngroups:,} player-rounds**
- Simulations per arm/row: **{args.simulations:,}**
- Strict-prior course tee rows: **{len(course_drives):,}**
- Strict-prior course APP donors: **{len(course_approaches):,}**
- Historical-to-forecast geometry rigid-alignment RMSE: median **{course_approaches.groupby('event_key')['geometry_alignment_rmse_yd'].first().median():.2f} yd**, max **{course_approaches.groupby('event_key')['geometry_alignment_rmse_yd'].first().max():.2f} yd**
- Tee profile coverage: **{profile_coverage['has_tee_profile']:.1%}**
- APP profile coverage: **{profile_coverage['has_app_profile']:.1%}**
- APP profile shrinkage: mean prior {app_globals['mean_prior']:.0f}, variance prior {app_globals['variance_prior']:.0f}; residual SD {app_globals['global_sg_sd']:.4f}

## Experiment

Each arm first samples a tee endpoint from the prior-edition hole distribution.
It samples the tee lie from a prior-edition spatial surface map, calculates
remaining yardage from that generated point, and passes exactly that generated
state to the APP model. The actual held-out tee result is never supplied to the
APP transition. It appears in the score file only in columns ending
`audit_only` so this can be verified.

APP donors are empirical paired outcomes (endpoint error, terminal atoms, and SG) from
editions before {args.holdout_year}. Conditional weights mix same round/hole,
same hole, same par, and whole-course histories, then smooth on the *simulated*
start distance. Player APP effects reweight paired donors using shrunk,
pre-{args.holdout_year} conditional SG residual location and scale.
Each historical edition's complete tee/pin geometry is first rigidly aligned
to the target forecast geometry; raw ShotLink XY origins are never assumed to
match across years. After the miss vector is rotated into the simulated start state, ordinary end
lies are re-sampled from a KNN map of prior endpoints in the common physical
hole frame. Holeouts stay linked to pin/green atoms. The filtered course donors
contain essentially no hazards (Wyndham 3/40,456 `other`; Sony 4/37,617;
Travelers 2/32,804; zero `water`), so preserved water/other labels are merely
unsupported-outcome audits. KNN pseudocount hazard probability is **not** an
estimate of penalty incidence.

The evaluation subset requires an observed, continuous stroke-2 APP transition
and a completed hole. It therefore tests the integrated core on conventional
par-4/par-5 tee-to-APP sequences; it is not yet a penalty/drop or all-hole test.

## Main proper-score results

All values are losses; lower is better. Confidence intervals cluster paired
deltas by player-round.

{_markdown_table(headline)}

## Filtered-case tee-state diagnostic

This scores the generated APP start state directly against the actual tee
result on the exact stroke-2 APP evaluation subset. It distinguishes a tee
transport/subset issue from degradation introduced by the downstream APP step.

{_markdown_table(tee_diagnostic)}

## Non-decision partial-hole diagnostic

These values are retained only to debug the transition chain. Hazard samples
are excluded because the prototype has no drop location or penalty-stroke cost,
which makes arm distributions differently conditioned and therefore **not
properly comparable**. They are excluded from the valid metric summary and
must not be used to select a model.

{_markdown_table(proxy_diagnostic)}

## Generated-state backoff audit

{_markdown_table(support)}

## Interpretation guardrails

- `partial_value_crps` and `hole_proxy_crps` are conditional non-hazard diagnostics only. They are not decision metrics until explicit penalty/drop, ARG, and putting transitions exist.
- The `_mean_anchored` diagnostics remove only the incremental fitted player-APP mean. `player_app` is anchored to this harness's `population`; `both` is anchored to this harness's `player_tee`. They isolate incremental APP shape from its fitted mean. Production skill is not loaded, so this does not validate replacing or preserving the production mean.
- Generated water/other/green tee states currently back off to recovery APP donors. This harness does **not** estimate penalty incidence, add a penalty stroke, or simulate a drop. Causal stroke -> penalty -> drop linkage is a hard prerequisite for any full-hole use, alongside explicit ARG branches.
- One recurring-course edition is a feasibility result, not production validation.
"""


def write_combined_report(
    output_dir: Path, diagnostic_dir: Path, sensitivity_dir: Path
) -> tuple[Path, Path, Path]:
    """Combine three strict-prior event runs using events as equal clusters."""
    event_slugs = (
        "wyndham_championship",
        "sony_open_in_hawaii",
        "travelers_championship",
    )
    focus = (
        "tee_endpoint_nll", "tee_remaining_crps", "tee_lie_logloss",
        "endpoint_nll", "remaining_crps", "lie_logloss",
    )

    def load_directory(directory: Path) -> pd.DataFrame:
        parts = []
        for event_slug in event_slugs:
            path = directory / f"integrated_metrics_{event_slug}_historical.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame["event"] = event_slug
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)

    primary = load_directory(diagnostic_dir)
    primary = primary[
        primary["metric"].isin(focus)
        & primary["arm"].isin(["player_tee", "player_app", "both"])
    ].copy()
    equal_rows = []
    for (arm, metric), group in primary.groupby(["arm", "metric"], sort=True):
        values = group["delta_vs_population"].to_numpy(float)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(len(values)))
        # df=2 Student-t critical. With three events this is descriptive, not a
        # generalization claim.
        half_width = 4.303 * se
        equal_rows.append(
            {
                "arm": arm,
                "metric": metric,
                "events": len(values),
                "equal_event_mean_delta": mean,
                "descriptive_ci_low": mean - half_width,
                "descriptive_ci_high": mean + half_width,
                "events_better": int(np.sum(values < 0)),
                "events_worse": int(np.sum(values > 0)),
            }
        )
    equal = pd.DataFrame(equal_rows)

    sensitivity_parts = []
    configurations = (
        ("128_seed_20260820", diagnostic_dir),
        ("128_seed_20260821", sensitivity_dir),
        ("256_seed_20260820", output_dir),
    )
    for label, directory in configurations:
        frame = load_directory(directory)
        frame = frame[
            frame["metric"].isin(["endpoint_nll", "remaining_crps", "lie_logloss"])
            & frame["arm"].isin(["player_app", "both"])
        ]
        summary = frame.groupby(["arm", "metric"], as_index=False)[
            "delta_vs_population"
        ].mean()
        summary["configuration"] = label
        sensitivity_parts.append(summary)
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)

    tee_sensitivity_parts = []
    for label, directory in configurations[:2]:
        frame = load_directory(directory)
        frame = frame[
            frame["arm"].eq("player_tee")
            & frame["metric"].isin(
                ["tee_endpoint_nll", "tee_remaining_crps", "tee_lie_logloss"]
            )
        ]
        summary = frame.groupby(["arm", "metric"], as_index=False)[
            "delta_vs_population"
        ].mean()
        summary["configuration"] = label
        tee_sensitivity_parts.append(summary)
    tee_sensitivity = pd.concat(tee_sensitivity_parts, ignore_index=True)

    ablation_parts = []
    for mode in ("combined", "mean_only", "scale_only"):
        suffix = "" if mode == "combined" else f"_{mode}"
        path = output_dir / (
            "integrated_metrics_wyndham_championship_historical" + suffix + ".csv"
        )
        frame = pd.read_csv(path)
        frame = frame[
            frame["arm"].eq("player_app")
            & frame["metric"].isin(
                ["endpoint_nll", "endpoint_error_yd", "remaining_crps", "lie_logloss"]
            )
        ][["metric", "delta_vs_population", "ci_low", "ci_high"]].copy()
        frame["app_profile_mode"] = mode
        ablation_parts.append(frame)
    ablations = pd.concat(ablation_parts, ignore_index=True)

    equal_path = output_dir / "three_event_equal_event_metrics.csv"
    sensitivity_path = output_dir / "three_event_seed_sensitivity.csv"
    ablation_path = output_dir / "wyndham_app_ablation_metrics.csv"
    equal.to_csv(equal_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    ablations.to_csv(ablation_path, index=False)

    tee_table = equal[
        equal["arm"].eq("player_tee") & equal["metric"].str.startswith("tee_")
    ]
    app_table = equal[
        equal["arm"].isin(["player_app", "both"])
        & equal["metric"].isin(["endpoint_nll", "remaining_crps", "lie_logloss"])
    ]
    sensitivity_table = sensitivity.pivot_table(
        index=["arm", "metric"],
        columns="configuration",
        values="delta_vs_population",
    ).reset_index()
    tee_sensitivity_table = tee_sensitivity.pivot_table(
        index=["arm", "metric"],
        columns="configuration",
        values="delta_vs_population",
    ).reset_index()
    report_path = output_dir / "THREE_EVENT_INTEGRATED_SUMMARY.md"
    report_path.write_text(
        f"""# Three-event strict-prior tee -> APP summary

## Verdict

The integrated mechanism is computationally and statistically testable, but
these results do **not** establish a player-specific APP improvement. Valid APP
score deltas are small, confidence intervals span zero, and several signs move
between 128/256 draws or seeds. The SG-importance player APP weighting remains
rejected/experimental. No production recommendation follows from this test.

The tee-state diagnostic is more encouraging: player tee transport improves
the Sony subset and is mostly neutral at Wyndham/Travelers. Downstream tee ->
APP scores are nevertheless neutral, so the next research step is a directly
estimated coordinate profile with stronger shrinkage, not more weight on the
current SG tilt.

## Primary three-event summary

The primary runs use strict pre-2026 setup geometry, 120 player-rounds per
event, 128 simulations, and the same held-out cases. Events receive equal
weight. Intervals use the event-mean standard error and a df=2 t critical;
with only three courses they are descriptive and intentionally wide. Negative
deltas are better.

### Generated APP start / tee diagnostic

{_markdown_table(tee_table)}

The tee signal is stable across the two 128-draw seeds: equal-event remaining
CRPS is about **-0.123** in both and lie log loss is **-0.00377/-0.00370**.

{_markdown_table(tee_sensitivity_table)}

### Valid shot-2 state scores

{_markdown_table(app_table)}

## Monte Carlo sensitivity

Equal-event mean valid-score deltas move materially across the following runs.
That instability is part of the result, not evidence to average away.

{_markdown_table(sensitivity_table)}

## Wyndham player-APP ablation

`mean_only` uses only the shrunk APP SG location tilt; `scale_only` uses only
the SG-residual dispersion tilt. Neither produces a robust valid-score gain.

{_markdown_table(ablations[["app_profile_mode", "metric", "delta_vs_population", "ci_low", "ci_high"]])}

## Validity boundaries

- Only endpoint density/error, remaining-distance CRPS, lie log loss/Brier,
  and the direct tee-state analogues are treated as valid.
- Partial-hole and hole-score proxies are excluded because this prototype has
  no explicit penalty/drop cost and therefore conditions hazards differently
  by arm.
- Ordinary APP lies are endpoint-linked through a prior-course KNN surface
  after per-edition rigid geometry alignment. Holeouts are terminal pin/green
  atoms. Penalty incidence is not identified: causal stroke -> penalty -> drop
  linkage is absent and required before any full-hole interpretation.
- The evaluation subset contains clean continuous stroke-2 APP sequences; it
  does not validate penalty, recovery, ARG, putting, or full-hole recursion.
""",
        encoding="utf-8",
    )
    return report_path, equal_path, sensitivity_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-name", default=DEFAULT_EVENT)
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument("--profile-start-year", type=int, default=2022)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--max-player-rounds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--case-sample-seed",
        type=int,
        default=20260820,
        help="Fixed target-subset seed, kept separate from Monte Carlo seed sensitivity.",
    )
    parser.add_argument("--setup-mode", choices=["actual", "historical"], default="actual")
    parser.add_argument(
        "--app-profile-mode",
        choices=["combined", "mean_only", "scale_only"],
        default="combined",
        help="Ablate player APP SG location versus dispersion reweighting.",
    )
    parser.add_argument(
        "--tee-profiles-csv",
        type=Path,
        default=Path("permanent_data/hole_sim_prototype/player_tee_profiles_wyndham_championship_actual.csv"),
    )
    parser.add_argument("--app-profiles-csv", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=DEFAULT_OUTPUT / "diagnostic_128",
    )
    parser.add_argument(
        "--sensitivity-dir",
        type=Path,
        default=DEFAULT_OUTPUT / "seed_20260821_strict_128",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.combine_only:
        paths = write_combined_report(
            args.output_dir, args.diagnostic_dir, args.sensitivity_dir
        )
        print("Wrote " + ", ".join(str(path) for path in paths))
        return
    event_keys = tee._event_keys(args.export_root, args.event_name, args.holdout_year)
    print(f"Selected recurring editions: {event_keys}")
    original_course_drives, target_drives = tee.load_course_panel(
        args.export_root, event_keys, args.holdout_year, args.course_start_year
    )
    course_drives, target_drives, forecast = _setup_forecast(
        original_course_drives, target_drives, args.setup_mode
    )
    course_approaches, target_raw, target_sg, target_geometry = build_course_approaches(
        args.export_root, event_keys, args.holdout_year, args.course_start_year
    )
    course_approaches = add_common_course_frame(course_approaches, forecast)
    if not args.tee_profiles_csv.exists():
        raise FileNotFoundError(
            f"Missing strict-prior tee profiles: {args.tee_profiles_csv}. "
            "Run hole_sim_tee_transition.py first or pass --tee-profiles-csv."
        )
    tee_profiles = pd.read_csv(args.tee_profiles_csv, dtype={"source_player_id": str})
    tour_training = load_tour_app_training(
        args.export_root,
        args.holdout_year - 1 if args.smoke_test else args.profile_start_year,
        args.holdout_year,
    )
    if args.app_profiles_csv:
        app_profiles = pd.read_csv(args.app_profiles_csv, dtype={"source_player_id": str})
        app_globals = {
            "global_sg_sd": float("nan"), "mean_prior": 100.0, "variance_prior": 250.0
        }
    else:
        app_profiles, app_globals = fit_app_profiles(tour_training)
    value_model = StateValueModel(tour_training)
    cases = build_target_cases(
        target_drives,
        target_raw,
        target_sg,
        target_geometry,
        forecast,
        args.setup_mode,
        8 if args.smoke_test else args.max_player_rounds,
        args.case_sample_seed,
    )
    if args.smoke_test:
        cases = cases.head(60).copy()
        args.simulations = min(args.simulations, 48)
    print(
        f"Tee train={len(course_drives):,}; APP donors={len(course_approaches):,}; "
        f"target cases={len(cases):,}; tee profiles={len(tee_profiles):,}; "
        f"APP profiles={len(app_profiles):,}"
    )
    app_model = HierarchicalApproachModel(course_approaches)
    app_surface = ApproachEndpointSurface(course_approaches)
    scores, metrics = evaluate(
        cases,
        course_drives,
        tee_profiles,
        app_profiles,
        app_model,
        app_surface,
        value_model,
        args.simulations,
        args.seed,
        args.app_profile_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", args.event_name.casefold()).strip("_")
    mode_suffix = "" if args.app_profile_mode == "combined" else f"_{args.app_profile_mode}"
    suffix = f"_{slug}_{args.setup_mode}{mode_suffix}" + ("_smoke" if args.smoke_test else "")
    score_path = args.output_dir / f"integrated_scores{suffix}.csv"
    metric_path = args.output_dir / f"integrated_metrics{suffix}.csv"
    profile_path = args.output_dir / f"player_app_profiles{suffix}.csv"
    report_path = args.output_dir / f"INTEGRATED_REPORT{suffix}.md"
    scores.to_csv(score_path, index=False)
    metrics.to_csv(metric_path, index=False)
    if not args.app_profiles_csv:
        app_profiles.to_csv(profile_path, index=False)
    report_path.write_text(
        report(
            args,
            course_drives,
            course_approaches,
            cases,
            tee_profiles,
            app_profiles,
            scores,
            metrics,
            app_globals,
        ),
        encoding="utf-8",
    )
    print(metrics[metrics["arm"].ne("population")].to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

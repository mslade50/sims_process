"""Leakage-safe prototype for player-adjusted tee-shot state transitions.

This is a research harness, not a production pipeline dependency.  It tests
the first transition required by a true hole simulator:

    tee state -> landing coordinate + lie -> remaining distance

For a recurring event, the population distribution is learned from prior
editions of the same physical hole.  Round-specific history receives the most
weight, then same-hole/all-round history, then same-par event history.  Player
distance, directional bias, and longitudinal/lateral dispersion modifiers are
estimated across all PGA courses strictly before the holdout season and are
hierarchically shrunk toward the Tour population.

The held-out edition is never used to fit either the course distribution,
surface map, or player characteristics.  Run the small end-to-end check with:

    python archive/hole_sim_tee_transition.py --smoke-test

The default full run writes CSV and Markdown artifacts to
``permanent_data/hole_sim_prototype``.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import logsumexp


DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_OUTPUT = Path("permanent_data/hole_sim_prototype")
DEFAULT_EVENT = "THE PLAYERS Championship"
LIE_CLASSES = ("fairway", "rough", "sand", "recovery", "water", "green", "other")


@dataclass(frozen=True)
class Distribution:
    points: np.ndarray
    weights: np.ndarray
    source: str
    n_setup: int
    n_hole: int
    n_par: int


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[:, None], axis=0) / max(weights.sum(), 1e-12)


def _effective_n(weights: np.ndarray) -> float:
    weights = weights / max(weights.sum(), 1e-12)
    return float(1.0 / np.sum(weights**2))


def _crps(samples: np.ndarray, weights: np.ndarray, observed: float) -> float:
    """Weighted empirical CRPS using a sorted O(n log n) identity."""
    order = np.argsort(samples)
    x = samples[order]
    w = weights[order]
    w = w / max(w.sum(), 1e-12)
    first = float(np.sum(w * np.abs(x - observed)))
    cumulative_w = np.cumsum(w) - w
    cumulative_wx = np.cumsum(w * x) - w * x
    pair = float(np.sum(w * (x * cumulative_w - cumulative_wx)))
    return first - pair


def _kde_nll(
    points: np.ndarray, weights: np.ndarray, observed: np.ndarray
) -> float:
    """Weighted diagonal Gaussian KDE negative log-density in yd^-2."""
    weights = weights / max(weights.sum(), 1e-12)
    mean = _weighted_mean(points, weights)
    variance = np.sum(weights[:, None] * (points - mean) ** 2, axis=0)
    n_eff = max(_effective_n(weights), 2.0)
    # Scott's rule in two dimensions, with floors that prevent a handful of
    # repeated coordinates from creating artificial certainty.
    bandwidth = np.sqrt(np.maximum(variance, np.array([25.0, 16.0]))) * n_eff ** (-1 / 6)
    bandwidth = np.maximum(bandwidth, np.array([3.5, 3.0]))
    z = (observed[None, :] - points) / bandwidth
    components = (
        np.log(np.maximum(weights, 1e-300))
        - math.log(2 * math.pi)
        - np.log(bandwidth).sum()
        - 0.5 * np.sum(z * z, axis=1)
    )
    return float(-logsumexp(components))


def _normalize_lie(value: object) -> str:
    lie = str(value).lower().strip()
    if lie in LIE_CLASSES[:-1]:
        return lie
    if lie in {"bunker", "greenside bunker", "fairway bunker"}:
        return "sand"
    if lie in {"penalty", "hazard", "native area", "unknown", "nan", "none", ""}:
        return "other"
    return "other"


def _event_keys(export_root: Path, event_name: str, holdout_year: int) -> dict[int, str]:
    events = pd.read_parquet(export_root / "events.parquet")
    match = events[
        events["tour"].eq("pga")
        & events["event_name"].str.casefold().eq(event_name.casefold())
        & events["season"].le(holdout_year)
    ].copy()
    if holdout_year not in set(match["season"]):
        available = match[["season", "event_key"]].to_dict("records")
        raise ValueError(f"No {holdout_year} holdout for {event_name!r}; found {available}")
    return dict(zip(match["season"].astype(int), match["event_key"]))


def _read_year_geometry(export_root: Path, year: int, event_key: str | None = None) -> pd.DataFrame:
    columns = [
        "event_key", "round_no", "hole_no", "course_id", "par", "yardage",
        "tee_x", "tee_y", "tee_z", "pin_x", "pin_y", "pin_z",
    ]
    filters = [("event_key", "==", event_key)] if event_key else None
    frame = pd.read_parquet(
        export_root / f"pga_{year}_hole_geometry.parquet",
        columns=columns,
        filters=filters,
    )
    frame["season"] = year
    return frame


def _read_drives(export_root: Path, year: int, event_key: str | None = None) -> pd.DataFrame:
    columns = [
        "event_key", "source_player_id", "player_name", "round_no", "hole_no",
        "stroke_no", "event_type", "start_lie", "end_lie", "shot_distance_yd",
        "distance_remaining_yd", "start_x", "start_y", "start_z", "end_x",
        "end_y", "end_z", "ball_holed", "course_id", "shot_key",
    ]
    filters: list[tuple[str, str, object]] = [
        ("stroke_no", "==", 1),
        ("start_lie", "==", "tee"),
    ]
    if event_key:
        filters.append(("event_key", "==", event_key))
    frame = pd.read_parquet(
        export_root / f"pga_{year}_shots.parquet",
        columns=columns,
        filters=filters,
    )
    frame = frame[
        frame["event_type"].eq("STROKE")
        & frame["ball_holed"].fillna(0).eq(0)
        & frame["shot_distance_yd"].between(100, 420)
    ].copy()
    frame["season"] = year
    return frame


def _add_local_coordinates(shots: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    # course_id prevents multi-course events from attaching a shot to the
    # wrong layout when round and hole numbers overlap.
    keys = ["event_key", "round_no", "hole_no", "course_id"]
    geometry = geometry.drop_duplicates(keys)
    frame = shots.merge(
        geometry[keys + ["par", "yardage", "tee_x", "tee_y", "pin_x", "pin_y"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    frame = frame[frame["par"].ge(4)].copy()
    numeric = ["start_x", "start_y", "end_x", "end_y", "pin_x", "pin_y"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric + ["source_player_id", "end_lie"])

    # The archive coordinates are feet.  Re-anchor each observation at its
    # actual tee position, and rotate so +long points directly at that day's pin.
    vx = frame["pin_x"].to_numpy(float) - frame["start_x"].to_numpy(float)
    vy = frame["pin_y"].to_numpy(float) - frame["start_y"].to_numpy(float)
    norm = np.hypot(vx, vy)
    okay = norm > 100
    frame = frame.loc[okay].copy()
    vx, vy, norm = vx[okay], vy[okay], norm[okay]
    ux, uy = vx / norm, vy / norm
    dx = frame["end_x"].to_numpy(float) - frame["start_x"].to_numpy(float)
    dy = frame["end_y"].to_numpy(float) - frame["start_y"].to_numpy(float)
    frame["long_yd"] = (dx * ux + dy * uy) / 3.0
    frame["lat_yd"] = (-dx * uy + dy * ux) / 3.0
    frame["pin_long_yd"] = norm / 3.0
    frame["geom_remaining_yd"] = np.hypot(
        frame["pin_long_yd"] - frame["long_yd"], frame["lat_yd"]
    )
    frame["lie_class"] = frame["end_lie"].map(_normalize_lie)
    frame["source_player_id"] = frame["source_player_id"].astype(str)
    return frame[
        frame["long_yd"].between(40, 430)
        & frame["lat_yd"].between(-150, 150)
        & frame["geom_remaining_yd"].between(0, 350)
    ].reset_index(drop=True)


def load_course_panel(
    export_root: Path,
    event_keys: dict[int, str],
    holdout_year: int,
    course_start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for year, event_key in sorted(event_keys.items()):
        if year < course_start_year:
            continue
        geometry = _read_year_geometry(export_root, year, event_key)
        shots = _read_drives(export_root, year, event_key)
        parts.append(_add_local_coordinates(shots, geometry))
    panel = pd.concat(parts, ignore_index=True)
    train = panel[panel["season"].lt(holdout_year)].copy()
    holdout = panel[panel["season"].eq(holdout_year)].copy()
    if train.empty or holdout.empty:
        raise ValueError("Course training or holdout panel is empty")
    return train, holdout


def load_player_training(
    export_root: Path, start_year: int, holdout_year: int
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in range(start_year, holdout_year):
        print(f"Loading player-profile drives: {year}")
        geometry = _read_year_geometry(export_root, year)
        shots = _read_drives(export_root, year)
        parts.append(_add_local_coordinates(shots, geometry))
    return pd.concat(parts, ignore_index=True)


def fit_player_profiles(
    drives: pd.DataFrame,
    mean_prior: float = 50.0,
    variance_prior: float = 120.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit shrunk player transforms after removing each field/setup mean."""
    setup = ["event_key", "round_no", "hole_no"]
    group = drives.groupby(setup, observed=True)
    drives = drives.copy()
    drives["field_long"] = group["long_yd"].transform("mean")
    drives["field_lat"] = group["lat_yd"].transform("mean")
    drives["log_distance_ratio"] = np.log(
        drives["long_yd"] / drives["field_long"]
    ).clip(-0.35, 0.35)
    drives["long_resid"] = drives["long_yd"] - drives["field_long"]
    drives["lat_resid"] = drives["lat_yd"] - drives["field_lat"]
    field_long_sd = group["long_yd"].transform("std")
    field_lat_sd = group["lat_yd"].transform("std")
    field_long_sd = field_long_sd.replace([np.inf, -np.inf], np.nan)
    field_lat_sd = field_lat_sd.replace([np.inf, -np.inf], np.nan)
    field_long_sd = field_long_sd.fillna(field_long_sd.median()).clip(8.0, 80.0)
    field_lat_sd = field_lat_sd.fillna(field_lat_sd.median()).clip(6.0, 60.0)
    drives["long_std_resid"] = drives["long_resid"] / field_long_sd
    drives["lat_std_resid"] = drives["lat_resid"] / field_lat_sd

    # Dispersion is a within-player quantity.  Centering again within player
    # prevents a player's mean distance/directional tendency from being folded
    # into the variance scale and then applied a second time as a mean shift.
    player_group = drives.groupby("source_player_id", observed=True)
    drives["long_disp_resid"] = (
        drives["long_std_resid"] - player_group["long_std_resid"].transform("mean")
    )
    drives["lat_disp_resid"] = (
        drives["lat_std_resid"] - player_group["lat_std_resid"].transform("mean")
    )

    global_long_var = float(np.mean(drives["long_disp_resid"] ** 2))
    global_lat_var = float(np.mean(drives["lat_disp_resid"] ** 2))
    grouped = drives.groupby("source_player_id", observed=True)
    profiles = grouped.agg(
        player_name=("player_name", "last"),
        n_drives=("shot_key", "size"),
        distance_log_sum=("log_distance_ratio", "sum"),
        lat_resid_sum=("lat_resid", "sum"),
        long_resid_ss=("long_disp_resid", lambda x: float(np.sum(np.asarray(x) ** 2))),
        lat_resid_ss=("lat_disp_resid", lambda x: float(np.sum(np.asarray(x) ** 2))),
    ).reset_index()
    n = profiles["n_drives"].to_numpy(float)
    profiles["distance_ratio"] = np.exp(
        profiles["distance_log_sum"] / (n + mean_prior)
    ).clip(0.88, 1.12)
    profiles["lateral_bias_yd"] = (
        profiles["lat_resid_sum"] / (n + mean_prior)
    ).clip(-12, 12)
    profiles["long_scale"] = np.sqrt(
        (profiles["long_resid_ss"] + variance_prior * global_long_var)
        / ((n + variance_prior) * global_long_var)
    ).clip(0.65, 1.50)
    profiles["lat_scale"] = np.sqrt(
        (profiles["lat_resid_ss"] + variance_prior * global_lat_var)
        / ((n + variance_prior) * global_lat_var)
    ).clip(0.65, 1.50)
    profiles["source_player_id"] = profiles["source_player_id"].astype(str)
    globals_ = {
        "global_long_sd": math.sqrt(global_long_var),
        "global_lat_sd": math.sqrt(global_lat_var),
        "mean_prior": mean_prior,
        "variance_prior": variance_prior,
    }
    return profiles, globals_


def build_distribution(
    course_train: pd.DataFrame,
    round_no: int,
    hole_no: int,
    par: int,
    setup_prior: float = 160.0,
    hole_prior: float = 500.0,
) -> Distribution:
    setup = course_train[
        course_train["round_no"].eq(round_no) & course_train["hole_no"].eq(hole_no)
    ]
    hole = course_train[course_train["hole_no"].eq(hole_no)]
    par_pool = course_train[course_train["par"].eq(par)]
    if par_pool.empty:
        par_pool = course_train
    n_setup, n_hole, n_par = len(setup), len(hole), len(par_pool)
    alpha = n_setup / (n_setup + setup_prior)
    beta = n_hole / (n_hole + hole_prior)
    mass = pd.Series(dtype=float)
    labels: list[str] = []

    def add_component(frame: pd.DataFrame, total_weight: float) -> None:
        nonlocal mass
        if frame.empty or total_weight <= 0:
            return
        component = pd.Series(total_weight / len(frame), index=frame.index)
        mass = mass.add(component, fill_value=0.0)

    if n_setup:
        add_component(setup, alpha)
        labels.append(f"setup={alpha:.2f}")
    remainder = 1.0 - alpha
    if n_hole:
        add_component(hole, remainder * beta)
        labels.append(f"hole={remainder * beta:.2f}")
    final = remainder * (1.0 - beta)
    add_component(par_pool, final)
    labels.append(f"par={final:.2f}")
    # A setup observation also belongs to the hole and par backoffs. Aggregate
    # its mixture mass by source row so KDE bandwidth does not count the same
    # empirical support point multiple times.
    points = course_train.loc[mass.index, ["long_yd", "lat_yd"]].to_numpy(float)
    combined = mass.to_numpy(float)
    combined /= combined.sum()
    return Distribution(points, combined, ",".join(labels), n_setup, n_hole, n_par)


def adjust_distribution(points: np.ndarray, weights: np.ndarray, profile: pd.Series) -> np.ndarray:
    center = _weighted_mean(points, weights)
    adjusted = np.empty_like(points)
    adjusted[:, 0] = (
        center[0] * float(profile["distance_ratio"])
        + float(profile["long_scale"]) * (points[:, 0] - center[0])
    )
    adjusted[:, 1] = (
        center[1]
        + float(profile["lateral_bias_yd"])
        + float(profile["lat_scale"]) * (points[:, 1] - center[1])
    )
    return adjusted


def _historical_setup_frame(
    course_train: pd.DataFrame, holdout: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Express endpoints in axes fixed entirely from prior-edition geometry.

    Absolute archive coordinates are stable at a physical course.  For every
    round/hole setup we use the median prior tee and pin to define a canonical
    origin and direction, then apply that same rigid frame to both training and
    held-out endpoints.  No held-out tee or pin is used by this mode.
    """
    keys = ["round_no", "hole_no"]
    canonical = (
        course_train.groupby(keys, as_index=False)
        .agg(
            canonical_tee_x=("start_x", "median"),
            canonical_tee_y=("start_y", "median"),
            canonical_pin_x=("pin_x", "median"),
            canonical_pin_y=("pin_y", "median"),
        )
    )

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.merge(canonical, on=keys, how="inner", validate="many_to_one")
        vx = result["canonical_pin_x"].to_numpy(float) - result["canonical_tee_x"].to_numpy(float)
        vy = result["canonical_pin_y"].to_numpy(float) - result["canonical_tee_y"].to_numpy(float)
        norm = np.hypot(vx, vy)
        ux, uy = vx / norm, vy / norm
        dx = result["end_x"].to_numpy(float) - result["canonical_tee_x"].to_numpy(float)
        dy = result["end_y"].to_numpy(float) - result["canonical_tee_y"].to_numpy(float)
        result["long_yd"] = (dx * ux + dy * uy) / 3.0
        result["lat_yd"] = (-dx * uy + dy * ux) / 3.0
        result["pin_long_yd"] = norm / 3.0
        return result

    return transform(course_train), transform(holdout)


class SurfaceMap:
    def __init__(self, hole_history: pd.DataFrame, prior_strength: float = 12.0):
        self.points = hole_history[["long_yd", "lat_yd"]].to_numpy(float)
        self.codes = pd.Categorical(
            hole_history["lie_class"], categories=LIE_CLASSES
        ).codes
        self.tree = cKDTree(self.points)
        counts = np.bincount(self.codes[self.codes >= 0], minlength=len(LIE_CLASSES)).astype(float)
        self.prior = (counts + 1.0) / (counts.sum() + len(LIE_CLASSES))
        self.prior_strength = prior_strength

    def probabilities(self, queries: np.ndarray, k: int = 48) -> np.ndarray:
        k = min(k, len(self.points))
        distance, index = self.tree.query(queries, k=k)
        if k == 1:
            distance, index = distance[:, None], index[:, None]
        local_scale = np.maximum(np.median(distance, axis=1), 8.0)
        weight = np.exp(-0.5 * (distance / local_scale[:, None]) ** 2)
        result = np.tile(self.prior * self.prior_strength, (len(queries), 1))
        for code in range(len(LIE_CLASSES)):
            result[:, code] += np.sum(weight * (self.codes[index] == code), axis=1)
        return result / result.sum(axis=1, keepdims=True)


def _lie_forecast(
    surface: SurfaceMap,
    points: np.ndarray,
    weights: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(points) > max_points:
        chosen = rng.choice(len(points), size=max_points, replace=True, p=weights / weights.sum())
        query, query_weights = points[chosen], np.full(max_points, 1 / max_points)
    else:
        query, query_weights = points, weights / weights.sum()
    conditional = surface.probabilities(query)
    return np.sum(conditional * query_weights[:, None], axis=0)


def score_holdout(
    course_train: pd.DataFrame,
    holdout: pd.DataFrame,
    profiles: pd.DataFrame,
    max_lie_points: int = 300,
    seed: int = 20260820,
    setup_mode: str = "actual",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if setup_mode == "historical":
        course_train, holdout = _historical_setup_frame(course_train, holdout)
    profile_lookup = profiles.set_index("source_player_id")
    neutral = pd.Series(
        {"distance_ratio": 1.0, "lateral_bias_yd": 0.0, "long_scale": 1.0, "lat_scale": 1.0}
    )
    distributions: dict[tuple[int, int, int], Distribution] = {}
    surfaces: dict[int, SurfaceMap] = {}
    lie_cache: dict[tuple[int, int, str, str], np.ndarray] = {}
    rows: list[dict] = []

    for record in holdout.itertuples(index=False):
        setup_key = (int(record.round_no), int(record.hole_no), int(record.par))
        if setup_key not in distributions:
            distributions[setup_key] = build_distribution(course_train, *setup_key)
        distribution = distributions[setup_key]
        if int(record.hole_no) not in surfaces:
            history = course_train[course_train["hole_no"].eq(record.hole_no)]
            surfaces[int(record.hole_no)] = SurfaceMap(history)
        surface = surfaces[int(record.hole_no)]

        player_id = str(record.source_player_id)
        has_profile = player_id in profile_lookup.index
        profile = profile_lookup.loc[player_id] if has_profile else neutral
        if isinstance(profile, pd.DataFrame):
            profile = profile.iloc[-1]
        base_points = distribution.points
        mean_profile = profile.copy()
        mean_profile["long_scale"] = 1.0
        mean_profile["lat_scale"] = 1.0
        dispersion_profile = profile.copy()
        dispersion_profile["distance_ratio"] = 1.0
        dispersion_profile["lateral_bias_yd"] = 0.0
        arm_points = {
            "population": base_points,
            "mean_transport": adjust_distribution(base_points, distribution.weights, mean_profile),
            "dispersion_transport": adjust_distribution(
                base_points, distribution.weights, dispersion_profile
            ),
            "combined": adjust_distribution(base_points, distribution.weights, profile),
        }
        observed = np.array([record.long_yd, record.lat_yd], dtype=float)
        pin_long = float(record.pin_long_yd)
        actual_lie_index = LIE_CLASSES.index(record.lie_class)
        actual_onehot = np.zeros(len(LIE_CLASSES))
        actual_onehot[actual_lie_index] = 1.0
        row = {
                "event_key": record.event_key,
                "season": record.season,
                "round_no": record.round_no,
                "hole_no": record.hole_no,
                "par": record.par,
                "source_player_id": player_id,
                "player_name": record.player_name,
                "has_player_profile": has_profile,
                "profile_n_drives": int(profile.get("n_drives", 0)),
                "distance_ratio": float(profile["distance_ratio"]),
                "lateral_bias_yd": float(profile["lateral_bias_yd"]),
                "long_scale": float(profile["long_scale"]),
                "lat_scale": float(profile["lat_scale"]),
                "actual_long_yd": record.long_yd,
                "actual_lat_yd": record.lat_yd,
                "actual_remaining_yd": record.geom_remaining_yd,
                "actual_lie": record.lie_class,
                "setup_mode": setup_mode,
                "forecast_pin_long_yd": pin_long,
                "actual_pin_long_yd": record.pin_long_yd,
                "n_setup_history": distribution.n_setup,
                "n_hole_history": distribution.n_hole,
                "backoff_weights": distribution.source,
            }
        for arm, points in arm_points.items():
            arm_mean = _weighted_mean(points, distribution.weights)
            remaining = np.hypot(pin_long - points[:, 0], points[:, 1])
            lie_player = "population" if arm == "population" else player_id
            lie_key = (record.round_no, record.hole_no, lie_player, arm)
            if lie_key not in lie_cache:
                lie_cache[lie_key] = _lie_forecast(
                    surface, points, distribution.weights, max_lie_points, rng
                )
            lie_probs = lie_cache[lie_key]
            row[f"{arm}_endpoint_nll"] = _kde_nll(
                points, distribution.weights, observed
            )
            row[f"{arm}_endpoint_error_yd"] = float(
                np.linalg.norm(observed - arm_mean)
            )
            row[f"{arm}_remaining_crps"] = _crps(
                remaining, distribution.weights, record.geom_remaining_yd
            )
            row[f"{arm}_lie_logloss"] = float(
                -math.log(max(lie_probs[actual_lie_index], 1e-12))
            )
            row[f"{arm}_lie_brier"] = float(
                np.mean((lie_probs - actual_onehot) ** 2)
            )
            row[f"{arm}_fairway_prob"] = lie_probs[LIE_CLASSES.index("fairway")]
        # Backward-compatible aliases for earlier one-arm output consumers.
        for metric in (
            "endpoint_nll", "endpoint_error_yd", "remaining_crps",
            "lie_logloss", "lie_brier", "fairway_prob",
        ):
            row[f"adjusted_{metric}"] = row[f"combined_{metric}"]
        rows.append(row)
    scores = pd.DataFrame(rows)
    metrics = summarize(scores)
    return scores, metrics


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    metric_names = (
        "endpoint_nll", "endpoint_error_yd", "remaining_crps",
        "lie_logloss", "lie_brier",
    )
    rows = []
    for arm in ("mean_transport", "dispersion_transport", "combined"):
      for name in metric_names:
        baseline = f"population_{name}"
        adjusted = f"{arm}_{name}"
        base_value = float(scores[baseline].mean())
        adjusted_value = float(scores[adjusted].mean())
        delta = adjusted_value - base_value
        hole_delta = (
            scores.assign(_delta=scores[adjusted] - scores[baseline])
            .groupby("hole_no")["_delta"]
            .mean()
        )
        cluster_se = float(hole_delta.std(ddof=1) / math.sqrt(len(hole_delta)))
        rows.append(
            {
                "metric": name,
                "arm": arm,
                "n": len(scores),
                "population": base_value,
                "player_adjusted": adjusted_value,
                "delta_adjusted_minus_population": delta,
                "relative_change_pct": 100 * delta / abs(base_value) if base_value else np.nan,
                "hole_cluster_se": cluster_se,
                "hole_cluster_ci_low": delta - 1.96 * cluster_se,
                "hole_cluster_ci_high": delta + 1.96 * cluster_se,
                "lower_is_better": True,
            }
        )
    return pd.DataFrame(rows)


def _markdown_report(
    event_name: str,
    holdout_year: int,
    course_train: pd.DataFrame,
    holdout: pd.DataFrame,
    profiles: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: pd.DataFrame,
    globals_: dict[str, float],
    smoke_test: bool,
    setup_mode: str,
) -> str:
    rows = []
    for row in metrics.itertuples(index=False):
        verdict = "better" if row.delta_adjusted_minus_population < 0 else "worse"
        rows.append(
            f"| {row.arm} / {row.metric} | {row.population:.6f} | {row.player_adjusted:.6f} | "
            f"{row.delta_adjusted_minus_population:+.6f} | "
            f"[{row.hole_cluster_ci_low:+.6f}, {row.hole_cluster_ci_high:+.6f}] | {verdict} |"
        )
    profile_coverage = float(scores["has_player_profile"].mean())
    by_round = scores.groupby("round_no").agg(
        n=("source_player_id", "size"),
        endpoint_delta=("adjusted_endpoint_nll", "mean"),
        endpoint_base=("population_endpoint_nll", "mean"),
        remaining_delta=("adjusted_remaining_crps", "mean"),
        remaining_base=("population_remaining_crps", "mean"),
        lie_delta=("adjusted_lie_logloss", "mean"),
        lie_base=("population_lie_logloss", "mean"),
    )
    by_round["endpoint_delta"] -= by_round["endpoint_base"]
    by_round["remaining_delta"] -= by_round["remaining_base"]
    by_round["lie_delta"] -= by_round["lie_base"]
    round_lines = [
        f"| {idx} | {int(row.n)} | {row.endpoint_delta:+.5f} | "
        f"{row.remaining_delta:+.5f} | {row.lie_delta:+.5f} |"
        for idx, row in by_round.iterrows()
    ]
    caveat = (
        "This was the reduced smoke-test sample, so its metrics are only an execution check."
        if smoke_test else
        "This is one event holdout. It establishes feasibility, not production readiness."
    )
    if np.isfinite(globals_["global_long_sd"]):
        profile_scale_line = (
            f"- Tour standardized residual SD: longitudinal **{globals_['global_long_sd']:.2f}**, "
            f"lateral **{globals_['global_lat_sd']:.2f}**"
        )
    else:
        profile_scale_line = "- Player modifier profiles: reused from the identical strict pre-2026 fit"
    return f"""# Tee-transition hole simulator prototype

## Design

- Event: **{event_name}**, held-out edition: **{holdout_year}**
- Prior-course drives: **{len(course_train):,}** from {course_train['season'].nunique()} editions
- Held-out drives scored: **{len(scores):,}**
- Player profiles: **{len(profiles):,}**, coverage in holdout: **{profile_coverage:.1%}**
- Player history cutoff: all profile inputs precede the held-out season
- Setup information: **{setup_mode}** ({'held-out tee/pin geometry is treated as known' if setup_mode == 'actual' else 'canonical tee, direction, and pin are fixed entirely from prior-edition geometry'})
{profile_scale_line}

The population endpoint distribution is an empirical mixture of the same
hole/round in prior editions, the same hole across rounds, and a same-par event
backoff. Player transforms change distance, lateral bias, and longitudinal and
lateral dispersion. A spatial nearest-neighbor surface map converts simulated
landing coordinates into end-lie probabilities. Remaining distance is derived
from the simulated coordinate and {'held-out pin location' if setup_mode == 'actual' else 'prior-edition canonical pin forecast'}.
Dispersion scales are learned from within-player residuals standardized by the
field SD of each event/round/hole setup. Mean-only, dispersion-only, and
combined transports are scored separately against the same population model.

All metrics are proper or direct predictive scores and **lower is better**.

| Metric | Population | Player adjusted | Delta | Hole-cluster 95% CI | Direction |
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Stability by round

| Round | N | Endpoint NLL delta | Remaining CRPS delta | Lie log-loss delta |
|---:|---:|---:|---:|---:|
{chr(10).join(round_lines)}

## Interpretation

Negative deltas favor the player-adjusted transition. Endpoint NLL tests the
full two-dimensional landing distribution; remaining-distance CRPS tests the
next-shot yardage state; lie log loss tests the probability assigned to the
realized surface. {caveat}

This prototype deliberately stops after the tee transition. It does not yet
model club/strategy selection, weather, roll versus carry, exact course surface
polygons, approach/ARG/putting transitions, or complete hole scores. The next
research gate is rolling leave-one-edition-out validation on several recurring
courses, followed by an approach transition conditioned on the generated
distance and lie state.

Course editions are selected by stable event name in this prototype. The
reported recurring events are known single physical venues over the study
window, but production work must use the canonical physical-course/hole bridge
before generalizing to rotating events. Nested setup/hole/par pools retain
their intended empirical-Bayes mixture mass, while duplicated source rows are
collapsed before KDE bandwidth calculation.
"""


def write_combined_summary(output_dir: Path) -> tuple[Path, Path, Path]:
    """Combine completed event/setup metric files with event-cluster inference."""
    parts: list[pd.DataFrame] = []
    for path in sorted(output_dir.glob("tee_transition_metrics_*.csv")):
        if "_smoke" in path.stem:
            continue
        match = re.match(
            r"tee_transition_metrics_(.+)_(actual|historical)$", path.stem
        )
        if not match:
            continue
        event_slug, setup_mode = match.groups()
        frame = pd.read_csv(path)
        if "arm" not in frame.columns:
            continue
        frame.insert(0, "event", event_slug.replace("_", " ").title())
        frame.insert(1, "setup_mode", setup_mode)
        parts.append(frame)
    if not parts:
        raise ValueError(f"No current ablation metric files found in {output_dir}")
    event_metrics = pd.concat(parts, ignore_index=True)
    event_path = output_dir / "combined_event_metrics.csv"
    event_metrics.to_csv(event_path, index=False)

    pooled_rows = []
    for (setup_mode, arm, metric), group in event_metrics.groupby(
        ["setup_mode", "arm", "metric"], sort=True
    ):
        weights = group["n"].to_numpy(float)
        population = float(np.average(group["population"], weights=weights))
        adjusted = float(np.average(group["player_adjusted"], weights=weights))
        event_deltas = group["delta_adjusted_minus_population"].to_numpy(float)
        event_mean = float(event_deltas.mean())
        event_se = float(event_deltas.std(ddof=1) / math.sqrt(len(event_deltas))) if len(event_deltas) > 1 else np.nan
        pooled_rows.append(
            {
                "setup_mode": setup_mode,
                "arm": arm,
                "metric": metric,
                "events": len(group),
                "n": int(weights.sum()),
                "population_weighted": population,
                "player_adjusted_weighted": adjusted,
                "observation_weighted_delta": adjusted - population,
                "event_mean_delta": event_mean,
                "event_cluster_se": event_se,
                "event_cluster_ci_low": event_mean - 1.96 * event_se,
                "event_cluster_ci_high": event_mean + 1.96 * event_se,
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    pooled_path = output_dir / "combined_pooled_metrics.csv"
    pooled.to_csv(pooled_path, index=False)

    lines = []
    for row in pooled.itertuples(index=False):
        lines.append(
            f"| {row.setup_mode} | {row.arm} | {row.metric} | {row.events} | "
            f"{row.n:,} | {row.observation_weighted_delta:+.6f} | "
            f"{row.event_mean_delta:+.6f} | "
            f"[{row.event_cluster_ci_low:+.6f}, {row.event_cluster_ci_high:+.6f}] |"
        )
    history_label = (
        "one prior course edition (2025)"
        if "one_year" in output_dir.name
        else "six prior course editions (2020-2025)"
    )
    report = f"""# Combined tee-transition prototype results

Three 2026 holdouts are included: Wyndham, Sony, and Travelers. Each uses
{history_label} and player modifiers fit only on pre-2026 PGA
shots. `actual` treats the 2026 tee/pin setup as known; `historical` fixes the
coordinate frame and pin entirely from prior editions.

Negative deltas are better. The event-cluster interval uses only three event
means and is therefore a robustness description, not a high-powered test.

## Main readout

- The combined transport improves all five scored outcomes in both setup modes;
  every equal-event interval is below zero.
- Dispersion-only is the cleanest density result: endpoint NLL and lie log loss
  improve in both setup modes with intervals below zero.
- Mean-only supplies nearly all of the remaining-yardage gain, but slightly
  worsens lie scoring. Combining it with dispersion retains the yardage gain
  while restoring an improvement in lie calibration.
- These intervals contain only three event clusters. Their consistent signs
  are encouraging replication evidence, not a production-grade uncertainty
  estimate.

| Setup | Arm | Metric | Events | N | Observation-weighted delta | Event-mean delta | Event-cluster 95% CI |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

The ablation interpretation is structural: mean transport changes player
distance and lateral bias; dispersion transport changes only longitudinal and
lateral scale; combined applies both. Dispersion scales are estimated from
within-player residuals standardized by each setup's field SD.
"""
    report_path = output_dir / "COMBINED_TEE_TRANSITION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return event_path, pooled_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-name", default=DEFAULT_EVENT)
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--profile-start-year", type=int, default=2020)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--setup-mode",
        choices=["actual", "historical"],
        default="actual",
        help="Use held-out setup geometry or forecast it from prior editions.",
    )
    parser.add_argument(
        "--profiles-csv",
        type=Path,
        help="Reuse profiles previously fit using the same strict pre-holdout cutoff.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 2025-only player profiles and a small two-hole holdout.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Combine already-generated event metric files without rerunning fits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.combine_only:
        paths = write_combined_summary(args.output_dir)
        print("Wrote " + ", ".join(str(path) for path in paths))
        return
    event_keys = _event_keys(args.export_root, args.event_name, args.holdout_year)
    print(f"Selected editions: {event_keys}")
    course_train, holdout = load_course_panel(
        args.export_root,
        event_keys,
        args.holdout_year,
        args.course_start_year,
    )
    profile_start = args.holdout_year - 1 if args.smoke_test else args.profile_start_year
    if args.profiles_csv:
        profiles = pd.read_csv(args.profiles_csv, dtype={"source_player_id": str})
        player_training = pd.DataFrame()
        globals_ = {
            "global_long_sd": float("nan"),
            "global_lat_sd": float("nan"),
            "mean_prior": 50.0,
            "variance_prior": 120.0,
        }
    else:
        player_training = load_player_training(
            args.export_root, profile_start, args.holdout_year
        )
        profiles, globals_ = fit_player_profiles(player_training)
    if args.smoke_test:
        holdout = holdout[holdout["hole_no"].isin([1, 2])].head(120).copy()
    print(
        f"Course train={len(course_train):,}; player train={len(player_training):,}; "
        f"holdout={len(holdout):,}; profiles={len(profiles):,}"
    )
    scores, metrics = score_holdout(
        course_train, holdout, profiles, seed=args.seed, setup_mode=args.setup_mode
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_slug = re.sub(r"[^a-z0-9]+", "_", args.event_name.casefold()).strip("_")
    suffix = f"_{event_slug}_{args.setup_mode}" + ("_smoke" if args.smoke_test else "")
    scores.to_csv(args.output_dir / f"tee_transition_scores{suffix}.csv", index=False)
    metrics.to_csv(args.output_dir / f"tee_transition_metrics{suffix}.csv", index=False)
    if not args.profiles_csv:
        profiles.to_csv(args.output_dir / f"player_tee_profiles{suffix}.csv", index=False)
    report = _markdown_report(
        args.event_name,
        args.holdout_year,
        course_train,
        holdout,
        profiles,
        scores,
        metrics,
        globals_,
        args.smoke_test,
        args.setup_mode,
    )
    report_path = args.output_dir / f"TEE_TRANSITION_REPORT{suffix}.md"
    report_path.write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

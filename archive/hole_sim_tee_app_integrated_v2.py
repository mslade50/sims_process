"""Strict-prior particle simulator for the spatial tee -> APP chain.

This research harness replaces the SG-importance APP layer in
``hole_sim_tee_app_integrated.py`` with the audited coordinate transition in
``hole_sim_app_transition.py``.  Every APP draw is conditioned on a generated
tee endpoint, generated lie and generated remaining distance.  The held-out
tee result is used only for scoring.

Four paired arms share random numbers within every player-hole case::

    population  population tee, population APP
    player_tee  player tee transport, population APP
    player_app  population tee, structured player APP transport
    both        player tee transport, structured player APP transport

All setup geometry, course donors and player profiles precede the holdout
season.  Historical editions are rigidly aligned into the latest prior
edition's physical-course frame.  Particle states are persisted so later ARG
and putting models can continue the same paths.

Penalty/drop resolution is deliberately gated.  A spatially predicted hazard
retains its physical endpoint, but its next lie, drop distance, penalty count
and stroke cost are unknown until a full raw-sequence resolver is connected.
No full-hole score is fabricated from those particles, and nothing in
production imports this archived module.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import hole_sim_app_transition as app
import hole_sim_tee_app_integrated as v1
import hole_sim_tee_transition as tee


DEFAULT_EXPORT = app.DEFAULT_EXPORT
DEFAULT_OUTPUT = Path("permanent_data/hole_sim_tee_app_integrated_v2")
DEFAULT_TEE_PROFILES = Path(
    "permanent_data/hole_sim_prototype/player_tee_profiles.csv"
)
DEFAULT_APP_PROFILES = Path(
    "permanent_data/hole_sim_app_transition_strict/player_app_profiles_structured.csv"
)
DEFAULT_EVENTS = app.DEFAULT_EVENTS
ARMS = ("population", "player_tee", "player_app", "both")
PLAYABLE_TEE_LIES = ("fairway", "rough", "sand", "recovery")
METRICS = (
    "tee_endpoint_nll",
    "tee_endpoint_error_yd",
    "tee_remaining_crps",
    "tee_lie_logloss",
    "tee_lie_brier",
    "app_endpoint_nll",
    "app_endpoint_error_yd",
    "app_remaining_crps_ordinary",
    "app_lie_logloss",
    "app_lie_brier",
)

# The imported full-sequence foundation supplies causal APP endpoints and
# exact next-physical-stroke states. A separate endpoint-linked spatial
# resolver owns donor selection; until it is attached here, hazard particles
# fail closed even though aligned resolution columns are preserved as hooks.
PENALTY_RESOLUTION_READY = False
COORDINATE_FRAME_VERSION = "verified_raw_to_latest_prior_rigid_v1"

HazardSupportChecker = Callable[[np.ndarray], np.ndarray]
HazardSupportFactory = Callable[[object, str], HazardSupportChecker | None]


def _supported_surface_probabilities(
    probabilities: np.ndarray,
    empirical_codes: np.ndarray,
    *,
    class_count: int,
    enforce_empirical_class_support: bool,
    endpoints: np.ndarray | None = None,
    hazard_class_indices: tuple[int, ...] = (),
    hazard_support_checker: HazardSupportChecker | None = None,
) -> np.ndarray:
    """Make the generative lie surface compatible with recursive mechanics.

    The transition-evaluation surfaces intentionally use pseudocounts for
    proper scoring.  A recursive generator has a stricter contract: it may not
    draw a class that has no physical donor on that hole, and it may not draw a
    hazard whose endpoint-linked resolver has no compatible mechanic.  This
    optional post-processing is therefore enabled by the full-chain wrapper,
    while the standalone transition evaluator retains its original smoothing.
    """

    result = np.asarray(probabilities, float).copy()
    if result.ndim != 2 or result.shape[1] != int(class_count):
        raise ValueError("Surface probability matrix has an unexpected shape")
    if enforce_empirical_class_support:
        codes = np.asarray(empirical_codes, int)
        counts = np.bincount(codes[codes >= 0], minlength=int(class_count))
        result[:, counts[: int(class_count)] == 0] = 0.0
    if hazard_support_checker is not None and hazard_class_indices:
        if endpoints is None:
            raise ValueError("Hazard support checks require generated endpoints")
        supported = np.asarray(hazard_support_checker(np.asarray(endpoints, float)), bool)
        if supported.shape != (len(result),):
            raise ValueError("Hazard support checker must return one flag per endpoint")
        unsupported_rows = np.flatnonzero(~supported)
        if len(unsupported_rows):
            result[np.ix_(unsupported_rows, np.asarray(hazard_class_indices, int))] = 0.0
    totals = result.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("Surface support masking removed all probability mass")
    return result / totals[:, None]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _strict_prior_event_scope(
    export_root: Path,
    event_name: str,
    holdout_year: int,
) -> tuple[dict[int, str], dict[str, object]]:
    """Verify donor venue against the latest prior edition, never holdout geometry."""
    prior_keys = app._event_keys(export_root, event_name, holdout_year - 1)
    events = pd.read_parquet(
        export_root / "events.parquet",
        columns=["event_key", "tour", "season", "event_name"],
    )
    target = events[
        events["tour"].str.casefold().eq("pga")
        & events["season"].eq(holdout_year)
        & events["event_name"].str.casefold().eq(event_name.casefold())
    ]
    if len(target) != 1:
        raise ValueError(
            f"Expected one metadata-only {holdout_year} event key for {event_name!r}; "
            f"found {len(target)}"
        )
    keys = dict(prior_keys)
    keys[holdout_year] = str(target.iloc[0]["event_key"])
    return keys, {
        "venue_scope_mode": "strict_prior_geometry_identity",
        "venue_identity_reference_year": int(max(prior_keys)),
        "holdout_event_key_source": "events_metadata_only",
        "holdout_geometry_used_for_predictor_eligibility": False,
    }


def _load_prior_tee_penalty_labels(
    export_root: Path,
    event_keys: dict[int, str],
    holdout_year: int,
) -> pd.DataFrame:
    """Return sequence-safe causal OTT penalty labels for prior tee shots."""
    labels: list[pd.DataFrame] = []
    raw_columns = [
        "shot_key", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "event_type", "start_lie", "end_lie",
        "distance_remaining_yd", "ball_holed", "start_x", "start_y", "end_x",
        "end_y", "course_id",
    ]
    sg_columns = [
        "shot_key", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "sg_category", "start_lie",
        "start_distance_yd", "end_lie", "end_distance_yd", "sg_valid",
        "sg_issue", "complete_hole", "state_inferred",
    ]
    state_columns = [
        "shot_key", "sg_category", "start_lie", "start_distance_yd",
        "end_lie", "end_distance_yd",
    ]
    for season, event_key in sorted(event_keys.items()):
        if season >= holdout_year:
            continue
        raw = pd.read_parquet(
            export_root / f"pga_{season}_shots.parquet",
            columns=raw_columns,
            filters=[("event_key", "==", event_key)],
        )
        app_sg = pd.read_parquet(
            export_root / f"pga_{season}_shot_sg.parquet",
            columns=sg_columns,
            filters=[("event_key", "==", event_key), ("sg_category", "==", "app")],
        )
        states = pd.read_parquet(
            export_root / f"pga_{season}_shot_sg.parquet",
            columns=state_columns,
            filters=[("event_key", "==", event_key)],
        )
        _, links, _ = app.link_app_penalties_full_sequence(
            raw, app_sg, states, int(season)
        )
        causal = links[
            links["causal_linked"]
            & links["causal_sg_sg_category"].eq("ott")
        ].copy()
        if causal.empty:
            continue
        grouped = causal.groupby("causal_shot_key", as_index=False).agg(
            tee_penalty_count=("penalty_shot_key", "size"),
        ).rename(columns={"causal_shot_key": "shot_key"})
        grouped["season"] = int(season)
        labels.append(grouped)
    if not labels:
        return pd.DataFrame(columns=["shot_key", "tee_penalty_count", "season"])
    result = pd.concat(labels, ignore_index=True)
    if result["shot_key"].duplicated().any():
        raise ValueError("Causal tee penalty labels are not unique by shot_key")
    return result


def _choice_from_uniform(weights: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, np.asarray(uniforms, float), side="right").clip(
        0, len(weights) - 1
    )


def _categorical_from_uniform(
    probabilities: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:
    cumulative = np.cumsum(np.asarray(probabilities, float), axis=1)
    return np.sum(np.asarray(uniforms)[:, None] > cumulative, axis=1).clip(
        0, probabilities.shape[1] - 1
    )


def _uniform_crps(samples: np.ndarray, observed: float) -> float:
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return np.nan
    return app._crps(samples, np.full(len(samples), 1.0 / len(samples)), observed)


def _markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("nan" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _actual_app_lie(value: object, holed: bool) -> str:
    if holed:
        return "hole"
    lie = str(value).strip().casefold()
    if lie in {"water", "penalty", "hazard", "native area"}:
        return "penalty"
    return app._normalize_end_lie(lie)


def _select_app_profile(
    player_id: str,
    start_lie: str,
    distance_yd: float,
    band_lookup: pd.DataFrame,
    base_lookup: pd.DataFrame,
    neutral: pd.Series,
) -> tuple[pd.Series, bool, bool]:
    band = app._audited_distance_band_scalar(float(distance_yd))
    key = (str(player_id), str(start_lie), band)
    has_player = str(player_id) in base_lookup.index
    has_band = key in band_lookup.index
    if has_band:
        profile = band_lookup.loc[key]
    elif has_player:
        profile = base_lookup.loc[str(player_id)].copy()
        profile["long_bias_yd"] = 0.0
        profile["n_band"] = 0
    else:
        profile = neutral
    if isinstance(profile, pd.DataFrame):
        profile = profile.iloc[-1]
    return profile, has_player, has_band


def _align_app_course_to_forecast(
    course_train_raw: pd.DataFrame,
    event_keys: dict[int, str],
    forecast: pd.DataFrame,
    export_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align causal APP donors and endpoints into the prior setup frame."""
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    pin_table = forecast[
        ["round_no", "hole_no", "forecast_pin_x", "forecast_pin_y"]
    ].rename(columns={"forecast_pin_x": "pin_x", "forecast_pin_y": "pin_y"})
    forecast_tee = forecast[
        ["round_no", "hole_no", "forecast_tee_x", "forecast_tee_y"]
    ]
    for season, event_key in sorted(event_keys.items()):
        if season >= int(course_train_raw["season"].max()) + 1:
            continue
        event = course_train_raw[
            course_train_raw["event_key"].eq(event_key)
        ].copy()
        if event.empty:
            continue
        geometry = tee._read_year_geometry(export_root, int(season), event_key)
        rigid = v1._fit_rigid_geometry(geometry, forecast)
        coordinate_prefixes = ["start", "end"]
        for optional_prefix in ("resolution_start", "drop_end"):
            if {
                f"{optional_prefix}_x", f"{optional_prefix}_y"
            }.issubset(event.columns):
                coordinate_prefixes.append(optional_prefix)
        for prefix in coordinate_prefixes:
            aligned = v1._apply_rigid(event[f"{prefix}_x"], event[f"{prefix}_y"], rigid)
            event[f"{prefix}_x"] = aligned[:, 0]
            event[f"{prefix}_y"] = aligned[:, 1]
        event["geometry_alignment_rmse_yd"] = rigid[3]
        parts.append(event)
        diagnostics.append(
            {
                "season": int(season),
                "event_key": event_key,
                "rows": len(event),
                "alignment_rmse_yd": rigid[3],
            }
        )
    if not parts:
        raise ValueError("No strictly prior APP rows survived physical-course alignment")
    aligned = pd.concat(parts, ignore_index=True)
    aligned = app._apply_reference_pins(aligned, pin_table)
    aligned = aligned.merge(
        forecast_tee, on=["round_no", "hole_no"], how="inner", validate="many_to_one"
    )
    dx = aligned["end_x"].to_numpy(float) - aligned["forecast_tee_x"].to_numpy(float)
    dy = aligned["end_y"].to_numpy(float) - aligned["forecast_tee_y"].to_numpy(float)
    vx = aligned["pin_x"].to_numpy(float) - aligned["forecast_tee_x"].to_numpy(float)
    vy = aligned["pin_y"].to_numpy(float) - aligned["forecast_tee_y"].to_numpy(float)
    norm = np.hypot(vx, vy)
    ux = vx / norm
    uy = vy / norm
    aligned["common_end_long_yd"] = (dx * ux + dy * uy) / 3.0
    aligned["common_end_lat_yd"] = (-dx * uy + dy * ux) / 3.0
    if {"resolution_start_x", "resolution_start_y"}.issubset(aligned.columns):
        resolution_dx = (
            aligned["resolution_start_x"].to_numpy(float)
            - aligned["forecast_tee_x"].to_numpy(float)
        )
        resolution_dy = (
            aligned["resolution_start_y"].to_numpy(float)
            - aligned["forecast_tee_y"].to_numpy(float)
        )
        aligned["resolution_common_long_yd"] = (
            resolution_dx * ux + resolution_dy * uy
        ) / 3.0
        aligned["resolution_common_lat_yd"] = (
            -resolution_dx * uy + resolution_dy * ux
        ) / 3.0
        aligned["resolution_reference_distance_yd"] = np.hypot(
            aligned["resolution_common_long_yd"]
            - norm / 3.0,
            aligned["resolution_common_lat_yd"],
        )
        resolution_lie = aligned.get(
            "resolution_start_lie", pd.Series(index=aligned.index, dtype=object)
        )
        resolution_lie = resolution_lie.fillna(
            aligned.get(
                "resolution_start_lie_raw",
                pd.Series(index=aligned.index, dtype=object),
            )
        )
        aligned["resolution_start_lie_class"] = resolution_lie.map(
            app._normalize_start_lie
        )
    aligned = aligned[
        aligned["reference_start_distance_yd"].between(0.1, 700)
        & aligned["common_end_long_yd"].between(-150, 850)
        & aligned["common_end_lat_yd"].between(-300, 300)
    ].copy()
    aligned["coordinate_frame_version"] = COORDINATE_FRAME_VERSION
    aligned["coordinate_frame_reference_year"] = int(
        forecast["setup_reference_year"].iloc[0]
    ) if "setup_reference_year" in forecast else int(aligned["season"].max())
    aligned["source_coordinate_provenance"] = "raw_shotlink_single_alignment"
    return aligned, pd.DataFrame(diagnostics)


def _load_verified_app_course_raw(
    export_root: Path,
    event_keys: dict[int, str],
    holdout_year: int,
    course_start_year: int,
) -> pd.DataFrame:
    """Load full-sequence APP rows without applying a coordinate transform.

    ``event_keys`` has already passed the state-foundation physical-course
    gate. The direct APP module's public course loader intentionally maps rows
    into the holdout frame; v2 cannot use that panel because its deployable
    frame is the latest verified *prior* setup. This loader preserves raw
    coordinates and the full-sequence causal/resolution fields so v2 applies
    exactly one edition -> prior-reference transform.
    """
    parts: list[pd.DataFrame] = []
    for season, event_key in sorted(event_keys.items()):
        if season < course_start_year or season >= holdout_year:
            continue
        geometry = app._read_geometry(export_root, season, [event_key])
        rows = app._read_approach_rows(export_root, season, [event_key])
        prepared = app._attach_geometry(rows, geometry)
        prepared["coordinate_frame_version"] = "raw_shotlink_unaligned"
        parts.append(prepared[prepared["state_valid"]].copy())
    if not parts:
        raise ValueError("No verified raw prior-edition APP rows")
    return pd.concat(parts, ignore_index=True)


def _build_all_hole_forecast(
    export_root: Path,
    event_keys: dict[int, str],
    holdout_year: int,
) -> tuple[pd.DataFrame, int]:
    """Use the latest verified prior edition for all 72 round-hole setups."""
    prior_years = sorted(year for year in event_keys if year < holdout_year)
    if not prior_years:
        raise ValueError("No verified prior edition is available for setup geometry")
    reference_year = prior_years[-1]
    geometry = tee._read_year_geometry(
        export_root, reference_year, event_keys[reference_year]
    )
    forecast = (
        geometry.groupby(["round_no", "hole_no"], as_index=False)
        .agg(
            par=("par", "first"),
            yardage=("yardage", "median"),
            forecast_tee_x=("tee_x", "median"),
            forecast_tee_y=("tee_y", "median"),
            forecast_pin_x=("pin_x", "median"),
            forecast_pin_y=("pin_y", "median"),
        )
        .sort_values(["round_no", "hole_no"])
        .reset_index(drop=True)
    )
    expected = pd.MultiIndex.from_product(
        [range(1, 5), range(1, 19)], names=["round_no", "hole_no"]
    )
    observed = pd.MultiIndex.from_frame(forecast[["round_no", "hole_no"]])
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(
            f"Latest verified prior geometry is not a 4x18 grid; missing {list(missing)}"
        )
    dx = forecast["forecast_pin_x"] - forecast["forecast_tee_x"]
    dy = forecast["forecast_pin_y"] - forecast["forecast_tee_y"]
    forecast["forecast_pin_long_yd"] = np.hypot(dx, dy) / 3.0
    forecast["setup_reference_year"] = reference_year
    return forecast, reference_year


def _align_drives_to_forecast(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    export_root: Path,
) -> pd.DataFrame:
    """Express par-4/5 tee endpoints in the all-hole prior setup frame."""
    parts: list[pd.DataFrame] = []
    for (season, event_key), event in frame.groupby(
        ["season", "event_key"], observed=True
    ):
        geometry = tee._read_year_geometry(export_root, int(season), str(event_key))
        rigid = v1._fit_rigid_geometry(geometry, forecast)
        aligned = v1._apply_rigid(event["end_x"], event["end_y"], rigid)
        event = event.copy()
        event["aligned_end_x"] = aligned[:, 0]
        event["aligned_end_y"] = aligned[:, 1]
        event["geometry_alignment_rmse_yd"] = rigid[3]
        parts.append(event)
    result = pd.concat(parts, ignore_index=True).merge(
        forecast[
            [
                "round_no", "hole_no", "forecast_tee_x", "forecast_tee_y",
                "forecast_pin_x", "forecast_pin_y", "forecast_pin_long_yd",
            ]
        ],
        on=["round_no", "hole_no"],
        how="inner",
        validate="many_to_one",
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
    return result.reset_index(drop=True)


def _surface_in_common_frame(course_app: pd.DataFrame) -> dict[int, app.AbsoluteSurfaceMap]:
    surfaces: dict[int, app.AbsoluteSurfaceMap] = {}
    for hole_no, history in course_app.groupby("hole_no", observed=True):
        surface_rows = history.copy()
        surface_rows["end_x"] = surface_rows["common_end_long_yd"] * 3.0
        surface_rows["end_y"] = surface_rows["common_end_lat_yd"] * 3.0
        if surface_rows["spatial_valid"].any():
            surfaces[int(hole_no)] = app.AbsoluteSurfaceMap(surface_rows)
    return surfaces


def _prepare_target_cases(
    target_drives: pd.DataFrame,
    target_raw: pd.DataFrame,
    target_sg: pd.DataFrame,
    target_geometry: pd.DataFrame,
    forecast: pd.DataFrame,
    max_player_rounds: int,
    seed: int,
) -> pd.DataFrame:
    cases = v1.build_target_cases(
        target_drives,
        target_raw,
        target_sg,
        target_geometry,
        forecast,
        "historical",
        max_player_rounds,
        seed,
    )
    second_flags = target_raw[
        ["shot_key", "ball_holed"]
    ].drop_duplicates("shot_key").rename(
        columns={"shot_key": "second_shot_key", "ball_holed": "second_ball_holed"}
    )
    cases = cases.merge(second_flags, on="second_shot_key", how="left", validate="many_to_one")
    sx = cases["second_start_x_for_score"].to_numpy(float)
    sy = cases["second_start_y_for_score"].to_numpy(float)
    ex = cases["second_end_x_for_score"].to_numpy(float)
    ey = cases["second_end_y_for_score"].to_numpy(float)
    px = cases["forecast_pin_x"].to_numpy(float)
    py = cases["forecast_pin_y"].to_numpy(float)
    vx, vy = px - sx, py - sy
    norm = np.hypot(vx, vy)
    ux, uy = vx / norm, vy / norm
    dx, dy = ex - px, ey - py
    cases["actual_app_miss_long_yd"] = (dx * ux + dy * uy) / 3.0
    cases["actual_app_miss_lat_yd"] = (-dx * uy + dy * ux) / 3.0
    cases["actual_app_holed"] = cases["second_ball_holed"].fillna(0).astype(bool)
    cases["actual_app_lie"] = [
        _actual_app_lie(value, holed)
        for value, holed in zip(cases["second_end_lie"], cases["actual_app_holed"])
    ]
    return cases.reset_index(drop=True)


def prepare_event_foundations(
    export_root: Path,
    event_name: str,
    holdout_year: int = 2026,
    course_start_year: int = 2020,
    include_scoring_targets: bool = True,
) -> dict[str, object]:
    """Prepare exactly-once aligned inputs for v2 and full-chain wrappers.

    The returned frames are safe to reuse without another rigid transform.
    ``provenance['alignment_count']`` is a hard contract, not an inferred
    diagnostic: every admitted raw prior edition is mapped directly into the
    latest verified prior setup frame exactly once.
    """
    event_keys, venue_scope = _strict_prior_event_scope(
        export_root, event_name, holdout_year
    )
    prior_drive_parts: list[pd.DataFrame] = []
    for season, event_key in sorted(event_keys.items()):
        if season < course_start_year or season >= holdout_year:
            continue
        geometry = tee._read_year_geometry(export_root, season, event_key)
        shots = tee._read_drives(export_root, season, event_key)
        prior_drive_parts.append(tee._add_local_coordinates(shots, geometry))
    if not prior_drive_parts:
        raise ValueError("No verified prior-edition tee donors")
    original_drives = pd.concat(prior_drive_parts, ignore_index=True)
    target_drives_raw: pd.DataFrame | None = None
    if include_scoring_targets:
        target_geometry_for_tee = tee._read_year_geometry(
            export_root, holdout_year, event_keys[holdout_year]
        )
        target_shots_for_tee = tee._read_drives(
            export_root, holdout_year, event_keys[holdout_year]
        )
        target_drives_raw = tee._add_local_coordinates(
            target_shots_for_tee, target_geometry_for_tee
        )
    forecast, reference_year = _build_all_hole_forecast(
        export_root, event_keys, holdout_year
    )
    if reference_year >= holdout_year:
        raise AssertionError("Setup reference must strictly precede the holdout")
    course_drives = _align_drives_to_forecast(
        original_drives, forecast, export_root
    )
    tee_penalties = _load_prior_tee_penalty_labels(
        export_root, event_keys, holdout_year
    )
    course_drives = course_drives.merge(
        tee_penalties[["shot_key", "tee_penalty_count"]],
        on="shot_key",
        how="left",
        validate="one_to_one",
    )
    course_drives["tee_penalty_count"] = (
        course_drives["tee_penalty_count"].fillna(0).astype(int)
    )
    course_drives["tee_causal_penalty"] = course_drives["tee_penalty_count"].gt(0)
    # tee.SurfaceMap has no penalty category. Its explicit `other` class is a
    # gated hazard state in v2; preserving the causal endpoint makes the local
    # surface learn where penalties occur without pretending recovery is safe.
    course_drives.loc[course_drives["tee_causal_penalty"], "lie_class"] = "other"
    target_drives: pd.DataFrame | None = None
    if include_scoring_targets:
        # Held-out geometry is allowed only on this audit path, to express
        # observed endpoints in the frozen prior predictor frame for scoring.
        if target_drives_raw is None:
            raise AssertionError("Target drives were not loaded for scoring")
        target_drives = _align_drives_to_forecast(
            target_drives_raw, forecast, export_root
        )
    course_app_raw = _load_verified_app_course_raw(
        export_root, event_keys, holdout_year, course_start_year
    )
    course_app, alignment = _align_app_course_to_forecast(
        course_app_raw, event_keys, forecast, export_root
    )
    frame_versions = set(course_app["coordinate_frame_version"].astype(str))
    if frame_versions != {COORDINATE_FRAME_VERSION}:
        raise AssertionError(f"Unexpected APP coordinate frames: {frame_versions}")
    if include_scoring_targets:
        _, target_raw, target_sg, target_geometry = v1.build_course_approaches(
            export_root, event_keys, holdout_year, course_start_year
        )
    else:
        # Forecast grids use target rows only to choose evaluation identities
        # and round participation. Do not read coordinates, lies, distances,
        # results, target SG or held-out geometry on this path.
        target_raw = pd.read_parquet(
            export_root / f"pga_{holdout_year}_shots.parquet",
            columns=[
                "event_key", "source_player_id", "player_name", "round_no",
                "event_type",
            ],
            filters=[("event_key", "==", event_keys[holdout_year])],
        )
        target_sg = None
        target_geometry = None
    provenance = {
        "coordinate_frame_version": COORDINATE_FRAME_VERSION,
        "alignment_count": 1,
        "source_coordinate_provenance": "raw_shotlink_single_alignment",
        "setup_reference_year": int(reference_year),
        "holdout_year": int(holdout_year),
        "verified_event_keys": {int(year): key for year, key in event_keys.items()},
        "state_foundation_version": getattr(app, "FOUNDATION_VERSION", "unknown"),
        "penalty_resolution_ready": PENALTY_RESOLUTION_READY,
        "predictor_reference_scope": "latest_verified_prior_only",
        "heldout_scoring_transform_applied": bool(include_scoring_targets),
        "heldout_geometry_role": (
            "observed_endpoint_scoring_only" if include_scoring_targets else "none"
        ),
        "tee_causal_penalty_rows": int(course_drives["tee_causal_penalty"].sum()),
        "tee_penalty_surface_label": "other_gated_hazard",
        **venue_scope,
    }
    return {
        "event_keys": event_keys,
        "forecast": forecast,
        "course_drives": course_drives,
        "target_drives": target_drives,
        "course_app": course_app,
        "app_alignment": alignment,
        "target_raw": target_raw,
        "target_sg": target_sg,
        "target_geometry": target_geometry,
        "provenance": provenance,
    }


class SpatialAppSampler:
    """Reusable direct APP transition for arbitrary generated particle states.

    The full-chain wrapper may call :meth:`sample` after a tee shot, a layup,
    or any later miss that remains in the APP category. Inputs and outputs use
    the same strict-prior course frame in yards. The sampler deliberately does
    not route categories or resolve penalties.
    """

    def __init__(
        self,
        course_app: pd.DataFrame,
        profiles: pd.DataFrame,
        surfaces: dict[int, app.AbsoluteSurfaceMap] | None = None,
        *,
        enforce_empirical_class_support: bool = False,
    ):
        self.course_app = course_app
        self.surfaces = surfaces or _surface_in_common_frame(course_app)
        self.enforce_empirical_class_support = bool(
            enforce_empirical_class_support
        )
        self.cache: dict[tuple[int, int, str, int], app.AppDistribution] = {}
        profile_frame = profiles.copy()
        profile_frame["source_player_id"] = profile_frame["source_player_id"].astype(str)
        profile_frame["profile_distance_band"] = profile_frame[
            "profile_distance_band"
        ].astype(str)
        self.band_lookup = profile_frame.set_index(
            ["source_player_id", "start_lie_class", "profile_distance_band"]
        )
        self.base_lookup = profile_frame.groupby("source_player_id", observed=True).first()
        self.neutral = pd.Series(
            {
                "n_app": 0,
                "n_band": 0,
                "long_bias_yd": 0.0,
                "lat_bias_yd": 0.0,
                "long_scale": 1.0,
                "lat_scale": 1.0,
            }
        )

    def sample(
        self,
        start_endpoints_yd: np.ndarray,
        start_lies: np.ndarray,
        start_remaining_yd: np.ndarray,
        pin_endpoint_yd: np.ndarray,
        round_no: int,
        hole_no: int,
        par: int,
        player_id: str,
        player_transport: bool,
        uniforms: dict[str, np.ndarray],
        active: np.ndarray | None = None,
        hazard_support_checker: HazardSupportChecker | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample one direct APP transition per active particle.

        ``uniforms`` contains equally sized ``index``, ``holeout`` and ``lie``
        arrays. Reusing those arrays across arms provides common random
        numbers. ``par`` is carried in the public state contract for category
        routing, although the audited APP hierarchy itself conditions on
        round, hole, lie and distance.
        """
        del par
        starts = np.asarray(start_endpoints_yd, float)
        start_lies = np.asarray(start_lies, object)
        start_remaining = np.asarray(start_remaining_yd, float)
        pin = np.asarray(pin_endpoint_yd, float)
        count = len(starts)
        if starts.shape != (count, 2):
            raise ValueError("start_endpoints_yd must have shape [particles, 2]")
        required_uniforms = ("index", "holeout", "lie")
        if any(len(np.asarray(uniforms[name])) != count for name in required_uniforms):
            raise ValueError("APP uniform arrays must match the particle count")
        if active is None:
            active = np.ones(count, dtype=bool)
        else:
            active = np.asarray(active, bool).copy()
        active &= np.isfinite(starts).all(axis=1) & np.isfinite(start_remaining)
        active &= np.isin(start_lies, app.START_LIES[:-1])

        endpoints = np.full((count, 2), np.nan)
        remaining = np.full(count, np.nan)
        geometric_remaining = np.full(count, np.nan)
        end_lies = np.full(count, "not_attempted", dtype=object)
        holed = np.zeros(count, dtype=bool)
        lie_probabilities = np.full((count, len(app.END_LIES)), np.nan)
        status = np.full(count, "app_not_attempted", dtype=object)
        has_profile = np.zeros(count, dtype=bool)
        has_band_profile = np.zeros(count, dtype=bool)
        n_setup = np.zeros(count, dtype=int)
        n_hole = np.zeros(count, dtype=int)
        n_lie = np.zeros(count, dtype=int)

        surface = self.surfaces.get(int(hole_no))
        if surface is None:
            status[active] = "app_no_surface_support"
            active[:] = False
        state_groups: dict[tuple[str, int], list[int]] = {}
        for particle in np.flatnonzero(active):
            lie = app._normalize_start_lie(start_lies[particle])
            bucket = max(1, int(round(float(start_remaining[particle]) / 15.0)))
            state_groups.setdefault((lie, bucket), []).append(int(particle))

        for (start_lie, distance_bucket), positions_list in state_groups.items():
            positions = np.asarray(positions_list, int)
            cache_key = (int(round_no), int(hole_no), start_lie, distance_bucket)
            if cache_key not in self.cache:
                self.cache[cache_key] = app.build_distribution(
                    self.course_app,
                    cache_key[0],
                    cache_key[1],
                    cache_key[2],
                    distance_bucket * 15.0,
                )
            distribution = self.cache[cache_key]
            support = distribution.rows
            weights = distribution.weights
            spatial_mask = (
                support["spatial_valid"].to_numpy(bool)
                & ~support["is_holeout"].to_numpy(bool)
            )
            if spatial_mask.sum() < 12:
                status[positions] = "app_no_spatial_support"
                continue
            spatial_points = support.loc[
                spatial_mask, ["miss_long_yd", "miss_lat_yd"]
            ].to_numpy(float)
            spatial_weights = weights[spatial_mask]
            spatial_weights /= spatial_weights.sum()
            holeout_probability = float(
                weights[support["is_holeout"].to_numpy(bool)].sum()
            )
            profile, player_found, band_found = _select_app_profile(
                str(player_id),
                start_lie,
                distance_bucket * 15.0,
                self.band_lookup,
                self.base_lookup,
                self.neutral,
            )
            transported = app.adjust_points(
                spatial_points,
                spatial_weights,
                profile,
                "combined" if player_transport else "population",
            )
            sampled = _choice_from_uniform(
                spatial_weights, np.asarray(uniforms["index"])[positions]
            )
            misses = transported[sampled]
            vector = pin[None, :] - starts[positions]
            vector_distance = np.maximum(np.linalg.norm(vector, axis=1), 1e-8)
            unit = vector / vector_distance[:, None]
            perpendicular = np.column_stack([-unit[:, 1], unit[:, 0]])
            group_endpoints = (
                pin[None, :]
                + unit * misses[:, 0, None]
                + perpendicular * misses[:, 1, None]
            )
            group_holed = (
                np.asarray(uniforms["holeout"])[positions] < holeout_probability
            )
            group_endpoints[group_holed] = pin
            conditional_lie = _supported_surface_probabilities(
                surface.probabilities(group_endpoints),
                surface.codes,
                class_count=len(app.END_LIES),
                enforce_empirical_class_support=(
                    self.enforce_empirical_class_support
                ),
                endpoints=group_endpoints,
                hazard_class_indices=(app.END_LIES.index("penalty"),),
                hazard_support_checker=hazard_support_checker,
            )
            if np.any(group_holed):
                conditional_lie[group_holed] = 0.0
                conditional_lie[group_holed, app.END_LIES.index("hole")] = 1.0
            lie_codes = _categorical_from_uniform(
                conditional_lie, np.asarray(uniforms["lie"])[positions]
            )
            group_lies = np.asarray(app.END_LIES, object)[lie_codes]
            group_lies[group_holed] = "hole"
            group_geometric = np.linalg.norm(group_endpoints - pin[None, :], axis=1)
            # Full-sequence APP preparation relabels every official causal
            # hazard as ``penalty``. ``other`` is a legitimate catch-all lie,
            # not an assessed penalty stroke.
            penalty = group_lies == "penalty"
            group_remaining = group_geometric.copy()
            group_remaining[penalty] = np.nan
            group_remaining[group_holed] = 0.0

            endpoints[positions] = group_endpoints
            remaining[positions] = group_remaining
            geometric_remaining[positions] = group_geometric
            end_lies[positions] = group_lies
            holed[positions] = group_holed
            lie_probabilities[positions] = conditional_lie
            status[positions] = np.where(
                penalty,
                "app_hazard_unresolved",
                np.where(
                    group_holed,
                    "app_complete_holed",
                    np.where(
                        group_lies == "green", "app_green_ready_putt", "app_ready_arg"
                    ),
                ),
            )
            has_profile[positions] = player_found
            has_band_profile[positions] = band_found
            n_setup[positions] = distribution.n_setup
            n_hole[positions] = distribution.n_hole
            n_lie[positions] = distribution.n_lie
        penalty_unknown = status == "app_hazard_unresolved"
        penalty_count = np.where(penalty_unknown, np.nan, 0.0)
        return {
            "endpoints": endpoints,
            "remaining": remaining,
            "geometric_remaining": geometric_remaining,
            "end_lies": end_lies,
            "holed": holed,
            "lie_probabilities": lie_probabilities,
            "status": status,
            "penalty_count": penalty_count,
            "penalty_cost_strokes": penalty_count.copy(),
            "has_profile": has_profile,
            "has_band_profile": has_band_profile,
            "n_setup": n_setup,
            "n_hole": n_hole,
            "n_lie": n_lie,
        }


def _simulate_case_arm(
    record: object,
    tee_distribution: tee.Distribution,
    tee_surface: tee.SurfaceMap,
    tee_profile: pd.Series,
    course_app: pd.DataFrame,
    app_surfaces: dict[int, app.AbsoluteSurfaceMap],
    app_cache: dict[tuple[int, int, str, int], app.AppDistribution],
    app_band_lookup: pd.DataFrame,
    app_base_lookup: pd.DataFrame,
    neutral_app: pd.Series,
    player_tee: bool,
    player_app: bool,
    randoms: dict[str, np.ndarray],
    *,
    enforce_empirical_class_support: bool = False,
    tee_hazard_support_checker: HazardSupportChecker | None = None,
    app_hazard_support_checker: HazardSupportChecker | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | float | bool]]:
    simulations = len(randoms["tee_index"])
    tee_points = tee_distribution.points
    if player_tee:
        tee_points = tee.adjust_distribution(
            tee_points, tee_distribution.weights, tee_profile
        )
    tee_indices = _choice_from_uniform(tee_distribution.weights, randoms["tee_index"])
    tee_endpoints = tee_points[tee_indices]
    tee_lie_probabilities = _supported_surface_probabilities(
        tee_surface.probabilities(tee_endpoints),
        tee_surface.codes,
        class_count=len(tee.LIE_CLASSES),
        enforce_empirical_class_support=enforce_empirical_class_support,
        endpoints=tee_endpoints,
        hazard_class_indices=tuple(
            tee.LIE_CLASSES.index(name) for name in ("water", "other")
        ),
        hazard_support_checker=tee_hazard_support_checker,
    )
    tee_lie_codes = _categorical_from_uniform(
        tee_lie_probabilities, randoms["tee_lie"]
    )
    tee_lies = np.asarray(tee.LIE_CLASSES, object)[tee_lie_codes]
    pin = np.array([float(record.forecast_pin_long_yd), 0.0])
    tee_remaining = np.linalg.norm(pin[None, :] - tee_endpoints, axis=1)

    app_endpoints = np.full((simulations, 2), np.nan)
    app_remaining = np.full(simulations, np.nan)
    app_geometric_remaining = np.full(simulations, np.nan)
    app_lies = np.full(simulations, "not_attempted", dtype=object)
    app_holed = np.zeros(simulations, dtype=bool)
    app_lie_probabilities = np.full((simulations, len(app.END_LIES)), np.nan)
    statuses = np.full(simulations, "", dtype=object)
    has_app_profile = np.zeros(simulations, dtype=bool)
    has_band_profile = np.zeros(simulations, dtype=bool)
    n_setup = np.zeros(simulations, dtype=int)
    n_hole = np.zeros(simulations, dtype=int)
    n_lie = np.zeros(simulations, dtype=int)

    playable = np.isin(tee_lies, PLAYABLE_TEE_LIES) & np.isfinite(tee_remaining)
    tee_green = tee_lies == "green"
    statuses[tee_green] = "tee_green_ready_putt"
    statuses[~playable & ~tee_green] = "tee_hazard_unresolved"

    state_groups: dict[tuple[str, int], list[int]] = {}
    for particle in np.flatnonzero(playable):
        start_lie = app._normalize_start_lie(tee_lies[particle])
        distance_bucket = max(1, int(round(float(tee_remaining[particle]) / 15.0)))
        state_groups.setdefault((start_lie, distance_bucket), []).append(int(particle))

    surface = app_surfaces.get(int(record.hole_no))
    if surface is None:
        statuses[playable] = "app_no_surface_support"
        playable[:] = False

    for (start_lie, distance_bucket), positions_list in state_groups.items():
        positions = np.asarray(positions_list, int)
        if not np.any(playable[positions]):
            continue
        cache_key = (
            int(record.round_no), int(record.hole_no), start_lie, distance_bucket
        )
        if cache_key not in app_cache:
            app_cache[cache_key] = app.build_distribution(
                course_app,
                cache_key[0],
                cache_key[1],
                cache_key[2],
                distance_bucket * 15.0,
            )
        distribution = app_cache[cache_key]
        support = distribution.rows
        all_weights = distribution.weights
        spatial_mask = (
            support["spatial_valid"].to_numpy(bool)
            & ~support["is_holeout"].to_numpy(bool)
        )
        if spatial_mask.sum() < 12:
            statuses[positions] = "app_no_spatial_support"
            playable[positions] = False
            continue
        spatial_points = support.loc[
            spatial_mask, ["miss_long_yd", "miss_lat_yd"]
        ].to_numpy(float)
        spatial_weights = all_weights[spatial_mask]
        spatial_weights /= spatial_weights.sum()
        holeout_probability = float(
            all_weights[support["is_holeout"].to_numpy(bool)].sum()
        )
        profile, has_player, has_band = _select_app_profile(
            str(record.source_player_id),
            start_lie,
            distance_bucket * 15.0,
            app_band_lookup,
            app_base_lookup,
            neutral_app,
        )
        arm_name = "combined" if player_app else "population"
        transported = app.adjust_points(
            spatial_points, spatial_weights, profile, arm_name
        )
        sampled = _choice_from_uniform(
            spatial_weights, randoms["app_index"][positions]
        )
        misses = transported[sampled]
        starts = tee_endpoints[positions]
        vector = pin[None, :] - starts
        distance = np.maximum(np.linalg.norm(vector, axis=1), 1e-8)
        unit = vector / distance[:, None]
        perpendicular = np.column_stack([-unit[:, 1], unit[:, 0]])
        endpoints = (
            pin[None, :]
            + unit * misses[:, 0, None]
            + perpendicular * misses[:, 1, None]
        )
        holeout = randoms["app_holeout"][positions] < holeout_probability
        endpoints[holeout] = pin
        conditional_lie = _supported_surface_probabilities(
            surface.probabilities(endpoints),
            surface.codes,
            class_count=len(app.END_LIES),
            enforce_empirical_class_support=enforce_empirical_class_support,
            endpoints=endpoints,
            hazard_class_indices=(app.END_LIES.index("penalty"),),
            hazard_support_checker=app_hazard_support_checker,
        )
        if np.any(holeout):
            conditional_lie[holeout] = 0.0
            conditional_lie[holeout, app.END_LIES.index("hole")] = 1.0
        lie_codes = _categorical_from_uniform(
            conditional_lie, randoms["app_lie"][positions]
        )
        end_lies = np.asarray(app.END_LIES, object)[lie_codes]
        end_lies[holeout] = "hole"
        geometric_remaining = np.linalg.norm(endpoints - pin[None, :], axis=1)
        penalty = end_lies == "penalty"
        ordinary_remaining = geometric_remaining.copy()
        ordinary_remaining[penalty] = np.nan
        ordinary_remaining[holeout] = 0.0

        app_endpoints[positions] = endpoints
        app_geometric_remaining[positions] = geometric_remaining
        app_remaining[positions] = ordinary_remaining
        app_lies[positions] = end_lies
        app_holed[positions] = holeout
        app_lie_probabilities[positions] = conditional_lie
        statuses[positions] = np.where(
            penalty,
            "app_hazard_unresolved",
            np.where(
                holeout,
                "app_complete_holed",
                np.where(end_lies == "green", "app_green_ready_putt", "app_ready_arg"),
            ),
        )
        has_app_profile[positions] = has_player
        has_band_profile[positions] = has_band
        n_setup[positions] = distribution.n_setup
        n_hole[positions] = distribution.n_hole
        n_lie[positions] = distribution.n_lie

    tee_penalty_unknown = statuses == "tee_hazard_unresolved"
    app_penalty_unknown = statuses == "app_hazard_unresolved"
    tee_penalty_count = np.where(tee_penalty_unknown, np.nan, 0.0)
    app_penalty_count = np.where(app_penalty_unknown, np.nan, 0.0)
    particle_frame = pd.DataFrame(
        {
            "particle_id": np.arange(simulations, dtype=int),
            "particle_weight": np.full(simulations, 1.0 / simulations),
            "tee_end_long_yd": tee_endpoints[:, 0],
            "tee_end_lat_yd": tee_endpoints[:, 1],
            "tee_end_lie": tee_lies,
            "tee_remaining_yd": tee_remaining,
            "tee_holed": False,
            "tee_penalty_count": tee_penalty_count,
            "tee_penalty_cost_strokes": tee_penalty_count.copy(),
            "app_start_long_yd": np.where(playable, tee_endpoints[:, 0], np.nan),
            "app_start_lat_yd": np.where(playable, tee_endpoints[:, 1], np.nan),
            "app_start_lie": np.where(playable, tee_lies, "not_attempted"),
            "app_start_remaining_yd": np.where(playable, tee_remaining, np.nan),
            "app_end_long_yd": app_endpoints[:, 0],
            "app_end_lat_yd": app_endpoints[:, 1],
            "app_end_lie": app_lies,
            "app_geometric_remaining_yd": app_geometric_remaining,
            "app_remaining_yd": app_remaining,
            "app_holed": app_holed,
            "app_penalty_count": app_penalty_count,
            "app_penalty_cost_strokes": app_penalty_count.copy(),
            "shots_taken": np.where(np.isfinite(app_endpoints[:, 0]), 2, 1),
            "chain_status": statuses,
            "has_app_profile": has_app_profile,
            "has_app_band_profile": has_band_profile,
            "app_n_setup_history": n_setup,
            "app_n_hole_history": n_hole,
            "app_n_lie_history": n_lie,
        }
    )
    diagnostics: dict[str, np.ndarray | float | bool] = {
        "tee_endpoints": tee_endpoints,
        "tee_remaining": tee_remaining,
        "tee_lie_probabilities": tee_lie_probabilities,
        "app_endpoints": app_endpoints,
        "app_remaining": app_remaining,
        "app_lie_probabilities": app_lie_probabilities,
        "app_attempted": np.isfinite(app_endpoints[:, 0]),
        "tee_hazard_rate": float(np.mean(tee_penalty_unknown)),
        "app_hazard_rate": float(np.mean(app_penalty_unknown)),
        "penalty_resolution_ready": PENALTY_RESOLUTION_READY,
    }
    return particle_frame, diagnostics


def evaluate_event(
    event_name: str,
    cases: pd.DataFrame,
    course_drives: pd.DataFrame,
    course_app: pd.DataFrame,
    tee_profiles: pd.DataFrame,
    app_profiles: pd.DataFrame,
    simulations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tee_lookup = tee_profiles.set_index("source_player_id")
    neutral_tee = pd.Series(
        {
            "distance_ratio": 1.0,
            "lateral_bias_yd": 0.0,
            "long_scale": 1.0,
            "lat_scale": 1.0,
        }
    )
    profile_frame = app_profiles.copy()
    profile_frame["source_player_id"] = profile_frame["source_player_id"].astype(str)
    profile_frame["profile_distance_band"] = profile_frame[
        "profile_distance_band"
    ].astype(str)
    app_band_lookup = profile_frame.set_index(
        ["source_player_id", "start_lie_class", "profile_distance_band"]
    )
    app_base_lookup = profile_frame.groupby("source_player_id", observed=True).first()
    neutral_app = pd.Series(
        {
            "n_app": 0,
            "n_band": 0,
            "long_bias_yd": 0.0,
            "lat_bias_yd": 0.0,
            "long_scale": 1.0,
            "lat_scale": 1.0,
        }
    )
    tee_distributions: dict[tuple[int, int, int], tee.Distribution] = {}
    tee_surfaces = {
        int(hole): tee.SurfaceMap(history)
        for hole, history in course_drives.groupby("hole_no", observed=True)
    }
    app_surfaces = _surface_in_common_frame(course_app)
    app_cache: dict[tuple[int, int, str, int], app.AppDistribution] = {}
    particle_parts: list[pd.DataFrame] = []
    case_rows: list[dict[str, object]] = []

    for case_index, record in enumerate(cases.itertuples(index=False)):
        if (case_index + 1) % 250 == 0:
            print(f"{event_name}: integrated case {case_index + 1:,}/{len(cases):,}")
        setup_key = (int(record.round_no), int(record.hole_no), int(record.par))
        if setup_key not in tee_distributions:
            tee_distributions[setup_key] = tee.build_distribution(
                course_drives, *setup_key
            )
        player_id = str(record.source_player_id)
        tee_profile = (
            tee_lookup.loc[player_id] if player_id in tee_lookup.index else neutral_tee
        )
        if isinstance(tee_profile, pd.DataFrame):
            tee_profile = tee_profile.iloc[-1]
        random_seed = np.random.SeedSequence(
            [seed, zlib.crc32(str(record.second_shot_key).encode("utf-8"))]
        )
        rng = np.random.default_rng(random_seed)
        randoms = {
            name: rng.random(simulations)
            for name in ("tee_index", "tee_lie", "app_index", "app_holeout", "app_lie")
        }
        simulations_by_arm: dict[str, dict[str, np.ndarray | float | bool]] = {}
        for arm_name in ARMS:
            particle_frame, diagnostic = _simulate_case_arm(
                record,
                tee_distributions[setup_key],
                tee_surfaces[int(record.hole_no)],
                tee_profile,
                course_app,
                app_surfaces,
                app_cache,
                app_band_lookup,
                app_base_lookup,
                neutral_app,
                player_tee=arm_name in {"player_tee", "both"},
                player_app=arm_name in {"player_app", "both"},
                randoms=randoms,
            )
            simulations_by_arm[arm_name] = diagnostic
            metadata = {
                "event_name": event_name,
                "event_key": record.event_key,
                "source_player_id": player_id,
                "player_name": record.player_name,
                "round_no": int(record.round_no),
                "hole_no": int(record.hole_no),
                "par": int(record.par),
                "case_id": case_index,
                "arm": arm_name,
                "coordinate_frame_version": COORDINATE_FRAME_VERSION,
                "coordinate_frame_reference_year": int(
                    course_app["coordinate_frame_reference_year"].iloc[0]
                ),
            }
            for key, value in metadata.items():
                particle_frame[key] = value
            particle_parts.append(particle_frame)

        actual_tee_endpoint = np.array([record.long_yd, record.lat_yd], float)
        actual_app_endpoint = np.array(
            [record.actual_second_long_yd, record.actual_second_lat_yd], float
        )
        actual_tee_lie_index = tee.LIE_CLASSES.index(str(record.lie_class))
        actual_app_lie = str(record.actual_app_lie)
        actual_app_lie_index = app.END_LIES.index(actual_app_lie)
        tee_onehot = np.zeros(len(tee.LIE_CLASSES))
        tee_onehot[actual_tee_lie_index] = 1.0
        app_onehot = np.zeros(len(app.END_LIES))
        app_onehot[actual_app_lie_index] = 1.0
        case_row: dict[str, object] = {
            "event_name": event_name,
            "event_key": record.event_key,
            "case_id": case_index,
            "source_player_id": player_id,
            "player_name": record.player_name,
            "round_no": int(record.round_no),
            "hole_no": int(record.hole_no),
            "par": int(record.par),
            "actual_tee_long_yd_audit_only": float(record.long_yd),
            "actual_tee_lat_yd_audit_only": float(record.lat_yd),
            "actual_tee_lie_audit_only": str(record.lie_class),
            "actual_tee_remaining_yd_audit_only": float(record.geom_remaining_yd),
            "actual_app_end_long_yd": float(record.actual_second_long_yd),
            "actual_app_end_lat_yd": float(record.actual_second_lat_yd),
            "actual_app_remaining_yd": float(record.actual_second_remaining_yd),
            "actual_app_lie": actual_app_lie,
            "actual_app_holed": bool(record.actual_app_holed),
            "has_tee_profile": player_id in tee_lookup.index,
            "coordinate_frame_version": COORDINATE_FRAME_VERSION,
            "coordinate_frame_reference_year": int(
                course_app["coordinate_frame_reference_year"].iloc[0]
            ),
        }
        for arm_name, simulated in simulations_by_arm.items():
            tee_endpoints = np.asarray(simulated["tee_endpoints"], float)
            tee_remaining = np.asarray(simulated["tee_remaining"], float)
            tee_lie_probs = np.asarray(simulated["tee_lie_probabilities"], float).mean(axis=0)
            tee_lie_probs = np.maximum(tee_lie_probs, 1e-9)
            tee_lie_probs /= tee_lie_probs.sum()
            weights = np.full(len(tee_endpoints), 1.0 / len(tee_endpoints))
            case_row[f"{arm_name}_tee_endpoint_nll"] = tee._kde_nll(
                tee_endpoints, weights.copy(), actual_tee_endpoint
            )
            case_row[f"{arm_name}_tee_endpoint_error_yd"] = float(
                np.linalg.norm(tee_endpoints.mean(axis=0) - actual_tee_endpoint)
            )
            case_row[f"{arm_name}_tee_remaining_crps"] = _uniform_crps(
                tee_remaining, float(record.geom_remaining_yd)
            )
            case_row[f"{arm_name}_tee_lie_logloss"] = -math.log(
                max(float(tee_lie_probs[actual_tee_lie_index]), 1e-12)
            )
            case_row[f"{arm_name}_tee_lie_brier"] = float(
                np.mean((tee_lie_probs - tee_onehot) ** 2)
            )

            attempted = np.asarray(simulated["app_attempted"], bool)
            endpoints = np.asarray(simulated["app_endpoints"], float)[attempted]
            remaining = np.asarray(simulated["app_remaining"], float)
            lie_matrix = np.asarray(simulated["app_lie_probabilities"], float)[attempted]
            lie_probs = np.nanmean(lie_matrix, axis=0) if len(lie_matrix) else np.full(len(app.END_LIES), np.nan)
            if np.isfinite(lie_probs).all():
                lie_probs = np.maximum(lie_probs, 1e-9)
                lie_probs /= lie_probs.sum()
                app_logloss = -math.log(
                    max(float(lie_probs[actual_app_lie_index]), 1e-12)
                )
                app_brier = float(np.mean((lie_probs - app_onehot) ** 2))
            else:
                app_logloss = np.nan
                app_brier = np.nan
            endpoint_valid = np.isfinite(endpoints).all(axis=1)
            endpoint_points = endpoints[endpoint_valid]
            if len(endpoint_points) >= 2 and not bool(record.actual_app_holed):
                endpoint_weights = np.full(len(endpoint_points), 1.0 / len(endpoint_points))
                app_nll = tee._kde_nll(
                    endpoint_points, endpoint_weights, actual_app_endpoint
                )
                app_error = float(
                    np.linalg.norm(endpoint_points.mean(axis=0) - actual_app_endpoint)
                )
            else:
                app_nll = np.nan
                app_error = np.nan
            case_row[f"{arm_name}_app_endpoint_nll"] = app_nll
            case_row[f"{arm_name}_app_endpoint_error_yd"] = app_error
            case_row[f"{arm_name}_app_remaining_crps_ordinary"] = _uniform_crps(
                remaining, float(record.actual_second_remaining_yd)
            )
            case_row[f"{arm_name}_app_lie_logloss"] = app_logloss
            case_row[f"{arm_name}_app_lie_brier"] = app_brier
            case_row[f"{arm_name}_tee_hazard_gate_rate"] = float(
                simulated["tee_hazard_rate"]
            )
            case_row[f"{arm_name}_app_hazard_gate_rate"] = float(
                simulated["app_hazard_rate"]
            )
            case_row[f"{arm_name}_app_attempt_rate"] = float(np.mean(attempted))
        case_rows.append(case_row)

    particles = pd.concat(particle_parts, ignore_index=True)
    id_columns = [
        "event_name", "event_key", "source_player_id", "player_name",
        "round_no", "hole_no", "par", "case_id", "arm", "particle_id",
        "coordinate_frame_version", "coordinate_frame_reference_year",
    ]
    particles = particles[id_columns + [
        column for column in particles.columns if column not in id_columns
    ]]
    case_scores = pd.DataFrame(case_rows)
    return particles, case_scores, summarize(case_scores)


def build_forecast_grid(
    event_name: str,
    event_key: str,
    target_raw: pd.DataFrame,
    forecast: pd.DataFrame,
    max_player_rounds: int,
    seed: int,
) -> pd.DataFrame:
    """Cross roster identities with the strict-prior 4x18 setup grid.

    Held-out rows establish only who played each round. No shot coordinate,
    lie, distance or result enters a forecast case.
    """
    roster = target_raw[
        target_raw["event_type"].eq("STROKE")
    ][["source_player_id", "player_name", "round_no"]].copy()
    roster["source_player_id"] = roster["source_player_id"].astype(str)
    roster = roster.drop_duplicates(["source_player_id", "round_no"])
    if max_player_rounds > 0 and len(roster) > max_player_rounds:
        hashes = pd.util.hash_pandas_object(
            roster[["source_player_id", "round_no"]], index=False
        ).to_numpy(np.uint64)
        rng = np.random.default_rng(seed)
        jitter = rng.integers(0, np.iinfo(np.uint32).max, size=len(roster), dtype=np.uint64)
        roster = roster.assign(_grid_hash=hashes ^ jitter).nsmallest(
            max_player_rounds, "_grid_hash"
        ).drop(columns="_grid_hash")
    grid = roster.merge(forecast, on="round_no", how="inner", validate="many_to_many")
    grid["event_name"] = event_name
    grid["event_key"] = event_key
    grid["roster_source"] = "holdout_identity_and_round_only"
    grid["coordinate_frame_version"] = COORDINATE_FRAME_VERSION
    grid["coordinate_frame_reference_year"] = grid["setup_reference_year"]
    grid = grid.sort_values(
        ["source_player_id", "round_no", "hole_no"]
    ).reset_index(drop=True)
    grid["case_id"] = np.arange(len(grid), dtype=int)
    expected_rows = len(roster) * 18
    if len(grid) != expected_rows:
        raise ValueError(f"Forecast grid has {len(grid)} rows; expected {expected_rows}")
    return grid


def simulate_forecast_grid(
    grid: pd.DataFrame,
    course_drives: pd.DataFrame,
    course_app: pd.DataFrame,
    tee_profiles: pd.DataFrame,
    app_profiles: pd.DataFrame,
    simulations: int,
    seed: int,
    *,
    enforce_empirical_class_support: bool = False,
    hazard_support_factory: HazardSupportFactory | None = None,
) -> pd.DataFrame:
    """Initialize every forecast hole without reading a held-out shot state."""
    tee_lookup = tee_profiles.set_index("source_player_id")
    neutral_tee = pd.Series(
        {
            "distance_ratio": 1.0,
            "lateral_bias_yd": 0.0,
            "long_scale": 1.0,
            "lat_scale": 1.0,
        }
    )
    profile_frame = app_profiles.copy()
    profile_frame["source_player_id"] = profile_frame["source_player_id"].astype(str)
    profile_frame["profile_distance_band"] = profile_frame[
        "profile_distance_band"
    ].astype(str)
    app_band_lookup = profile_frame.set_index(
        ["source_player_id", "start_lie_class", "profile_distance_band"]
    )
    app_base_lookup = profile_frame.groupby("source_player_id", observed=True).first()
    neutral_app = pd.Series(
        {
            "n_app": 0,
            "n_band": 0,
            "long_bias_yd": 0.0,
            "lat_bias_yd": 0.0,
            "long_scale": 1.0,
            "lat_scale": 1.0,
        }
    )
    tee_distributions: dict[tuple[int, int, int], tee.Distribution] = {}
    tee_surfaces = {
        int(hole): tee.SurfaceMap(history)
        for hole, history in course_drives.groupby("hole_no", observed=True)
    }
    app_surfaces = _surface_in_common_frame(course_app)
    app_cache: dict[tuple[int, int, str, int], app.AppDistribution] = {}
    recursive_app = SpatialAppSampler(
        course_app,
        app_profiles,
        app_surfaces,
        enforce_empirical_class_support=enforce_empirical_class_support,
    )
    parts: list[pd.DataFrame] = []

    for grid_index, record in enumerate(grid.itertuples(index=False)):
        if (grid_index + 1) % 500 == 0:
            print(f"Forecast grid case {grid_index + 1:,}/{len(grid):,}")
        player_id = str(record.source_player_id)
        tee_hazard_support_checker = (
            hazard_support_factory(record, "ott")
            if hazard_support_factory is not None
            else None
        )
        app_hazard_support_checker = (
            hazard_support_factory(record, "app")
            if hazard_support_factory is not None
            else None
        )
        tee_profile = (
            tee_lookup.loc[player_id] if player_id in tee_lookup.index else neutral_tee
        )
        if isinstance(tee_profile, pd.DataFrame):
            tee_profile = tee_profile.iloc[-1]
        case_seed = np.random.SeedSequence(
            [seed, zlib.crc32(f"{player_id}|{record.round_no}|{record.hole_no}".encode())]
        )
        rng = np.random.default_rng(case_seed)
        randoms = {
            name: rng.random(simulations)
            for name in ("tee_index", "tee_lie", "app_index", "app_holeout", "app_lie")
        }
        for arm_name in ARMS:
            if int(record.par) == 3:
                starts = np.zeros((simulations, 2), dtype=float)
                start_lies = np.full(simulations, "tee", dtype=object)
                start_remaining = np.full(
                    simulations, float(record.forecast_pin_long_yd), dtype=float
                )
                result = recursive_app.sample(
                    starts,
                    start_lies,
                    start_remaining,
                    np.array([float(record.forecast_pin_long_yd), 0.0]),
                    int(record.round_no),
                    int(record.hole_no),
                    int(record.par),
                    player_id,
                    arm_name in {"player_app", "both"},
                    {
                        "index": randoms["app_index"],
                        "holeout": randoms["app_holeout"],
                        "lie": randoms["app_lie"],
                    },
                    hazard_support_checker=app_hazard_support_checker,
                )
                app_endpoints = np.asarray(result["endpoints"], float)
                particle_frame = pd.DataFrame(
                    {
                        "particle_id": np.arange(simulations, dtype=int),
                        "particle_weight": np.full(simulations, 1.0 / simulations),
                        "tee_end_long_yd": np.nan,
                        "tee_end_lat_yd": np.nan,
                        "tee_end_lie": "not_applicable_par3",
                        "tee_remaining_yd": np.nan,
                        "tee_holed": False,
                        "tee_penalty_count": 0.0,
                        "tee_penalty_cost_strokes": 0.0,
                        "app_start_long_yd": 0.0,
                        "app_start_lat_yd": 0.0,
                        "app_start_lie": "tee",
                        "app_start_remaining_yd": start_remaining,
                        "app_end_long_yd": app_endpoints[:, 0],
                        "app_end_lat_yd": app_endpoints[:, 1],
                        "app_end_lie": result["end_lies"],
                        "app_geometric_remaining_yd": result["geometric_remaining"],
                        "app_remaining_yd": result["remaining"],
                        "app_holed": result["holed"],
                        "app_penalty_count": result["penalty_count"],
                        "app_penalty_cost_strokes": result["penalty_cost_strokes"],
                        "shots_taken": np.where(np.isfinite(app_endpoints[:, 0]), 1, 0),
                        "chain_status": result["status"],
                        "has_app_profile": result["has_profile"],
                        "has_app_band_profile": result["has_band_profile"],
                        "app_n_setup_history": result["n_setup"],
                        "app_n_hole_history": result["n_hole"],
                        "app_n_lie_history": result["n_lie"],
                    }
                )
                initial_category = "app"
            else:
                setup_key = (int(record.round_no), int(record.hole_no), int(record.par))
                if setup_key not in tee_distributions:
                    tee_distributions[setup_key] = tee.build_distribution(
                        course_drives, *setup_key
                    )
                particle_frame, _ = _simulate_case_arm(
                    record,
                    tee_distributions[setup_key],
                    tee_surfaces[int(record.hole_no)],
                    tee_profile,
                    course_app,
                    app_surfaces,
                    app_cache,
                    app_band_lookup,
                    app_base_lookup,
                    neutral_app,
                    player_tee=arm_name in {"player_tee", "both"},
                    player_app=arm_name in {"player_app", "both"},
                    randoms=randoms,
                    enforce_empirical_class_support=(
                        enforce_empirical_class_support
                    ),
                    tee_hazard_support_checker=tee_hazard_support_checker,
                    app_hazard_support_checker=app_hazard_support_checker,
                )
                initial_category = "tee"
            metadata = {
                "event_name": record.event_name,
                "event_key": record.event_key,
                "source_player_id": player_id,
                "player_name": record.player_name,
                "round_no": int(record.round_no),
                "hole_no": int(record.hole_no),
                "par": int(record.par),
                "case_id": int(record.case_id),
                "arm": arm_name,
                "initial_category": initial_category,
                "setup_reference_year": int(record.setup_reference_year),
                "forecast_tee_x": float(record.forecast_tee_x),
                "forecast_tee_y": float(record.forecast_tee_y),
                "forecast_pin_x": float(record.forecast_pin_x),
                "forecast_pin_y": float(record.forecast_pin_y),
                "forecast_pin_long_yd": float(record.forecast_pin_long_yd),
                "roster_source": record.roster_source,
                "coordinate_frame_version": record.coordinate_frame_version,
                "coordinate_frame_reference_year": int(
                    record.coordinate_frame_reference_year
                ),
            }
            for key, value in metadata.items():
                particle_frame[key] = value
            parts.append(particle_frame)
    particles = pd.concat(parts, ignore_index=True)
    id_columns = [
        "event_name", "event_key", "source_player_id", "player_name",
        "round_no", "hole_no", "par", "case_id", "arm", "particle_id",
        "initial_category", "setup_reference_year", "forecast_tee_x",
        "forecast_tee_y", "forecast_pin_x", "forecast_pin_y", "forecast_pin_long_yd",
        "roster_source",
        "coordinate_frame_version", "coordinate_frame_reference_year",
    ]
    return particles[id_columns + [
        column for column in particles.columns if column not in id_columns
    ]]


def summarize(case_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clusters = (
        case_scores["source_player_id"].astype(str)
        + ":R" + case_scores["round_no"].astype(str)
    )
    for arm_name in ARMS:
        for metric in METRICS:
            column = f"{arm_name}_{metric}"
            baseline = f"population_{metric}"
            valid = case_scores[column].notna() & case_scores[baseline].notna()
            paired_index = case_scores.index[valid]
            arm_values = case_scores.loc[paired_index, column]
            population_values = case_scores.loc[paired_index, baseline]
            delta_values = arm_values - population_values
            cluster_delta = delta_values.groupby(clusters.loc[paired_index]).mean()
            delta = float(delta_values.mean()) if len(delta_values) else np.nan
            se = (
                float(cluster_delta.std(ddof=1) / math.sqrt(len(cluster_delta)))
                if len(cluster_delta) > 1 else np.nan
            )
            rows.append(
                {
                    "arm": arm_name,
                    "metric": metric,
                    "n": len(paired_index),
                    "player_round_clusters": len(cluster_delta),
                    "population": float(population_values.mean()) if len(paired_index) else np.nan,
                    "arm_value": float(arm_values.mean()) if len(paired_index) else np.nan,
                    "delta_vs_population": delta,
                    "cluster_se": se,
                    "ci_low": delta - 1.96 * se,
                    "ci_high": delta + 1.96 * se,
                }
            )
    return pd.DataFrame(rows)


def _write_particle_npz(particles: pd.DataFrame, path: Path) -> None:
    ordered = particles.sort_values(["case_id", "arm", "particle_id"]).copy()
    case_meta = (
        ordered.sort_values(["case_id", "arm", "particle_id"])
        .drop_duplicates("case_id")
        .sort_values("case_id")
    )
    cases = case_meta["case_id"].to_numpy(int)
    arms = np.asarray(ARMS)
    simulations = int(ordered["particle_id"].max()) + 1
    shape = (len(cases), len(arms), simulations)
    arm_index = {arm_name: index for index, arm_name in enumerate(ARMS)}
    numeric_columns = [
        "particle_weight", "tee_end_long_yd", "tee_end_lat_yd",
        "tee_remaining_yd", "tee_penalty_count", "tee_penalty_cost_strokes",
        "app_start_long_yd", "app_start_lat_yd", "app_start_remaining_yd",
        "app_end_long_yd", "app_end_lat_yd", "app_geometric_remaining_yd",
        "app_remaining_yd", "app_penalty_count", "app_penalty_cost_strokes",
        "shots_taken",
    ]
    string_columns = ["tee_end_lie", "app_start_lie", "app_end_lie", "chain_status"]
    boolean_columns = ["tee_holed", "app_holed"]
    arrays: dict[str, np.ndarray] = {}
    for column in numeric_columns:
        arrays[column] = np.full(shape, np.nan)
    for column in string_columns:
        arrays[column] = np.full(shape, "", dtype="<U32")
    for column in boolean_columns:
        arrays[column] = np.zeros(shape, dtype=bool)
    case_index = {case_id: index for index, case_id in enumerate(cases)}
    for arm_name, frame in ordered.groupby("arm", observed=True):
        ci = frame["case_id"].map(case_index).to_numpy(int)
        ai = np.full(len(frame), arm_index[str(arm_name)], dtype=int)
        pi = frame["particle_id"].to_numpy(int)
        for column in numeric_columns:
            arrays[column][ci, ai, pi] = pd.to_numeric(
                frame[column], errors="coerce"
            ).to_numpy(float)
        for column in string_columns:
            arrays[column][ci, ai, pi] = frame[column].fillna("").astype(str).to_numpy()
        for column in boolean_columns:
            arrays[column][ci, ai, pi] = frame[column].fillna(False).to_numpy(bool)
    arrays.update(
        {
            "arms": arms,
            "case_id": cases,
            "event_name": np.asarray(case_meta["event_name"].astype(str).tolist(), dtype=str),
            "event_key": np.asarray(case_meta["event_key"].astype(str).tolist(), dtype=str),
            "source_player_id": np.asarray(
                case_meta["source_player_id"].astype(str).tolist(), dtype=str
            ),
            "player_name": np.asarray(case_meta["player_name"].astype(str).tolist(), dtype=str),
            "round_no": case_meta["round_no"].to_numpy(int),
            "hole_no": case_meta["hole_no"].to_numpy(int),
            "par": case_meta["par"].to_numpy(int),
            "axis_order": np.asarray(["case", "arm", "particle"]),
            "coordinate_units": np.asarray("yards"),
            "penalty_resolution_ready": np.asarray(PENALTY_RESOLUTION_READY),
        }
    )
    for column in (
        "initial_category", "roster_source", "coordinate_frame_version",
        "setup_reference_year", "coordinate_frame_reference_year",
        "forecast_tee_x", "forecast_tee_y", "forecast_pin_x", "forecast_pin_y",
        "forecast_pin_long_yd",
    ):
        if column not in case_meta:
            continue
        if column in {"initial_category", "roster_source", "coordinate_frame_version"}:
            arrays[column] = np.asarray(case_meta[column].astype(str).tolist(), dtype=str)
        else:
            arrays[column] = pd.to_numeric(
                case_meta[column], errors="coerce"
            ).to_numpy(float)
    np.savez_compressed(path, **arrays)


def _report(
    event_name: str,
    course_drives: pd.DataFrame,
    course_app: pd.DataFrame,
    cases: pd.DataFrame,
    particles: pd.DataFrame,
    metrics: pd.DataFrame,
    alignment: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    primary = metrics[
        metrics["arm"].ne("population")
        & metrics["metric"].isin(
            ["app_endpoint_nll", "app_remaining_crps_ordinary", "app_lie_logloss"]
        )
    ][["arm", "metric", "n", "delta_vs_population", "ci_low", "ci_high"]]
    gate = []
    for arm_name in ARMS:
        arm = particles[particles["arm"].eq(arm_name)]
        gate.append(
            {
                "arm": arm_name,
                "tee_hazard_gate": arm["chain_status"].eq("tee_hazard_unresolved").mean(),
                "app_hazard_gate": arm["chain_status"].eq("app_hazard_unresolved").mean(),
                "APP_attempted": arm["app_end_long_yd"].notna().mean(),
            }
        )
    gate_frame = pd.DataFrame(gate)
    return f"""# Strict-prior spatial tee -> APP v2: {event_name}

## Design

- Holdout: **{args.holdout_year}**; setup mode: **historical only**
- Prior-course tee rows: **{len(course_drives):,}**
- Prior-course causal APP rows: **{len(course_app):,}**
- Held-out tee-to-stroke-2 APP cases: **{len(cases):,}**
- Particles per case/arm: **{args.simulations:,}**
- Persisted particle rows: **{len(particles):,}**
- Coordinate frame: **{COORDINATE_FRAME_VERSION}** (one raw-edition → latest-prior transform)
- APP edition alignment RMSE range: **{alignment['alignment_rmse_yd'].min():.3f}–{alignment['alignment_rmse_yd'].max():.3f} yd**

The model samples a strict-prior course tee endpoint, applies the audited
player distance/dispersion transport when requested, reclassifies its lie on
the physical course surface and calculates remaining distance to the prior pin
forecast.  That generated state—not the realized 2026 drive—selects the direct
APP distance × lie donor distribution.  The structured player APP mean and
dispersion transform is then applied in the generated shot axis, and the
landing lie is reclassified on the aligned physical-course endpoint surface.
All four arms use identical underlying uniforms within a case.

## Paired results versus population

{_markdown_table(primary)}

## Particle gates

{_markdown_table(gate_frame, digits=5)}

Endpoint metrics use the common course frame in yards.  Remaining-distance
CRPS is explicitly conditional on ordinary resolved outcomes because no drop
or replay state is fabricated for a hazard.  Lie probabilities still retain
the modeled penalty class, and every unresolved path remains in the persisted
particle table with its physical endpoint and a null penalty count/cost.

## Hard boundary

`PENALTY_RESOLUTION_READY = {PENALTY_RESOLUTION_READY}`.  The causal APP
reader now uses the complete raw sequence and retains the exact linked
resolution state. V2 intentionally leaves endpoint-linked hazard donor
selection external: the foundation and resolver must pass their coherence
audit before recursive scoring. `app_ready_arg` is a coarse historical status
name, **not** a downstream shot category; many such states are still 50–100+
yards from the pin. Every continuation must recompute APP versus ARG from the
particle's lie and remaining distance. Green/hole statuses alone are terminal
category cues.
"""


def combined_report(output_dir: Path) -> tuple[pd.DataFrame, Path]:
    parts: list[pd.DataFrame] = []
    for path in sorted(output_dir.glob("integrated_v2_metrics_*_historical.csv")):
        frame = pd.read_csv(path)
        frame["event"] = path.stem.removeprefix("integrated_v2_metrics_").removesuffix(
            "_historical"
        )
        parts.append(frame)
    if not parts:
        raise FileNotFoundError("No v2 event metric files to combine")
    event_metrics = pd.concat(parts, ignore_index=True)
    primary = event_metrics[event_metrics["arm"].ne("population")].copy()
    rows: list[dict[str, object]] = []
    for (arm_name, metric), frame in primary.groupby(["arm", "metric"], observed=True):
        values = frame["delta_vs_population"].to_numpy(float)
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan
        rows.append(
            {
                "arm": arm_name,
                "metric": metric,
                "events": len(values),
                "equal_event_delta": mean,
                "event_se": se,
                "ci_low": mean - 1.96 * se,
                "ci_high": mean + 1.96 * se,
                "events_favorable": int(np.sum(values < 0)),
                "case_n_min": int(frame["n"].min()),
                "case_n_max": int(frame["n"].max()),
                "player_round_clusters_min": int(frame["player_round_clusters"].min()),
                "player_round_clusters_max": int(frame["player_round_clusters"].max()),
            }
        )
    combined = pd.DataFrame(rows)
    combined.to_csv(output_dir / "three_event_equal_event_metrics.csv", index=False)

    hazard_parts: list[pd.DataFrame] = []
    for path in sorted(output_dir.glob("forecast_grid_particles_*_historical.parquet")):
        frame = pd.read_parquet(path, columns=["event_name", "arm", "chain_status"])
        summary_rows = []
        for (event_name, arm_name), group in frame.groupby(
            ["event_name", "arm"], observed=True
        ):
            summary_rows.append(
                {
                    "event": event_name,
                    "arm": arm_name,
                    "particles": len(group),
                    "tee_hazard_rate": group["chain_status"].eq(
                        "tee_hazard_unresolved"
                    ).mean(),
                    "app_hazard_rate": group["chain_status"].eq(
                        "app_hazard_unresolved"
                    ).mean(),
                }
            )
        hazard_parts.append(pd.DataFrame(summary_rows))
    hazard_by_event = pd.concat(hazard_parts, ignore_index=True) if hazard_parts else pd.DataFrame()
    hazard_equal = (
        hazard_by_event.groupby("arm", observed=True, as_index=False)
        .agg(
            events=("event", "nunique"),
            particles=("particles", "sum"),
            equal_event_tee_hazard_rate=("tee_hazard_rate", "mean"),
            equal_event_app_hazard_rate=("app_hazard_rate", "mean"),
        )
        if len(hazard_by_event) else pd.DataFrame()
    )
    hazard_by_event.to_csv(output_dir / "three_event_hazard_rates.csv", index=False)

    provenance_rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("coordinate_provenance_*_historical.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assertions = {
            "alignment_once": data.get("alignment_count") == 1,
            "prior_reference": int(data.get("setup_reference_year", 9999))
            < int(data.get("holdout_year", 0)),
            "prior_predictor_scope": data.get("predictor_reference_scope")
            == "latest_verified_prior_only",
            "no_holdout_geometry_eligibility": not bool(
                data.get("holdout_geometry_used_for_predictor_eligibility", True)
            ),
        }
        provenance_rows.append(
            {
                "event": path.stem.removeprefix("coordinate_provenance_").removesuffix(
                    "_historical"
                ),
                "frame": data.get("coordinate_frame_version"),
                "reference_year": data.get("setup_reference_year"),
                "tee_causal_penalty_rows": data.get("tee_causal_penalty_rows"),
                **assertions,
                "all_assertions_pass": all(assertions.values()),
            }
        )
    provenance = pd.DataFrame(provenance_rows)
    if len(provenance) != 3 or not provenance["all_assertions_pass"].all():
        raise AssertionError(
            "Three-event coordinate provenance is incomplete or failed: "
            + provenance.to_dict("records").__repr__()
        )
    recommendations = pd.DataFrame(
        [
            {
                "arm": "population",
                "full_chain_role": "required baseline",
                "recommendation": "include",
            },
            {
                "arm": "player_tee",
                "full_chain_role": "primary candidate",
                "recommendation": "include; tee transport has the strongest prior validation",
            },
            {
                "arm": "player_app",
                "full_chain_role": "APP isolation ablation",
                "recommendation": "include for benchmark only; not production-ready",
            },
            {
                "arm": "both",
                "full_chain_role": "combined research candidate",
                "recommendation": "include as secondary ablation; require full-round win",
            },
        ]
    )
    report = f"""# Three-event strict-prior spatial tee -> APP v2

This equal-event summary combines the bounded 2026 Wyndham, Sony and Travelers
historical-setup holdouts: **6 player-round clusters per event and 48 particles
per case/arm**. Negative proper-score deltas favor the indicated player
transport arm over the population tee + population APP chain. Earlier v2
reports and smoke directories are superseded by this strict-prior run.

## Equal-event arm-by-metric deltas

{_markdown_table(combined)}

## Provenance assertions

{_markdown_table(provenance, digits=0)}

Every predictor pool and setup reference is strictly prior. The holdout event
key comes from metadata only; heldout geometry is used solely to express
observed validation endpoints in the already-frozen prior frame.

## Forecast-grid hazard gates

{_markdown_table(hazard_equal)}

Tee hazards use full-sequence causal OTT penalty labels at the preserved
physical endpoint. APP/tee penalty resolution remains external to v2 and must
pass the foundation/resolver coherence audit before recursive scoring.

## Full-chain benchmark recommendation

{_markdown_table(recommendations)}

`app_ready_arg` is not a downstream category. The full-chain router must
recompute APP versus ARG from lie and remaining distance for every particle.
These are transition diagnostics, not production-price results; promotion
requires complete-hole and complete-round gains after production mean anchors
and shared round/week latents are restored.
"""
    report_path = output_dir / "THREE_EVENT_INTEGRATED_V2_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return combined, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-name", action="append")
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--max-player-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--case-sample-seed", type=int, default=20260820)
    parser.add_argument("--tee-profiles-csv", type=Path, default=DEFAULT_TEE_PROFILES)
    parser.add_argument("--app-profiles-csv", type=Path, default=DEFAULT_APP_PROFILES)
    parser.add_argument(
        "--forecast-grid",
        action="store_true",
        help="Also initialize every prior-setup hole for a bounded player-round roster.",
    )
    parser.add_argument(
        "--forecast-grid-only",
        action="store_true",
        help="Write the all-hole forecast grid without scoring observed stroke-2 cases.",
    )
    parser.add_argument(
        "--grid-player-rounds",
        type=int,
        default=12,
        help="Bound roster player-rounds for forecast particles; 0 uses every observed roster identity.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.combine_only:
        combined, report_path = combined_report(args.output_dir)
        print(combined.to_string(index=False))
        print(f"Wrote {report_path}")
        return
    for profile_path in (args.tee_profiles_csv, args.app_profiles_csv):
        if not profile_path.exists():
            raise FileNotFoundError(f"Missing strict-prior profile file: {profile_path}")
    tee_profiles = pd.read_csv(args.tee_profiles_csv, dtype={"source_player_id": str})
    app_profiles = pd.read_csv(args.app_profiles_csv, dtype={"source_player_id": str})
    required_app = {
        "source_player_id", "start_lie_class", "profile_distance_band",
        "long_bias_yd", "lat_bias_yd", "long_scale", "lat_scale",
    }
    missing = required_app.difference(app_profiles.columns)
    if missing:
        raise ValueError(f"APP profile file is missing {sorted(missing)}")
    events = list(args.event_name or DEFAULT_EVENTS)
    if args.smoke_test:
        events = events[:1]
        args.simulations = min(args.simulations, 24)
        args.max_player_rounds = min(args.max_player_rounds, 4)
        args.grid_player_rounds = min(args.grid_player_rounds, 2)
    for event_index, event_name in enumerate(events):
        print(f"Preparing strict-prior v2 chain: {event_name}")
        prepared = prepare_event_foundations(
            args.export_root,
            event_name,
            args.holdout_year,
            args.course_start_year,
            include_scoring_targets=not args.forecast_grid_only,
        )
        event_keys = prepared["event_keys"]
        forecast = prepared["forecast"]
        course_drives = prepared["course_drives"]
        target_drives = prepared["target_drives"]
        course_app = prepared["course_app"]
        alignment = prepared["app_alignment"]
        target_raw = prepared["target_raw"]
        target_sg = prepared["target_sg"]
        target_geometry = prepared["target_geometry"]
        provenance = prepared["provenance"]
        setup_reference_year = int(provenance["setup_reference_year"])
        slug = _slug(event_name)
        suffix = f"_{slug}_historical" + ("_smoke" if args.smoke_test else "")
        provenance_path = args.output_dir / f"coordinate_provenance{suffix}.json"
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        if not args.forecast_grid_only:
            if target_drives is None or target_sg is None or target_geometry is None:
                raise AssertionError("Scoring targets were not prepared")
            cases = _prepare_target_cases(
                target_drives,
                target_raw,
                target_sg,
                target_geometry,
                forecast,
                args.max_player_rounds,
                args.case_sample_seed,
            )
            if args.smoke_test:
                cases = cases.head(30).copy()
            print(
                f"{event_name}: setup reference={setup_reference_year}; "
                f"tee donors={len(course_drives):,}; causal APP donors={len(course_app):,}; "
                f"cases={len(cases):,}; particles={args.simulations}"
            )
            particles, case_scores, metrics = evaluate_event(
                event_name,
                cases,
                course_drives,
                course_app,
                tee_profiles,
                app_profiles,
                args.simulations,
                args.seed + event_index * 100,
            )
            particle_path = args.output_dir / f"integrated_v2_particles{suffix}.parquet"
            npz_path = args.output_dir / f"integrated_v2_particles{suffix}.npz"
            score_path = args.output_dir / f"integrated_v2_scores{suffix}.csv"
            metric_path = args.output_dir / f"integrated_v2_metrics{suffix}.csv"
            alignment_path = args.output_dir / f"integrated_v2_alignment{suffix}.csv"
            report_path = args.output_dir / f"INTEGRATED_V2_REPORT{suffix}.md"
            particles.to_parquet(particle_path, index=False)
            _write_particle_npz(particles, npz_path)
            case_scores.to_csv(score_path, index=False)
            metrics.to_csv(metric_path, index=False)
            alignment.to_csv(alignment_path, index=False)
            report_path.write_text(
                _report(
                    event_name,
                    course_drives,
                    course_app,
                    cases,
                    particles,
                    metrics,
                    alignment,
                    args,
                ),
                encoding="utf-8",
            )
            print(metrics[metrics["arm"].ne("population")].to_string(index=False))
            print(f"Wrote {report_path}, {particle_path}, {npz_path}")
        if args.forecast_grid or args.forecast_grid_only:
            grid = build_forecast_grid(
                event_name,
                event_keys[args.holdout_year],
                target_raw,
                forecast,
                args.grid_player_rounds,
                args.case_sample_seed,
            )
            print(
                f"{event_name}: full forecast grid={len(grid):,} cases "
                f"({grid[['source_player_id', 'round_no']].drop_duplicates().shape[0]} player-rounds x 18)"
            )
            grid_particles = simulate_forecast_grid(
                grid,
                course_drives,
                course_app,
                tee_profiles,
                app_profiles,
                args.simulations,
                args.seed + event_index * 100 + 50,
            )
            grid_case_path = args.output_dir / f"forecast_grid_cases{suffix}.csv"
            grid_particle_path = args.output_dir / f"forecast_grid_particles{suffix}.parquet"
            grid_npz_path = args.output_dir / f"forecast_grid_particles{suffix}.npz"
            grid_report_path = args.output_dir / f"FORECAST_GRID_REPORT{suffix}.md"
            grid.to_csv(grid_case_path, index=False)
            grid_particles.to_parquet(grid_particle_path, index=False)
            _write_particle_npz(grid_particles, grid_npz_path)
            statuses = (
                grid_particles.groupby(["initial_category", "arm", "chain_status"], observed=True)
                .size().rename("particles").reset_index()
            )
            grid_report_path.write_text(
                f"""# Full-grid strict-prior initializer: {event_name}

- Setup reference: verified **{setup_reference_year}** edition
- Grid: **{len(grid):,}** player-round-hole cases; all 18 holes in every selected round
- Particles: **{len(grid_particles):,}** across {len(ARMS)} arms
- Par 3 initial category: **APP from tee**
- Par 4/5 initial category: **tee then generated-state APP**
- Held-out information used: **player identity and round participation only**
- Coordinate frame: **{COORDINATE_FRAME_VERSION}** (one raw-edition → latest-prior transform)

{_markdown_table(statuses, digits=0)}

Round participation selects evaluation cases only; it is not a transition
feature. No held-out coordinate, lie, distance or shot result enters these particles.
They stop at the first unresolved hazard or after the first APP transition and
are intended for the recursive APP/ARG/putting wrapper.
""",
                encoding="utf-8",
            )
            print(f"Wrote {grid_report_path}, {grid_particle_path}, {grid_npz_path}")
        schema_path = args.output_dir / "particle_schema.json"
        if not schema_path.exists():
            schema = {
                "axis_order_npz": ["case", "arm", "particle"],
                "coordinate_frame": "strict-prior common course frame; tee=(0,0), pin=(forecast_pin_long_yd,0)",
                "coordinate_frame_version": COORDINATE_FRAME_VERSION,
                "coordinate_units": "yards",
                "case_id": "deterministic sort by source_player_id, round_no, hole_no after seeded roster selection",
                "forecast_grid_case_fields": [
                    "forecast_tee_x", "forecast_tee_y", "forecast_pin_x",
                    "forecast_pin_y", "forecast_pin_long_yd", "par",
                    "setup_reference_year", "initial_category", "roster_source",
                ],
                "roster_scope": "heldout player identity and round participation select evaluation cases only; no shot state or outcome is read",
                "penalty_fields": "nullable until raw-sequence drop/replay resolver is connected",
                "continuable_status": [
                    "app_ready_arg", "app_green_ready_putt",
                    "tee_green_ready_putt", "app_complete_holed",
                ],
                "category_routing": "app_ready_arg is not an ARG category; recompute APP/ARG from lie and remaining distance",
                "blocked_status": [
                    "tee_hazard_unresolved", "app_hazard_unresolved",
                    "app_no_surface_support", "app_no_spatial_support",
                ],
            }
            schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    if not args.smoke_test and len(events) > 1 and not args.forecast_grid_only:
        combined, report_path = combined_report(args.output_dir)
        print(combined.to_string(index=False))
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

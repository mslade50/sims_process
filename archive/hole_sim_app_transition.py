"""Leakage-safe spatial approach-transition research harness.

This archived script evaluates the second transition needed by a true
shot-by-shot golf simulator::

    (course, hole, round, start lie, start yardage)
        -> pin-relative endpoint + end lie + next-shot yardage

The course distribution is learned only from prior editions of the recurring
event.  Player approach bias and dispersion profiles are learned from PGA
shots strictly before the holdout season.  The 2026 Wyndham, Sony and
Travelers editions are never used for fitting.

Two setup modes are supported.  ``actual`` treats the held-out pin location as
known, which isolates transition quality.  ``historical`` fixes the pin for
each round/hole entirely from prior-edition geometry, which is the honest
pre-event setup forecast.

Explicit penalty rows are sequenced against every raw row in the same
event/player/round/hole and collapsed into the preceding played stroke only
when that causal stroke is APP.  The real landing endpoint, penalty count and
exact following STROKE/PROVISIONAL raw start state are retained separately. Recurring
event names generate candidate donors, but every edition must pass a
single-course structure check and a rigid-aligned geometry identity check
against the holdout before it enters the exact-course panel. Hole-outs remain
fixed discrete atoms at the cup and are not translated away from zero by a
player transform.

Nothing in production imports this module.  A short execution check is::

    python archive/hole_sim_app_transition.py --smoke-test

The full three-event study writes to
``permanent_data/hole_sim_app_transition``.
"""

from __future__ import annotations

import argparse
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import logsumexp

try:
    from archive.hole_sim_state_foundations import (
        FOUNDATION_VERSION,
        align_coordinates_to_target,
        link_app_penalties_full_sequence,
        load_verified_recurring_event,
    )
except ModuleNotFoundError:  # Direct execution: python archive/<script>.py
    from hole_sim_state_foundations import (  # type: ignore
        FOUNDATION_VERSION,
        align_coordinates_to_target,
        link_app_penalties_full_sequence,
        load_verified_recurring_event,
    )


DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_OUTPUT = Path("permanent_data/hole_sim_app_transition")
DEFAULT_EVENTS = (
    "Wyndham Championship",
    "Sony Open in Hawaii",
    "Travelers Championship",
)
START_LIES = ("tee", "fairway", "rough", "sand", "recovery", "other")
END_LIES = (
    "green", "fairway", "rough", "sand", "recovery", "tee", "hole",
    "penalty", "other",
)
ARMS = ("population", "mean_transport", "dispersion_transport", "combined")


@dataclass(frozen=True)
class AppDistribution:
    rows: pd.DataFrame
    weights: np.ndarray
    source: str
    n_setup: int
    n_hole: int
    n_lie: int


def _normalize_start_lie(value: object) -> str:
    lie = str(value).strip().lower()
    if lie in START_LIES[:-1]:
        return lie
    if "sand" in lie or "bunker" in lie:
        return "sand"
    return "other"


def _normalize_end_lie(value: object) -> str:
    lie = str(value).strip().lower()
    if lie in END_LIES[:-2]:
        return lie
    if lie in {"holed", "cup"}:
        return "hole"
    if "sand" in lie or "bunker" in lie:
        return "sand"
    return "other"


def _weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.sum(points * weights[:, None], axis=0)


def _effective_n(weights: np.ndarray) -> float:
    weights = weights / max(float(weights.sum()), 1e-12)
    return float(1.0 / max(float(np.sum(weights * weights)), 1e-12))


def _crps(samples: np.ndarray, weights: np.ndarray, observed: float) -> float:
    finite = np.isfinite(samples) & np.isfinite(weights) & (weights > 0)
    samples, weights = samples[finite], weights[finite]
    if not len(samples):
        return np.nan
    order = np.argsort(samples)
    x = samples[order]
    w = weights[order]
    w /= w.sum()
    first = float(np.sum(w * np.abs(x - observed)))
    before_w = np.cumsum(w) - w
    before_wx = np.cumsum(w * x) - w * x
    pair = float(np.sum(w * (x * before_w - before_wx)))
    return first - pair


def _kde_nll(points: np.ndarray, weights: np.ndarray, observed: np.ndarray) -> float:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0)
    points, weights = points[finite], weights[finite]
    if len(points) < 2:
        return np.nan
    weights /= weights.sum()
    center = _weighted_mean(points, weights)
    variance = np.sum(weights[:, None] * (points - center) ** 2, axis=0)
    bandwidth = np.sqrt(np.maximum(variance, np.array([9.0, 9.0])))
    bandwidth *= max(_effective_n(weights), 2.0) ** (-1 / 6)
    bandwidth = np.maximum(bandwidth, np.array([2.25, 2.25]))
    z = (observed[None, :] - points) / bandwidth
    components = (
        np.log(np.maximum(weights, 1e-300))
        - math.log(2 * math.pi)
        - np.log(bandwidth).sum()
        - 0.5 * np.sum(z * z, axis=1)
    )
    return float(-logsumexp(components))


def _verified_event_scope(
    export_root: Path,
    event_name: str,
    holdout_year: int,
    reference_mode: str = "latest_prior",
) -> tuple[dict[int, str], pd.DataFrame]:
    verified, _, audit = load_verified_recurring_event(
        export_root, event_name, holdout_year, reference_mode=reference_mode
    )
    physical_ids = audit.loc[audit["accepted"], "physical_course_id"].unique()
    if len(physical_ids) != 1:
        raise ValueError(
            f"Ambiguous physical-course identity for {event_name!r}: "
            f"{physical_ids.tolist()}"
        )
    for row in audit.loc[~audit["accepted"]].itertuples(index=False):
        print(
            f"{event_name}: rejected {row.season} {row.event_key} as "
            f"{row.status} (aligned RMSE={row.aligned_rmse_yd:.1f} yd)"
        )
    print(
        f"{event_name}: physical course {physical_ids[0]} verified across "
        f"{len(verified):,} editions"
    )
    return dict(zip(verified["season"].astype(int), verified["event_key"])), audit


def _event_keys(export_root: Path, event_name: str, holdout_year: int) -> dict[int, str]:
    event_keys, _ = _verified_event_scope(
        export_root, event_name, holdout_year, reference_mode="latest_prior"
    )
    return event_keys


def _read_geometry(
    export_root: Path, year: int, event_keys: list[str] | None = None
) -> pd.DataFrame:
    columns = [
        "event_key", "round_no", "hole_no", "course_id", "par", "yardage",
        "tee_x", "tee_y", "pin_x", "pin_y",
    ]
    filters = [("event_key", "in", event_keys)] if event_keys else None
    return pd.read_parquet(
        export_root / f"pga_{year}_hole_geometry.parquet",
        columns=columns,
        filters=filters,
    )


def _read_approach_rows(
    export_root: Path, year: int, event_keys: list[str] | None = None
) -> pd.DataFrame:
    sg_columns = [
        "event_key", "round_no", "hole_no", "source_player_id", "player_name",
        "stroke_no", "sg_category", "start_lie", "start_distance_yd", "end_lie",
        "end_distance_yd", "sg_valid", "sg_issue", "complete_hole",
        "state_inferred", "shot_key",
    ]
    sg_filters: list[tuple[str, str, object]] = [("sg_category", "==", "app")]
    if event_keys:
        sg_filters.append(("event_key", "in", event_keys))
    sg = pd.read_parquet(
        export_root / f"pga_{year}_shot_sg.parquet",
        columns=sg_columns,
        filters=sg_filters,
    )
    if sg.empty:
        return sg
    raw_columns = [
        "shot_key", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "event_type", "start_lie", "end_lie",
        "distance_remaining_yd", "ball_holed", "start_x", "start_y", "end_x",
        "end_y", "course_id",
    ]
    raw_filters = [("event_key", "in", event_keys)] if event_keys else None
    raw = pd.read_parquet(
        export_root / f"pga_{year}_shots.parquet",
        columns=raw_columns,
        filters=raw_filters,
    )
    state_columns = [
        "shot_key", "sg_category", "start_lie", "start_distance_yd",
        "end_lie", "end_distance_yd",
    ]
    state_filters = [("event_key", "in", event_keys)] if event_keys else None
    states = pd.read_parquet(
        export_root / f"pga_{year}_shot_sg.parquet",
        columns=state_columns,
        filters=state_filters,
    )
    frame, _, audit = link_app_penalties_full_sequence(raw, sg, states, year)
    frame["season"] = year
    frame["source_player_id"] = frame["source_player_id"].astype(str)
    if audit["penalty_rows"]:
        print(
            f"{year}: full-sequence penalties={audit['penalty_rows']:,}; "
            f"causal linked={audit['full_causal_link_rate']:.2%}; "
            f"APP hazards={audit['attached_app_hazard_strokes']:,}; "
            f"resolution raw lie={audit['resolution_raw_lie_rate_app']:.2%}; "
            f"legacy mismatch={audit['legacy_causal_mismatch_rate']:.2%}"
        )
    return frame


def _attach_geometry(frame: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    keys = ["event_key", "round_no", "hole_no", "course_id"]
    geometry = geometry.drop_duplicates(keys)
    out = frame.merge(
        geometry[keys + ["par", "yardage", "pin_x", "pin_y"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    numeric = [
        "start_distance_yd", "end_distance_yd", "start_x", "start_y",
        "end_x", "end_y", "pin_x", "pin_y", "resolution_start_x",
        "resolution_start_y",
    ]
    for column in numeric:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["start_lie_class"] = out["start_lie"].map(_normalize_start_lie)
    out["end_lie_class"] = out["end_lie"].map(_normalize_end_lie)
    penalty = out["penalty_after"].fillna(False).astype(bool)
    holed = out["ball_holed"].fillna(0).astype(bool)
    out.loc[penalty, "end_lie_class"] = "penalty"
    out.loc[holed & ~penalty, "end_lie_class"] = "hole"
    out["is_penalty"] = penalty
    out["is_holeout"] = holed & ~penalty
    regular_end_valid = out["end_distance_yd"].between(0, 700)
    resolution_lie = out.get(
        "resolution_start_lie",
        pd.Series(index=out.index, dtype=object),
    )
    resolution_lie_valid = (
        resolution_lie.notna()
        & resolution_lie.astype(str).str.strip().ne("")
    )
    resolution_xy_valid = np.isfinite(
        out.reindex(columns=["resolution_start_x", "resolution_start_y"])
        .apply(pd.to_numeric, errors="coerce")
    ).all(axis=1)
    recursive_end_valid = (~penalty & regular_end_valid) | (
        penalty & resolution_xy_valid & resolution_lie_valid
    )
    out["state_valid"] = (
        out["start_distance_yd"].between(0.1, 600)
        & recursive_end_valid
        & out["complete_hole"].fillna(0).astype(bool)
        & out["sg_valid"].fillna(0).astype(bool)
    )
    return out


def _apply_reference_pins(
    frame: pd.DataFrame,
    pin_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create pin-relative endpoint coordinates for actual or forecast pins."""
    out = frame.copy()
    if pin_table is not None:
        out = out.drop(columns=["pin_x", "pin_y"]).merge(
            pin_table,
            on=["round_no", "hole_no"],
            how="inner",
            validate="many_to_one",
        )
    vx = out["pin_x"].to_numpy(float) - out["start_x"].to_numpy(float)
    vy = out["pin_y"].to_numpy(float) - out["start_y"].to_numpy(float)
    norm = np.hypot(vx, vy)
    valid_axis = np.isfinite(norm) & (norm > 3.0)
    ux = np.divide(vx, norm, out=np.full(len(out), np.nan), where=valid_axis)
    uy = np.divide(vy, norm, out=np.full(len(out), np.nan), where=valid_axis)
    ex = out["end_x"].to_numpy(float) - out["pin_x"].to_numpy(float)
    ey = out["end_y"].to_numpy(float) - out["pin_y"].to_numpy(float)
    out["miss_long_yd"] = (ex * ux + ey * uy) / 3.0
    out["miss_lat_yd"] = (-ex * uy + ey * ux) / 3.0
    out["axis_ux"] = ux
    out["axis_uy"] = uy
    out["reference_start_distance_yd"] = norm / 3.0
    out["spatial_valid"] = (
        out["state_valid"]
        & np.isfinite(out[["miss_long_yd", "miss_lat_yd"]]).all(axis=1)
        & out["miss_long_yd"].between(-250, 250)
        & out["miss_lat_yd"].between(-250, 250)
    )
    # In actual mode this agrees closely with the SG state.  In historical
    # mode it intentionally measures the realized endpoint against the pin
    # forecast that was available before the held-out edition.
    out["scored_remaining_yd"] = np.nan
    regular = out["spatial_valid"] & ~out["is_penalty"]
    out.loc[regular, "scored_remaining_yd"] = np.hypot(
        out.loc[regular, "miss_long_yd"], out.loc[regular, "miss_lat_yd"]
    )
    penalty = out["spatial_valid"] & out["is_penalty"]
    out.loc[penalty, "scored_remaining_yd"] = np.hypot(
        out.loc[penalty, "resolution_start_x"] - out.loc[penalty, "pin_x"],
        out.loc[penalty, "resolution_start_y"] - out.loc[penalty, "pin_y"],
    ) / 3.0
    out.loc[out["is_holeout"], "scored_remaining_yd"] = 0.0
    return out


def load_course_panel(
    export_root: Path,
    event_name: str,
    holdout_year: int,
    course_start_year: int,
    reference_mode: str = "latest_prior",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    event_keys, identity_audit = _verified_event_scope(
        export_root, event_name, holdout_year, reference_mode=reference_mode
    )
    identity_by_key = identity_audit.set_index("event_key")
    parts: list[pd.DataFrame] = []
    for year, event_key in sorted(event_keys.items()):
        if year < course_start_year:
            continue
        geometry = _read_geometry(export_root, year, [event_key])
        rows = _read_approach_rows(export_root, year, [event_key])
        identity = identity_by_key.loc[event_key]
        # Same-venue verification is rigid-frame invariant. In strict mode,
        # normalize every admitted edition into the latest-prior frame before
        # historical pins or absolute endpoint-to-surface maps are pooled.
        geometry = align_coordinates_to_target(
            geometry, identity, ("tee", "pin")
        )
        rows = align_coordinates_to_target(
            rows,
            identity,
            ("start", "end", "causal_end", "resolution_start", "drop_end"),
        )
        parts.append(_attach_geometry(rows, geometry))
    panel = pd.concat(parts, ignore_index=True)
    panel["coordinate_reference_mode"] = reference_mode
    train = panel[panel["season"].lt(holdout_year) & panel["state_valid"]].copy()
    holdout = panel[panel["season"].eq(holdout_year) & panel["state_valid"]].copy()
    if train.empty or holdout.empty:
        raise ValueError(f"Empty course train/holdout for {event_name}")
    return train, holdout, event_keys


def setup_mode_panel(
    course_train: pd.DataFrame,
    holdout: pd.DataFrame,
    setup_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if setup_mode == "actual":
        train_out = _apply_reference_pins(course_train)
        holdout_out = _apply_reference_pins(holdout)
    else:
        if setup_mode != "historical":
            raise ValueError(f"Unknown setup mode: {setup_mode}")
        canonical = (
            course_train.groupby(["round_no", "hole_no"], as_index=False)
            .agg(pin_x=("pin_x", "median"), pin_y=("pin_y", "median"))
        )
        train_out = _apply_reference_pins(course_train, canonical)
        holdout_out = _apply_reference_pins(holdout, canonical)
    train_out = train_out[
        train_out["reference_start_distance_yd"].between(0.1, 700)
    ].copy()
    holdout_out = holdout_out[
        holdout_out["reference_start_distance_yd"].between(0.1, 700)
    ].copy()
    return train_out, holdout_out


def load_player_training(
    export_root: Path,
    start_year: int,
    holdout_year: int,
    smoke_test: bool = False,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    first = holdout_year - 1 if smoke_test else start_year
    keep = [
        "event_key", "round_no", "hole_no", "source_player_id", "player_name",
        "shot_key", "start_lie_class", "start_distance_yd", "miss_long_yd",
        "miss_lat_yd", "spatial_valid",
    ]
    for year in range(first, holdout_year):
        print(f"Loading all-course APP profiles: {year}")
        geometry = _read_geometry(export_root, year)
        rows = _read_approach_rows(export_root, year)
        prepared = _apply_reference_pins(_attach_geometry(rows, geometry))
        continuous = prepared["spatial_valid"] & ~prepared["is_holeout"]
        parts.append(prepared.loc[continuous, keep])
    return pd.concat(parts, ignore_index=True)


def build_tour_transition_cache(
    export_root: Path,
    start_year: int,
    holdout_year: int,
    cache_path: Path,
    regular_per_year_cell: int = 600,
) -> pd.DataFrame:
    """Build a compact, survey-weighted strict-prior Tour APP donor set.

    Regular shots are deterministically sampled within year x 15-yard x lie
    cells. Causal penalties and hole-outs are all retained. The sampling
    weight restores each cell's original mass.
    """
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        versions = (
            set(cached["foundation_version"].dropna().astype(str))
            if "foundation_version" in cached else set()
        )
        if versions != {FOUNDATION_VERSION}:
            raise ValueError(
                f"Tour APP cache {cache_path} predates the full-sequence state "
                f"foundation ({sorted(versions)}). Use a new cache path."
            )
        print(f"Loaded Tour APP cache: {len(cached):,} rows from {cache_path}")
        return cached
    parts: list[pd.DataFrame] = []
    keep = [
        "season", "event_key", "shot_key", "start_lie_class",
        "reference_start_distance_yd", "miss_long_yd", "miss_lat_yd",
        "scored_remaining_yd", "end_lie_class", "spatial_valid",
        "is_penalty", "is_holeout", "foundation_version",
    ]
    for year in range(start_year, holdout_year):
        print(f"Building Tour-only APP donors: {year}")
        geometry = _read_geometry(export_root, year)
        rows = _read_approach_rows(export_root, year)
        prepared = _apply_reference_pins(_attach_geometry(rows, geometry))
        prepared = prepared.loc[
            prepared["state_valid"]
            & prepared["reference_start_distance_yd"].between(0.1, 700),
            keep,
        ].copy()
        prepared["tour_distance_bucket"] = (
            prepared["reference_start_distance_yd"] / 15.0
        ).round().astype(int)
        prepared["_sample_hash"] = pd.util.hash_pandas_object(
            prepared["shot_key"], index=False
        ).to_numpy(np.uint64)
        cell_keys = ["start_lie_class", "tour_distance_bucket"]
        selected_parts: list[pd.DataFrame] = []
        for _, cell in prepared.groupby(cell_keys, observed=True, sort=False):
            special = cell[cell["is_penalty"] | cell["is_holeout"]].copy()
            regular = cell[~(cell["is_penalty"] | cell["is_holeout"])].copy()
            original_n = len(regular)
            if original_n > regular_per_year_cell:
                regular = regular.nsmallest(regular_per_year_cell, "_sample_hash")
            regular["tour_sampling_weight"] = (
                original_n / max(len(regular), 1)
            )
            special["tour_sampling_weight"] = 1.0
            selected_parts.extend([regular, special])
        sampled = pd.concat(selected_parts, ignore_index=True)
        parts.append(sampled.drop(columns="_sample_hash"))
    cache = pd.concat(parts, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(cache_path, index=False)
    print(f"Wrote Tour APP cache: {len(cache):,} rows to {cache_path}")
    return cache


def _fallback_transform(
    frame: pd.DataFrame,
    value: str,
    detailed_keys: list[str],
    broad_keys: list[str],
    statistic: str,
    minimum_count: int,
) -> pd.Series:
    detailed = frame.groupby(detailed_keys, observed=True)[value]
    broad = frame.groupby(broad_keys, observed=True)[value]
    if statistic == "mean":
        primary = detailed.transform("mean")
        fallback = broad.transform("mean")
    elif statistic == "std":
        primary = detailed.transform("std")
        fallback = broad.transform("std")
    else:
        raise ValueError(statistic)
    count = detailed.transform("size")
    return primary.where(count.ge(minimum_count), fallback)


def _audited_distance_band(values: pd.Series) -> pd.Series:
    """The four APP yardage bands validated in the independent audit."""
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 100.0, 150.0, 215.0, np.inf],
        labels=["<=100", "100-150", "150-215", ">215"],
        include_lowest=True,
        right=True,
    ).astype("string")


def _audited_distance_band_scalar(value: float) -> str:
    if value <= 100.0:
        return "<=100"
    if value <= 150.0:
        return "100-150"
    if value <= 215.0:
        return "150-215"
    return ">215"


def fit_player_profiles(
    approaches: pd.DataFrame,
    cross_mean_prior: float = 200.0,
    along_mean_prior: float = 200.0,
    variance_prior: float = 450.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit audit-selected, strongly partial-pooled APP transforms.

    Cross-track bias is a player-wide effect shrunk to the global zero mean.
    Along-track bias is allowed to differ by 50-yard band and start lie, also
    shrunk to zero with 200 equivalent shots. Dispersion is player-wide and
    receives a still stronger 450-shot variance prior.
    """
    frame = approaches.copy()
    frame["distance_bucket"] = (
        (frame["start_distance_yd"] / 20).round().clip(1, 30).astype(int)
    )
    detailed = [
        "event_key", "round_no", "hole_no", "start_lie_class", "distance_bucket",
    ]
    broad = ["event_key", "start_lie_class", "distance_bucket"]
    for axis in ("long", "lat"):
        value = f"miss_{axis}_yd"
        center = _fallback_transform(frame, value, detailed, broad, "mean", 8)
        scale = _fallback_transform(frame, value, detailed, broad, "std", 8)
        global_center = frame.groupby(
            ["start_lie_class", "distance_bucket"], observed=True
        )[value].transform("mean")
        global_scale = frame.groupby(
            ["start_lie_class", "distance_bucket"], observed=True
        )[value].transform("std")
        center = center.fillna(global_center).fillna(frame[value].mean())
        scale = scale.fillna(global_scale).fillna(frame[value].std()).clip(3.0, 80.0)
        frame[f"{axis}_resid"] = frame[value] - center
        frame[f"{axis}_std_resid"] = frame[f"{axis}_resid"] / scale

    players = frame.groupby("source_player_id", observed=True)
    for axis in ("long", "lat"):
        standardized_sum = players[f"{axis}_std_resid"].transform("sum")
        standardized_n = players[f"{axis}_std_resid"].transform("size")
        shrunk_center = standardized_sum / (standardized_n + 200.0)
        frame[f"{axis}_disp_resid"] = (
            frame[f"{axis}_std_resid"]
            - shrunk_center
        )
    global_long_var = float(np.mean(frame["long_disp_resid"] ** 2))
    global_lat_var = float(np.mean(frame["lat_disp_resid"] ** 2))
    profiles = players.agg(
        player_name=("player_name", "last"),
        n_app=("shot_key", "size"),
        long_resid_sum=("long_resid", "sum"),
        lat_resid_sum=("lat_resid", "sum"),
        long_resid_ss=("long_disp_resid", lambda x: float(np.sum(np.asarray(x) ** 2))),
        lat_resid_ss=("lat_disp_resid", lambda x: float(np.sum(np.asarray(x) ** 2))),
    ).reset_index()
    n = profiles["n_app"].to_numpy(float)
    profiles["lat_bias_yd"] = (
        profiles["lat_resid_sum"] / (n + cross_mean_prior)
    ).clip(-8, 8)
    profiles["long_scale"] = np.sqrt(
        (profiles["long_resid_ss"] + variance_prior * global_long_var)
        / ((n + variance_prior) * global_long_var)
    ).clip(0.72, 1.35)
    profiles["lat_scale"] = np.sqrt(
        (profiles["lat_resid_ss"] + variance_prior * global_lat_var)
        / ((n + variance_prior) * global_lat_var)
    ).clip(0.72, 1.35)
    profiles["source_player_id"] = profiles["source_player_id"].astype(str)

    # A single signed long miss is not stable across wedge, mid-iron and long-
    # iron states. Fit it only inside the four audited distance x lie cells; the large
    # prior leaves unseen cells exactly neutral instead of borrowing an
    # inappropriate whole-player direction.
    frame["profile_distance_band"] = _audited_distance_band(
        frame["start_distance_yd"]
    )
    band_profiles = frame.groupby(
        ["source_player_id", "start_lie_class", "profile_distance_band"],
        observed=True,
        as_index=False,
    ).agg(
        n_band=("shot_key", "size"),
        long_band_resid_sum=("long_resid", "sum"),
    )
    band_n = band_profiles["n_band"].to_numpy(float)
    band_profiles["long_bias_yd"] = (
        band_profiles["long_band_resid_sum"] / (band_n + along_mean_prior)
    ).clip(-8, 8)
    profiles = band_profiles.merge(
        profiles.drop(columns=["long_resid_sum", "lat_resid_sum"]),
        on="source_player_id",
        how="left",
        validate="many_to_one",
    )
    return profiles, {
        "cross_mean_prior": cross_mean_prior,
        "along_mean_prior": along_mean_prior,
        "variance_prior": variance_prior,
        "global_long_sd": math.sqrt(global_long_var),
        "global_lat_sd": math.sqrt(global_lat_var),
        "training_shots": len(frame),
    }


def _distance_kernel(values: pd.Series, center: float, bandwidth: float = 24.0) -> np.ndarray:
    delta = (values.to_numpy(float) - center) / bandwidth
    weights = np.exp(-0.5 * delta * delta)
    weights[np.abs(delta) > 5] = 0.0
    return weights


def build_distribution(
    course_train: pd.DataFrame,
    round_no: int,
    hole_no: int,
    start_lie: str,
    distance_center: float,
    setup_prior: float = 70.0,
    hole_prior: float = 220.0,
    lie_prior: float = 700.0,
    max_support: int = 3000,
) -> AppDistribution:
    same_lie = course_train["start_lie_class"].eq(start_lie)
    setup_mask = (
        course_train["round_no"].eq(round_no)
        & course_train["hole_no"].eq(hole_no)
        & same_lie
    )
    hole_mask = course_train["hole_no"].eq(hole_no) & same_lie
    lie_mask = same_lie
    pools = [
        ("setup", setup_mask, setup_prior),
        ("hole", hole_mask, hole_prior),
        ("lie", lie_mask, lie_prior),
        ("event", pd.Series(True, index=course_train.index), np.inf),
    ]
    remaining_mass = 1.0
    mass = pd.Series(0.0, index=course_train.index)
    labels: list[str] = []
    counts: list[int] = []
    for name, mask, prior in pools:
        subset = course_train.loc[mask]
        kernel = _distance_kernel(subset["reference_start_distance_yd"], distance_center)
        positive = kernel > 1e-5
        subset = subset.loc[positive]
        kernel = kernel[positive]
        counts.append(int(len(subset)))
        if not len(subset):
            labels.append(f"{name}=0.00")
            continue
        kernel /= kernel.sum()
        n_eff = float(np.sum(positive))
        fraction = 1.0 if not np.isfinite(prior) else n_eff / (n_eff + prior)
        component_mass = remaining_mass * fraction
        mass.loc[subset.index] += component_mass * kernel
        remaining_mass -= component_mass
        labels.append(f"{name}={component_mass:.2f}")
        if remaining_mass < 1e-6:
            break
    positive_mass = mass[mass > 0]
    if positive_mass.empty:
        raise ValueError("No APP course support for requested state")
    if len(positive_mass) > max_support:
        seed_text = f"{round_no}|{hole_no}|{start_lie}|{distance_center:.1f}"
        rng = np.random.default_rng(zlib.crc32(seed_text.encode("utf-8")))
        probability = positive_mass.to_numpy(float)
        probability /= probability.sum()
        # Compression must preserve the target empirical measure. Sampling
        # without replacement and then retaining the original masses would
        # select high-mass rows more often *and* weight them more heavily,
        # effectively squaring their influence. Draw with replacement and use
        # the aggregated draw frequencies as the compressed distribution.
        chosen = rng.choice(
            len(positive_mass), size=max_support, replace=True, p=probability
        )
        draw_counts = np.bincount(chosen, minlength=len(positive_mass))
        selected = np.flatnonzero(draw_counts)
        positive_mass = pd.Series(
            draw_counts[selected] / float(max_support),
            index=positive_mass.index[selected],
            dtype=float,
        )
    rows = course_train.loc[positive_mass.index].copy().reset_index(drop=True)
    weights = positive_mass.to_numpy(float)
    weights /= weights.sum()
    while len(counts) < 3:
        counts.append(0)
    return AppDistribution(
        rows=rows,
        weights=weights,
        source=",".join(labels),
        n_setup=counts[0],
        n_hole=counts[1],
        n_lie=counts[2],
    )


def build_tour_distribution(
    tour_train: pd.DataFrame,
    start_lie: str,
    distance_center: float,
    max_support: int = 3000,
) -> AppDistribution:
    """Tour-only distance x lie baseline with no course or player inputs."""
    subset = tour_train[tour_train["start_lie_class"].eq(start_lie)].copy()
    if len(subset) < 50:
        subset = tour_train.copy()
    kernel = _distance_kernel(
        subset["reference_start_distance_yd"], distance_center, bandwidth=24.0
    )
    mass = kernel * subset["tour_sampling_weight"].to_numpy(float)
    keep = mass > 1e-8
    subset = subset.loc[keep].copy()
    mass = mass[keep]
    if not len(subset):
        raise ValueError("No Tour APP support for requested distance/lie state")
    mass /= mass.sum()
    if len(subset) > max_support:
        seed_text = f"tour|{start_lie}|{distance_center:.1f}"
        rng = np.random.default_rng(zlib.crc32(seed_text.encode("utf-8")))
        chosen = rng.choice(len(subset), size=max_support, replace=True, p=mass)
        counts = np.bincount(chosen, minlength=len(subset))
        selected = np.flatnonzero(counts)
        subset = subset.iloc[selected].copy()
        mass = counts[selected].astype(float) / max_support
    return AppDistribution(
        rows=subset.reset_index(drop=True),
        weights=mass / mass.sum(),
        source="tour_distance_x_lie",
        n_setup=0,
        n_hole=0,
        n_lie=len(subset),
    )


def adjust_points(
    points: np.ndarray,
    weights: np.ndarray,
    profile: pd.Series,
    arm: str,
) -> np.ndarray:
    if arm == "population":
        return points.copy()
    center = _weighted_mean(points, weights)
    use_mean = arm in {"mean_transport", "combined"}
    use_dispersion = arm in {"dispersion_transport", "combined"}
    long_bias = float(profile["long_bias_yd"]) if use_mean else 0.0
    lat_bias = float(profile["lat_bias_yd"]) if use_mean else 0.0
    long_scale = float(profile["long_scale"]) if use_dispersion else 1.0
    lat_scale = float(profile["lat_scale"]) if use_dispersion else 1.0
    adjusted = np.empty_like(points)
    adjusted[:, 0] = center[0] + long_bias + long_scale * (points[:, 0] - center[0])
    adjusted[:, 1] = center[1] + lat_bias + lat_scale * (points[:, 1] - center[1])
    return adjusted


class AbsoluteSurfaceMap:
    """Historical endpoint-to-lie map in stable course coordinates."""

    def __init__(self, hole_history: pd.DataFrame, prior_strength: float = 10.0):
        regular = hole_history[
            hole_history["spatial_valid"]
            & ~hole_history["is_holeout"]
        ].copy()
        if regular.empty:
            raise ValueError("No spatial course history")
        self.points = regular[["end_x", "end_y"]].to_numpy(float) / 3.0
        self.codes = pd.Categorical(regular["end_lie_class"], categories=END_LIES).codes
        self.tree = cKDTree(self.points)
        counts = np.bincount(
            self.codes[self.codes >= 0], minlength=len(END_LIES)
        ).astype(float)
        counts[END_LIES.index("hole")] = 0.0
        self.prior = (counts + 0.5) / (counts.sum() + 0.5 * len(END_LIES))
        self.prior[END_LIES.index("hole")] = 0.0
        self.prior /= self.prior.sum()
        self.prior_strength = prior_strength

    def probabilities(self, query_yd: np.ndarray, k: int = 120) -> np.ndarray:
        k = min(k, len(self.points))
        distance, index = self.tree.query(query_yd, k=k)
        if k == 1:
            distance, index = distance[:, None], index[:, None]
        local_scale = np.maximum(np.median(distance, axis=1), 3.0)
        weight = np.exp(-0.5 * (distance / local_scale[:, None]) ** 2)
        result = np.tile(self.prior * self.prior_strength, (len(query_yd), 1))
        for code in range(len(END_LIES)):
            result[:, code] += np.sum(weight * (self.codes[index] == code), axis=1)
        result[:, END_LIES.index("hole")] = 0.0
        return result / result.sum(axis=1, keepdims=True)


def _endpoint_to_absolute(
    points: np.ndarray,
    pin_x: float,
    pin_y: float,
    ux: float,
    uy: float,
) -> np.ndarray:
    # miss_lat is positive to the player's left: perpendicular is (-uy, ux).
    x = pin_x / 3.0 + points[:, 0] * ux - points[:, 1] * uy
    y = pin_y / 3.0 + points[:, 0] * uy + points[:, 1] * ux
    return np.column_stack([x, y])


def _lie_probabilities(
    surface: AbsoluteSurfaceMap,
    adjusted_points: np.ndarray,
    spatial_weights: np.ndarray,
    holeout_probability: float,
    record: object,
    max_queries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(adjusted_points) > max_queries:
        chosen = rng.choice(
            len(adjusted_points),
            size=max_queries,
            replace=True,
            p=spatial_weights / spatial_weights.sum(),
        )
        query_points = adjusted_points[chosen]
        query_weights = np.full(max_queries, 1.0 / max_queries)
    else:
        query_points = adjusted_points
        query_weights = spatial_weights / spatial_weights.sum()
    absolute = _endpoint_to_absolute(
        query_points,
        float(record.pin_x),
        float(record.pin_y),
        float(record.axis_ux),
        float(record.axis_uy),
    )
    conditional = surface.probabilities(absolute)
    probabilities = np.sum(conditional * query_weights[:, None], axis=0)
    regular_probability = max(1.0 - holeout_probability, 0.0)
    probabilities *= regular_probability
    probabilities[END_LIES.index("hole")] = holeout_probability
    probabilities = np.maximum(probabilities, 1e-10)
    return probabilities / probabilities.sum()


def score_holdout(
    course_train: pd.DataFrame,
    holdout: pd.DataFrame,
    profiles: pd.DataFrame,
    tour_train: pd.DataFrame | None,
    setup_mode: str,
    seed: int,
    max_lie_queries: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_frame = profiles.copy()
    profile_frame["source_player_id"] = profile_frame["source_player_id"].astype(str)
    profile_frame["profile_distance_band"] = profile_frame[
        "profile_distance_band"
    ].astype(str)
    band_lookup = profile_frame.set_index(
        ["source_player_id", "start_lie_class", "profile_distance_band"]
    )
    base_lookup = profile_frame.groupby("source_player_id", observed=True).first()
    neutral = pd.Series(
        {
            "n_app": 0,
            "n_band": 0,
            "long_bias_yd": 0.0,
            "lat_bias_yd": 0.0,
            "long_scale": 1.0,
            "lat_scale": 1.0,
        }
    )
    surfaces = {
        int(hole): AbsoluteSurfaceMap(history)
        for hole, history in course_train.groupby("hole_no", observed=True)
        if history["spatial_valid"].any()
    }
    cache: dict[tuple[int, int, str, int], AppDistribution] = {}
    tour_cache: dict[tuple[str, int], AppDistribution] = {}
    rows: list[dict] = []
    for record in holdout.itertuples(index=False):
        distance_bucket = int(round(float(record.reference_start_distance_yd) / 15.0))
        key = (
            int(record.round_no), int(record.hole_no), str(record.start_lie_class),
            distance_bucket,
        )
        if key not in cache:
            cache[key] = build_distribution(
                course_train,
                key[0], key[1], key[2], distance_bucket * 15.0,
            )
        distribution = cache[key]
        support = distribution.rows
        all_weights = distribution.weights
        # A causal penalty keeps the prior APP stroke's real landing endpoint
        # and is therefore part of the continuous spatial support. A hole-out
        # remains a fixed atom at the cup and must never be transported.
        spatial_mask = (
            support["spatial_valid"].to_numpy(bool)
            & ~support["is_holeout"].to_numpy(bool)
        )
        if spatial_mask.sum() < 12 or int(record.hole_no) not in surfaces:
            continue
        spatial_points = support.loc[
            spatial_mask, ["miss_long_yd", "miss_lat_yd"]
        ].to_numpy(float)
        spatial_weights = all_weights[spatial_mask]
        spatial_weights /= spatial_weights.sum()
        penalty_probability = float(all_weights[support["is_penalty"].to_numpy(bool)].sum())
        holeout_probability = float(all_weights[support["is_holeout"].to_numpy(bool)].sum())
        player_id = str(record.source_player_id)
        has_profile = player_id in base_lookup.index
        profile_band = _audited_distance_band_scalar(
            float(record.reference_start_distance_yd)
        )
        profile_key = (player_id, str(record.start_lie_class), profile_band)
        has_band_profile = profile_key in band_lookup.index
        if has_band_profile:
            profile = band_lookup.loc[profile_key]
        elif has_profile:
            # Whole-player cross bias and dispersion remain valid fallbacks;
            # unseen along-distance cells stay neutral by design.
            profile = base_lookup.loc[player_id].copy()
            profile["long_bias_yd"] = 0.0
            profile["n_band"] = 0
        else:
            profile = neutral
        if isinstance(profile, pd.DataFrame):
            profile = profile.iloc[-1]
        actual_lie = str(record.end_lie_class)
        actual_lie_index = END_LIES.index(actual_lie)
        onehot = np.zeros(len(END_LIES))
        onehot[actual_lie_index] = 1.0
        penalty_strokes = getattr(record, "penalty_stroke_cost", np.nan)
        if not np.isfinite(penalty_strokes):
            penalty_strokes = record.penalty_count
        row = {
            "event_key": record.event_key,
            "season": int(record.season),
            "round_no": int(record.round_no),
            "hole_no": int(record.hole_no),
            "source_player_id": player_id,
            "player_name": record.player_name,
            "start_lie": record.start_lie_class,
            "start_distance_yd": float(record.reference_start_distance_yd),
            "actual_end_lie": actual_lie,
            "actual_remaining_yd": float(record.scored_remaining_yd),
            "actual_miss_long_yd": float(record.miss_long_yd),
            "actual_miss_lat_yd": float(record.miss_lat_yd),
            "spatial_actual": bool(record.spatial_valid),
            "continuous_endpoint_actual": bool(record.spatial_valid) and not bool(record.is_holeout),
            "is_penalty": bool(record.is_penalty),
            "penalty_strokes": int(penalty_strokes),
            "is_holeout": bool(record.is_holeout),
            "setup_mode": setup_mode,
            "has_player_profile": has_profile,
            "has_band_profile": has_band_profile,
            "profile_n_app": int(profile.get("n_app", 0)),
            "profile_n_band": int(profile.get("n_band", 0)),
            "long_bias_yd": float(profile["long_bias_yd"]),
            "lat_bias_yd": float(profile["lat_bias_yd"]),
            "long_scale": float(profile["long_scale"]),
            "lat_scale": float(profile["lat_scale"]),
            "n_setup_history": distribution.n_setup,
            "n_hole_history": distribution.n_hole,
            "n_lie_history": distribution.n_lie,
            "backoff_weights": distribution.source,
            "forecast_penalty_probability": penalty_probability,
            "forecast_holeout_probability": holeout_probability,
        }

        # Separately test whether exact-course population donors add value over
        # a strict-prior Tour distance x lie transition. This baseline receives
        # no current-course and no player-profile information.
        if tour_train is not None:
            tour_key = (str(record.start_lie_class), distance_bucket)
            if tour_key not in tour_cache:
                tour_cache[tour_key] = build_tour_distribution(
                    tour_train, tour_key[0], distance_bucket * 15.0
                )
            tour_distribution = tour_cache[tour_key]
            tour_support = tour_distribution.rows
            tour_weights = tour_distribution.weights
            tour_spatial_mask = (
                tour_support["spatial_valid"].to_numpy(bool)
                & ~tour_support["is_holeout"].to_numpy(bool)
            )
            tour_points = tour_support.loc[
                tour_spatial_mask, ["miss_long_yd", "miss_lat_yd"]
            ].to_numpy(float)
            tour_spatial_weights = tour_weights[tour_spatial_mask]
            tour_spatial_weights /= tour_spatial_weights.sum()
            continuous_actual = bool(record.spatial_valid) and not bool(record.is_holeout)
            if continuous_actual:
                observed = np.array(
                    [record.miss_long_yd, record.miss_lat_yd], dtype=float
                )
                row["tour_endpoint_nll"] = _kde_nll(
                    tour_points, tour_spatial_weights.copy(), observed
                )
                row["tour_endpoint_error_yd"] = float(np.linalg.norm(
                    observed - _weighted_mean(tour_points, tour_spatial_weights)
                ))
            else:
                row["tour_endpoint_nll"] = np.nan
                row["tour_endpoint_error_yd"] = np.nan
            tour_remaining = tour_support["scored_remaining_yd"].to_numpy(float).copy()
            tour_regular = tour_spatial_mask & ~tour_support["is_penalty"].to_numpy(bool)
            tour_point_regular = ~tour_support.loc[
                tour_spatial_mask, "is_penalty"
            ].to_numpy(bool)
            tour_remaining[tour_regular] = np.hypot(
                tour_points[tour_point_regular, 0],
                tour_points[tour_point_regular, 1],
            )
            row["tour_remaining_crps"] = _crps(
                tour_remaining, tour_weights, float(record.scored_remaining_yd)
            )
            tour_lie_probs = np.array([
                float(tour_weights[tour_support["end_lie_class"].eq(lie)].sum())
                for lie in END_LIES
            ])
            tour_lie_probs = np.maximum(tour_lie_probs, 1e-8)
            tour_lie_probs /= tour_lie_probs.sum()
            row["tour_lie_logloss"] = -math.log(
                max(float(tour_lie_probs[actual_lie_index]), 1e-12)
            )
            row["tour_lie_brier"] = float(
                np.mean((tour_lie_probs - onehot) ** 2)
            )
        regular_remaining_mask = spatial_mask & ~support["is_penalty"].to_numpy(bool)
        point_regular_remaining_mask = ~support.loc[spatial_mask, "is_penalty"].to_numpy(bool)
        special_remaining = support["scored_remaining_yd"].to_numpy(float)
        # Use common random numbers for the surface-map integration so arm
        # deltas reflect the transport rather than independent Monte Carlo
        # subsamples of the same empirical support.
        lie_seed = seed ^ zlib.crc32(
            f"{record.shot_key}|{setup_mode}".encode("utf-8")
        )
        for arm in ARMS:
            points = adjust_points(spatial_points, spatial_weights, profile, arm)
            continuous_actual = bool(record.spatial_valid) and not bool(record.is_holeout)
            if continuous_actual:
                observed = np.array(
                    [record.miss_long_yd, record.miss_lat_yd], dtype=float
                )
                row[f"{arm}_endpoint_nll"] = _kde_nll(
                    points, spatial_weights.copy(), observed
                )
                row[f"{arm}_endpoint_error_yd"] = float(
                    np.linalg.norm(observed - _weighted_mean(points, spatial_weights))
                )
            else:
                row[f"{arm}_endpoint_nll"] = np.nan
                row[f"{arm}_endpoint_error_yd"] = np.nan
            remaining_samples = special_remaining.copy()
            remaining_samples[regular_remaining_mask] = np.hypot(
                points[point_regular_remaining_mask, 0],
                points[point_regular_remaining_mask, 1],
            )
            row[f"{arm}_remaining_crps"] = _crps(
                remaining_samples, all_weights, float(record.scored_remaining_yd)
            )
            lie_probs = _lie_probabilities(
                surfaces[int(record.hole_no)],
                points,
                spatial_weights,
                holeout_probability,
                record,
                max_lie_queries,
                np.random.default_rng(lie_seed),
            )
            row[f"{arm}_lie_logloss"] = -math.log(
                max(float(lie_probs[actual_lie_index]), 1e-12)
            )
            row[f"{arm}_lie_brier"] = float(np.mean((lie_probs - onehot) ** 2))
            for lie, probability in zip(END_LIES, lie_probs):
                row[f"{arm}_{lie}_prob"] = float(probability)
        rows.append(row)
    scores = pd.DataFrame(rows)
    return scores, summarize(scores)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "endpoint_nll", "endpoint_error_yd", "remaining_crps", "lie_logloss",
        "lie_brier",
    )
    rows: list[dict] = []
    cluster_keys = ["round_no", "hole_no"]
    subsets = {
        "all_states": pd.Series(True, index=scores.index),
        "continuous_no_penalty": (
            scores["continuous_endpoint_actual"] & ~scores["is_penalty"]
        ),
    }
    for subset_name, subset_mask in subsets.items():
      subset_scores = scores.loc[subset_mask]
      for arm in ARMS[1:]:
        for metric in metrics:
            baseline = f"population_{metric}"
            adjusted = f"{arm}_{metric}"
            valid = subset_scores[[baseline, adjusted] + cluster_keys].dropna()
            delta_values = valid[adjusted] - valid[baseline]
            cluster_delta = valid.assign(_delta=delta_values).groupby(cluster_keys)["_delta"].mean()
            delta = float(delta_values.mean())
            se = (
                float(cluster_delta.std(ddof=1) / math.sqrt(len(cluster_delta)))
                if len(cluster_delta) > 1 else np.nan
            )
            rows.append(
                {
                    "evaluation_subset": subset_name,
                    "comparison": "player_transport_vs_course_population",
                    "arm": arm,
                    "metric": metric,
                    "n": len(valid),
                    "population": float(valid[baseline].mean()),
                    "player_adjusted": float(valid[adjusted].mean()),
                    "delta_adjusted_minus_population": delta,
                    "round_hole_cluster_se": se,
                    "cluster_ci_low": delta - 1.96 * se,
                    "cluster_ci_high": delta + 1.96 * se,
                    "lower_is_better": True,
                }
            )
      if "tour_endpoint_nll" in subset_scores.columns:
        # Endpoint and next-yardage scores are directly comparable. Lie is
        # intentionally omitted: the course arm uses an endpoint-conditioned
        # spatial surface, while this generic Tour baseline uses donor lie
        # mass, so a delta would conflate donor scope with lie estimator.
        for metric in metrics[:3]:
            baseline = f"tour_{metric}"
            adjusted = f"population_{metric}"
            valid = subset_scores[[baseline, adjusted] + cluster_keys].dropna()
            delta_values = valid[adjusted] - valid[baseline]
            cluster_delta = valid.assign(_delta=delta_values).groupby(cluster_keys)["_delta"].mean()
            delta = float(delta_values.mean())
            se = (
                float(cluster_delta.std(ddof=1) / math.sqrt(len(cluster_delta)))
                if len(cluster_delta) > 1 else np.nan
            )
            rows.append(
                {
                    "evaluation_subset": subset_name,
                    "comparison": "course_population_vs_tour",
                    "arm": "exact_course_population",
                    "metric": metric,
                    "n": len(valid),
                    "population": float(valid[baseline].mean()),
                    "player_adjusted": float(valid[adjusted].mean()),
                    "delta_adjusted_minus_population": delta,
                    "round_hole_cluster_se": se,
                    "cluster_ci_low": delta - 1.96 * se,
                    "cluster_ci_high": delta + 1.96 * se,
                    "lower_is_better": True,
                }
            )
    return pd.DataFrame(rows)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def event_report(
    event_name: str,
    setup_mode: str,
    course_train: pd.DataFrame,
    holdout: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: pd.DataFrame,
    globals_: dict[str, float],
    smoke_test: bool,
) -> str:
    lines = []
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_subset} | {row.comparison} | {row.arm} | {row.metric} | {row.n:,} | {row.population:.6f} | "
            f"{row.player_adjusted:.6f} | {row.delta_adjusted_minus_population:+.6f} | "
            f"[{row.cluster_ci_low:+.6f}, {row.cluster_ci_high:+.6f}] |"
        )
    penalty_count = int(holdout["is_penalty"].sum())
    holeout_count = int(holdout["is_holeout"].sum())
    spatial_count = int((holdout["spatial_valid"] & ~holdout["is_holeout"]).sum())
    return f"""# Spatial APP transition: {event_name}

- Holdout: **2026**; setup mode: **{setup_mode}**
- Prior course states: **{len(course_train):,}** from **{course_train['season'].nunique()}** editions
- Held-out APP states: **{len(holdout):,}**; scored after support checks: **{len(scores):,}**
- Continuous spatial endpoints: **{spatial_count:,}**; penalties retained: **{penalty_count:,}**; hole-outs retained: **{holeout_count:,}**
- Holdout profile coverage: **{scores['has_player_profile'].mean():.1%}**
- Profile training shots: **{int(globals_['training_shots']):,}**
- Shrinkage: cross-bias prior **{globals_['cross_mean_prior']:.0f}**, along band x lie prior **{globals_['along_mean_prior']:.0f}**, variance prior **{globals_['variance_prior']:.0f}** APP shots
- State foundation: **{FOUNDATION_VERSION}**
- Coordinate reference: **{course_train['coordinate_reference_mode'].iloc[0]}** (latest-prior mode fixes the donor frame without heldout setup geometry)
- Smoke test: **{smoke_test}**

All player and course inputs precede 2026. In historical mode, canonical pins
are medians of prior-edition geometry only. Lower scores are better. Explicit
penalty bookkeeping rows are not independent observations: they are collapsed
using the complete raw sequence into the causal preceding played APP shot.
Its real endpoint enters the continuous KDE, the penalty label enters the
endpoint-linked surface, and its remaining state uses the exact following
STROKE/PROVISIONAL raw start lie and coordinates joined by shot key. Remaining
distance is recomputed from those coordinates to the active forecast pin; the
post-penalty SG annotation is diagnostic only. Hole-outs alone are held as
fixed discrete atoms at the cup.

| Subset | Comparison | Candidate | Metric | N | Baseline | Candidate score | Delta | Round-hole cluster 95% CI |
|---|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

The population transition is a continuously distance-weighted empirical
mixture of the same round/hole/lie, same hole/lie, same course/lie, and full
course history. Mean transport applies shrunk player long/left-right biases;
dispersion transport applies shrunk longitudinal/lateral scales; combined
applies both. End-lie probabilities come from a historical spatial surface map
in absolute course coordinates, including causal penalty endpoints. The
post-drop distance for a penalty is recomputed from the exact next playable
raw start coordinates to the active pin, although this one-step evaluator does not yet resample
drop placement when a transported endpoint changes hazard membership.

Rows labeled `course_population_vs_tour` compare that exact-course population
transition with a separate strict-prior Tour donor pool conditioned only on
distance and start lie. In `actual` mode both arms use realized pins, so this
is a matched donor-scope sensitivity. In `historical` mode the course arm also
contains prior-only pin/setup forecast uncertainty while Tour donors were
normalized to their realized pins; that delta tests the full deployable
course+setup model but does **not** isolate course value by itself. The player-
transport rows remain matched within either setup mode.
The target event's own prior editions are removed from the Tour pool for this
comparison. Lie metrics are not reported course-vs-Tour because the course arm
uses a spatial surface whereas the generic arm uses direct donor lie mass.

Limitations: there are no surface polygons, aim or club labels; coordinates
mix carry and roll; weather and shared round/week latent effects are absent;
the pin forecast is deliberately simple; and this evaluates a one-step APP
transition rather than a recursively generated state from the tee model.
"""


def combined_report(output_dir: Path) -> tuple[pd.DataFrame, Path]:
    parts = []
    for path in sorted(output_dir.glob("app_transition_metrics_*.csv")):
        if "_smoke" in path.stem:
            continue
        match = re.fullmatch(
            r"app_transition_metrics_(.+)_(actual|historical)", path.stem
        )
        if not match:
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "event", match.group(1).replace("_", " ").title())
        frame.insert(1, "setup_mode", match.group(2))
        parts.append(frame)
    if not parts:
        raise ValueError("No full APP metric files found")
    event_metrics = pd.concat(parts, ignore_index=True)
    event_metrics.to_csv(output_dir / "combined_event_metrics.csv", index=False)
    rows = []
    for (mode, subset, comparison, arm, metric), group in event_metrics.groupby(
        ["setup_mode", "evaluation_subset", "comparison", "arm", "metric"], sort=True
    ):
        weight = group["n"].to_numpy(float)
        deltas = group["delta_adjusted_minus_population"].to_numpy(float)
        event_se = float(deltas.std(ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else np.nan
        rows.append(
            {
                "setup_mode": mode,
                "evaluation_subset": subset,
                "comparison": comparison,
                "arm": arm,
                "metric": metric,
                "events": len(group),
                "n": int(weight.sum()),
                "observation_weighted_delta": float(np.average(deltas, weights=weight)),
                "event_mean_delta": float(deltas.mean()),
                "event_cluster_se": event_se,
                # Only three recurring-event holdouts are available. Keep this
                # normal-approximation band as a descriptive stability check;
                # it is not a formal small-sample 95% confidence interval.
                "event_cluster_ci_low": float(deltas.mean() - 1.96 * event_se),
                "event_cluster_ci_high": float(deltas.mean() + 1.96 * event_se),
            }
        )
    pooled = pd.DataFrame(rows)
    pooled.to_csv(output_dir / "combined_pooled_metrics.csv", index=False)
    table = []
    for row in pooled.itertuples(index=False):
        table.append(
            f"| {row.setup_mode} | {row.evaluation_subset} | {row.comparison} | {row.arm} | {row.metric} | {row.events} | {row.n:,} | "
            f"{row.observation_weighted_delta:+.6f} | {row.event_mean_delta:+.6f} | "
            f"[{row.event_cluster_ci_low:+.6f}, {row.event_cluster_ci_high:+.6f}] |"
        )
    report = f"""# Combined spatial APP transition holdouts

The study contains strict 2026 Wyndham, Sony and Travelers holdouts. Player
profiles and course distributions use only pre-2026 shots. `actual` isolates
transition quality with the held-out pin known; `historical` forecasts the pin
from prior-edition geometry. Negative deltas favor the player adjustment.

For `course_population_vs_tour`, only the `actual` setup is a matched estimate
of exact-course donor value. The historical course arm includes setup forecast
uncertainty whereas generic Tour donors are normalized to actual historical
pins, so that row is a full-model sensitivity rather than a course-only effect.

| Setup | Subset | Comparison | Candidate | Metric | Events | N | Observation-weighted delta | Event-mean delta | Descriptive normal-approx interval |
|---|---|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Intervals are event mean plus/minus `1.96 x SE` using only three event means.
They are descriptive, not formal 95% confidence intervals; a t critical value
with two degrees of freedom is materially wider. Penalties and hole-outs are
included in remaining-distance and lie scores. A penalty donor retains its
exact following physical-stroke state, which is not yet resampled when player
transport moves an endpoint into or out of a hazard. This test still starts from each
player's realized approach state; the next gate is feeding it states generated
by the tee transition and recursively finishing the hole.
"""
    path = output_dir / "COMBINED_APP_TRANSITION_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return pooled, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-name", action="append")
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--profile-start-year", type=int, default=2020)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument(
        "--course-last-editions",
        type=int,
        default=0,
        help="Restrict course donors to the N latest pre-holdout editions; 0 uses all.",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--setup-mode", choices=["actual", "historical", "both"], default="both"
    )
    parser.add_argument("--profiles-csv", type=Path)
    parser.add_argument(
        "--max-holdout-per-event",
        type=int,
        default=0,
        help="Deterministic round-hole-balanced cap; 0 scores every holdout state.",
    )
    parser.add_argument("--max-lie-queries", type=int, default=300)
    parser.add_argument("--fit-profiles-only", action="store_true")
    parser.add_argument("--tour-cache", type=Path)
    parser.add_argument("--no-tour-baseline", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.combine_only:
        pooled, report_path = combined_report(args.output_dir)
        print(pooled.to_string(index=False))
        print(f"Wrote {report_path}")
        return
    if args.profiles_csv:
        profiles = pd.read_csv(args.profiles_csv, dtype={"source_player_id": str})
        required = {
            "start_lie_class", "profile_distance_band", "n_band",
            "long_bias_yd", "lat_bias_yd", "long_scale", "lat_scale",
        }
        missing = required.difference(profiles.columns)
        if missing:
            raise ValueError(
                f"Profile file predates structured APP shrinkage; missing {sorted(missing)}"
            )
        globals_ = {
            "cross_mean_prior": 200.0,
            "along_mean_prior": 200.0,
            "variance_prior": 450.0,
            "global_long_sd": np.nan,
            "global_lat_sd": np.nan,
            "training_shots": int(
                profiles.groupby("source_player_id", observed=True)["n_app"].first().sum()
            ),
        }
    else:
        training = load_player_training(
            args.export_root,
            args.profile_start_year,
            args.holdout_year,
            smoke_test=args.smoke_test,
        )
        profiles, globals_ = fit_player_profiles(training)
        profile_path = args.output_dir / (
            "player_app_profiles_structured_smoke.csv"
            if args.smoke_test else
            "player_app_profiles_structured.csv"
        )
        profiles.to_csv(profile_path, index=False)
        print(f"Wrote {profile_path}")
    if args.fit_profiles_only:
        return
    tour_train = None
    if not args.no_tour_baseline:
        tour_cache_path = args.tour_cache or (
            args.output_dir / (
                "tour_app_donors_fullseq_smoke.parquet"
                if args.smoke_test else
                "tour_app_donors_fullseq_2020_2025.parquet"
            )
        )
        tour_train = build_tour_transition_cache(
            args.export_root,
            args.holdout_year - 1 if args.smoke_test else args.profile_start_year,
            args.holdout_year,
            tour_cache_path,
        )
    events = args.event_name or list(DEFAULT_EVENTS)
    if args.smoke_test:
        events = events[:1]
    modes = ("actual", "historical") if args.setup_mode == "both" else (args.setup_mode,)
    for event_index, event_name in enumerate(events):
        train_raw, holdout_raw, event_keys = load_course_panel(
            args.export_root,
            event_name,
            args.holdout_year,
            args.course_start_year,
        )
        if args.course_last_editions > 0:
            donor_years = sorted(train_raw["season"].unique())[-args.course_last_editions:]
            train_raw = train_raw[train_raw["season"].isin(donor_years)].copy()
            print(f"Restricted course donors to editions {donor_years}")
        print(f"{event_name}: editions={event_keys}")
        tour_for_event = tour_train
        if tour_train is not None:
            # A genuine Tour-only comparator cannot contain prior editions of
            # the target event/course whose incremental value is being tested.
            tour_for_event = tour_train[
                ~tour_train["event_key"].isin(set(event_keys.values()))
            ].copy()
        if args.smoke_test:
            holdout_raw = holdout_raw[
                holdout_raw["hole_no"].isin([1, 2])
            ].head(160).copy()
        elif args.max_holdout_per_event > 0 and len(holdout_raw) > args.max_holdout_per_event:
            # Preserve round/hole coverage while selecting deterministic rows.
            group_count = holdout_raw.groupby(["round_no", "hole_no"]).ngroups
            per_group = max(1, math.ceil(args.max_holdout_per_event / group_count))
            sampled = holdout_raw.assign(
                _sample_hash=pd.util.hash_pandas_object(
                    holdout_raw["shot_key"], index=False
                ).to_numpy(np.uint64)
            ).sort_values("_sample_hash")
            sampled = sampled.groupby(
                ["round_no", "hole_no"], observed=True, group_keys=False
            ).head(per_group)
            holdout_raw = sampled.sort_values("_sample_hash").head(
                args.max_holdout_per_event
            ).drop(columns="_sample_hash").copy()
            print(f"Bounded holdout to {len(holdout_raw):,} round-hole-balanced states")
        for mode_index, mode in enumerate(modes):
            course_train, holdout = setup_mode_panel(train_raw, holdout_raw, mode)
            scores, metrics = score_holdout(
                course_train,
                holdout,
                profiles,
                tour_for_event,
                mode,
                seed=args.seed + event_index * 100 + mode_index,
                max_lie_queries=args.max_lie_queries,
            )
            suffix = f"{_slug(event_name)}_{mode}" + ("_smoke" if args.smoke_test else "")
            score_path = args.output_dir / f"app_transition_scores_{suffix}.csv"
            metric_path = args.output_dir / f"app_transition_metrics_{suffix}.csv"
            report_path = args.output_dir / f"APP_TRANSITION_REPORT_{suffix}.md"
            scores.to_csv(score_path, index=False)
            metrics.to_csv(metric_path, index=False)
            report_path.write_text(
                event_report(
                    event_name, mode, course_train, holdout, scores, metrics,
                    globals_, args.smoke_test,
                ),
                encoding="utf-8",
            )
            print(metrics.to_string(index=False))
            print(f"Wrote {report_path}")
    if not args.smoke_test and len(events) > 1:
        pooled, report_path = combined_report(args.output_dir)
        print(pooled.to_string(index=False))
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

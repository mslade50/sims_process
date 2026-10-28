"""Archive-only normalized TEE/APP atom precompute and raw parity harness.

The existing full chain aligns raw recurring-course donors before sampling.
This research module inserts the leakage-safe normalization boundary earlier:

    donor-native as-of profile/materialization
      -> beta-zero neutralization
      -> exactly one donor-to-latest-prior rigid alignment
      -> target-player reapplication on a query support
      -> stochastic endpoint surface classification in the sampler

APP caches contain pin-relative miss vectors, not fabricated absolute
endpoints.  A generated start point/direction is particle-state-dependent, so
the sampler rotates a selected miss vector into the physical setup and only
then queries lie/hazard probabilities.  TEE uses the same stochastic surface
step to preserve the current model exactly.

The default CLI gate is intentionally small: Travelers 2026, one frozen
pre-event player, round one, 18 holes, 128 particles.  It requires raw cached
support to reproduce selected atom IDs, initializer state and terminal hole
scores exactly.  The normalized arm is diagnostic cache construction only.
Nothing here is imported by production.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping
import zlib

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_app_transition as app  # noqa: E402
import hole_sim_asof_spatial_profiles as asof  # noqa: E402
import hole_sim_field_normalization as norm  # noqa: E402
import hole_sim_full_chain as full  # noqa: E402
import hole_sim_hazard_resolution as hazard  # noqa: E402
import hole_sim_normalized_chain_hook as contract  # noqa: E402
import hole_sim_state_foundations as foundations  # noqa: E402
import hole_sim_tail_policy as tail_policy  # noqa: E402
import hole_sim_tee_app_integrated_v2 as spatial  # noqa: E402
import hole_sim_tee_transition as tee  # noqa: E402


DEFAULT_EXPORT = app.DEFAULT_EXPORT
DEFAULT_PROFILES = REPO / "permanent_data" / "hole_sim_asof_spatial_profiles_v2"
DEFAULT_BASELINES = full.DEFAULT_BASELINES
DEFAULT_OUTPUT = (
    REPO
    / "permanent_data"
    / "hole_sim_normalized_atom_precompute_travelers_smoke_v2_unbounded_score_tail"
)
# ``REFERENCE_PRECOMPUTE_VERSION`` names the immutable 128-particle artifact
# already on disk.  New precomputes use a new version because their terminal
# parity path is governed by the shared unbounded-score tail policy.
REFERENCE_PRECOMPUTE_VERSION = "normalized_atom_precompute_v1"
PRECOMPUTE_VERSION = "normalized_atom_precompute_v2_unbounded_score_tail"
DEFERRED_CLASSIFIER_VERSION = "shot_relative_surface_deferred_v1"
DEFAULT_EVENT = "Travelers Championship"
DEFAULT_HOLDOUT = 2026
DEFAULT_MODE = "player_state_residualized"
NATIVE_FRAME_VERSION = "donor_native_actual_setup_local_v1"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = list(dict.fromkeys(columns))
    work = frame[selected].copy()
    schema = [(column, str(work[column].dtype)) for column in selected]
    hashes = pd.util.hash_pandas_object(work, index=False).to_numpy("uint64")
    hashes.sort()
    digest = sha256(json.dumps(schema, separators=(",", ":")).encode())
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _event_course_id(
    export_root: Path,
    event_keys: Mapping[int, str],
    reference_year: int,
) -> str:
    geometry = tee._read_year_geometry(
        export_root, int(reference_year), str(event_keys[int(reference_year)])
    )
    # Raw course_id can vary by setup/layout inside a single physical venue.
    # The foundation's rigid-normalized geometry fingerprint is the audited
    # recurring-course identity used by the hazard resolver and setup keys.
    return str(foundations.physical_course_fingerprint(geometry))


def _event_dates(
    history_db: Path,
    export_root: Path,
    event_keys: Mapping[int, str],
) -> pd.DataFrame:
    calendar = asof.load_event_calendar(
        history_db, export_root / "events.parquet"
    )
    selected = calendar[calendar["event_key"].isin(set(event_keys.values()))].copy()
    missing = sorted(set(event_keys.values()) - set(selected["event_key"]))
    if missing:
        raise ValueError(f"Calendar misses verified event keys: {missing}")
    if selected.duplicated("event_key").any():
        raise ValueError("Calendar has duplicate event keys")
    return selected


def _read_sg_reward(
    export_root: Path,
    event_keys: Mapping[int, str],
    holdout_year: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for season, event_key in sorted(event_keys.items()):
        if int(season) >= int(holdout_year):
            continue
        frame = pd.read_parquet(
            export_root / f"pga_{int(season)}_shot_sg.parquet",
            columns=["event_key", "shot_key", "sg_raw", "sg_valid"],
            filters=[("event_key", "==", str(event_key))],
        )
        parts.append(frame)
    reward = pd.concat(parts, ignore_index=True)
    reward = reward[reward["sg_valid"].fillna(False) & reward["sg_raw"].notna()].copy()
    reward = reward.drop_duplicates("shot_key")
    return reward[["shot_key", "sg_raw"]]


def _load_native_course_atoms(
    export_root: Path,
    event_keys: Mapping[int, str],
    holdout_year: int,
    course_start_year: int,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tee_parts: list[pd.DataFrame] = []
    for season, event_key in sorted(event_keys.items()):
        if int(season) < int(course_start_year) or int(season) >= int(holdout_year):
            continue
        geometry = tee._read_year_geometry(export_root, int(season), str(event_key))
        shots = tee._read_drives(export_root, int(season), str(event_key))
        tee_parts.append(tee._add_local_coordinates(shots, geometry))
    if not tee_parts:
        raise ValueError("No native TEE donors")
    native_tee = pd.concat(tee_parts, ignore_index=True)
    native_app_raw = spatial._load_verified_app_course_raw(
        export_root, dict(event_keys), holdout_year, course_start_year
    )
    native_app = app._apply_reference_pins(native_app_raw)
    reward = _read_sg_reward(export_root, event_keys, holdout_year)
    dates = calendar[["event_key", "event_start", "event_end"]]
    native_tee = native_tee.merge(
        dates, on="event_key", how="left", validate="many_to_one"
    ).merge(reward, on="shot_key", how="left", validate="one_to_one")
    native_app = native_app.merge(
        dates, on="event_key", how="left", validate="many_to_one"
    ).merge(reward, on="shot_key", how="left", validate="one_to_one")
    tee_penalties = spatial._load_prior_tee_penalty_labels(
        export_root, dict(event_keys), holdout_year
    )
    native_tee = native_tee.merge(
        tee_penalties[["shot_key", "tee_penalty_count"]],
        on="shot_key",
        how="left",
        validate="one_to_one",
    )
    native_tee["tee_penalty_count"] = (
        native_tee["tee_penalty_count"].fillna(0).astype(int)
    )
    native_tee["tee_causal_penalty"] = native_tee["tee_penalty_count"].gt(0)
    native_tee.loc[native_tee["tee_causal_penalty"], "lie_class"] = "other"
    for label, frame in (("TEE", native_tee), ("APP", native_app)):
        if frame[["event_start", "event_end"]].isna().any().any():
            raise ValueError(f"{label} donor dates are incomplete")
        if frame["shot_key"].duplicated().any():
            raise ValueError(f"{label} shot_key is not unique")
    # Fixed-reference reward is required by the generic core even at beta zero.
    # Missing reward rows are explicitly zero and ineligible for any future
    # quality arm; beta-zero spatial transforms do not consume the value.
    native_tee["reward_available"] = native_tee["sg_raw"].notna()
    native_app["reward_available"] = native_app["sg_raw"].notna()
    native_tee["sg_raw"] = native_tee["sg_raw"].fillna(0.0)
    native_app["sg_raw"] = native_app["sg_raw"].fillna(0.0)
    native_tee["native_row_id"] = np.arange(len(native_tee), dtype=np.int64)
    native_app["native_row_id"] = np.arange(len(native_app), dtype=np.int64)
    native_tee["native_coordinate_frame"] = NATIVE_FRAME_VERSION
    native_app["native_coordinate_frame"] = NATIVE_FRAME_VERSION
    return native_tee, native_app


def _tee_reference_pool(
    prior: pd.DataFrame,
    record: object,
) -> tuple[pd.DataFrame, str]:
    masks = (
        (
            prior["round_no"].eq(record.round_no)
            & prior["hole_no"].eq(record.hole_no)
            & prior["par"].eq(record.par),
            "prior_course_round_hole_par",
        ),
        (
            prior["hole_no"].eq(record.hole_no) & prior["par"].eq(record.par),
            "prior_course_hole_par",
        ),
        (prior["par"].eq(record.par), "prior_course_par"),
        (pd.Series(True, index=prior.index), "prior_course_all_tee"),
    )
    for mask, source in masks:
        pool = prior.loc[mask]
        if len(pool) >= 12:
            return pool, source
    return prior.iloc[0:0], "predeclared_tour_state_prior"


def _app_reference_pool(
    prior: pd.DataFrame,
    record: object,
) -> tuple[pd.DataFrame, str]:
    lie = str(record.start_lie_class)
    band = str(record.profile_distance_band)
    masks = (
        (
            prior["round_no"].eq(record.round_no)
            & prior["hole_no"].eq(record.hole_no)
            & prior["start_lie_class"].eq(lie)
            & prior["profile_distance_band"].eq(band),
            "prior_course_round_hole_lie_band",
        ),
        (
            prior["hole_no"].eq(record.hole_no)
            & prior["start_lie_class"].eq(lie)
            & prior["profile_distance_band"].eq(band),
            "prior_course_hole_lie_band",
        ),
        (
            prior["start_lie_class"].eq(lie)
            & prior["profile_distance_band"].eq(band),
            "prior_course_lie_band",
        ),
        (
            prior["profile_distance_band"].eq(band),
            "prior_course_band",
        ),
        (pd.Series(True, index=prior.index), "prior_course_all_app"),
    )
    for mask, source in masks:
        pool = prior.loc[mask]
        if len(pool) >= 12:
            return pool, source
    return prior.iloc[0:0], "predeclared_pin_relative_prior"


def _donor_conditions(
    donors: pd.DataFrame,
    *,
    category: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = donors.copy()
    if category == "ott":
        frame["sg_category"] = "ott"
        frame["state_key"] = frame["par"].map(lambda value: contract.tee_state_key(int(value)))
        frame["profile_distance_band"] = pd.NA
        frame["end_hazard"] = frame.get(
            "tee_causal_penalty", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        frame["end_lie"] = frame["lie_class"].astype(str)
        frame["long_for_reference"] = frame["long_yd"]
        frame["lat_for_reference"] = frame["lat_yd"]
    elif category == "app":
        frame["sg_category"] = "app"
        frame["profile_distance_band"] = frame["reference_start_distance_yd"].map(
            contract.app_distance_band
        )
        frame["state_key"] = [
            contract.app_state_key(lie, distance)
            for lie, distance in zip(
                frame["start_lie_class"], frame["reference_start_distance_yd"]
            )
        ]
        frame["end_hazard"] = frame["is_penalty"].fillna(False).astype(bool)
        frame["end_lie"] = frame["end_lie_class"].astype(str)
        frame["long_for_reference"] = frame["miss_long_yd"]
        frame["lat_for_reference"] = frame["miss_lat_yd"]
    else:
        raise ValueError(category)
    frame["source_player_id"] = frame["source_player_id"].astype(str)
    frame["setup_key"] = [
        contract.setup_key(event, course, round_no, hole_no)
        for event, course, round_no, hole_no in zip(
            frame["event_key"], frame["course_id"], frame["round_no"], frame["hole_no"]
        )
    ]
    condition_cols = [
        "event_key", "sg_category", "state_key", "setup_key",
        "round_no", "hole_no", "par", "event_start",
    ]
    if category == "app":
        condition_cols += ["start_lie_class", "profile_distance_band"]
    unique = frame[condition_cols].drop_duplicates().sort_values(
        ["event_start", "setup_key", "state_key"]
    )
    rows: list[dict[str, object]] = []
    for record in unique.itertuples(index=False):
        prior = frame[frame["event_end"] < pd.Timestamp(record.event_start)]
        if category == "ott":
            pool, source = _tee_reference_pool(prior, record)
            if pool.empty:
                center_long = 275.0 if int(record.par) == 4 else 290.0
                center_lat = 0.0
            else:
                center_long = float(pool["long_for_reference"].mean())
                center_lat = float(pool["lat_for_reference"].mean())
        else:
            pool, source = _app_reference_pool(prior, record)
            if pool.empty:
                center_long = 0.0
                center_lat = 0.0
            else:
                center_long = float(pool["long_for_reference"].mean())
                center_lat = float(pool["lat_for_reference"].mean())
        cutoff = (
            asof.PRIOR_ORIGIN
            if pool.empty
            else pd.Timestamp(pool["event_end"].max())
        )
        if not cutoff < pd.Timestamp(record.event_start):
            raise AssertionError("Donor reference cutoff is not strictly prior")
        rows.append(
            {
                "event_key": record.event_key,
                "sg_category": category,
                "state_key": record.state_key,
                "setup_key": record.setup_key,
                "reference_long_yd": center_long,
                "reference_lat_yd": center_lat,
                "reference_cutoff": cutoff,
                "reference_source": source,
                "reference_pool_rows": int(len(pool)),
            }
        )
    conditions = pd.DataFrame(rows)
    return frame, conditions


def _native_tee_model_endpoint(frame: pd.DataFrame) -> np.ndarray:
    vx = frame["pin_x"].to_numpy(float) - frame["start_x"].to_numpy(float)
    vy = frame["pin_y"].to_numpy(float) - frame["start_y"].to_numpy(float)
    length = np.hypot(vx, vy)
    ux, uy = vx / length, vy / length
    long = frame["model_long_yd"].to_numpy(float)
    lat = frame["model_lat_yd"].to_numpy(float)
    x = frame["start_x"].to_numpy(float) + 3.0 * (long * ux - lat * uy)
    y = frame["start_y"].to_numpy(float) + 3.0 * (long * uy + lat * ux)
    return np.column_stack([x, y])


def _native_app_model_endpoint(frame: pd.DataFrame) -> np.ndarray:
    vx = frame["pin_x"].to_numpy(float) - frame["start_x"].to_numpy(float)
    vy = frame["pin_y"].to_numpy(float) - frame["start_y"].to_numpy(float)
    length = np.hypot(vx, vy)
    ux, uy = vx / length, vy / length
    long = frame["model_long_yd"].to_numpy(float)
    lat = frame["model_lat_yd"].to_numpy(float)
    x = frame["pin_x"].to_numpy(float) + 3.0 * (long * ux - lat * uy)
    y = frame["pin_y"].to_numpy(float) + 3.0 * (long * uy + lat * ux)
    return np.column_stack([x, y])


def _deferred_surface_classifier(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit shot-relative contract; no physical label is fabricated."""

    out = frame.copy()
    out["model_end_lie"] = "deferred_runtime_surface_probability"
    out["model_end_hazard"] = pd.NA
    return out


@dataclass(frozen=True)
class SupportArrays:
    key: str
    atom_ids: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    cdf: np.ndarray
    source: str
    n_setup: int
    n_hole: int
    n_broad: int
    holeout_probability: float = 0.0
    cache_sha256: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        key: str,
        atom_ids: np.ndarray,
        points: np.ndarray,
        weights: np.ndarray,
        source: str,
        n_setup: int,
        n_hole: int,
        n_broad: int,
        holeout_probability: float = 0.0,
        metadata: Mapping[str, object] | None = None,
    ) -> "SupportArrays":
        ids = np.ascontiguousarray(np.asarray(atom_ids).astype(str))
        xy = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        probability = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
        if xy.shape != (len(ids), 2) or probability.shape != (len(ids),):
            raise ValueError("Support array shapes disagree")
        if not len(ids) or not np.isfinite(xy).all() or not np.isfinite(probability).all():
            raise ValueError("Support arrays are empty/non-finite")
        if (probability < 0.0).any() or probability.sum() <= 0.0:
            raise ValueError("Support weights are invalid")
        probability = np.ascontiguousarray(probability / probability.sum())
        cumulative = np.ascontiguousarray(np.cumsum(probability))
        cumulative[-1] = 1.0
        digest = sha256()
        digest.update(str(key).encode())
        digest.update("\0".join(ids.tolist()).encode())
        digest.update(xy.tobytes())
        digest.update(probability.tobytes())
        digest.update(str(float(holeout_probability)).encode())
        return cls(
            key=str(key),
            atom_ids=ids,
            points=xy,
            weights=probability,
            cdf=cumulative,
            source=str(source),
            n_setup=int(n_setup),
            n_hole=int(n_hole),
            n_broad=int(n_broad),
            holeout_probability=float(holeout_probability),
            cache_sha256=digest.hexdigest(),
            metadata=dict(metadata or {}),
        )

    def choose(self, uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(uniforms, dtype=float)
        indices = np.searchsorted(self.cdf, values, side="right")
        indices = np.minimum(indices, len(self.atom_ids) - 1)
        return indices, self.atom_ids[indices]


def _tee_distribution_rows(
    course: pd.DataFrame,
    round_no: int,
    hole_no: int,
    par: int,
    setup_prior: float = 160.0,
    hole_prior: float = 500.0,
) -> tuple[pd.DataFrame, np.ndarray, tee.Distribution]:
    course = course.reset_index(drop=True)
    setup = course[course["round_no"].eq(round_no) & course["hole_no"].eq(hole_no)]
    hole = course[course["hole_no"].eq(hole_no)]
    par_pool = course[course["par"].eq(par)]
    if par_pool.empty:
        par_pool = course
    alpha = len(setup) / (len(setup) + setup_prior)
    beta = len(hole) / (len(hole) + hole_prior)
    mass = pd.Series(dtype=float)

    def add_component(frame: pd.DataFrame, total_weight: float) -> None:
        nonlocal mass
        if frame.empty or total_weight <= 0.0:
            return
        component = pd.Series(total_weight / len(frame), index=frame.index)
        mass = mass.add(component, fill_value=0.0)

    add_component(setup, alpha)
    remainder = 1.0 - alpha
    add_component(hole, remainder * beta)
    add_component(par_pool, remainder * (1.0 - beta))
    rows = course.loc[mass.index].copy().reset_index(drop=True)
    weights = mass.to_numpy(float)
    weights /= weights.sum()
    reference = tee.build_distribution(course, round_no, hole_no, par)
    if not np.array_equal(rows[["long_yd", "lat_yd"]].to_numpy(float), reference.points):
        raise AssertionError("TEE row support does not reproduce current distribution")
    if not np.allclose(weights, reference.weights, atol=0.0, rtol=0.0):
        raise AssertionError("TEE weights do not reproduce current distribution")
    return rows, weights, reference


@dataclass
class NormalizedEventAtoms:
    """Exactly-once aligned atom panels and their strict-prior profiles."""

    prepared: dict[str, object]
    calendar: pd.DataFrame
    tee_profiles: pd.DataFrame
    app_profiles: pd.DataFrame
    native_tee: pd.DataFrame
    native_app: pd.DataFrame
    neutral_tee: pd.DataFrame
    neutral_app: pd.DataFrame
    normalization_metadata: dict[str, Mapping[str, object]]
    profile_manifest: dict[str, object]
    event_name: str
    holdout_year: int
    target_event_key: str
    target_event_start: pd.Timestamp
    physical_course_id: str
    timings: dict[str, float]

    @property
    def raw_course_drives(self) -> pd.DataFrame:
        return self.prepared["course_drives"]  # type: ignore[return-value]

    @property
    def raw_course_app(self) -> pd.DataFrame:
        return self.prepared["course_app"]  # type: ignore[return-value]

    @property
    def forecast(self) -> pd.DataFrame:
        return self.prepared["forecast"]  # type: ignore[return-value]


def _load_profile_bundle(
    profile_dir: Path,
    event_keys: Mapping[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest_path = profile_dir / "manifest.json"
    tee_path = profile_dir / "tee_player_spatial_profiles_asof.parquet"
    app_path = profile_dir / "app_player_spatial_profiles_asof.parquet"
    for path in (manifest_path, tee_path, app_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != asof.PROFILE_VERSION:
        raise ValueError(f"Unexpected profile contract: {manifest.get('contract')!r}")
    if manifest.get("snapshot_kind") != "strict_prior_asof":
        raise ValueError("Profile snapshot is not strict_prior_asof")
    if int(manifest.get("strict_cutoff_violations", -1)) != 0:
        raise ValueError("Profile manifest reports cutoff violations")
    if _file_sha256(tee_path) != manifest.get("tee_artifact_sha256"):
        raise ValueError("TEE profile file hash disagrees with manifest")
    if _file_sha256(app_path) != manifest.get("app_artifact_sha256"):
        raise ValueError("APP profile file hash disagrees with manifest")
    tee_profiles = pd.read_parquet(tee_path)
    app_profiles = pd.read_parquet(app_path)
    if asof._frame_sha256(tee_profiles) != manifest.get("tee_profiles_sha256"):
        raise ValueError("TEE profile logical hash disagrees with manifest")
    if asof._frame_sha256(app_profiles) != manifest.get("app_profiles_sha256"):
        raise ValueError("APP profile logical hash disagrees with manifest")
    admitted = set(map(str, event_keys.values()))
    tee_profiles = tee_profiles[tee_profiles["event_key"].astype(str).isin(admitted)].copy()
    app_profiles = app_profiles[app_profiles["event_key"].astype(str).isin(admitted)].copy()
    for label, frame in (("TEE", tee_profiles), ("APP", app_profiles)):
        frame["source_player_id"] = frame["source_player_id"].astype(str)
        frame["event_start"] = pd.to_datetime(frame["event_start"], utc=True)
        frame["profile_cutoff"] = pd.to_datetime(frame["profile_cutoff"], utc=True)
        if frame.empty or not (frame["profile_cutoff"] < frame["event_start"]).all():
            raise ValueError(f"{label} filtered profile snapshot violates cutoff")
    return tee_profiles, app_profiles, manifest


def _normalization_input(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "native_row_id", "event_key", "event_start", "source_player_id",
        "sg_category", "state_key", "setup_key", "long_for_reference",
        "lat_for_reference", "end_lie", "end_hazard", "sg_raw",
    ]
    out = frame[columns].rename(
        columns={"long_for_reference": "long_yd", "lat_for_reference": "lat_yd"}
    ).copy()
    out["source_player_id"] = out["source_player_id"].astype(str)
    return out


def _merge_native_geometry(
    normalized: pd.DataFrame,
    native: pd.DataFrame,
) -> pd.DataFrame:
    profile_output = set(normalized.columns)
    native_columns = [
        column for column in native.columns
        if column == "native_row_id" or column not in profile_output
    ]
    return normalized.merge(
        native[native_columns],
        on="native_row_id",
        how="left",
        validate="one_to_one",
    )


def _align_normalized_tee(
    normalized: pd.DataFrame,
    native: pd.DataFrame,
    forecast: pd.DataFrame,
    export_root: Path,
) -> pd.DataFrame:
    work = _merge_native_geometry(normalized, native)
    endpoints = _native_tee_model_endpoint(work)
    work["end_x"] = endpoints[:, 0]
    work["end_y"] = endpoints[:, 1]
    aligned = spatial._align_drives_to_forecast(work, forecast, export_root)
    aligned["atom_coordinate_role"] = "neutral_endpoint_aligned_once"
    aligned["alignment_count"] = 1
    return aligned


def _align_normalized_app(
    normalized: pd.DataFrame,
    native: pd.DataFrame,
    forecast: pd.DataFrame,
    event_keys: Mapping[int, str],
    export_root: Path,
) -> pd.DataFrame:
    work = _merge_native_geometry(normalized, native)
    endpoints = _native_app_model_endpoint(work)
    work["end_x"] = endpoints[:, 0]
    work["end_y"] = endpoints[:, 1]
    # Mirror v2's exactly-once rigid transform but deliberately do not apply
    # its raw-data plausibility filter after moving a normalized atom. The
    # donor universe was already restricted to legacy-accepted spatial rows;
    # silently dropping a transformed tail atom would change hierarchy mass.
    parts: list[pd.DataFrame] = []
    for (season, event_key), event in work.groupby(
        ["season", "event_key"], observed=True, sort=True
    ):
        if str(event_key) != str(event_keys[int(season)]):
            raise ValueError("APP atom event key disagrees with verified edition")
        geometry = tee._read_year_geometry(export_root, int(season), str(event_key))
        rigid = spatial.v1._fit_rigid_geometry(geometry, forecast)
        event = event.copy()
        prefixes = ["start", "end"]
        for optional in ("resolution_start", "drop_end"):
            if {f"{optional}_x", f"{optional}_y"}.issubset(event.columns):
                prefixes.append(optional)
        for prefix in prefixes:
            moved = spatial.v1._apply_rigid(
                event[f"{prefix}_x"], event[f"{prefix}_y"], rigid
            )
            event[f"{prefix}_x"] = moved[:, 0]
            event[f"{prefix}_y"] = moved[:, 1]
        event["geometry_alignment_rmse_yd"] = rigid[3]
        parts.append(event)
    aligned = pd.concat(parts, ignore_index=True)
    pin_table = forecast[
        ["round_no", "hole_no", "forecast_pin_x", "forecast_pin_y"]
    ].rename(columns={"forecast_pin_x": "pin_x", "forecast_pin_y": "pin_y"})
    aligned = app._apply_reference_pins(aligned, pin_table)
    aligned = aligned.merge(
        forecast[
            ["round_no", "hole_no", "forecast_tee_x", "forecast_tee_y"]
        ],
        on=["round_no", "hole_no"],
        how="inner",
        validate="many_to_one",
    )
    dx = aligned["end_x"].to_numpy(float) - aligned["forecast_tee_x"].to_numpy(float)
    dy = aligned["end_y"].to_numpy(float) - aligned["forecast_tee_y"].to_numpy(float)
    vx = aligned["pin_x"].to_numpy(float) - aligned["forecast_tee_x"].to_numpy(float)
    vy = aligned["pin_y"].to_numpy(float) - aligned["forecast_tee_y"].to_numpy(float)
    length = np.hypot(vx, vy)
    ux, uy = vx / length, vy / length
    aligned["common_end_long_yd"] = (dx * ux + dy * uy) / 3.0
    aligned["common_end_lat_yd"] = (-dx * uy + dy * ux) / 3.0
    aligned["coordinate_frame_version"] = spatial.COORDINATE_FRAME_VERSION
    aligned["coordinate_frame_reference_year"] = int(
        forecast["setup_reference_year"].iloc[0]
    )
    aligned["source_coordinate_provenance"] = "raw_shotlink_single_alignment"
    aligned["atom_coordinate_role"] = "neutral_miss_endpoint_aligned_once"
    aligned["alignment_count"] = 1
    return aligned


def _assert_raw_alignment_equivalence(
    raw_tee: pd.DataFrame,
    raw_app: pd.DataFrame,
    prepared: Mapping[str, object],
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    reference_tee = prepared["course_drives"]
    tee_join = raw_tee[["shot_key", "long_yd", "lat_yd"]].merge(
        reference_tee[["shot_key", "long_yd", "lat_yd"]],
        on="shot_key",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_candidate", "_reference"),
    )
    if not tee_join["_merge"].eq("both").all():
        raise AssertionError("Raw TEE normalizer changed aligned donor membership")
    tee_error = np.abs(
        tee_join[["long_yd_candidate", "lat_yd_candidate"]].to_numpy(float)
        - tee_join[["long_yd_reference", "lat_yd_reference"]].to_numpy(float)
    )
    if float(tee_error.max(initial=0.0)) > 1e-10:
        raise AssertionError("Raw TEE normalizer changed the aligned endpoint")
    reference_app = prepared["course_app"]
    reference_app = reference_app[
        reference_app["spatial_valid"].fillna(False)
        & ~reference_app["is_holeout"].fillna(False)
    ]
    app_join = raw_app[
        ["shot_key", "miss_long_yd", "miss_lat_yd", "common_end_long_yd", "common_end_lat_yd"]
    ].merge(
        reference_app[
            ["shot_key", "miss_long_yd", "miss_lat_yd", "common_end_long_yd", "common_end_lat_yd"]
        ],
        on="shot_key",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_candidate", "_reference"),
    )
    if not app_join["_merge"].eq("both").all():
        counts = app_join["_merge"].astype(str).value_counts().to_dict()
        examples = app_join.loc[
            ~app_join["_merge"].eq("both"), ["shot_key", "_merge"]
        ].head(8).to_dict("records")
        raise AssertionError(
            "Raw APP normalizer changed aligned donor membership: "
            f"counts={counts}, examples={examples}"
        )
    app_candidate = app_join[
        [
            "miss_long_yd_candidate", "miss_lat_yd_candidate",
            "common_end_long_yd_candidate", "common_end_lat_yd_candidate",
        ]
    ].to_numpy(float)
    app_reference = app_join[
        [
            "miss_long_yd_reference", "miss_lat_yd_reference",
            "common_end_long_yd_reference", "common_end_lat_yd_reference",
        ]
    ].to_numpy(float)
    app_error = np.abs(app_candidate - app_reference)
    if float(app_error.max(initial=0.0)) > 1e-10:
        raise AssertionError("Raw APP normalizer changed the aligned endpoint")
    comparisons.update(
        {
            "tee_rows": int(len(tee_join)),
            "app_spatial_rows": int(len(app_join)),
            "tee_max_abs_endpoint_error_yd": float(tee_error.max(initial=0.0)),
            "app_max_abs_endpoint_error_yd": float(app_error.max(initial=0.0)),
        }
    )
    return comparisons


def precompute_event_atoms(
    *,
    export_root: Path,
    history_db: Path,
    profile_dir: Path,
    event_name: str,
    holdout_year: int,
    course_start_year: int,
    mode: str = DEFAULT_MODE,
) -> NormalizedEventAtoms:
    """Materialize donor-native neutral atoms and align each exactly once."""

    started = time.perf_counter()
    prepared = spatial.prepare_event_foundations(
        export_root,
        event_name,
        holdout_year,
        course_start_year,
        include_scoring_targets=False,
    )
    provenance = prepared["provenance"]
    if int(provenance.get("alignment_count", -1)) != 1:
        raise ValueError("Prepared raw course inputs are not exactly-once aligned")
    event_keys = prepared["event_keys"]
    target_event_key = str(event_keys[int(holdout_year)])
    calendar = _event_dates(history_db, export_root, event_keys)
    target_dates = calendar[calendar["event_key"].eq(target_event_key)]
    if len(target_dates) != 1:
        raise ValueError("Target event date is absent or duplicated")
    target_event_start = pd.Timestamp(target_dates.iloc[0]["event_start"])
    physical_course_id = _event_course_id(
        export_root, event_keys, int(provenance["setup_reference_year"])
    )
    tee_profiles, app_profiles, profile_manifest = _load_profile_bundle(
        profile_dir, event_keys
    )
    loaded_at = time.perf_counter()
    native_tee, native_app_all = _load_native_course_atoms(
        export_root, event_keys, holdout_year, course_start_year, calendar
    )
    native_tee, tee_conditions = _donor_conditions(native_tee, category="ott")
    app_numeric = native_app_all[
        ["reference_start_distance_yd", "miss_long_yd", "miss_lat_yd"]
    ].apply(pd.to_numeric, errors="coerce")
    app_normalizable = (
        native_app_all["spatial_valid"].fillna(False)
        & ~native_app_all["is_holeout"].fillna(False)
        & np.isfinite(app_numeric).all(axis=1)
        & app_numeric["reference_start_distance_yd"].ge(0.0)
    )
    legacy_spatial = prepared["course_app"]
    legacy_spatial = legacy_spatial[
        legacy_spatial["spatial_valid"].fillna(False)
        & ~legacy_spatial["is_holeout"].fillna(False)
    ]
    legacy_spatial_ids = set(legacy_spatial["shot_key"].astype(str))
    accepted_alignment = native_app_all["shot_key"].astype(str).isin(
        legacy_spatial_ids
    )
    native_app = native_app_all.loc[app_normalizable & accepted_alignment].copy()
    app_exclusions = {
        "native_app_rows": int(len(native_app_all)),
        "legacy_spatial_support_rows": int(len(legacy_spatial_ids)),
        "excluded_nonspatial_holeout_or_nonfinite": int((~app_normalizable).sum()),
        "excluded_by_legacy_alignment_bounds_after_spatial_valid": int(
            (app_normalizable & ~accepted_alignment).sum()
        ),
        "normalized_spatial_rows": int(len(native_app)),
        "policy": "normalize_exact_legacy_spatial_support; holeout_mass_accounted_separately",
    }
    native_app, app_conditions = _donor_conditions(native_app, category="app")
    app_spatial = native_app
    tee_materialized = asof.materialize_normalization_profiles(
        tee_profiles, tee_conditions
    )
    app_materialized = asof.materialize_normalization_profiles(
        app_profiles, app_conditions
    )
    config = norm.NormalizationConfig(ability_log_weight_per_sg2=0.0)
    raw_tee_native, raw_tee_meta = norm.normalize_donor_points(
        _normalization_input(native_tee), tee_materialized, mode="raw", config=config
    )
    raw_app_native, raw_app_meta = norm.normalize_donor_points(
        _normalization_input(app_spatial), app_materialized, mode="raw", config=config
    )
    neutral_tee_native, tee_meta = norm.normalize_donor_points(
        _normalization_input(native_tee), tee_materialized, mode=mode, config=config
    )
    neutral_app_native, app_meta = norm.normalize_donor_points(
        _normalization_input(app_spatial), app_materialized, mode=mode, config=config
    )
    normalized_at = time.perf_counter()
    forecast = prepared["forecast"]
    raw_tee_aligned = _align_normalized_tee(
        raw_tee_native, native_tee, forecast, export_root
    )
    raw_app_aligned = _align_normalized_app(
        raw_app_native, app_spatial, forecast, event_keys, export_root
    )
    neutral_tee = _align_normalized_tee(
        neutral_tee_native, native_tee, forecast, export_root
    )
    neutral_app = _align_normalized_app(
        neutral_app_native, app_spatial, forecast, event_keys, export_root
    )
    raw_equivalence = _assert_raw_alignment_equivalence(
        raw_tee_aligned, raw_app_aligned, prepared
    )
    aligned_at = time.perf_counter()
    metadata: dict[str, Mapping[str, object]] = {
        "raw_tee": raw_tee_meta,
        "raw_app": raw_app_meta,
        "neutral_tee": tee_meta,
        "neutral_app": app_meta,
        "raw_alignment_equivalence": raw_equivalence,
        "app_normalization_exclusions": app_exclusions,
    }
    return NormalizedEventAtoms(
        prepared=prepared,
        calendar=calendar,
        tee_profiles=tee_profiles,
        app_profiles=app_profiles,
        native_tee=native_tee,
        native_app=native_app,
        neutral_tee=neutral_tee,
        neutral_app=neutral_app,
        normalization_metadata=metadata,
        profile_manifest=profile_manifest,
        event_name=event_name,
        holdout_year=int(holdout_year),
        target_event_key=target_event_key,
        target_event_start=target_event_start,
        physical_course_id=physical_course_id,
        timings={
            "prepare_and_profiles_seconds": loaded_at - started,
            "native_load_and_normalize_seconds": normalized_at - loaded_at,
            "align_and_raw_audit_seconds": aligned_at - normalized_at,
            "total_precompute_seconds": aligned_at - started,
        },
    )


class AtomSupportProvider:
    """Build contiguous TEE/APP supports for raw or normalized sampling.

    The hierarchy mass from the existing sampler remains the base measure.
    Core ``model_weight`` is only a mean-one multiplier.  Final probability is
    therefore ``base_mass * model_weight``, normalized once in
    :meth:`SupportArrays.create`.  At beta zero/raw this is audited against the
    existing support and CDF exactly.
    """

    def __init__(
        self,
        atoms: NormalizedEventAtoms,
        *,
        arm: str,
        cache_enabled: bool = True,
    ) -> None:
        if arm not in {"raw", "normalized"}:
            raise ValueError("arm must be raw or normalized")
        self.atoms = atoms
        self.arm = arm
        self.cache_enabled = bool(cache_enabled)
        self.tee_cache: dict[tuple[int, int, int, str], SupportArrays] = {}
        self.app_cache: dict[tuple[int, int, str, int, str], SupportArrays] = {}
        self.materialized_rows: list[pd.DataFrame] = []
        self.inventory: list[dict[str, object]] = []
        self.tee_surfaces = {
            int(hole): tee.SurfaceMap(history)
            for hole, history in atoms.raw_course_drives.groupby(
                "hole_no", observed=True
            )
        }
        self.app_surfaces = spatial._surface_in_common_frame(atoms.raw_course_app)
        self.target_tee_profiles = atoms.tee_profiles[
            atoms.tee_profiles["event_key"].astype(str).eq(atoms.target_event_key)
        ].copy()
        self.target_app_profiles = atoms.app_profiles[
            atoms.app_profiles["event_key"].astype(str).eq(atoms.target_event_key)
        ].copy()
        if self.target_tee_profiles.empty or self.target_app_profiles.empty:
            raise ValueError("Target event profiles are missing")
        self.neutral_tee_by_id = atoms.neutral_tee.set_index("shot_key", drop=False)
        self.neutral_app_by_id = atoms.neutral_app.set_index("shot_key", drop=False)

    def _target_condition(
        self,
        *,
        category: str,
        state_key: str,
        round_no: int,
        hole_no: int,
        center: np.ndarray,
        support_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        cutoff = pd.to_datetime(support_rows["event_end"], utc=True).max()
        if not cutoff < self.atoms.target_event_start:
            raise ValueError("Target support reference is not strictly prior")
        return pd.DataFrame(
            {
                "event_key": [self.atoms.target_event_key],
                "sg_category": [category],
                "state_key": [state_key],
                "setup_key": [
                    contract.setup_key(
                        self.atoms.target_event_key,
                        self.atoms.physical_course_id,
                        int(round_no),
                        int(hole_no),
                    )
                ],
                "reference_long_yd": [float(center[0])],
                "reference_lat_yd": [float(center[1])],
                "reference_cutoff": [cutoff],
            }
        )

    @staticmethod
    def _retag_neutral_support(
        rows: pd.DataFrame,
        points: np.ndarray,
        center: np.ndarray,
        condition: pd.DataFrame,
    ) -> pd.DataFrame:
        out = rows.copy().reset_index(drop=True)
        for column in ("sg_category", "state_key", "setup_key"):
            out[column] = condition.iloc[0][column]
        out["neutral_resid_long_yd"] = points[:, 0] - float(center[0])
        out["neutral_resid_lat_yd"] = points[:, 1] - float(center[1])
        if "neutral_log_weight" not in out:
            out["neutral_log_weight"] = 0.0
        return out

    def _tee_geometry_classifier(
        self,
        hole_no: int,
    ):
        surface = self.tee_surfaces[int(hole_no)]

        def classify(frame: pd.DataFrame) -> pd.DataFrame:
            points = frame[["model_long_yd", "model_lat_yd"]].to_numpy(float)
            probability = surface.probabilities(points)
            codes = np.argmax(probability, axis=1)
            lies = np.asarray(tee.LIE_CLASSES, object)[codes]
            out = frame.copy()
            out["model_end_lie"] = lies
            out["model_end_hazard"] = np.isin(lies, ["water", "other"])
            return out

        return classify

    def tee_support(
        self,
        *,
        round_no: int,
        hole_no: int,
        par: int,
        player_id: str,
    ) -> SupportArrays:
        key = (int(round_no), int(hole_no), int(par), str(player_id))
        if self.cache_enabled and key in self.tee_cache:
            return self.tee_cache[key]
        raw_rows, raw_mass, distribution = _tee_distribution_rows(
            self.atoms.raw_course_drives,
            int(round_no),
            int(hole_no),
            int(par),
        )
        atom_ids = raw_rows["shot_key"].astype(str).to_numpy()
        if self.arm == "raw":
            points = distribution.points
            final_mass = raw_mass
            metadata: dict[str, object] = {
                "mass_contract": "legacy_hierarchy_mass_exact",
                "model_weight_role": "unity_raw",
                "classifier_role": "runtime_stochastic_surface_only",
            }
        else:
            selected = self.neutral_tee_by_id.reindex(atom_ids)
            if selected["shot_key"].isna().any():
                raise ValueError("Normalized TEE panel misses raw hierarchy atoms")
            selected = selected.reset_index(drop=True)
            neutral_points = selected[["long_yd", "lat_yd"]].to_numpy(float)
            center = np.average(neutral_points, axis=0, weights=raw_mass)
            state = contract.tee_state_key(int(par))
            condition = self._target_condition(
                category="ott",
                state_key=state,
                round_no=round_no,
                hole_no=hole_no,
                center=center,
                support_rows=selected,
            )
            target_profiles = asof.materialize_normalization_profiles(
                self.target_tee_profiles, condition
            )
            target_atoms = self._retag_neutral_support(
                selected, neutral_points, center, condition
            )
            reapplied, target_meta = norm.reapply_target_profile(
                target_atoms,
                target_profiles,
                target_player_id=str(player_id),
                target_event_start=self.atoms.target_event_start,
                geometry_classifier=self._tee_geometry_classifier(int(hole_no)),
                geometry_classification_mode="physical",
                config=norm.NormalizationConfig(ability_log_weight_per_sg2=0.0),
            )
            # The scalar classifier is a coherence audit only. Sampling still
            # queries the complete stochastic surface at the moved endpoint.
            points = reapplied[["model_long_yd", "model_lat_yd"]].to_numpy(float)
            multipliers = reapplied["model_weight"].to_numpy(float)
            final_mass = raw_mass * multipliers
            reapplied["base_hierarchy_mass"] = raw_mass
            reapplied["final_unnormalized_mass"] = final_mass
            reapplied["support_category"] = "tee"
            self.materialized_rows.append(reapplied)
            metadata = {
                "mass_contract": "base_hierarchy_mass_times_model_weight",
                "model_weight_role": "mean_one_multiplier_not_probability",
                "target_reapplication_provenance_sha256": target_meta[
                    "provenance_sha256"
                ],
                "classifier_version": "tee_argmax_audit_runtime_stochastic_v1",
                "runtime_classifier": "tee_SurfaceMap_probability_then_shared_lie_uniform",
            }
        support = SupportArrays.create(
            key=f"tee|{self.arm}|R{round_no}|H{hole_no}|P{par}|{player_id}",
            atom_ids=atom_ids,
            points=points,
            weights=final_mass,
            source=distribution.source,
            n_setup=distribution.n_setup,
            n_hole=distribution.n_hole,
            n_broad=distribution.n_par,
            metadata=metadata,
        )
        if self.arm == "raw":
            expected_mass = np.asarray(distribution.weights, float)
            expected_mass = expected_mass / expected_mass.sum()
            expected_cdf = np.cumsum(expected_mass)
            expected_cdf[-1] = 1.0
            if not np.array_equal(support.weights, expected_mass):
                raise AssertionError("Raw TEE cached weights differ from legacy mass")
            if not np.array_equal(support.cdf, expected_cdf):
                raise AssertionError("Raw TEE cached CDF differs from legacy CDF")
        if self.cache_enabled:
            self.tee_cache[key] = support
        self.inventory.append(
            {
                "category": "tee", "arm": self.arm, "round_no": int(round_no),
                "hole_no": int(hole_no), "state_key": contract.tee_state_key(int(par)),
                "player_id": str(player_id), "atoms": len(support.atom_ids),
                "cache_sha256": support.cache_sha256,
            }
        )
        return support

    def app_support(
        self,
        *,
        round_no: int,
        hole_no: int,
        start_lie: str,
        distance_bucket: int,
        player_id: str,
    ) -> SupportArrays:
        return _app_support_impl(
            self,
            round_no=round_no,
            hole_no=hole_no,
            start_lie=start_lie,
            distance_bucket=distance_bucket,
            player_id=player_id,
        )


class CachedSpatialTeeSampler:
    """TEE sampler backed by contiguous support/CDF arrays."""

    def __init__(
        self,
        provider: AtomSupportProvider,
        *,
        enforce_empirical_class_support: bool = False,
    ) -> None:
        self.provider = provider
        self.surfaces = provider.tee_surfaces
        self.enforce_empirical_class_support = bool(enforce_empirical_class_support)
        self.selection_rows: list[dict[str, object]] = []

    def sample(
        self,
        *,
        round_no: int,
        hole_no: int,
        par: int,
        player_id: str,
        player_transport: bool,
        pin_long_yd: float,
        index_uniform: np.ndarray,
        lie_uniform: np.ndarray,
        active: np.ndarray,
        hazard_support_checker: spatial.HazardSupportChecker | None = None,
    ) -> dict[str, np.ndarray]:
        if self.provider.arm == "raw" and player_transport:
            raise ValueError("Raw cached parity supports population transport only")
        count = len(active)
        endpoints = np.full((count, 2), np.nan)
        remaining = np.full(count, np.nan)
        lies = np.full(count, "not_attempted", dtype=object)
        statuses = np.full(count, "tee_not_attempted", dtype=object)
        atom_ids = np.full(count, "not_attempted", dtype=object)
        positions = np.flatnonzero(np.asarray(active, bool))
        if not len(positions):
            return {
                "endpoints": endpoints, "remaining": remaining,
                "end_lies": lies, "status": statuses, "atom_ids": atom_ids,
            }
        support = self.provider.tee_support(
            round_no=round_no,
            hole_no=hole_no,
            par=par,
            player_id=str(player_id),
        )
        selected, selected_ids = support.choose(
            np.asarray(index_uniform, float)[positions]
        )
        group_endpoints = support.points[selected]
        surface = self.surfaces.get(int(hole_no))
        if surface is None:
            statuses[positions] = "tee_no_surface_support"
            return {
                "endpoints": endpoints, "remaining": remaining,
                "end_lies": lies, "status": statuses, "atom_ids": atom_ids,
            }
        probabilities = spatial._supported_surface_probabilities(
            surface.probabilities(group_endpoints),
            surface.codes,
            class_count=len(tee.LIE_CLASSES),
            enforce_empirical_class_support=self.enforce_empirical_class_support,
            endpoints=group_endpoints,
            hazard_class_indices=tuple(
                tee.LIE_CLASSES.index(name) for name in ("water", "other")
            ),
            hazard_support_checker=hazard_support_checker,
        )
        codes = spatial._categorical_from_uniform(
            probabilities, np.asarray(lie_uniform, float)[positions]
        )
        group_lies = np.asarray(tee.LIE_CLASSES, object)[codes]
        group_remaining = np.linalg.norm(
            np.asarray([float(pin_long_yd), 0.0])[None, :] - group_endpoints,
            axis=1,
        )
        group_status = np.where(
            group_lies == "green",
            "tee_green_ready_putt",
            np.where(
                np.isin(group_lies, list(full.PLAYABLE_TEE_LIES)),
                "tee_ready_state",
                "tee_hazard_unresolved",
            ),
        )
        endpoints[positions] = group_endpoints
        remaining[positions] = group_remaining
        lies[positions] = group_lies
        statuses[positions] = group_status
        atom_ids[positions] = selected_ids
        return {
            "endpoints": endpoints,
            "remaining": remaining,
            "end_lies": lies,
            "status": statuses,
            "atom_ids": atom_ids,
        }


class CachedSpatialAppSampler:
    """APP sampler retaining shot-relative cache and runtime classification."""

    def __init__(
        self,
        provider: AtomSupportProvider,
        *,
        enforce_empirical_class_support: bool = False,
    ) -> None:
        self.provider = provider
        self.surfaces = provider.app_surfaces
        self.enforce_empirical_class_support = bool(enforce_empirical_class_support)
        self.selection_rows: list[dict[str, object]] = []

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
        hazard_support_checker: spatial.HazardSupportChecker | None = None,
    ) -> dict[str, np.ndarray]:
        del par
        if self.provider.arm == "raw" and player_transport:
            raise ValueError("Raw cached parity supports population transport only")
        starts = np.asarray(start_endpoints_yd, float)
        start_lies = np.asarray(start_lies, object)
        start_remaining = np.asarray(start_remaining_yd, float)
        pin = np.asarray(pin_endpoint_yd, float)
        count = len(starts)
        if starts.shape != (count, 2):
            raise ValueError("start_endpoints_yd must have shape [particles, 2]")
        for name in ("index", "holeout", "lie"):
            if len(np.asarray(uniforms[name])) != count:
                raise ValueError("APP uniform arrays must match particle count")
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
        atom_ids = np.full(count, "not_attempted", dtype=object)
        surface = self.surfaces.get(int(hole_no))
        if surface is None:
            status[active] = "app_no_surface_support"
            active[:] = False
        groups: dict[tuple[str, int], list[int]] = {}
        for particle in np.flatnonzero(active):
            lie = app._normalize_start_lie(start_lies[particle])
            bucket = max(1, int(round(float(start_remaining[particle]) / 15.0)))
            groups.setdefault((lie, bucket), []).append(int(particle))
        for (start_lie, distance_bucket), positions_list in groups.items():
            positions = np.asarray(positions_list, int)
            try:
                support = self.provider.app_support(
                    round_no=round_no,
                    hole_no=hole_no,
                    start_lie=start_lie,
                    distance_bucket=distance_bucket,
                    player_id=str(player_id),
                )
            except ValueError:
                status[positions] = "app_no_spatial_support"
                continue
            selected, selected_ids = support.choose(
                np.asarray(uniforms["index"], float)[positions]
            )
            misses = support.points[selected]
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
                np.asarray(uniforms["holeout"], float)[positions]
                < support.holeout_probability
            )
            group_endpoints[group_holed] = pin
            conditional_lie = spatial._supported_surface_probabilities(
                surface.probabilities(group_endpoints),
                surface.codes,
                class_count=len(app.END_LIES),
                enforce_empirical_class_support=self.enforce_empirical_class_support,
                endpoints=group_endpoints,
                hazard_class_indices=(app.END_LIES.index("penalty"),),
                hazard_support_checker=hazard_support_checker,
            )
            if np.any(group_holed):
                conditional_lie[group_holed] = 0.0
                conditional_lie[group_holed, app.END_LIES.index("hole")] = 1.0
            lie_codes = spatial._categorical_from_uniform(
                conditional_lie, np.asarray(uniforms["lie"], float)[positions]
            )
            group_lies = np.asarray(app.END_LIES, object)[lie_codes]
            group_lies[group_holed] = "hole"
            group_geometric = np.linalg.norm(group_endpoints - pin[None, :], axis=1)
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
            has_player = self.provider.target_app_profiles[
                "source_player_id"
            ].astype(str).eq(str(player_id)).any()
            state = contract.app_state_key(start_lie, distance_bucket * 15.0)
            has_state = self.provider.target_app_profiles[
                "source_player_id"
            ].astype(str).eq(str(player_id)) & self.provider.target_app_profiles[
                "state_key"
            ].astype(str).eq(state)
            has_profile[positions] = bool(has_player)
            has_band_profile[positions] = bool(has_state.any())
            n_setup[positions] = support.n_setup
            n_hole[positions] = support.n_hole
            n_lie[positions] = support.n_broad
            atom_ids[positions] = selected_ids
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
            "atom_ids": atom_ids,
        }

def _app_support_impl(
        self: AtomSupportProvider,
        *,
        round_no: int,
        hole_no: int,
        start_lie: str,
        distance_bucket: int,
        player_id: str,
    ) -> SupportArrays:
        normalized_lie = app._normalize_start_lie(start_lie)
        key = (
            int(round_no), int(hole_no), normalized_lie,
            int(distance_bucket), str(player_id),
        )
        if self.cache_enabled and key in self.app_cache:
            return self.app_cache[key]
        distance_center = int(distance_bucket) * 15.0
        distribution = app.build_distribution(
            self.atoms.raw_course_app,
            int(round_no),
            int(hole_no),
            normalized_lie,
            distance_center,
        )
        rows = distribution.rows
        spatial_mask = (
            rows["spatial_valid"].to_numpy(bool)
            & ~rows["is_holeout"].to_numpy(bool)
        )
        if int(spatial_mask.sum()) < 12:
            raise ValueError("APP support has fewer than 12 spatial atoms")
        atom_rows = rows.loc[spatial_mask].copy().reset_index(drop=True)
        atom_ids = atom_rows["shot_key"].astype(str).to_numpy()
        base_mass = np.asarray(distribution.weights[spatial_mask], float)
        base_mass /= base_mass.sum()
        holeout_probability = float(
            distribution.weights[rows["is_holeout"].to_numpy(bool)].sum()
        )
        if self.arm == "raw":
            points = atom_rows[["miss_long_yd", "miss_lat_yd"]].to_numpy(float)
            final_mass = base_mass
            metadata: dict[str, object] = {
                "mass_contract": "legacy_conditional_spatial_mass_exact",
                "holeout_mass_contract": "legacy_unconditional_mass_exact",
                "model_weight_role": "unity_raw",
                "physical_classification_stage": "runtime_after_start_to_pin_rotation",
            }
        else:
            selected = self.neutral_app_by_id.reindex(atom_ids)
            available = selected["shot_key"].notna().to_numpy(bool)
            if not available.all():
                missing_ids = atom_ids[~available][:8].tolist()
                raise ValueError(
                    "Normalized APP panel must map every legacy spatial atom; "
                    f"missing={int((~available).sum())}, examples={missing_ids}"
                )
            selected = selected.reset_index(drop=True)
            neutral_points = selected[["miss_long_yd", "miss_lat_yd"]].to_numpy(float)
            center = np.average(neutral_points, axis=0, weights=base_mass)
            state = contract.app_state_key(normalized_lie, distance_center)
            condition = self._target_condition(
                category="app",
                state_key=state,
                round_no=round_no,
                hole_no=hole_no,
                center=center,
                support_rows=selected,
            )
            target_profiles = asof.materialize_normalization_profiles(
                self.target_app_profiles, condition
            )
            target_atoms = self._retag_neutral_support(
                selected, neutral_points, center, condition
            )
            reapplied, target_meta = norm.reapply_target_profile(
                target_atoms,
                target_profiles,
                target_player_id=str(player_id),
                target_event_start=self.atoms.target_event_start,
                geometry_classification_mode="deferred_shot_relative",
                config=norm.NormalizationConfig(ability_log_weight_per_sg2=0.0),
            )
            points = reapplied[
                ["deferred_miss_long_yd", "deferred_miss_lat_yd"]
            ].to_numpy(float)
            multipliers = reapplied["model_weight"].to_numpy(float)
            final_mass = base_mass * multipliers
            reapplied["base_hierarchy_mass"] = base_mass
            reapplied["final_unnormalized_mass"] = final_mass
            reapplied["support_category"] = "app"
            self.materialized_rows.append(reapplied)
            metadata = {
                "mass_contract": "base_hierarchy_mass_times_model_weight",
                "holeout_mass_contract": "legacy_unconditional_mass_retained_beta_zero",
                "model_weight_role": "mean_one_multiplier_not_probability",
                "target_reapplication_provenance_sha256": target_meta[
                    "provenance_sha256"
                ],
                "classifier_version": DEFERRED_CLASSIFIER_VERSION,
                "physical_classification_stage": "runtime_after_start_to_pin_rotation",
                "app_beta_gate": "beta_zero_state_agnostic_cdf_only",
            }
        support = SupportArrays.create(
            key=(
                f"app|{self.arm}|R{round_no}|H{hole_no}|{normalized_lie}|"
                f"B{distance_bucket}|{player_id}"
            ),
            atom_ids=atom_ids,
            points=points,
            weights=final_mass,
            source=distribution.source,
            n_setup=distribution.n_setup,
            n_hole=distribution.n_hole,
            n_broad=distribution.n_lie,
            holeout_probability=holeout_probability,
            metadata=metadata,
        )
        if self.arm == "raw":
            expected_mass = np.asarray(distribution.weights[spatial_mask], float)
            expected_mass /= expected_mass.sum()
            # Existing SpatialAppSampler normalizes the conditional spatial
            # mass once here and `_choice_from_uniform` normalizes it again.
            expected_mass /= expected_mass.sum()
            expected_cdf = np.cumsum(expected_mass)
            expected_cdf[-1] = 1.0
            if not np.array_equal(support.weights, expected_mass):
                raise AssertionError("Raw APP cached weights differ from legacy mass")
            if not np.array_equal(support.cdf, expected_cdf):
                raise AssertionError("Raw APP cached CDF differs from legacy CDF")
            if support.holeout_probability != holeout_probability:
                raise AssertionError("Raw APP cached holeout mass differs from legacy")
        if self.cache_enabled:
            self.app_cache[key] = support
        self.inventory.append(
            {
                "category": "app", "arm": self.arm, "round_no": int(round_no),
                "hole_no": int(hole_no), "state_key": contract.app_state_key(
                    normalized_lie, distance_center
                ),
                "player_id": str(player_id), "atoms": len(support.atom_ids),
                "holeout_probability": support.holeout_probability,
                "cache_sha256": support.cache_sha256,
            }
        )
        return support


def simulate_cached_population_grid(
    grid: pd.DataFrame,
    provider: AtomSupportProvider,
    *,
    simulations: int,
    seed: int,
    enforce_empirical_class_support: bool = False,
    hazard_support_factory: spatial.HazardSupportFactory | None = None,
) -> pd.DataFrame:
    """Initialize population particles and persist the chosen source atom IDs."""

    tee_sampler = CachedSpatialTeeSampler(
        provider,
        enforce_empirical_class_support=enforce_empirical_class_support,
    )
    app_sampler = CachedSpatialAppSampler(
        provider,
        enforce_empirical_class_support=enforce_empirical_class_support,
    )
    parts: list[pd.DataFrame] = []
    for record in grid.itertuples(index=False):
        player_id = str(record.source_player_id)
        case_seed = np.random.SeedSequence(
            [seed, zlib.crc32(f"{player_id}|{record.round_no}|{record.hole_no}".encode())]
        )
        rng = np.random.default_rng(case_seed)
        randoms = {
            name: rng.random(simulations)
            for name in ("tee_index", "tee_lie", "app_index", "app_holeout", "app_lie")
        }
        tee_checker = (
            hazard_support_factory(record, "ott")
            if hazard_support_factory is not None else None
        )
        app_checker = (
            hazard_support_factory(record, "app")
            if hazard_support_factory is not None else None
        )
        pin = np.asarray([float(record.forecast_pin_long_yd), 0.0])
        if int(record.par) == 3:
            starts = np.zeros((simulations, 2), dtype=float)
            start_lies = np.full(simulations, "tee", dtype=object)
            start_remaining = np.full(simulations, float(pin[0]), dtype=float)
            app_result = app_sampler.sample(
                starts,
                start_lies,
                start_remaining,
                pin,
                int(record.round_no),
                int(record.hole_no),
                int(record.par),
                player_id,
                False,
                {
                    "index": randoms["app_index"],
                    "holeout": randoms["app_holeout"],
                    "lie": randoms["app_lie"],
                },
                hazard_support_checker=app_checker,
            )
            app_endpoints = np.asarray(app_result["endpoints"], float)
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
                    "app_end_lie": app_result["end_lies"],
                    "app_geometric_remaining_yd": app_result["geometric_remaining"],
                    "app_remaining_yd": app_result["remaining"],
                    "app_holed": app_result["holed"],
                    "app_penalty_count": app_result["penalty_count"],
                    "app_penalty_cost_strokes": app_result["penalty_cost_strokes"],
                    "shots_taken": np.where(np.isfinite(app_endpoints[:, 0]), 1, 0),
                    "chain_status": app_result["status"],
                    "has_app_profile": app_result["has_profile"],
                    "has_app_band_profile": app_result["has_band_profile"],
                    "app_n_setup_history": app_result["n_setup"],
                    "app_n_hole_history": app_result["n_hole"],
                    "app_n_lie_history": app_result["n_lie"],
                    "tee_atom_id": "not_applicable_par3",
                    "app_atom_id": app_result["atom_ids"],
                }
            )
            initial_category = "app"
        else:
            tee_result = tee_sampler.sample(
                round_no=int(record.round_no),
                hole_no=int(record.hole_no),
                par=int(record.par),
                player_id=player_id,
                player_transport=False,
                pin_long_yd=float(pin[0]),
                index_uniform=randoms["tee_index"],
                lie_uniform=randoms["tee_lie"],
                active=np.ones(simulations, dtype=bool),
                hazard_support_checker=tee_checker,
            )
            tee_endpoints = np.asarray(tee_result["endpoints"], float)
            tee_lies = np.asarray(tee_result["end_lies"], object)
            tee_remaining = np.asarray(tee_result["remaining"], float)
            playable = np.isin(tee_lies, spatial.PLAYABLE_TEE_LIES) & np.isfinite(
                tee_remaining
            )
            app_result = app_sampler.sample(
                tee_endpoints,
                tee_lies,
                tee_remaining,
                pin,
                int(record.round_no),
                int(record.hole_no),
                int(record.par),
                player_id,
                False,
                {
                    "index": randoms["app_index"],
                    "holeout": randoms["app_holeout"],
                    "lie": randoms["app_lie"],
                },
                active=playable,
                hazard_support_checker=app_checker,
            )
            app_endpoints = np.asarray(app_result["endpoints"], float)
            attempted = np.isfinite(app_endpoints).all(axis=1)
            statuses = np.asarray(tee_result["status"], object).copy()
            statuses[playable] = np.asarray(app_result["status"], object)[playable]
            tee_hazard = statuses == "tee_hazard_unresolved"
            app_hazard = statuses == "app_hazard_unresolved"
            tee_penalty_count = np.where(tee_hazard, np.nan, 0.0)
            app_penalty_count = np.where(app_hazard, np.nan, 0.0)
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
                    "app_start_long_yd": np.where(attempted, tee_endpoints[:, 0], np.nan),
                    "app_start_lat_yd": np.where(attempted, tee_endpoints[:, 1], np.nan),
                    "app_start_lie": np.where(attempted, tee_lies, "not_attempted"),
                    "app_start_remaining_yd": np.where(attempted, tee_remaining, np.nan),
                    "app_end_long_yd": app_endpoints[:, 0],
                    "app_end_lat_yd": app_endpoints[:, 1],
                    "app_end_lie": app_result["end_lies"],
                    "app_geometric_remaining_yd": app_result["geometric_remaining"],
                    "app_remaining_yd": app_result["remaining"],
                    "app_holed": app_result["holed"],
                    "app_penalty_count": app_penalty_count,
                    "app_penalty_cost_strokes": app_penalty_count.copy(),
                    "shots_taken": np.where(attempted, 2, 1),
                    "chain_status": statuses,
                    "has_app_profile": app_result["has_profile"],
                    "has_app_band_profile": app_result["has_band_profile"],
                    "app_n_setup_history": app_result["n_setup"],
                    "app_n_hole_history": app_result["n_hole"],
                    "app_n_lie_history": app_result["n_lie"],
                    "tee_atom_id": tee_result["atom_ids"],
                    "app_atom_id": app_result["atom_ids"],
                }
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
            "arm": "population",
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
        for column, value in metadata.items():
            particle_frame[column] = value
        parts.append(particle_frame)
    particles = pd.concat(parts, ignore_index=True)
    id_columns = [
        "event_name", "event_key", "source_player_id", "player_name",
        "round_no", "hole_no", "par", "case_id", "arm", "particle_id",
        "initial_category", "setup_reference_year", "forecast_tee_x",
        "forecast_tee_y", "forecast_pin_x", "forecast_pin_y",
        "forecast_pin_long_yd", "roster_source", "coordinate_frame_version",
        "coordinate_frame_reference_year",
    ]
    return particles[id_columns + [
        column for column in particles.columns if column not in id_columns
    ]]


def _compare_particle_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    columns: Iterable[str],
    label: str,
) -> dict[str, object]:
    keys = ["case_id", "particle_id"]
    left = reference.sort_values(keys).reset_index(drop=True)
    right = candidate.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise AssertionError(f"{label} particle keys differ")
    mismatches: dict[str, int] = {}
    max_abs: dict[str, float] = {}
    for column in columns:
        if column not in left or column not in right:
            raise ValueError(f"{label} misses comparison column {column}")
        if pd.api.types.is_numeric_dtype(left[column]):
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
            equal = (a == b) | (np.isnan(a) & np.isnan(b))
            mismatches[column] = int((~equal).sum())
            finite = np.isfinite(a) & np.isfinite(b)
            max_abs[column] = (
                float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            )
        else:
            a = left[column].astype(str).to_numpy()
            b = right[column].astype(str).to_numpy()
            mismatches[column] = int((a != b).sum())
    bad = {column: count for column, count in mismatches.items() if count}
    if bad:
        raise AssertionError(f"{label} is not exact: {bad}")
    return {
        "label": label,
        "rows": int(len(left)),
        "columns": list(columns),
        "mismatches": mismatches,
        "max_abs_error": max_abs,
        "exact": True,
    }


def _support_npz(provider: AtomSupportProvider, path: Path) -> dict[str, object]:
    items: list[SupportArrays] = [
        *provider.tee_cache.values(), *provider.app_cache.values()
    ]
    items = sorted(items, key=lambda item: item.key)
    offsets = [0]
    for item in items:
        offsets.append(offsets[-1] + len(item.atom_ids))
    if items:
        atom_ids = np.concatenate([item.atom_ids for item in items])
        points = np.concatenate([item.points for item in items], axis=0)
        weights = np.concatenate([item.weights for item in items])
        cdf = np.concatenate([item.cdf for item in items])
    else:
        atom_ids = np.asarray([], dtype=str)
        points = np.empty((0, 2), dtype=float)
        weights = np.asarray([], dtype=float)
        cdf = np.asarray([], dtype=float)
    np.savez_compressed(
        path,
        support_keys=np.asarray([item.key for item in items]),
        offsets=np.asarray(offsets, dtype=np.int64),
        atom_ids=atom_ids,
        points_yd=points,
        probability=weights,
        cdf=cdf,
        holeout_probability=np.asarray(
            [item.holeout_probability for item in items], dtype=float
        ),
        cache_sha256=np.asarray([item.cache_sha256 for item in items]),
    )
    return {
        "supports": len(items),
        "atoms": int(len(atom_ids)),
        "sha256": _file_sha256(path),
        "contiguous_points": bool(points.flags.c_contiguous),
        "contiguous_weights": bool(weights.flags.c_contiguous),
        "contiguous_cdf": bool(cdf.flags.c_contiguous),
    }


def _build_smoke_grid(
    atoms: NormalizedEventAtoms,
    baseline_path: Path,
    *,
    roster_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    roster, roster_meta = full._load_pre_event_roster(
        baseline_path, atoms.event_name, 1, roster_seed
    )
    grid = full._build_complete_grid(
        atoms.event_name,
        atoms.target_event_key,
        atoms.prepared["target_raw"],
        atoms.forecast,
        roster,
    )
    grid = grid[grid["round_no"].eq(1)].copy().reset_index(drop=True)
    grid["case_id"] = np.arange(len(grid), dtype=int)
    if len(grid) != 18:
        raise ValueError(f"Smoke grid has {len(grid)} rows rather than 18")
    return grid, roster_meta


def _hazard_resolver(
    *,
    export_root: Path,
    event_name: str,
    holdout_year: int,
    course_start_year: int,
) -> tuple[hazard.EndpointLinkedHazardResolver, dict[str, object]]:
    donors, heldout, identity, diagnostics = hazard.load_normalized_hazard_panel(
        export_root, event_name, holdout_year, course_start_year
    )
    resolver = hazard.EndpointLinkedHazardResolver(
        donors,
        same_hole_fallback_max_distance_yd=150.0,
    )
    return resolver, {
        "predictor_donor_rows": int(len(donors)),
        "heldout_rows_loaded_for_resolution_audit_only": int(len(heldout)),
        "physical_course_id": resolver.physical_course_id,
        "coordinate_reference_year": resolver.coordinate_reference_year,
        "identity_rows": int(len(identity)),
        "diagnostics": diagnostics,
    }


def _run_terminal_parity(
    initial: pd.DataFrame,
    atoms: NormalizedEventAtoms,
    raw_provider: AtomSupportProvider,
    *,
    legacy_tee_profiles: pd.DataFrame,
    legacy_app_profiles: pd.DataFrame,
    resolver: hazard.EndpointLinkedHazardResolver,
    export_root: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    state_model, state_rows = full._course_population_model(
        export_root,
        atoms.prepared["event_keys"],
        atoms.holdout_year,
    )
    old_tee = full.SpatialTeeSampler(
        atoms.raw_course_drives,
        legacy_tee_profiles,
        enforce_empirical_class_support=True,
    )
    old_app = spatial.SpatialAppSampler(
        atoms.raw_course_app,
        legacy_app_profiles,
        enforce_empirical_class_support=True,
    )
    cached_tee = CachedSpatialTeeSampler(
        raw_provider, enforce_empirical_class_support=True
    )
    cached_app = CachedSpatialAppSampler(
        raw_provider, enforce_empirical_class_support=True
    )
    old_state = full.PopulationTransitionSampler(state_model)
    cached_state = full.PopulationTransitionSampler(state_model)
    old_parts: list[pd.DataFrame] = []
    cached_parts: list[pd.DataFrame] = []
    old_hazards: list[dict[str, object]] = []
    cached_hazards: list[dict[str, object]] = []
    for _, case in initial.groupby("case_id", observed=True, sort=True):
        old_result, old_rows = full._continue_case_arm(
            case,
            app_sampler=old_app,
            tee_sampler=old_tee,
            state_sampler=old_state,
            hazard_resolver=resolver,
            seed=seed,
        )
        cached_result, cached_rows = full._continue_case_arm(
            case,
            app_sampler=cached_app,
            tee_sampler=cached_tee,
            state_sampler=cached_state,
            hazard_resolver=resolver,
            seed=seed,
        )
        old_parts.append(old_result)
        cached_parts.append(cached_result)
        old_hazards.extend(old_rows)
        cached_hazards.extend(cached_rows)
    old_terminal = pd.concat(old_parts, ignore_index=True)
    cached_terminal = pd.concat(cached_parts, ignore_index=True)
    columns = [
        "hole_strokes", "terminal", "chain_status", "hazard_involved",
        "hazard_transitions", "penalty_strokes", "transition_count",
        "return_to_app_radial_projections", "final_lie", "final_remaining_yd",
    ]
    audit = _compare_particle_frames(
        old_terminal,
        cached_terminal,
        columns=columns,
        label="raw_cached_terminal_chain",
    )
    audit.update(
        {
            "state_transition_rows": int(len(state_rows)),
            "old_backoff_counts": dict(old_state.backoff_counts),
            "cached_backoff_counts": dict(cached_state.backoff_counts),
            "old_hazard_rows": len(old_hazards),
            "cached_hazard_rows": len(cached_hazards),
        }
    )
    comparison = old_terminal[
        ["case_id", "particle_id", *columns]
    ].merge(
        cached_terminal[["case_id", "particle_id", *columns]],
        on=["case_id", "particle_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_legacy", "_cached"),
    )
    return old_terminal, comparison, audit


def run_smoke(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"Refusing non-empty or non-directory output path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = precompute_event_atoms(
        export_root=Path(args.export_root),
        history_db=Path(args.history_db),
        profile_dir=Path(args.profile_dir),
        event_name=str(args.event_name),
        holdout_year=int(args.holdout_year),
        course_start_year=int(args.course_start_year),
        mode=str(args.mode),
    )
    grid, roster_meta = _build_smoke_grid(
        atoms, Path(args.baseline_cases), roster_seed=int(args.roster_seed)
    )
    legacy_tee_profiles = pd.read_csv(
        args.tee_profiles_csv, dtype={"source_player_id": str}
    )
    legacy_app_profiles = pd.read_csv(
        args.app_profiles_csv, dtype={"source_player_id": str}
    )
    resolver, hazard_meta = _hazard_resolver(
        export_root=Path(args.export_root),
        event_name=str(args.event_name),
        holdout_year=int(args.holdout_year),
        course_start_year=int(args.course_start_year),
    )
    hazard_factory = full._forecast_hazard_support_factory(resolver)
    reference_initial = spatial.simulate_forecast_grid(
        grid,
        atoms.raw_course_drives,
        atoms.raw_course_app,
        legacy_tee_profiles,
        legacy_app_profiles,
        int(args.simulations),
        int(args.seed),
        enforce_empirical_class_support=True,
        hazard_support_factory=hazard_factory,
    )
    reference_initial = reference_initial[
        reference_initial["arm"].eq("population")
    ].copy().reset_index(drop=True)
    fresh_provider = AtomSupportProvider(atoms, arm="raw", cache_enabled=False)
    cached_provider = AtomSupportProvider(atoms, arm="raw", cache_enabled=True)
    fresh_initial = simulate_cached_population_grid(
        grid,
        fresh_provider,
        simulations=int(args.simulations),
        seed=int(args.seed),
        enforce_empirical_class_support=True,
        hazard_support_factory=hazard_factory,
    )
    cached_initial = simulate_cached_population_grid(
        grid,
        cached_provider,
        simulations=int(args.simulations),
        seed=int(args.seed),
        enforce_empirical_class_support=True,
        hazard_support_factory=hazard_factory,
    )
    physical_columns = [
        "tee_end_long_yd", "tee_end_lat_yd", "tee_end_lie",
        "tee_remaining_yd", "tee_penalty_count", "app_start_long_yd",
        "app_start_lat_yd", "app_start_lie", "app_start_remaining_yd",
        "app_end_long_yd", "app_end_lat_yd", "app_end_lie",
        "app_geometric_remaining_yd", "app_remaining_yd", "app_holed",
        "app_penalty_count", "shots_taken", "chain_status",
        "app_n_setup_history", "app_n_hole_history", "app_n_lie_history",
    ]
    legacy_initializer_audit = _compare_particle_frames(
        reference_initial,
        cached_initial,
        columns=physical_columns,
        label="legacy_vs_raw_cached_initializer",
    )
    atom_id_audit = _compare_particle_frames(
        fresh_initial,
        cached_initial,
        columns=["tee_atom_id", "app_atom_id"],
        label="fresh_vs_cached_selected_atom_ids",
    )
    terminal_scores, terminal_comparison, terminal_audit = _run_terminal_parity(
        reference_initial,
        atoms,
        cached_provider,
        legacy_tee_profiles=legacy_tee_profiles,
        legacy_app_profiles=legacy_app_profiles,
        resolver=resolver,
        export_root=Path(args.export_root),
        seed=int(args.seed) + 500,
    )
    normalized_provider = AtomSupportProvider(
        atoms, arm="normalized", cache_enabled=True
    )
    normalized_initial = simulate_cached_population_grid(
        grid,
        normalized_provider,
        simulations=int(args.simulations),
        seed=int(args.seed),
        enforce_empirical_class_support=True,
        hazard_support_factory=hazard_factory,
    )
    raw_npz_meta = _support_npz(cached_provider, output_dir / "raw_support_arrays.npz")
    normalized_npz_meta = _support_npz(
        normalized_provider, output_dir / "normalized_support_arrays.npz"
    )
    selected = fresh_initial[
        ["case_id", "particle_id", "tee_atom_id", "app_atom_id"]
    ].merge(
        cached_initial[
            ["case_id", "particle_id", "tee_atom_id", "app_atom_id"]
        ],
        on=["case_id", "particle_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_fresh", "_cached"),
    )
    selected.to_parquet(output_dir / "selected_atom_ids.parquet", index=False)
    terminal_comparison.to_parquet(
        output_dir / "terminal_comparison.parquet", index=False
    )
    terminal_scores.to_parquet(output_dir / "legacy_terminal_scores.parquet", index=False)
    normalized_initial.to_parquet(
        output_dir / "normalized_initializer_diagnostic.parquet", index=False
    )
    pd.DataFrame(cached_provider.inventory).to_csv(
        output_dir / "raw_support_inventory.csv", index=False
    )
    pd.DataFrame(normalized_provider.inventory).to_csv(
        output_dir / "normalized_support_inventory.csv", index=False
    )
    neutral_columns = [
        "shot_key", "event_key", "season", "round_no", "hole_no",
        "source_player_id", "state_key", "setup_key", "normalization_mode",
        "normalization_provenance_sha256", "alignment_count",
    ]
    atoms.neutral_tee[
        neutral_columns + ["long_yd", "lat_yd"]
    ].to_parquet(output_dir / "neutral_tee_atoms.parquet", index=False)
    atoms.neutral_app[
        neutral_columns + ["miss_long_yd", "miss_lat_yd"]
    ].to_parquet(output_dir / "neutral_app_atoms.parquet", index=False)
    if normalized_provider.materialized_rows:
        pd.concat(normalized_provider.materialized_rows, ignore_index=True).to_parquet(
            output_dir / "target_reapplied_atoms.parquet", index=False
        )
    elapsed = time.perf_counter() - started
    profile_bridge = atoms.profile_manifest.get("bridge", {})
    manifest: dict[str, object] = {
        "artifact": PRECOMPUTE_VERSION,
        "research_only": True,
        "production_touched": False,
        "event": atoms.event_name,
        "event_key": atoms.target_event_key,
        "holdout_year": atoms.holdout_year,
        "player_id": str(grid.iloc[0]["source_player_id"]),
        "rounds": [1],
        "holes": 18,
        "particles": int(args.simulations),
        "tail_policy": tail_policy.manifest_policy(),
        "normalization_mode": str(args.mode),
        "ability_log_weight_per_sg2": 0.0,
        "normalized_arm_role": "diagnostic_not_promotion_evidence",
        "transform_order": [
            "donor_native_asof_normalize",
            "rigid_align_neutral_endpoint_exactly_once",
            "target_profile_reapply",
            "runtime_physical_surface_reclassification",
        ],
        "app_cache_contract": (
            "pin_relative_miss_vectors_only; endpoint/lie/hazard generated at runtime"
        ),
        "app_nonzero_beta_gate": (
            "not_permitted_in_state_agnostic_CDF; requires generated-state reward audit"
        ),
        "mass_contract": "base_hierarchy_mass_times_mean_one_model_weight",
        "profile_contract": atoms.profile_manifest.get("contract"),
        "profile_manifest_sha256": _file_sha256(Path(args.profile_dir) / "manifest.json"),
        "identity_bridge_contract": profile_bridge.get("bridge_contract"),
        "identity_bridge_role": "season_wide_identity_alias_only_never_predictor",
        "skill_cutoff_rule": "latest_skill_date_strictly_before_event_start",
        "physical_course_id": atoms.physical_course_id,
        "coordinate_frame_version": atoms.prepared["provenance"].get(
            "coordinate_frame_version"
        ),
        "coordinate_reference_year": int(
            atoms.prepared["provenance"]["setup_reference_year"]
        ),
        "alignment_count": 1,
        "raw_alignment_equivalence": atoms.normalization_metadata[
            "raw_alignment_equivalence"
        ],
        "app_normalization_exclusions": atoms.normalization_metadata[
            "app_normalization_exclusions"
        ],
        "initializer_parity": legacy_initializer_audit,
        "selected_atom_id_parity": atom_id_audit,
        "terminal_parity": terminal_audit,
        "raw_support_npz": raw_npz_meta,
        "normalized_support_npz": normalized_npz_meta,
        "hazard_resolver": hazard_meta,
        "roster": roster_meta,
        "timing_seconds": {**atoms.timings, "full_smoke": elapsed},
        "code_sha256": _file_sha256(Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    report = "# Normalized atom precompute smoke\n\n"
    report += (
        f"Travelers strict-prior R1 smoke: 1 player x 18 holes x "
        f"{int(args.simulations)} particles. Raw cached supports reproduced legacy "
        "initializer states, selected source atom IDs, and terminal chain results "
        "exactly.\n\n"
        "The normalized output is diagnostic only. TEE atoms are physically "
        "classified for a coherence audit but the sampler still draws lie from "
        "the stochastic surface. APP atoms remain pin-relative miss vectors and "
        "are rotated/classified only after the generated start state is known.\n\n"
        f"- Raw support NPZ: {raw_npz_meta['supports']} supports, "
        f"{raw_npz_meta['atoms']:,} flattened atoms.\n"
        f"- Normalized support NPZ: {normalized_npz_meta['supports']} supports, "
        f"{normalized_npz_meta['atoms']:,} flattened atoms.\n"
        f"- Continuation guard: {tail_policy.MAX_CONTINUATION_TRANSITIONS} "
        "transitions; active survivors fail the whole run.\n"
        "- Hole-score cap/clipping: none.\n"
        f"- High-score diagnostic threshold: "
        f"{tail_policy.HIGH_SCORE_DIAGNOSTIC_STROKES} strokes (diagnostic only).\n"
        f"- Runtime: {elapsed:.1f} seconds.\n"
        "- Production files/deployments touched: no.\n"
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--history-db", type=Path, default=asof.DEFAULT_HISTORY_DB)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--baseline-cases", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--tee-profiles-csv", type=Path, default=full.DEFAULT_TEE_PROFILES)
    parser.add_argument("--app-profiles-csv", type=Path, default=full.DEFAULT_APP_PROFILES)
    parser.add_argument("--event-name", default=DEFAULT_EVENT)
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument("--mode", choices=("event_centered", DEFAULT_MODE), default=DEFAULT_MODE)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--roster-seed", type=int, default=20260824)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.simulations) != 128:
        print(
            "Warning: the promotion gate is defined at 128 particles; "
            f"running requested diagnostic size {args.simulations}."
        )
    output = run_smoke(args)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

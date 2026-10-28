"""Research-only complete spatial hole and round chain.

This wrapper continues the strict-prior spatial tee -> APP initializer through
terminal holes.  It is deliberately outside the production pipeline.  The
four initializer arms share random numbers for every later transition:

``population``, ``player_tee``, ``player_app`` and ``both``.

Spatial APP is called whenever the generated state routes back to APP.  ARG
and putting use only exact-physical-course population transitions from
``hole_sim_markov``.  Spatial hazards are resolved from a nearby causal
full-sequence donor; unsupported or incoherent states fail closed.  Hole
scores remain integer physical strokes and round files satisfy the frozen
evaluator's long-particle contract.

Continuation has one shared 18-transition computational guard.  It has no
physical hole-score cap: 15+ stroke holes are diagnostics only, and a path
still active after transition 17 fails the whole run.

Nothing here imports production simulation, pricing, storage or publishing
code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_app_transition as app  # noqa: E402
import hole_sim_hazard_resolution as hazard  # noqa: E402
import hole_sim_markov as markov  # noqa: E402
import hole_sim_state_foundations as foundations  # noqa: E402
import hole_sim_tail_policy as tail_policy  # noqa: E402
import hole_sim_tee_app_integrated_v2 as spatial  # noqa: E402
import hole_sim_tee_transition as tee  # noqa: E402
from evaluate_spatial_chain_rounds import (  # noqa: E402
    COMPLETE_ROUND_COLUMNS,
    load_complete_round_particles,
)
from hole_sim_calibration_adapter import (  # noqa: E402
    aggregate_hole_strokes,
    normalise_player,
    sha256_file,
)


DEFAULT_EXPORT = app.DEFAULT_EXPORT
DEFAULT_DB = markov.DEFAULT_DB
DEFAULT_OUTPUT = REPO / "permanent_data" / "hole_sim_full_chain"
DEFAULT_BASELINES = (
    REPO
    / "permanent_data"
    / "spatial_chain_common_baselines"
    / "round_case_pmfs.parquet"
)
DEFAULT_TEE_PROFILES = spatial.DEFAULT_TEE_PROFILES
DEFAULT_APP_PROFILES = spatial.DEFAULT_APP_PROFILES
DEFAULT_EVENTS = (
    "Sony Open in Hawaii",
    "Travelers Championship",
    "Wyndham Championship",
)
ARMS = spatial.ARMS
CHAIN_VERSION = "spatial_full_chain_feasibility_v3_unbounded_score_tail"
EXPECTED_FRAME = spatial.COORDINATE_FRAME_VERSION
PLAYABLE_TEE_LIES = set(spatial.PLAYABLE_TEE_LIES)


class ChainIncomplete(RuntimeError):
    """Raised when a particle cannot reach a coherent terminal state."""


@dataclass(frozen=True)
class CourseFrame:
    tee_x: float
    tee_y: float
    pin_x: float
    pin_y: float

    @property
    def unit(self) -> np.ndarray:
        vector = np.asarray([self.pin_x - self.tee_x, self.pin_y - self.tee_y])
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 3.0:
            raise ChainIncomplete("Forecast tee/pin axis is degenerate")
        return vector / norm

    @property
    def perpendicular(self) -> np.ndarray:
        unit = self.unit
        return np.asarray([-unit[1], unit[0]])

    def local_yards_to_absolute_feet(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, float)
        return (
            np.asarray([self.tee_x, self.tee_y])
            + 3.0 * point[0] * self.unit
            + 3.0 * point[1] * self.perpendicular
        )

    def local_yards_to_absolute_feet_many(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Local course points must have shape [n, 2]")
        return (
            np.asarray([self.tee_x, self.tee_y])[None, :]
            + 3.0 * values[:, 0, None] * self.unit[None, :]
            + 3.0 * values[:, 1, None] * self.perpendicular[None, :]
        )

    def absolute_feet_to_local_yards(self, point: np.ndarray) -> np.ndarray:
        delta = np.asarray(point, float) - np.asarray([self.tee_x, self.tee_y])
        return np.asarray(
            [float(delta @ self.unit) / 3.0, float(delta @ self.perpendicular) / 3.0]
        )

    def geometric_remaining_yd(self, absolute_point: np.ndarray) -> float:
        return float(
            np.linalg.norm(
                np.asarray(absolute_point, float)
                - np.asarray([self.pin_x, self.pin_y], float)
            )
            / 3.0
        )


def _hazard_support_checker(
    resolver: hazard.EndpointLinkedHazardResolver,
    frame: CourseFrame,
    *,
    hole_no: int,
    round_no: int,
    category: str,
) -> spatial.HazardSupportChecker:
    """Bridge local-yard generator endpoints to the absolute-feet resolver."""

    def checker(local_endpoints_yd: np.ndarray) -> np.ndarray:
        absolute = frame.local_yards_to_absolute_feet_many(local_endpoints_yd)
        return np.asarray(
            resolver.has_support_many(
                absolute[:, 0],
                absolute[:, 1],
                hole_no=int(hole_no),
                round_no=int(round_no),
                category=str(category),
            ),
            bool,
        )

    return checker


def _forecast_hazard_support_factory(
    resolver: hazard.EndpointLinkedHazardResolver,
) -> spatial.HazardSupportFactory:
    def factory(record: object, category: str) -> spatial.HazardSupportChecker:
        frame = CourseFrame(
            float(record.forecast_tee_x),
            float(record.forecast_tee_y),
            float(record.forecast_pin_x),
            float(record.forecast_pin_y),
        )
        return _hazard_support_checker(
            resolver,
            frame,
            hole_no=int(record.hole_no),
            round_no=int(record.round_no),
            category=category,
        )

    return factory


def _slug(value: str) -> str:
    return spatial._slug(value)


def _normalize_lie(value: object) -> str:
    text = str(value).strip().casefold()
    if text in {"hole", "holed", "cup"}:
        return "hole"
    if text in {"tee", "fairway", "rough", "sand", "recovery", "green"}:
        return text
    # Direct APP reserves ``penalty`` for causal assessed hazards. Its
    # ``other`` class is a non-penalty catch-all and is safely pooled with the
    # recovery state for downstream recursion.
    if text == "other":
        return "recovery"
    if "bunker" in text or "sand" in text:
        return "sand"
    return "unknown"


def _state_category(lie: str, distance_yd: float, par: int) -> str:
    lie = _normalize_lie(lie)
    if lie == "hole":
        return "terminal"
    if lie == "green":
        return "putt"
    if lie == "tee":
        return "app" if int(par) == 3 else "ott"
    if lie == "unknown" or not np.isfinite(distance_yd):
        return "invalid"
    return "arg" if float(distance_yd) <= 50.0 else "app"


def _charge_spatial_hazard(strokes_before: int, penalty_stroke_cost: int) -> int:
    """Advance from a pre-charged causal shot to the next uncharged play."""

    cost = int(penalty_stroke_cost)
    if cost < 1:
        raise AssertionError("Resolved hazard must advance at least one penalty stroke")
    return int(strokes_before) + cost


def _common_uniforms(
    *,
    seed: int,
    event_key: str,
    player_id: str,
    round_no: int,
    hole_no: int,
    particle_ids: np.ndarray,
    transition_index: int,
    stream: str,
) -> np.ndarray:
    """Stable arm-independent uniforms keyed at the particle-transition grain."""

    prefix = (
        f"{seed}|{event_key}|{player_id}|R{round_no}|H{hole_no}|"
        f"T{transition_index}|{stream}|"
    )
    values = [
        zlib.crc32(f"{prefix}{int(particle)}".encode("utf-8")) / float(2**32)
        for particle in np.asarray(particle_ids, int)
    ]
    return np.asarray(values, float)


def _validate_provenance(
    provenance: Mapping[str, object],
    holdout_year: int,
) -> None:
    expected = {
        "coordinate_frame_version": EXPECTED_FRAME,
        "alignment_count": 1,
        "predictor_reference_scope": "latest_verified_prior_only",
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"Strict-prior provenance failure: {key}={provenance.get(key)!r}; "
                f"expected {value!r}"
            )
    reference_year = int(provenance["setup_reference_year"])
    if reference_year >= int(holdout_year):
        raise ValueError("Setup reference must strictly precede holdout")
    heldout_role = str(provenance.get("heldout_geometry_role", ""))
    if heldout_role not in {"none", "course_identity_audit_only"}:
        raise ValueError(f"Heldout geometry entered the predictor path: {heldout_role}")
    if bool(provenance.get("heldout_scoring_transform_applied", False)):
        raise ValueError("Full-chain predictor preparation applied a heldout transform")


def _load_pre_event_roster(
    baseline_path: Path,
    event_name: str,
    max_players: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select deterministically from the frozen pre-event R1 field.

    Round-3/4 participation is an outcome of making the cut and must never be
    an eligibility condition. Every selected R1 player is simulated for all
    four rounds; the evaluator later scores only independently observed
    player-round outcomes.
    """

    baseline = pd.read_parquet(
        baseline_path,
        columns=[
            "event_key", "event", "event_start", "source_player_id",
            "player_name", "round_num", "skill_date",
        ],
    )
    baseline = baseline[baseline["event"].eq(event_name)].copy()
    baseline["source_player_id"] = baseline["source_player_id"].astype(str)
    baseline["player_name"] = baseline["player_name"].map(normalise_player)
    cases = baseline[baseline["round_num"].eq(1)].drop_duplicates(
        ["event_key", "source_player_id", "player_name", "round_num"]
    )
    roster = cases[["event_key", "source_player_id", "player_name"]].copy()
    if roster.empty:
        raise ValueError(f"No frozen R1 field for {event_name}")
    roster["_hash"] = pd.util.hash_pandas_object(
        roster[["event_key", "source_player_id"]], index=False
    ).to_numpy(np.uint64) ^ np.uint64(seed)
    roster = roster.sort_values("_hash")
    if max_players > 0:
        roster = roster.head(max_players)
    roster = roster.drop(columns="_hash").reset_index(drop=True)
    event_keys = roster["event_key"].unique()
    if len(event_keys) != 1:
        raise ValueError(f"Frozen baseline event key is ambiguous: {event_keys}")
    fixed = baseline[
        baseline["source_player_id"].isin(roster["source_player_id"])
    ][["event_start"]]
    event_start = pd.to_datetime(fixed["event_start"]).min()
    # This field describes the candidate spatial distributions only. Frozen
    # mean skill is joined independently by the evaluator and is date-grained
    # at event_start, so it must not contaminate the candidate cutoff.
    input_cutoff = pd.Timestamp(event_start.year - 1, 12, 31)
    if input_cutoff >= event_start:
        raise ValueError("Frozen complete-case input cutoff is not pre-event")
    metadata = {
        "event_key": str(event_keys[0]),
        "event_start": str(pd.Timestamp(event_start).date()),
        "input_cutoff": str(pd.Timestamp(input_cutoff).date()),
        "input_cutoff_source": (
            "verified strict-prior producer policy (season < holdout); "
            "not inferred from CSV contents"
        ),
        "input_cutoff_is_content_derived": False,
        "eligible_r1_players": int(len(cases)),
        "selected_players": int(len(roster)),
        "selection_role": "frozen_pre_event_r1_field_only",
        "heldout_round_participation_used_for_selection": False,
    }
    return roster, metadata


def _input_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "full_chain_source": Path(__file__),
        "spatial_v2_source": Path(spatial.__file__),
        "state_foundation_source": Path(foundations.__file__),
        "hazard_resolver_source": Path(hazard.__file__),
        "markov_source": Path(markov.__file__),
        "baseline_case_pmfs": Path(args.baseline_cases),
        "tee_profile_csv": Path(args.tee_profiles_csv),
        "app_profile_csv": Path(args.app_profiles_csv),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _build_complete_grid(
    event_name: str,
    shotlink_event_key: str,
    target_raw: pd.DataFrame,
    forecast: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    selected_ids = set(roster["source_player_id"].astype(str))
    identities = target_raw[
        target_raw["event_type"].eq("STROKE")
    ][["source_player_id", "player_name"]].copy()
    identities["source_player_id"] = identities["source_player_id"].astype(str)
    identities = identities.drop_duplicates("source_player_id")
    roster = roster.merge(
        identities,
        on="source_player_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_frozen", "_shotlink"),
    )
    matched_ids = set(roster["source_player_id"].astype(str))
    if matched_ids != selected_ids:
        missing = sorted(selected_ids - matched_ids)
        raise ValueError(
            f"Pre-event roster did not exactly match {event_name} export IDs: {missing}"
        )
    rounds = pd.DataFrame({"round_no": [1, 2, 3, 4], "_join": 1})
    roster = roster.assign(_join=1).merge(rounds, on="_join").drop(columns="_join")
    grid = roster.merge(forecast, on="round_no", how="inner", validate="many_to_many")
    grid["player_name"] = grid["player_name_frozen"].map(normalise_player)
    grid["event_name"] = event_name
    grid["event_key"] = shotlink_event_key
    grid["roster_source"] = "frozen_pre_event_r1_field_simulated_r1_through_r4"
    grid["coordinate_frame_version"] = EXPECTED_FRAME
    grid["coordinate_frame_reference_year"] = grid["setup_reference_year"]
    grid = grid.sort_values(
        ["source_player_id", "round_no", "hole_no"]
    ).reset_index(drop=True)
    grid["case_id"] = np.arange(len(grid), dtype=int)
    expected = roster["source_player_id"].nunique() * 4 * 18
    if len(grid) != expected:
        raise ValueError(f"Full grid has {len(grid)} cases; expected {expected}")
    complete = grid.groupby("source_player_id", observed=True).agg(
        rounds=("round_no", "nunique"), holes=("case_id", "size")
    )
    if not (complete["rounds"].eq(4) & complete["holes"].eq(72)).all():
        raise ValueError("Complete-player grid invariant failed")
    return grid


class SpatialTeeSampler:
    """Reusable strict-prior tee transition for replay/retee states."""

    def __init__(
        self,
        course_drives: pd.DataFrame,
        profiles: pd.DataFrame,
        *,
        enforce_empirical_class_support: bool = False,
    ):
        self.course_drives = course_drives
        profiles = profiles.copy()
        profiles["source_player_id"] = profiles["source_player_id"].astype(str)
        self.profiles = profiles.set_index("source_player_id")
        self.enforce_empirical_class_support = bool(
            enforce_empirical_class_support
        )
        self.neutral = pd.Series(
            {
                "distance_ratio": 1.0,
                "lateral_bias_yd": 0.0,
                "long_scale": 1.0,
                "lat_scale": 1.0,
            }
        )
        self.distributions: dict[tuple[int, int, int], tee.Distribution] = {}
        self.surfaces = {
            int(hole): tee.SurfaceMap(history)
            for hole, history in course_drives.groupby("hole_no", observed=True)
        }

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
        count = len(active)
        endpoints = np.full((count, 2), np.nan)
        remaining = np.full(count, np.nan)
        lies = np.full(count, "not_attempted", dtype=object)
        statuses = np.full(count, "tee_not_attempted", dtype=object)
        positions = np.flatnonzero(np.asarray(active, bool))
        if not len(positions):
            return {
                "endpoints": endpoints,
                "remaining": remaining,
                "end_lies": lies,
                "status": statuses,
            }
        key = (int(round_no), int(hole_no), int(par))
        if key not in self.distributions:
            self.distributions[key] = tee.build_distribution(self.course_drives, *key)
        distribution = self.distributions[key]
        points = distribution.points
        profile = self.profiles.loc[player_id] if player_id in self.profiles.index else self.neutral
        if isinstance(profile, pd.DataFrame):
            profile = profile.iloc[-1]
        if player_transport:
            points = tee.adjust_distribution(points, distribution.weights, profile)
        sampled = spatial._choice_from_uniform(
            distribution.weights, np.asarray(index_uniform)[positions]
        )
        group_endpoints = points[sampled]
        surface = self.surfaces.get(int(hole_no))
        if surface is None:
            statuses[positions] = "tee_no_surface_support"
            return {
                "endpoints": endpoints,
                "remaining": remaining,
                "end_lies": lies,
                "status": statuses,
            }
        probabilities = spatial._supported_surface_probabilities(
            surface.probabilities(group_endpoints),
            surface.codes,
            class_count=len(tee.LIE_CLASSES),
            enforce_empirical_class_support=(
                self.enforce_empirical_class_support
            ),
            endpoints=group_endpoints,
            hazard_class_indices=tuple(
                tee.LIE_CLASSES.index(name) for name in ("water", "other")
            ),
            hazard_support_checker=hazard_support_checker,
        )
        codes = spatial._categorical_from_uniform(
            probabilities, np.asarray(lie_uniform)[positions]
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
                np.isin(group_lies, list(PLAYABLE_TEE_LIES)),
                "tee_ready_state",
                "tee_hazard_unresolved",
            ),
        )
        endpoints[positions] = group_endpoints
        remaining[positions] = group_remaining
        lies[positions] = group_lies
        statuses[positions] = group_status
        return {
            "endpoints": endpoints,
            "remaining": remaining,
            "end_lies": lies,
            "status": statuses,
        }


class PopulationTransitionSampler:
    """Uniform-driven ARG/PUTT sampler using Markov's exact-course hierarchy."""

    def __init__(self, model: markov.MarkovHoleModel):
        self.model = model
        self.backoff_counts: Counter[str] = Counter()

    def sample(
        self,
        *,
        round_no: int,
        hole_no: int,
        category: str,
        lie: str,
        distance_yd: float,
        common_uniform: float,
    ) -> tuple[bool, str, float, int, str]:
        if category not in {"arg", "putt"}:
            raise ValueError(f"Population state sampler cannot draw {category}")
        pool, level = self.model._pool(
            int(round_no),
            int(hole_no),
            category,
            _normalize_lie(lie),
            markov.distance_bucket(float(distance_yd)),
        )
        self.backoff_counts[level] += 1
        metric = np.abs(
            np.log1p(np.maximum(pool.start_distance, 0.0))
            - math.log1p(max(float(distance_yd), 0.0))
        )
        k = min(self.model.knn, len(pool))
        neighbours = (
            np.argpartition(metric, k - 1)[:k]
            if k < len(pool)
            else np.arange(len(pool))
        )
        neighbours = np.sort(neighbours)
        weights = np.exp(-metric[neighbours] / 0.18)
        weights = np.maximum(weights, 1e-12)
        weights /= weights.sum()
        selected = min(
            int(np.searchsorted(np.cumsum(weights), common_uniform, side="right")),
            len(neighbours) - 1,
        )
        choice = int(neighbours[selected])
        return (
            bool(pool.holed[choice]),
            str(pool.next_lie[choice]),
            float(pool.next_distance[choice]),
            int(pool.cost[choice]),
            level,
        )


def _course_population_model(
    export_root: Path,
    event_keys: Mapping[int, str],
    holdout_year: int,
) -> tuple[markov.MarkovHoleModel, pd.DataFrame]:
    prior = pd.DataFrame(
        [
            {"year": int(year), "event_key": str(key)}
            for year, key in sorted(event_keys.items())
            if int(year) < int(holdout_year)
        ]
    )
    target_key = str(event_keys[int(holdout_year)])
    transitions, _, _ = markov.build_course_transitions(export_root, prior, target_key)
    transitions = transitions[transitions["category"].isin(["arg", "putt"])].copy()
    if transitions.empty:
        raise ValueError("Verified course has no strict-prior ARG/PUTT transitions")
    # Empty Tour indices make every missing exact/course state fail closed;
    # no other venue can enter the downstream transition model.
    empty = transitions.iloc[0:0].copy()
    model = markov.MarkovHoleModel(transitions, empty, min_exact=12, knn=64)
    return model, transitions


def _project_endpoint(
    current_endpoint: np.ndarray,
    pin_local: np.ndarray,
    next_distance_yd: float,
) -> np.ndarray:
    vector = np.asarray(current_endpoint, float) - np.asarray(pin_local, float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-9:
        vector = np.asarray([1.0, 0.0])
    else:
        vector /= norm
    return np.asarray(pin_local, float) + vector * max(float(next_distance_yd), 0.0)


def _continue_case_arm(
    frame: pd.DataFrame,
    *,
    app_sampler: spatial.SpatialAppSampler,
    tee_sampler: SpatialTeeSampler,
    state_sampler: PopulationTransitionSampler,
    hazard_resolver: hazard.EndpointLinkedHazardResolver,
    seed: int,
    max_transitions: int | None = None,
    max_hole_strokes: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Continue one player/round/hole/arm particle block to terminal states."""

    # Parameters remain only so old archive callers fail explicitly instead
    # of silently changing semantics.  The current policy is not configurable.
    if (
        max_transitions is not None
        and int(max_transitions) != tail_policy.MAX_CONTINUATION_TRANSITIONS
    ):
        raise ValueError(
            "max_transitions is fixed by the shared tail policy at "
            f"{tail_policy.MAX_CONTINUATION_TRANSITIONS}"
        )
    if max_hole_strokes is not None:
        raise ValueError(
            "Physical hole-score caps are forbidden; "
            "HIGH_SCORE_DIAGNOSTIC_STROKES is diagnostic only"
        )

    frame = frame.sort_values("particle_id").reset_index(drop=True)
    fixed_columns = [
        "event_name", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "par", "case_id", "arm", "forecast_pin_long_yd",
        "forecast_tee_x", "forecast_tee_y", "forecast_pin_x", "forecast_pin_y",
        "coordinate_frame_version", "coordinate_frame_reference_year",
    ]
    fixed = frame[fixed_columns].drop_duplicates()
    if len(fixed) != 1:
        raise ValueError("Case/arm metadata is not unique")
    record = fixed.iloc[0]
    if str(record["coordinate_frame_version"]) != EXPECTED_FRAME:
        raise ValueError("Initializer particle coordinate frame is not authoritative")
    n = len(frame)
    particle_ids = frame["particle_id"].to_numpy(int)
    if len(np.unique(particle_ids)) != n:
        raise ValueError("Duplicate particle IDs inside case/arm")
    frame_axis = CourseFrame(
        float(record["forecast_tee_x"]),
        float(record["forecast_tee_y"]),
        float(record["forecast_pin_x"]),
        float(record["forecast_pin_y"]),
    )
    tee_hazard_support_checker = _hazard_support_checker(
        hazard_resolver,
        frame_axis,
        hole_no=int(record["hole_no"]),
        round_no=int(record["round_no"]),
        category="ott",
    )
    app_hazard_support_checker = _hazard_support_checker(
        hazard_resolver,
        frame_axis,
        hole_no=int(record["hole_no"]),
        round_no=int(record["round_no"]),
        category="app",
    )
    pin_local = np.asarray([float(record["forecast_pin_long_yd"]), 0.0])
    # Round-trip checks make unit/frame mistakes fail before resolving a real
    # hazard. Coordinates are local yards in v2 and absolute feet in v2's
    # resolver.
    if not np.allclose(
        frame_axis.absolute_feet_to_local_yards(
            frame_axis.local_yards_to_absolute_feet(pin_local)
        ),
        pin_local,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("Local/absolute course-frame round trip failed")

    tee_endpoint = frame[["tee_end_long_yd", "tee_end_lat_yd"]].to_numpy(float)
    app_endpoint = frame[["app_end_long_yd", "app_end_lat_yd"]].to_numpy(float)
    app_attempted = np.isfinite(app_endpoint).all(axis=1)
    endpoints = tee_endpoint.copy()
    endpoints[app_attempted] = app_endpoint[app_attempted]
    lies = frame["tee_end_lie"].astype(str).to_numpy(object)
    lies[app_attempted] = frame.loc[app_attempted, "app_end_lie"].astype(str)
    lies = np.asarray([_normalize_lie(value) for value in lies], object)
    remaining = pd.to_numeric(frame["tee_remaining_yd"], errors="coerce").to_numpy(float)
    app_remaining = pd.to_numeric(frame["app_remaining_yd"], errors="coerce").to_numpy(float)
    app_geometric = pd.to_numeric(
        frame["app_geometric_remaining_yd"], errors="coerce"
    ).to_numpy(float)
    app_state_remaining = np.where(np.isfinite(app_remaining), app_remaining, app_geometric)
    remaining[app_attempted] = app_state_remaining[app_attempted]
    status = frame["chain_status"].astype(str).to_numpy(object)
    strokes = pd.to_numeric(frame["shots_taken"], errors="raise").to_numpy(int)
    terminal = frame["app_holed"].fillna(False).to_numpy(bool)
    terminal |= status == "app_complete_holed"
    invalid = np.zeros(n, dtype=bool)
    invalid |= ~np.isfinite(endpoints).all(axis=1)
    invalid |= ~np.isfinite(remaining) & ~terminal
    invalid |= np.isin(
        status,
        ["app_no_surface_support", "app_no_spatial_support", "tee_no_surface_support"],
    )
    terminal_lie = lies == "hole"
    terminal |= terminal_lie
    remaining[terminal] = 0.0
    hazard_count = np.zeros(n, dtype=np.int32)
    penalty_strokes = np.zeros(n, dtype=np.int32)
    transition_count = np.zeros(n, dtype=np.int32)
    return_to_app_projection_count = np.zeros(n, dtype=np.int32)
    hazard_rows: list[dict[str, object]] = []

    def uniforms(transition_index: int, stream: str) -> np.ndarray:
        return _common_uniforms(
            seed=seed,
            event_key=str(record["event_key"]),
            player_id=str(record["source_player_id"]),
            round_no=int(record["round_no"]),
            hole_no=int(record["hole_no"]),
            particle_ids=particle_ids,
            transition_index=transition_index,
            stream=stream,
        )

    for transition_index in range(
        tail_policy.MAX_CONTINUATION_TRANSITIONS
    ):
        active = ~(terminal | invalid)
        if not active.any():
            break
        transition_count[active] += 1

        spatial_hazard = active & np.isin(
            status, ["tee_hazard_unresolved", "app_hazard_unresolved"]
        )
        if spatial_hazard.any():
            common = uniforms(transition_index, "hazard")
            for position in np.flatnonzero(spatial_hazard):
                category = (
                    "ott" if status[position] == "tee_hazard_unresolved" else "app"
                )
                absolute_endpoint = frame_axis.local_yards_to_absolute_feet(
                    endpoints[position]
                )
                strokes_before = int(strokes[position])
                try:
                    resolved = hazard_resolver.resolve(
                        endpoint_x=float(absolute_endpoint[0]),
                        endpoint_y=float(absolute_endpoint[1]),
                        hole_no=int(record["hole_no"]),
                        round_no=int(record["round_no"]),
                        category=category,
                        common_uniform=float(common[position]),
                        forecast_pin_x=float(record["forecast_pin_x"]),
                        forecast_pin_y=float(record["forecast_pin_y"]),
                        physical_course_id=hazard_resolver.physical_course_id,
                    )
                except hazard.HazardResolutionUnavailable as exc:
                    hazard_rows.append(
                        {
                            "event": str(record["event_name"]),
                            "event_key": str(record["event_key"]),
                            "source_player_id": str(record["source_player_id"]),
                            "round_num": int(record["round_no"]),
                            "hole_no": int(record["hole_no"]),
                            "particle_id": int(particle_ids[position]),
                            "arm": str(record["arm"]),
                            "transition_index": int(transition_index),
                            "category": category,
                            "resolver_version": "hazard_resolution_failure",
                            "resolution_available": False,
                            "raw_lie": str(lies[position]),
                            "query_endpoint_x": float(absolute_endpoint[0]),
                            "query_endpoint_y": float(absolute_endpoint[1]),
                            "exception_type": type(exc).__name__,
                            "exception_text": str(exc),
                        }
                    )
                    invalid[position] = True
                    status[position] = f"hazard_unavailable:{type(exc).__name__}"
                    continue
                absolute_resolution = np.asarray(
                    [resolved.resolution_x, resolved.resolution_y], float
                )
                geometric_remaining = frame_axis.geometric_remaining_yd(
                    absolute_resolution
                )
                if not np.isclose(
                    geometric_remaining,
                    float(resolved.resolution_distance_yd),
                    atol=1e-8,
                    rtol=0.0,
                ):
                    raise AssertionError(
                        "Resolver distance is not geometric to this forecast pin"
                    )
                resolution_lie = _normalize_lie(resolved.resolution_lie)
                if resolution_lie == "unknown":
                    invalid[position] = True
                    status[position] = "hazard_resolution_unknown_lie"
                    continue
                # The causal physical shot was charged when its spatial endpoint
                # was drawn. Resolution adds only assessed penalty rows; the next
                # playable stroke is not charged until the next recursion step.
                # Official stroke advancement can exceed the number of explicit
                # PENALTY rows (for example provisional/replay bookkeeping), so
                # penalty_stroke_cost is the accounting field.
                strokes[position] = _charge_spatial_hazard(
                    strokes_before, int(resolved.penalty_stroke_cost)
                )
                penalty_strokes[position] += int(resolved.penalty_stroke_cost)
                if int(strokes[position] - strokes_before) != int(
                    resolved.penalty_stroke_cost
                ):
                    raise AssertionError("Hazard stroke-cost telescoping failed")
                hazard_count[position] += 1
                endpoints[position] = frame_axis.absolute_feet_to_local_yards(
                    absolute_resolution
                )
                remaining[position] = geometric_remaining
                lies[position] = resolution_lie
                status[position] = "hazard_resolved_playable_state"
                hazard_row = asdict(resolved)
                hazard_row.update(
                    {
                        "event": str(record["event_name"]),
                        "event_key": str(record["event_key"]),
                        "source_player_id": str(record["source_player_id"]),
                        "round_num": int(record["round_no"]),
                        "hole_no": int(record["hole_no"]),
                        "particle_id": int(particle_ids[position]),
                        "arm": str(record["arm"]),
                        "transition_index": int(transition_index),
                        "causal_physical_shot_already_charged": True,
                        "strokes_before_resolution": strokes_before,
                        "semantic_penalty_rows": int(resolved.penalty_count),
                        "penalty_strokes_charged": int(
                            resolved.penalty_stroke_cost
                        ),
                        "strokes_after_resolution": int(strokes[position]),
                        "recursive_distance_source": "resolution_xy_to_forecast_pin",
                    }
                )
                hazard_rows.append(hazard_row)

        active = ~(terminal | invalid)
        if not active.any():
            continue
        categories = np.full(n, "inactive", dtype=object)
        for position in np.flatnonzero(active):
            categories[position] = _state_category(
                str(lies[position]), float(remaining[position]), int(record["par"])
            )
        newly_terminal = active & (categories == "terminal")
        terminal[newly_terminal] = True
        status[newly_terminal] = "terminal"
        invalid_state = active & (categories == "invalid")
        invalid[invalid_state] = True
        status[invalid_state] = "invalid_state_category"

        ott = ~(terminal | invalid) & (categories == "ott")
        if ott.any():
            result = tee_sampler.sample(
                round_no=int(record["round_no"]),
                hole_no=int(record["hole_no"]),
                par=int(record["par"]),
                player_id=str(record["source_player_id"]),
                player_transport=str(record["arm"]) in {"player_tee", "both"},
                pin_long_yd=float(record["forecast_pin_long_yd"]),
                index_uniform=uniforms(transition_index, "tee_index"),
                lie_uniform=uniforms(transition_index, "tee_lie"),
                active=ott,
                hazard_support_checker=tee_hazard_support_checker,
            )
            attempted = ott & np.isfinite(result["endpoints"]).all(axis=1)
            strokes[attempted] += 1
            endpoints[attempted] = result["endpoints"][attempted]
            remaining[attempted] = result["remaining"][attempted]
            lies[attempted] = result["end_lies"][attempted]
            status[ott] = result["status"][ott]
            unsupported = ott & ~attempted
            invalid[unsupported] = True

        app_active = ~(terminal | invalid) & (categories == "app")
        if app_active.any():
            result = app_sampler.sample(
                endpoints,
                lies,
                remaining,
                pin_local,
                int(record["round_no"]),
                int(record["hole_no"]),
                int(record["par"]),
                str(record["source_player_id"]),
                str(record["arm"]) in {"player_app", "both"},
                {
                    "index": uniforms(transition_index, "app_index"),
                    "holeout": uniforms(transition_index, "app_holeout"),
                    "lie": uniforms(transition_index, "app_lie"),
                },
                active=app_active,
                hazard_support_checker=app_hazard_support_checker,
            )
            attempted = app_active & np.isfinite(result["endpoints"]).all(axis=1)
            strokes[attempted] += 1
            endpoints[attempted] = result["endpoints"][attempted]
            remaining[attempted] = np.where(
                np.isfinite(result["remaining"][attempted]),
                result["remaining"][attempted],
                result["geometric_remaining"][attempted],
            )
            lies[attempted] = result["end_lies"][attempted]
            status[app_active] = result["status"][app_active]
            terminal |= app_active & result["holed"]
            unsupported = app_active & ~attempted
            invalid[unsupported] = True

        for category in ("arg", "putt"):
            category_active = ~(terminal | invalid) & (categories == category)
            common = uniforms(transition_index, f"state_{category}")
            for position in np.flatnonzero(category_active):
                strokes_before = int(strokes[position])
                try:
                    holed, next_lie, next_distance, cost, level = state_sampler.sample(
                        round_no=int(record["round_no"]),
                        hole_no=int(record["hole_no"]),
                        category=category,
                        lie=str(lies[position]),
                        distance_yd=float(remaining[position]),
                        common_uniform=float(common[position]),
                    )
                except RuntimeError:
                    invalid[position] = True
                    status[position] = f"{category}_transition_unavailable"
                    continue
                if cost < 1:
                    raise AssertionError("Population transition cost must include the shot")
                strokes[position] += int(cost)
                if cost > 1:
                    hazard_count[position] += 1
                    penalty_strokes[position] += int(cost - 1)
                    hazard_rows.append(
                        {
                            "event": str(record["event_name"]),
                            "event_key": str(record["event_key"]),
                            "source_player_id": str(record["source_player_id"]),
                            "round_num": int(record["round_no"]),
                            "hole_no": int(record["hole_no"]),
                            "particle_id": int(particle_ids[position]),
                            "arm": str(record["arm"]),
                            "transition_index": int(transition_index),
                            "category": category,
                            "resolver_version": "markov_collapsed_population_transition",
                            "causal_physical_shot_already_charged": False,
                            "strokes_before_resolution": strokes_before,
                            "penalty_strokes_charged": int(cost - 1),
                            "strokes_after_resolution": int(strokes[position]),
                            "transition_cost_including_causal_shot": int(cost),
                            "backoff_level": level,
                        }
                    )
                if holed or _normalize_lie(next_lie) == "hole":
                    terminal[position] = True
                    lies[position] = "hole"
                    remaining[position] = 0.0
                    endpoints[position] = pin_local
                    status[position] = "terminal"
                    continue
                normalized = _normalize_lie(next_lie)
                if normalized == "unknown" or not np.isfinite(next_distance):
                    invalid[position] = True
                    status[position] = f"{category}_invalid_next_state"
                    continue
                lies[position] = normalized
                remaining[position] = max(float(next_distance), 0.0)
                endpoints[position] = _project_endpoint(
                    endpoints[position], pin_local, remaining[position]
                )
                if _state_category(
                    normalized, remaining[position], int(record["par"])
                ) == "app":
                    return_to_app_projection_count[position] += 1
                status[position] = f"{category}_population_transition"

    unresolved = ~(terminal | invalid)
    invalid[unresolved] = True
    status[unresolved] = tail_policy.TRANSITION_LOOP_STATUS
    complete = terminal & ~invalid
    # Validate terminal scores before the nullable diagnostic representation.
    terminal_strokes = tail_policy.int16_exact(
        strokes[complete], label="full-chain terminal hole strokes"
    )
    hazard_count_out = tail_policy.int16_exact(
        hazard_count, label="full-chain hazard counts"
    )
    penalty_strokes_out = tail_policy.int16_exact(
        penalty_strokes, label="full-chain penalty strokes"
    )
    transition_count_out = tail_policy.int16_exact(
        transition_count, label="full-chain transition counts"
    )
    return_to_app_out = tail_policy.int16_exact(
        return_to_app_projection_count,
        label="full-chain return-to-APP counts",
    )
    hole_strokes = np.full(n, np.nan, dtype=float)
    hole_strokes[complete] = terminal_strokes
    result = pd.DataFrame(
        {
            "event_name": str(record["event_name"]),
            "shotlink_event_key": str(record["event_key"]),
            "source_player_id": str(record["source_player_id"]),
            "player_name": normalise_player(record["player_name"]),
            "round_num": int(record["round_no"]),
            "hole_no": int(record["hole_no"]),
            "par": int(record["par"]),
            "case_id": int(record["case_id"]),
            "arm": str(record["arm"]),
            "particle_id": particle_ids,
            "hole_strokes": hole_strokes,
            "terminal": complete,
            "chain_status": np.where(complete, "terminal", status),
            "high_score_diagnostic": complete & tail_policy.high_score_diagnostic_mask(
                strokes
            ),
            "hazard_involved": hazard_count_out > 0,
            "hazard_transitions": hazard_count_out,
            "penalty_strokes": penalty_strokes_out,
            "transition_count": transition_count_out,
            "return_to_app_radial_projections": return_to_app_out,
            "coordinate_frame_version": str(record["coordinate_frame_version"]),
            "coordinate_frame_reference_year": int(
                record["coordinate_frame_reference_year"]
            ),
            "final_lie": lies,
            "final_remaining_yd": remaining,
        }
    )
    return result, hazard_rows


def continue_event_particles(
    initial_particles: pd.DataFrame,
    *,
    course_drives: pd.DataFrame,
    course_app: pd.DataFrame,
    tee_profiles: pd.DataFrame,
    app_profiles: pd.DataFrame,
    state_model: markov.MarkovHoleModel,
    hazard_resolver: hazard.EndpointLinkedHazardResolver,
    seed: int,
    max_transitions: int | None = None,
    max_hole_strokes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = {
        "case_id", "arm", "particle_id", "forecast_tee_x", "forecast_tee_y",
        "forecast_pin_x", "forecast_pin_y", "coordinate_frame_version",
        "coordinate_frame_reference_year",
    }
    if missing := required.difference(initial_particles.columns):
        raise ValueError(f"Initializer particles miss {sorted(missing)}")
    frame_versions = set(initial_particles["coordinate_frame_version"].astype(str))
    if frame_versions != {EXPECTED_FRAME}:
        raise ValueError(f"Unexpected particle coordinate frames: {frame_versions}")
    tee_sampler = SpatialTeeSampler(
        course_drives,
        tee_profiles,
        enforce_empirical_class_support=True,
    )
    app_sampler = spatial.SpatialAppSampler(
        course_app,
        app_profiles,
        enforce_empirical_class_support=True,
    )
    state_sampler = PopulationTransitionSampler(state_model)
    hole_parts: list[pd.DataFrame] = []
    hazard_rows: list[dict[str, object]] = []
    grouped = initial_particles.groupby(["case_id", "arm"], observed=True, sort=True)
    for group_index, (_, group) in enumerate(grouped, start=1):
        if group_index % 250 == 0:
            print(f"Continued {group_index:,}/{grouped.ngroups:,} case/arm blocks")
        holes, hazards = _continue_case_arm(
            group,
            app_sampler=app_sampler,
            tee_sampler=tee_sampler,
            state_sampler=state_sampler,
            hazard_resolver=hazard_resolver,
            seed=seed,
            max_transitions=max_transitions,
            max_hole_strokes=max_hole_strokes,
        )
        hole_parts.append(holes)
        hazard_rows.extend(hazards)
    holes = pd.concat(hole_parts, ignore_index=True)
    hazards = pd.DataFrame(hazard_rows)
    diagnostics = {
        "case_arm_blocks": int(grouped.ngroups),
        "hole_particles": int(len(holes)),
        "terminal_rate": float(holes["terminal"].mean()),
        "hazard_hole_rate": float(holes["hazard_involved"].mean()),
        "penalty_strokes": int(holes["penalty_strokes"].sum()),
        "high_score_diagnostic_holes": int(
            holes["high_score_diagnostic"].sum()
        ),
        "tail_policy": tail_policy.manifest_policy(),
        "return_to_app_radial_projections": int(
            holes["return_to_app_radial_projections"].sum()
        ),
        "arg_putt_backoff_counts": dict(state_sampler.backoff_counts),
    }
    return holes, hazards, diagnostics


def _aggregate_round_particles(
    holes: pd.DataFrame,
    *,
    frozen_event_key: str,
    input_cutoff: str,
    simulations: int,
) -> pd.DataFrame:
    if not holes["terminal"].all():
        counts = holes.loc[~holes["terminal"], "chain_status"].value_counts().to_dict()
        raise ChainIncomplete(f"Cannot emit round particles; incomplete holes: {counts}")
    if not np.allclose(
        holes["hole_strokes"], np.rint(holes["hole_strokes"]), atol=1e-8, rtol=0.0
    ):
        raise AssertionError("Terminal hole strokes are not integer")
    holes = holes.copy()
    holes["hole_strokes"] = tail_policy.int16_exact(
        np.rint(holes["hole_strokes"]).to_numpy(),
        label="full-chain terminal hole strokes",
    )
    group_keys = [
        "event_name", "source_player_id", "player_name", "round_num", "arm",
        "particle_id",
    ]
    rounds = holes.groupby(group_keys, observed=True, as_index=False).agg(
        round_strokes=("hole_strokes", "sum"),
        round_par=("par", "sum"),
        terminal_holes=("terminal", "sum"),
        hazard_involved=("hazard_involved", "max"),
        hazard_transitions=("hazard_transitions", "sum"),
        penalty_strokes=("penalty_strokes", "sum"),
        return_to_app_radial_projections=(
            "return_to_app_radial_projections", "sum"
        ),
    )
    if not rounds["terminal_holes"].eq(18).all():
        raise ChainIncomplete("Round aggregation did not contain 18 terminal holes")
    rounds["round_strokes"] = tail_policy.round_totals_int16(
        rounds["round_strokes"].to_numpy(),
        label="full-chain round totals",
    )
    arm_key_columns = ["source_player_id", "round_num", "particle_id"]
    arm_key_sets = {
        str(arm_name): set(
            map(tuple, arm_frame[arm_key_columns].astype(object).to_numpy())
        )
        for arm_name, arm_frame in rounds.groupby("arm", observed=True)
    }
    if set(arm_key_sets) != set(ARMS):
        raise ValueError(f"Candidate arms are incomplete: {sorted(arm_key_sets)}")
    reference_keys = arm_key_sets[ARMS[0]]
    if any(keys != reference_keys for keys in arm_key_sets.values()):
        raise ValueError("Candidate arms do not share the exact player/round/particle roster")
    # Independently exercise the calibration adapter's canonical 4-D hole
    # aggregation and compare it with the long-form groupby result.
    for arm_name, arm_holes in holes.groupby("arm", observed=True):
        players = sorted(arm_holes["source_player_id"].astype(str).unique())
        particles = sorted(arm_holes["particle_id"].astype(int).unique())
        expected = pd.MultiIndex.from_product(
            [players, [1, 2, 3, 4], range(1, 19), particles],
            names=["source_player_id", "round_num", "hole_no", "particle_id"],
        )
        indexed = arm_holes.set_index(
            ["source_player_id", "round_num", "hole_no", "particle_id"]
        )["hole_strokes"].reindex(expected)
        if indexed.isna().any():
            raise ChainIncomplete(f"{arm_name} has an incomplete 4x18 particle cube")
        tensor = indexed.to_numpy(int).reshape(
            len(players), 4, 18, len(particles)
        )
        adapter_rounds = aggregate_hole_strokes(tensor)
        lookup = {
            (player, round_no, particle): int(adapter_rounds[pi, round_no - 1, pj])
            for pi, player in enumerate(players)
            for round_no in range(1, 5)
            for pj, particle in enumerate(particles)
        }
        compare = rounds[rounds["arm"].eq(arm_name)]
        expected_values = np.asarray(
            [
                lookup[(str(row.source_player_id), int(row.round_num), int(row.particle_id))]
                for row in compare.itertuples(index=False)
            ],
            int,
        )
        if not np.array_equal(compare["round_strokes"].to_numpy(int), expected_values):
            raise AssertionError("Long and adapter round aggregation disagree")

    rounds = rounds.rename(columns={"event_name": "event"})
    rounds["event_key"] = str(frozen_event_key)
    rounds["input_cutoff"] = str(input_cutoff)
    rounds["chain_version"] = CHAIN_VERSION
    rounds["particle_weight"] = 1.0 / int(simulations)
    required_order = [
        "event_key", "event", "source_player_id", "player_name", "round_num",
        "particle_id", "round_strokes", "round_par", "terminal_holes",
        "input_cutoff", "chain_version",
    ]
    optional = [
        "arm", "particle_weight", "hazard_involved", "hazard_transitions",
        "penalty_strokes",
        "return_to_app_radial_projections",
    ]
    output = rounds[required_order + optional].copy()
    if set(COMPLETE_ROUND_COLUMNS).difference(output.columns):
        raise AssertionError("Round output drifted from evaluator schema")
    return output


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{float(value):.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _hazard_summary(hazards: pd.DataFrame) -> pd.DataFrame:
    if hazards.empty:
        return pd.DataFrame(
            columns=[
                "arm", "category", "hazard_transitions", "penalty_strokes",
                "fallback_resolutions", "mean_endpoint_distance_yd",
            ]
        )
    frame = hazards.copy()
    frame["penalty_strokes_charged"] = pd.to_numeric(
        frame.get("penalty_strokes_charged", 0), errors="coerce"
    ).fillna(0)
    frame["endpoint_distance_yd"] = pd.to_numeric(
        frame.get("endpoint_distance_yd", np.nan), errors="coerce"
    )
    fallback = frame.get("fallback_used", pd.Series(False, index=frame.index))
    frame["fallback_used"] = fallback.fillna(False).astype(bool)
    return frame.groupby(["arm", "category"], observed=True, as_index=False).agg(
        hazard_transitions=("particle_id", "size"),
        penalty_strokes=("penalty_strokes_charged", "sum"),
        fallback_resolutions=("fallback_used", "sum"),
        mean_endpoint_distance_yd=("endpoint_distance_yd", "mean"),
    )


def _audit_hazard_accounting(hazards: pd.DataFrame) -> dict[str, object]:
    if hazards.empty:
        return {
            "spatial_hazard_rows": 0,
            "resolution_failure_rows": 0,
            "same_hole_fallback_rows": 0,
            "telescoping_failures": 0,
            "provisional_resolutions": 0,
            "multi_stroke_penalty_resolutions": 0,
        }
    frame = hazards.copy()
    spatial_mask = frame["resolver_version"].astype(str).str.startswith(
        "endpoint_linked_hazard"
    )
    failure_rows = int(
        frame["resolver_version"].astype(str).eq("hazard_resolution_failure").sum()
    )
    spatial_rows = frame[spatial_mask].copy()
    for column in (
        "strokes_before_resolution", "strokes_after_resolution",
        "penalty_stroke_cost",
    ):
        spatial_rows[column] = pd.to_numeric(spatial_rows[column], errors="coerce")
    if spatial_rows[
        ["strokes_before_resolution", "strokes_after_resolution", "penalty_stroke_cost"]
    ].isna().any().any():
        raise AssertionError("Spatial hazard accounting fields are incomplete")
    charged = (
        spatial_rows["strokes_after_resolution"]
        - spatial_rows["strokes_before_resolution"]
    )
    telescoping_failures = int(
        (~np.isclose(charged, spatial_rows["penalty_stroke_cost"], atol=0.0)).sum()
    )
    if telescoping_failures:
        raise AssertionError(
            f"Spatial hazard accounting has {telescoping_failures} telescoping failures"
        )
    if not spatial_rows["causal_physical_shot_already_charged"].fillna(False).all():
        raise AssertionError("Spatial hazard rows did not mark causal shot as pre-charged")
    event_types = set(spatial_rows["resolution_event_type"].dropna().astype(str))
    if not event_types.issubset({"STROKE", "PROVISIONAL"}):
        raise AssertionError(f"Unexpected hazard resolution event types: {event_types}")
    geometry_error = np.abs(
        pd.to_numeric(spatial_rows["resolution_distance_yd"], errors="coerce")
        - np.hypot(
            pd.to_numeric(spatial_rows["resolution_x"], errors="coerce")
            - pd.to_numeric(spatial_rows["forecast_pin_x"], errors="coerce"),
            pd.to_numeric(spatial_rows["resolution_y"], errors="coerce")
            - pd.to_numeric(spatial_rows["forecast_pin_y"], errors="coerce"),
        ) / 3.0
    )
    if len(geometry_error) and float(geometry_error.max()) > 1e-8:
        raise AssertionError("Spatial hazard recursive distance is not forecast-pin geometry")
    return {
        "spatial_hazard_rows": int(len(spatial_rows)),
        "resolution_failure_rows": failure_rows,
        "same_hole_fallback_rows": int(
            spatial_rows.get(
                "fallback_used", pd.Series(False, index=spatial_rows.index)
            ).fillna(False).astype(bool).sum()
        ),
        "telescoping_failures": telescoping_failures,
        "geometry_distance_max_error_yd": (
            float(geometry_error.max()) if len(geometry_error) else 0.0
        ),
        "provisional_resolutions": int(
            spatial_rows["resolution_event_type"].eq("PROVISIONAL").sum()
        ),
        "multi_stroke_penalty_resolutions": int(
            spatial_rows["penalty_stroke_cost"].gt(1).sum()
        ),
        "semantic_count_below_official_cost": int(
            (
                pd.to_numeric(spatial_rows["penalty_count"], errors="coerce")
                < spatial_rows["penalty_stroke_cost"]
            ).sum()
        ),
    }


def _run_internal_contract_tests() -> dict[str, object]:
    """Exercise the decision-critical multi-stroke/provisional semantics."""

    if _charge_spatial_hazard(1, 2) != 3:
        raise AssertionError("Multi-stroke hazard charge self-test failed")
    projected = _project_endpoint(
        np.asarray([95.0, 0.0]), np.asarray([100.0, 0.0]), 60.0
    )
    if not np.isclose(np.linalg.norm(projected - np.asarray([100.0, 0.0])), 60.0):
        raise AssertionError("State-only endpoint projection self-test failed")
    if _state_category("rough", 60.0, 4) != "app":
        raise AssertionError("Projected >50yd state did not route back to APP")
    synthetic = pd.DataFrame(
        [
            {
                "resolver_version": hazard.RESOLVER_VERSION,
                "strokes_before_resolution": 1,
                "strokes_after_resolution": 3,
                "penalty_count": 1,
                "penalty_stroke_cost": 2,
                "causal_physical_shot_already_charged": True,
                "resolution_event_type": "PROVISIONAL",
                "resolution_distance_yd": 5.0,
                "resolution_x": 15.0,
                "resolution_y": 0.0,
                "forecast_pin_x": 0.0,
                "forecast_pin_y": 0.0,
            }
        ]
    )
    audit = _audit_hazard_accounting(synthetic)
    expected = {
        "telescoping_failures": 0,
        "provisional_resolutions": 1,
        "multi_stroke_penalty_resolutions": 1,
        "semantic_count_below_official_cost": 1,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise AssertionError(
                f"Hazard accounting self-test {key}={audit.get(key)} != {value}"
            )
    masked = spatial._supported_surface_probabilities(
        np.asarray([[0.70, 0.20, 0.05, 0.05], [0.70, 0.20, 0.05, 0.05]]),
        np.asarray([0, 1, 1]),
        class_count=4,
        enforce_empirical_class_support=True,
        endpoints=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        hazard_class_indices=(1,),
        hazard_support_checker=lambda points: np.asarray([False, True]),
    )
    if not np.allclose(masked[0], [1.0, 0.0, 0.0, 0.0]):
        raise AssertionError("Unsupported hazard probability was not masked")
    if not np.allclose(masked[1], [0.70 / 0.90, 0.20 / 0.90, 0.0, 0.0]):
        raise AssertionError("Supported empirical hazard probability changed")
    if _normalize_lie("other") != "recovery":
        raise AssertionError("Non-penalty APP other lie did not route to recovery")
    return {
        "passed": True,
        "return_to_app_projection_test": True,
        "surface_support_mask_test": True,
        "app_other_nonpenalty_route_test": True,
        **expected,
    }


def _write_event_report(
    path: Path,
    *,
    event_name: str,
    roster_meta: Mapping[str, object],
    provenance: Mapping[str, object],
    transition_rows: int,
    initial_particles: pd.DataFrame,
    holes: pd.DataFrame,
    rounds: pd.DataFrame,
    hazards: pd.DataFrame,
    diagnostics: Mapping[str, object],
) -> None:
    arm_summary = rounds.groupby("arm", observed=True, as_index=False).agg(
        round_particles=("round_strokes", "size"),
        players=("source_player_id", "nunique"),
        mean_strokes=("round_strokes", "mean"),
        sd_strokes=("round_strokes", "std"),
        hazard_round_rate=("hazard_involved", "mean"),
        penalty_strokes=("penalty_strokes", "sum"),
    )
    hazard_summary = _hazard_summary(hazards)
    initial_status = (
        initial_particles.groupby(["arm", "chain_status"], observed=True)
        .size().rename("particles").reset_index()
    )
    incomplete = (
        holes.loc[~holes["terminal"], "chain_status"].value_counts()
        .rename_axis("chain_status").rename("particles").reset_index()
    )
    path.write_text(
        f"""# Complete spatial-chain feasibility: {event_name}

This is a **research feasibility output**, not a validated pricing model.

- Pre-event R1 roster: **{roster_meta['selected_players']}** players, each simulated for 4 rounds x 18 holes
- Initial spatial particles: **{len(initial_particles):,}**
- Terminal hole particles: **{int(holes['terminal'].sum()):,}/{len(holes):,}**
- Strict-prior ARG/PUTT transition rows: **{transition_rows:,}**
- Coordinate frame: **{provenance['coordinate_frame_version']}**
- Reference setup: **{provenance['setup_reference_year']}** (< {provenance['holdout_year']})
- Alignment count: **{provenance['alignment_count']}**
- Candidate input cutoff: **{roster_meta['input_cutoff']}**
- Cutoff semantics: **verified producer provenance; not inferred from CSV contents**
- Player-profile fit window: **2020-2025**, enforced by strict `season < holdout` producer code
- Tee/APP profile coverage: **{roster_meta['tee_profile_players']}/{roster_meta['selected_players']}** / **{roster_meta['app_profile_players']}/{roster_meta['selected_players']}**
- Heldout geometry role: **{provenance.get('heldout_geometry_role', 'unspecified')}**
- ARG/PUTT state-only endpoints projected back into recursive APP: **{int(holes['return_to_app_radial_projections'].sum()):,}**
- Generative lie support: **zero-empirical classes masked; every sampled spatial hazard pre-cleared by the resolver**
- Hazard fallback: **same physical hole and category only, 120-150 yd**

## Round particles

{_markdown_table(arm_summary)}

## Hazard stroke accounting

The spatial causal shot is charged when its endpoint is drawn. The resolver
then adds only `penalty_stroke_cost`; its returned STROKE/PROVISIONAL is the
next uncharged playable shot. Recursive distance is always recomputed from
resolution XY to this forecast pin. Common uniforms exclude arm from their
key and include event/player/round/hole/particle/transition index.
APP `other` is an ordinary catch-all lie pooled into recovery, never an
assessed penalty. The table reports every same-hole fallback explicitly.

{_markdown_table(hazard_summary)}

## Initial chain statuses

{_markdown_table(initial_status, digits=0)}

## Incomplete states

{_markdown_table(incomplete, digits=0)}

## Interpretation

ARG and putting are population-only exact-physical-course transitions. APP is
called recursively whenever the generated lie/distance routes to APP; the
coarse initializer status `app_ready_arg` is never treated as a category.
Round means shown above are raw generative means. The frozen evaluator joins
truth/dated skill independently and applies the discrete mean/shared-latent
adapter before any distribution comparison.

Diagnostics: `{json.dumps(dict(diagnostics), sort_keys=True)}`
""",
        encoding="utf-8",
    )


def run_event(
    *,
    event_name: str,
    args: argparse.Namespace,
    tee_profiles: pd.DataFrame,
    app_profiles: pd.DataFrame,
    event_index: int,
) -> dict[str, object]:
    print(f"Preparing full strict-prior chain: {event_name}")
    prepared = spatial.prepare_event_foundations(
        args.export_root,
        event_name,
        args.holdout_year,
        args.course_start_year,
        include_scoring_targets=False,
    )
    provenance = dict(prepared["provenance"])
    _validate_provenance(provenance, args.holdout_year)
    event_keys = dict(prepared["event_keys"])
    roster, roster_meta = _load_pre_event_roster(
        args.baseline_cases,
        event_name,
        args.max_players,
        args.roster_seed + event_index * 101,
    )
    selected_ids = set(roster["source_player_id"].astype(str))
    tee_profile_ids = set(tee_profiles["source_player_id"].astype(str))
    app_profile_ids = set(app_profiles["source_player_id"].astype(str))
    roster_meta["tee_profile_players"] = int(len(selected_ids & tee_profile_ids))
    roster_meta["app_profile_players"] = int(len(selected_ids & app_profile_ids))
    roster_meta["tee_profile_coverage"] = float(
        len(selected_ids & tee_profile_ids) / max(len(selected_ids), 1)
    )
    roster_meta["app_profile_coverage"] = float(
        len(selected_ids & app_profile_ids) / max(len(selected_ids), 1)
    )
    grid = _build_complete_grid(
        event_name,
        str(event_keys[args.holdout_year]),
        prepared["target_raw"],
        prepared["forecast"],
        roster,
    )
    print(
        f"{event_name}: {grid['source_player_id'].nunique()} pre-event R1 players, "
        f"{len(grid):,} hole cases, {args.simulations} particles, {len(ARMS)} arms"
    )
    donors, heldout_hazards, identity, hazard_load = hazard.load_normalized_hazard_panel(
        args.export_root,
        event_name,
        args.holdout_year,
        args.course_start_year,
    )
    del heldout_hazards, identity
    resolver = hazard.EndpointLinkedHazardResolver(
        donors,
        same_hole_fallback_max_distance_yd=150.0,
    )
    if int(resolver.coordinate_reference_year) != int(provenance["setup_reference_year"]):
        raise ValueError("Hazard and spatial coordinate reference years disagree")
    expected_reference_key = str(event_keys[int(provenance["setup_reference_year"])])
    if str(resolver.coordinate_reference_event_key) != expected_reference_key:
        raise ValueError("Hazard and spatial coordinate reference event keys disagree")
    initial = spatial.simulate_forecast_grid(
        grid,
        prepared["course_drives"],
        prepared["course_app"],
        tee_profiles,
        app_profiles,
        args.simulations,
        args.seed + event_index * 1000,
        enforce_empirical_class_support=True,
        hazard_support_factory=_forecast_hazard_support_factory(resolver),
    )
    coordinate_years = set(initial["coordinate_frame_reference_year"].astype(int))
    if coordinate_years != {int(provenance["setup_reference_year"])}:
        raise ValueError("Initializer reference year disagrees with preparation provenance")
    state_model, transitions = _course_population_model(
        args.export_root, event_keys, args.holdout_year
    )
    holes, hazards, diagnostics = continue_event_particles(
        initial,
        course_drives=prepared["course_drives"],
        course_app=prepared["course_app"],
        tee_profiles=tee_profiles,
        app_profiles=app_profiles,
        state_model=state_model,
        hazard_resolver=resolver,
        seed=args.seed + event_index * 1000 + 500,
    )
    diagnostics["hazard_loader"] = hazard_load
    diagnostics["hazard_accounting"] = _audit_hazard_accounting(hazards)
    slug = _slug(event_name)
    event_dir = args.output_dir / slug
    event_dir.mkdir(parents=True, exist_ok=True)
    hole_path = event_dir / "hole_particles.parquet"
    hazard_path = event_dir / "hazard_accounting.parquet"
    initial_path = event_dir / "initializer_particles.parquet"
    holes.to_parquet(hole_path, index=False)
    hazards.to_parquet(hazard_path, index=False)
    if args.write_initial:
        initial.to_parquet(initial_path, index=False)
    # Fail closed after preserving diagnostic particles. No scorer-ready round
    # file is emitted from an incomplete chain.
    if not holes["terminal"].all():
        incomplete = holes.loc[~holes["terminal"], "chain_status"].value_counts()
        (event_dir / "INCOMPLETE.json").write_text(
            json.dumps(incomplete.to_dict(), indent=2), encoding="utf-8"
        )
        raise ChainIncomplete(
            f"{event_name}: {int((~holes['terminal']).sum())} incomplete hole particles"
        )
    rounds = _aggregate_round_particles(
        holes,
        frozen_event_key=str(roster_meta["event_key"]),
        input_cutoff=str(roster_meta["input_cutoff"]),
        simulations=args.simulations,
    )
    round_path = event_dir / "complete_round_particles.parquet"
    rounds.to_parquet(round_path, index=False)
    # Exercise the exact persisted evaluator loader, not just an in-memory
    # schema assertion.
    loaded = load_complete_round_particles(round_path)
    if len(loaded) != len(rounds):
        raise AssertionError("Evaluator loader changed complete round row count")
    provenance_payload = {
        **provenance,
        # The v2 initializer deliberately leaves penalty paths unresolved.
        # This full-chain stage has now attached the strict-prior v3 resolver,
        # charged its stroke cost, and passed the accounting/geometry audit.
        "penalty_resolution_ready": True,
        "penalty_resolution_stage": "full_chain_endpoint_linked_v3",
        "hazard_coordinate_reference_year": int(resolver.coordinate_reference_year),
        "hazard_coordinate_reference_event_key": resolver.coordinate_reference_event_key,
        "hazard_resolver_version": hazard.RESOLVER_VERSION,
        "hazard_generation_support_policy": (
            "zero empirical lie classes masked; hazards require resolver support"
        ),
        "hazard_primary_radius_yd": float(resolver.max_nearest_distance_yd),
        "hazard_same_hole_fallback_radius_yd": (
            float(resolver.same_hole_fallback_max_distance_yd)
            if resolver.same_hole_fallback_max_distance_yd is not None
            else None
        ),
        "app_other_is_penalty": False,
        "chain_version": CHAIN_VERSION,
        "profile_fit_window": {
            "start_year": int(args.course_start_year),
            "end_year": int(args.holdout_year - 1),
            "producer_policy": "strict season < holdout_year",
            "cutoff_is_content_derived": False,
        },
        "input_hashes_sha256": _input_hashes(args),
        "complete_round_parquet_sha256": sha256_file(round_path),
        "random_seeds": {
            "base_seed": int(args.seed),
            "roster_base_seed": int(args.roster_seed),
            "event_index": int(event_index),
            "roster_seed_used": int(args.roster_seed + event_index * 101),
            "initializer_seed_used": int(args.seed + event_index * 1000),
            "continuation_seed_used": int(args.seed + event_index * 1000 + 500),
        },
        "common_random_key": (
            "seed|event|source_player_id|round|hole|transition|stream|particle; arm excluded"
        ),
        "tail_policy": tail_policy.manifest_policy(),
        "roster": roster_meta,
    }
    (event_dir / "provenance.json").write_text(
        json.dumps(provenance_payload, indent=2), encoding="utf-8"
    )
    _write_event_report(
        event_dir / "REPORT.md",
        event_name=event_name,
        roster_meta=roster_meta,
        provenance=provenance,
        transition_rows=len(transitions),
        initial_particles=initial,
        holes=holes,
        rounds=rounds,
        hazards=hazards,
        diagnostics=diagnostics,
    )
    print(f"Wrote {round_path} ({len(rounds):,} evaluator-ready rows)")
    return {
        "event": event_name,
        "event_dir": str(event_dir),
        "players": int(rounds["source_player_id"].nunique()),
        "round_rows": int(len(rounds)),
        "hole_rows": int(len(holes)),
        "hazard_rows": int(len(hazards)),
        "terminal_rate": float(holes["terminal"].mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-cases", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--event-name", action="append")
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--roster-seed", type=int, default=20260824)
    parser.add_argument("--tee-profiles-csv", type=Path, default=DEFAULT_TEE_PROFILES)
    parser.add_argument("--app-profiles-csv", type=Path, default=DEFAULT_APP_PROFILES)
    parser.add_argument("--write-initial", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    internal_contract_tests = _run_internal_contract_tests()
    if args.smoke_test:
        args.simulations = min(args.simulations, 24)
        args.max_players = min(args.max_players, 2)
        args.event_name = [
            (args.event_name or ["Wyndham Championship"])[0]
        ]
        args.output_dir = Path(args.output_dir)
    events = list(args.event_name or DEFAULT_EVENTS)
    for path in (
        args.baseline_cases,
        args.tee_profiles_csv,
        args.app_profiles_csv,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tee_profiles = pd.read_csv(
        args.tee_profiles_csv, dtype={"source_player_id": str}
    )
    app_profiles = pd.read_csv(
        args.app_profiles_csv, dtype={"source_player_id": str}
    )
    summaries: list[dict[str, object]] = []
    for event_index, event_name in enumerate(events):
        summaries.append(
            run_event(
                event_name=event_name,
                args=args,
                tee_profiles=tee_profiles,
                app_profiles=app_profiles,
                event_index=event_index,
            )
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "event_summary.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "chain_version": CHAIN_VERSION,
                "events": events,
                "simulations": int(args.simulations),
                "max_players": int(args.max_players),
                "arms": list(ARMS),
                "research_only": True,
                "production_touched": False,
                "random_seeds": {
                    "base_seed": int(args.seed),
                    "roster_base_seed": int(args.roster_seed),
                    "event_offsets": {
                        event: {
                            "event_index": int(index),
                            "roster_seed_used": int(args.roster_seed + index * 101),
                            "initializer_seed_used": int(args.seed + index * 1000),
                            "continuation_seed_used": int(args.seed + index * 1000 + 500),
                        }
                        for index, event in enumerate(events)
                    },
                },
                "internal_contract_tests": internal_contract_tests,
                "profile_fit_window": {
                    "start_year": int(args.course_start_year),
                    "end_year": int(args.holdout_year - 1),
                    "producer_policy": "strict season < holdout_year",
                    "cutoff_is_content_derived": False,
                },
                "input_hashes_sha256": _input_hashes(args),
                "complete_round_parquet_sha256": {
                    str(path.parent.name): sha256_file(path)
                    for path in args.output_dir.glob(
                        "*/complete_round_particles.parquet"
                    )
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

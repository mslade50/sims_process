"""Identity-preserving trace seam for archive-only ARG/PUTT transitions.

This module deliberately wraps, rather than modifies, the frozen research
implementations in :mod:`hole_sim_markov` and
:mod:`hole_sim_batched_population_transition`.  It provides three pieces that
the fast sampler intentionally omits:

* reconstruction of the strict-prior course transition frame while retaining
  the causal ShotLink identity columns;
* a fail-closed mapping from every Markov ``TransitionPool`` row back to the
  corresponding source transition row; and
* an aligned batch trace containing the selected donor, its exact normalized
  dynamic kNN mass, and the CDF interval used by the existing sampler.

The wrapped ``BatchTransitionResult`` remains authoritative.  Every traced
choice is checked for exact parity on holed status, next lie, next distance,
cost, and hierarchy level before it is returned.  This is research/archive
code; no production or streaming module imports it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_batched_population_transition as batched  # noqa: E402
import hole_sim_markov as markov  # noqa: E402


TRACE_IDENTITY_COLUMNS = (
    "transition_row_id",
    "event_key",
    "source_player_id",
    "player_name",
    "round_no",
    "hole_no",
    "stroke_no",
    "shot_key",
)

POOL_VALUE_COLUMNS = (
    "start_distance",
    "next_distance",
    "next_lie",
    "sg",
    "cost",
    "holed",
    "sg_weight_valid",
)

MARKOV_TRANSITION_COLUMNS = (
    "event_key",
    "source_player_id",
    "round_no",
    "hole_no",
    "category",
    "lie",
    "bucket",
    *POOL_VALUE_COLUMNS,
)

_INDEX_SPECS = (
    ("exact", "course", ("round_no", "hole_no", "category", "lie", "bucket")),
    ("exact_lie", "course", ("round_no", "hole_no", "category", "lie")),
    ("course", "course", ("category", "lie", "bucket")),
    ("course_relaxed", "course", ("category", "lie")),
    ("tour", "tour", ("category", "lie", "bucket")),
    ("tour_relaxed", "tour", ("category", "lie")),
)


class TransitionTraceParityError(AssertionError):
    """Raised when trace reconstruction is not exactly aligned to Markov."""


def _tuple_key(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_key(value: Sequence[object]) -> str:
    return json.dumps(
        [_json_scalar(item) for item in value],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def reconstruct_strict_prior_course_transitions(
    export_root: Path,
    prior_events: pd.DataFrame,
    target_event_key: str,
) -> pd.DataFrame:
    """Rebuild Markov's course frame and retain causal source identity.

    The transformations and row ordering intentionally mirror
    ``hole_sim_markov.build_course_transitions``.  ``player_name``,
    ``stroke_no``, ``shot_key``, and a stable pre-filter row identifier are
    carried alongside the original Markov columns.  Only seasons before the
    target season are accepted, which makes the strict-prior claim fail closed
    for this recurring-course use case.
    """

    _require_columns(prior_events, ("year", "event_key"), "prior_events")
    target_year, _ = markov.parse_event_key(str(target_event_key))
    prior = prior_events.loc[:, ["year", "event_key"]].copy()
    prior["year"] = pd.to_numeric(prior["year"], errors="raise").astype(int)
    prior["event_key"] = prior["event_key"].astype(str)
    if prior.empty:
        raise ValueError("prior_events cannot be empty")
    if prior["event_key"].duplicated().any():
        raise ValueError("prior_events contains duplicate event_key rows")
    if (prior["year"] >= int(target_year)).any():
        offenders = prior.loc[prior["year"] >= int(target_year), "event_key"].tolist()
        raise ValueError(
            "Strict-prior course transitions require season < target season; "
            f"offenders={offenders}"
        )

    keys = set(prior["event_key"])
    transition_parts: list[pd.DataFrame] = []
    for year in sorted(set(prior["year"].astype(int))):
        raw = markov.read_raw_year(export_root, int(year), markov.RAW_COLUMNS)
        if raw.empty:
            continue
        raw = raw[raw["event_key"].isin(keys)].copy()
        if raw.empty:
            continue
        raw["source_season"] = int(year)
        raw["source_raw_row"] = raw.index.to_numpy()

        sg = markov.read_sg_year(export_root, int(year), markov.SG_COLUMNS)
        sg = sg[sg["event_key"].isin(keys)].copy()
        stroke = raw[raw["event_type"].eq("STROKE")].copy()
        stroke = stroke.merge(
            sg[
                [
                    "shot_key",
                    "player_name",
                    "sg_category",
                    "start_distance_yd",
                    "sg_raw",
                    "sg_valid",
                ]
            ],
            on="shot_key",
            how="left",
            validate="one_to_one",
        )
        group_cols = ["event_key", "source_player_id", "round_no", "hole_no"]
        stroke = stroke.sort_values(group_cols + ["stroke_no"])
        complete_keys = raw.groupby(group_cols, observed=True)["ball_holed"].max()
        complete_keys = complete_keys[complete_keys.eq(1)].reset_index()[group_cols]
        stroke = stroke.merge(
            complete_keys.assign(_complete=1),
            on=group_cols,
            how="inner",
        )
        stroke["next_stroke_no"] = stroke.groupby(
            group_cols, observed=True
        )["stroke_no"].shift(-1)
        stroke["next_lie"] = stroke.groupby(
            group_cols, observed=True
        )["start_lie"].shift(-1)
        stroke["next_distance"] = stroke.groupby(
            group_cols, observed=True
        )["start_distance_yd"].shift(-1)
        stroke["holed"] = stroke["ball_holed"].eq(1)
        stroke["cost"] = np.where(
            stroke["holed"],
            1,
            pd.to_numeric(stroke["next_stroke_no"], errors="coerce")
            - stroke["stroke_no"],
        )
        stroke["cost"] = (
            pd.to_numeric(stroke["cost"], errors="coerce")
            .fillna(1)
            .clip(1, 4)
            .astype(int)
        )
        stroke["next_lie"] = np.where(
            stroke["holed"], "hole", stroke["next_lie"]
        )
        stroke["next_distance"] = np.where(
            stroke["holed"], 0.0, stroke["next_distance"]
        )
        stroke["category"] = stroke["sg_category"].astype(str).str.lower()
        stroke["lie"] = stroke["start_lie"].map(markov.lie_group)
        stroke["start_distance"] = pd.to_numeric(
            stroke["start_distance_yd"], errors="coerce"
        )
        stroke["sg"] = pd.to_numeric(stroke["sg_raw"], errors="coerce")
        stroke["sg_weight_valid"] = stroke["sg_valid"].eq(1) & stroke["sg"].notna()
        stroke["bucket"] = stroke["start_distance"].map(markov.distance_bucket)
        valid = (
            stroke["category"].isin(markov.CATEGORIES)
            & stroke["lie"].ne("unknown")
            & stroke["start_distance"].notna()
            & stroke["next_distance"].notna()
        )
        transition_parts.append(
            stroke.loc[
                valid,
                [
                    "event_key",
                    "source_player_id",
                    "player_name",
                    "round_no",
                    "hole_no",
                    "stroke_no",
                    "shot_key",
                    "source_season",
                    "source_raw_row",
                    "category",
                    "lie",
                    "bucket",
                    "start_distance",
                    "next_lie",
                    "next_distance",
                    "sg",
                    "cost",
                    "holed",
                    "sg_weight_valid",
                ],
            ]
        )

    if not transition_parts:
        raise ValueError("Strict-prior course transition rows were unavailable")
    transitions = pd.concat(transition_parts, ignore_index=True)
    transitions["sg"] = transitions.groupby("category", observed=True)["sg"].transform(
        lambda values: values.fillna(values.median())
    )
    transitions.insert(0, "transition_row_id", np.arange(len(transitions), dtype=np.int64))
    if transitions["shot_key"].isna().any() or transitions["shot_key"].duplicated().any():
        raise ValueError("Causal shot_key must be present and unique in the traced frame")
    return transitions


def assert_markov_frame_parity(
    canonical: pd.DataFrame,
    traced: pd.DataFrame,
) -> None:
    """Assert exact row/value equality against Markov's canonical frame."""

    _require_columns(canonical, MARKOV_TRANSITION_COLUMNS, "canonical transitions")
    _require_columns(traced, MARKOV_TRANSITION_COLUMNS, "traced transitions")
    try:
        pd.testing.assert_frame_equal(
            canonical.loc[:, list(MARKOV_TRANSITION_COLUMNS)].reset_index(drop=True),
            traced.loc[:, list(MARKOV_TRANSITION_COLUMNS)].reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise TransitionTraceParityError(
            "Identity-preserving reconstruction diverged from Markov row order/values"
        ) from exc


@dataclass(frozen=True)
class PoolTrace:
    """Source rows aligned exactly to one ``TransitionPool``."""

    source_scope: str
    pool_index_level: str
    pool_key: tuple[object, ...]
    rows: pd.DataFrame


class TransitionPoolTraceRegistry:
    """Fail-closed ``id(TransitionPool) -> ordered source rows`` registry."""

    def __init__(
        self,
        model: markov.MarkovHoleModel,
        *,
        course_transitions: pd.DataFrame,
        tour_transitions: pd.DataFrame | None = None,
    ) -> None:
        self.model = model
        self._by_pool_id: dict[int, PoolTrace] = {}
        sources: Mapping[str, pd.DataFrame | None] = {
            "course": course_transitions,
            "tour": tour_transitions,
        }
        for label, frame in sources.items():
            if frame is not None and not frame.empty:
                _require_columns(
                    frame,
                    (*TRACE_IDENTITY_COLUMNS, "category", "lie", "bucket", *POOL_VALUE_COLUMNS),
                    f"{label}_transitions",
                )
                if frame["transition_row_id"].duplicated().any():
                    raise ValueError(
                        f"{label}_transitions transition_row_id must be unique"
                    )

        for index_level, source_scope, group_columns in _INDEX_SPECS:
            pool_index: Mapping[tuple[object, ...], markov.TransitionPool] = getattr(
                model, index_level
            )
            if not pool_index:
                continue
            frame = sources[source_scope]
            if frame is None or frame.empty:
                raise ValueError(
                    f"Model has {index_level} pools but no traced {source_scope} frame"
                )
            groups = {
                _tuple_key(key): group.copy()
                for key, group in frame.groupby(
                    list(group_columns), observed=True, sort=False
                )
            }
            if set(pool_index) != set(groups):
                missing = sorted(set(pool_index) - set(groups), key=str)
                extra = sorted(set(groups) - set(pool_index), key=str)
                raise TransitionTraceParityError(
                    f"{index_level} pool keys diverged; missing={missing[:3]}, "
                    f"extra={extra[:3]}"
                )
            for pool_key, pool in pool_index.items():
                rows = groups[_tuple_key(pool_key)].reset_index(drop=True)
                self._assert_pool_parity(pool, rows, index_level, _tuple_key(pool_key))
                pool_id = id(pool)
                if pool_id in self._by_pool_id:
                    raise TransitionTraceParityError(
                        f"TransitionPool object was registered twice: id={pool_id}"
                    )
                self._by_pool_id[pool_id] = PoolTrace(
                    source_scope=source_scope,
                    pool_index_level=index_level,
                    pool_key=_tuple_key(pool_key),
                    rows=rows,
                )

    @staticmethod
    def _assert_pool_parity(
        pool: markov.TransitionPool,
        rows: pd.DataFrame,
        index_level: str,
        pool_key: tuple[object, ...],
    ) -> None:
        expected = {
            "start_distance": rows["start_distance"].to_numpy(float),
            "next_distance": rows["next_distance"].to_numpy(float),
            "next_lie": rows["next_lie"].astype(str).to_numpy(),
            "sg": rows["sg"].to_numpy(float),
            "cost": rows["cost"].to_numpy(np.int16),
            "holed": rows["holed"].to_numpy(bool),
            "sg_weight_valid": rows["sg_weight_valid"].to_numpy(bool),
        }
        for column, values in expected.items():
            actual = np.asarray(getattr(pool, column))
            equal = (
                np.array_equal(actual, values, equal_nan=True)
                if actual.dtype.kind in {"f", "c"}
                else np.array_equal(actual, values)
            )
            if not equal:
                raise TransitionTraceParityError(
                    f"Pool row order/value mismatch at {index_level}/{pool_key}: {column}"
                )

    def trace_for_pool(self, pool: markov.TransitionPool) -> PoolTrace:
        try:
            return self._by_pool_id[id(pool)]
        except KeyError as exc:
            raise TransitionTraceParityError(
                "Resolved TransitionPool has no identity trace mapping"
            ) from exc

    @property
    def mapped_pool_count(self) -> int:
        return len(self._by_pool_id)


@dataclass(frozen=True)
class TracedBatchTransitionResult:
    """Existing batch result plus an input-aligned selected-donor trace."""

    transition: batched.BatchTransitionResult
    donor_pool_row: np.ndarray
    donor_transition_row_id: np.ndarray
    normalized_knn_probability: np.ndarray
    cdf_lower: np.ndarray
    cdf_upper: np.ndarray
    query_state_key: np.ndarray
    pool_key: np.ndarray
    pool_index_level: np.ndarray
    source_scope: np.ndarray
    source_event_key: np.ndarray
    source_player_id: np.ndarray
    source_player_name: np.ndarray
    source_round_no: np.ndarray
    source_hole_no: np.ndarray
    source_stroke_no: np.ndarray
    source_shot_key: np.ndarray
    donor_category: np.ndarray
    donor_start_lie: np.ndarray
    donor_start_distance: np.ndarray
    donor_next_lie: np.ndarray
    donor_next_distance: np.ndarray
    donor_cost: np.ndarray
    donor_holed: np.ndarray
    donor_sg: np.ndarray
    donor_sg_weight_valid: np.ndarray

    def __len__(self) -> int:
        return len(self.transition)

    @property
    def base_hierarchy_mass(self) -> np.ndarray:
        """Alias used by downstream atom-ledger prototypes."""

        return self.normalized_knn_probability

    def donor_frame(self) -> pd.DataFrame:
        """Materialize the aligned trace as a bounded audit DataFrame."""

        return pd.DataFrame(
            {
                "query_row": np.arange(len(self), dtype=np.int64),
                "available": self.transition.available,
                "error": self.transition.error,
                "pool_level": self.transition.level,
                "query_state_key": self.query_state_key,
                "pool_key": self.pool_key,
                "pool_index_level": self.pool_index_level,
                "source_scope": self.source_scope,
                "donor_pool_row": self.donor_pool_row,
                "donor_transition_row_id": self.donor_transition_row_id,
                "normalized_knn_probability": self.normalized_knn_probability,
                "cdf_lower": self.cdf_lower,
                "cdf_upper": self.cdf_upper,
                "source_event_key": self.source_event_key,
                "source_player_id": self.source_player_id,
                "source_player_name": self.source_player_name,
                "source_round_no": self.source_round_no,
                "source_hole_no": self.source_hole_no,
                "source_stroke_no": self.source_stroke_no,
                "source_shot_key": self.source_shot_key,
                "donor_category": self.donor_category,
                "donor_start_lie": self.donor_start_lie,
                "donor_start_distance": self.donor_start_distance,
                "donor_next_lie": self.donor_next_lie,
                "donor_next_distance": self.donor_next_distance,
                "donor_cost": self.donor_cost,
                "donor_holed": self.donor_holed,
                "donor_sg": self.donor_sg,
                "donor_sg_weight_valid": self.donor_sg_weight_valid,
            }
        )


class TracedBatchedPopulationTransitionSampler:
    """Wrap the existing batch sampler and add exact donor trace columns."""

    def __init__(
        self,
        model: markov.MarkovHoleModel,
        *,
        course_transitions: pd.DataFrame,
        tour_transitions: pd.DataFrame | None = None,
        chunk_size: int = batched.DEFAULT_CHUNK_SIZE,
        round_values: Sequence[int] = batched.DEFAULT_ROUNDS,
        hole_values: Sequence[int] = batched.DEFAULT_HOLES,
    ) -> None:
        self.base_sampler = batched.BatchedPopulationTransitionSampler(
            model,
            chunk_size=chunk_size,
            round_values=round_values,
            hole_values=hole_values,
        )
        self.registry = TransitionPoolTraceRegistry(
            model,
            course_transitions=course_transitions,
            tour_transitions=tour_transitions,
        )

    @property
    def backoff_counts(self):
        return self.base_sampler.backoff_counts

    @property
    def last_sample_stats(self):
        return self.base_sampler.last_sample_stats

    def sample_many(
        self,
        *,
        round_no: np.ndarray | int,
        hole_no: np.ndarray | int,
        category: np.ndarray | str,
        lie: np.ndarray | str,
        distance_yd: np.ndarray | float,
        common_uniform: np.ndarray | float,
        chunk_size: int | None = None,
    ) -> TracedBatchTransitionResult:
        """Return the authoritative batch result and its selected donor rows."""

        query = {
            "round_no": round_no,
            "hole_no": hole_no,
            "category": category,
            "lie": lie,
            "distance_yd": distance_yd,
            "common_uniform": common_uniform,
        }
        transition = self.base_sampler.sample_many(**query, chunk_size=chunk_size)
        raw = batched._broadcast_flat(
            round_no, hole_no, category, lie, distance_yd, common_uniform
        )
        return self._trace(transition, *raw, chunk_size=chunk_size)

    def _trace(
        self,
        transition: batched.BatchTransitionResult,
        raw_round: np.ndarray,
        raw_hole: np.ndarray,
        raw_category: np.ndarray,
        raw_lie: np.ndarray,
        raw_distance: np.ndarray,
        raw_uniform: np.ndarray,
        *,
        chunk_size: int | None,
    ) -> TracedBatchTransitionResult:
        size = len(transition)
        chunks = self.base_sampler.chunk_size if chunk_size is None else int(chunk_size)
        if chunks <= 0:
            raise ValueError("chunk_size must be positive")
        rounds = raw_round.astype(np.int64)
        holes = raw_hole.astype(np.int64)
        categories = raw_category.astype(str)
        lies = batched.normalize_population_lies(raw_lie)
        distances = raw_distance.astype(float)
        uniforms = raw_uniform.astype(float)
        buckets = batched.distance_bucket_many(distances)

        pool_row = np.full(size, -1, dtype=np.int64)
        transition_row = np.full(size, -1, dtype=np.int64)
        probability = np.full(size, np.nan, dtype=float)
        cdf_lower = np.full(size, np.nan, dtype=float)
        cdf_upper = np.full(size, np.nan, dtype=float)
        query_state_key = np.full(size, "", dtype=object)
        pool_key = np.full(size, "", dtype=object)
        pool_index_level = np.full(size, "", dtype=object)
        source_scope = np.full(size, "", dtype=object)
        source_event_key = np.full(size, "", dtype=object)
        source_player_id = np.full(size, None, dtype=object)
        source_player_name = np.full(size, "", dtype=object)
        source_round_no = np.full(size, -1, dtype=np.int16)
        source_hole_no = np.full(size, -1, dtype=np.int16)
        source_stroke_no = np.full(size, -1, dtype=np.int16)
        source_shot_key = np.full(size, "", dtype=object)
        donor_category = np.full(size, "", dtype=object)
        donor_start_lie = np.full(size, "", dtype=object)
        donor_start_distance = np.full(size, np.nan, dtype=float)
        donor_next_lie = np.full(size, "", dtype=object)
        donor_next_distance = np.full(size, np.nan, dtype=float)
        donor_cost = np.zeros(size, dtype=np.int16)
        donor_holed = np.zeros(size, dtype=bool)
        donor_sg = np.full(size, np.nan, dtype=float)
        donor_sg_weight_valid = np.zeros(size, dtype=bool)

        if size:
            category_code = (categories == "putt").astype(np.int8)
            lie_code = np.zeros(size, dtype=np.int8)
            for code, label in enumerate(batched.NORMALIZED_LIES):
                lie_code[lies == label] = code
            state = np.empty(
                size,
                dtype=[
                    ("round_no", "<i8"),
                    ("hole_no", "<i8"),
                    ("category", "i1"),
                    ("lie", "i1"),
                    ("bucket", "i1"),
                ],
            )
            state["round_no"] = rounds
            state["hole_no"] = holes
            state["category"] = category_code
            state["lie"] = lie_code
            state["bucket"] = buckets
            unique_state, inverse = np.unique(state, return_inverse=True)

            for group_index, encoded in enumerate(unique_state):
                positions = np.flatnonzero(inverse == group_index)
                query_key = (
                    int(encoded["round_no"]),
                    int(encoded["hole_no"]),
                    batched.CATEGORIES[int(encoded["category"])],
                    batched.NORMALIZED_LIES[int(encoded["lie"])],
                    int(encoded["bucket"]),
                )
                encoded_query_key = _json_key(query_key)
                query_state_key[positions] = encoded_query_key
                resolution = self.base_sampler._resolution.get(query_key)
                if resolution is None:
                    resolution = self.base_sampler._cache_resolution(query_key)
                if resolution.pool_id is None:
                    if transition.available[positions].any():
                        raise TransitionTraceParityError(
                            f"Trace resolution unavailable but base result available: {query_key}"
                        )
                    continue
                pool_cache = self.base_sampler._pool_by_id[int(resolution.pool_id)]
                pool = pool_cache.pool
                pool_trace = self.registry.trace_for_pool(pool)
                pool_key[positions] = _json_key(pool_trace.pool_key)
                pool_index_level[positions] = pool_trace.pool_index_level
                source_scope[positions] = pool_trace.source_scope
                pool_rows = len(pool)
                k = min(int(self.base_sampler.model.knn), pool_rows)

                for start in range(0, len(positions), chunks):
                    selected_positions = positions[start : start + chunks]
                    query_log = np.log1p(np.maximum(distances[selected_positions], 0.0))
                    metric = np.abs(
                        query_log[:, None] - pool_cache.start_log_distance[None, :]
                    )
                    if k < pool_rows:
                        neighbours = np.argpartition(metric, k - 1, axis=1)[:, :k]
                    else:
                        neighbours = np.broadcast_to(
                            np.arange(pool_rows, dtype=np.int64),
                            (len(selected_positions), pool_rows),
                        ).copy()
                    neighbours.sort(axis=1)
                    selected_metric = np.take_along_axis(metric, neighbours, axis=1)
                    weights = np.exp(-selected_metric / 0.18)
                    np.maximum(weights, 1e-12, out=weights)
                    weights /= weights.sum(axis=1, keepdims=True)
                    cumulative = np.cumsum(weights, axis=1)
                    selected = np.count_nonzero(
                        cumulative <= uniforms[selected_positions, None], axis=1
                    )
                    selected[np.isnan(uniforms[selected_positions])] = k
                    np.minimum(selected, k - 1, out=selected)
                    row_index = np.arange(len(selected_positions), dtype=np.int64)
                    choices = neighbours[row_index, selected]
                    lower = np.where(selected > 0, cumulative[row_index, selected - 1], 0.0)

                    pool_row[selected_positions] = choices
                    probability[selected_positions] = weights[row_index, selected]
                    cdf_lower[selected_positions] = lower
                    cdf_upper[selected_positions] = cumulative[row_index, selected]
                    chosen_rows = pool_trace.rows.iloc[choices]
                    transition_row[selected_positions] = chosen_rows[
                        "transition_row_id"
                    ].to_numpy(np.int64)
                    source_event_key[selected_positions] = chosen_rows[
                        "event_key"
                    ].astype(str).to_numpy()
                    source_player_id[selected_positions] = chosen_rows[
                        "source_player_id"
                    ].to_numpy(object)
                    source_player_name[selected_positions] = chosen_rows[
                        "player_name"
                    ].fillna("").astype(str).to_numpy()
                    source_round_no[selected_positions] = chosen_rows[
                        "round_no"
                    ].to_numpy(np.int16)
                    source_hole_no[selected_positions] = chosen_rows[
                        "hole_no"
                    ].to_numpy(np.int16)
                    source_stroke_no[selected_positions] = chosen_rows[
                        "stroke_no"
                    ].to_numpy(np.int16)
                    source_shot_key[selected_positions] = chosen_rows[
                        "shot_key"
                    ].astype(str).to_numpy()
                    donor_category[selected_positions] = chosen_rows[
                        "category"
                    ].astype(str).to_numpy()
                    donor_start_lie[selected_positions] = chosen_rows[
                        "lie"
                    ].astype(str).to_numpy()
                    donor_start_distance[selected_positions] = chosen_rows[
                        "start_distance"
                    ].to_numpy(float)
                    donor_next_lie[selected_positions] = chosen_rows[
                        "next_lie"
                    ].astype(str).to_numpy()
                    donor_next_distance[selected_positions] = chosen_rows[
                        "next_distance"
                    ].to_numpy(float)
                    donor_cost[selected_positions] = chosen_rows["cost"].to_numpy(np.int16)
                    donor_holed[selected_positions] = chosen_rows["holed"].to_numpy(bool)
                    donor_sg[selected_positions] = chosen_rows["sg"].to_numpy(float)
                    donor_sg_weight_valid[selected_positions] = chosen_rows[
                        "sg_weight_valid"
                    ].to_numpy(bool)

                    self._assert_selected_parity(
                        transition,
                        selected_positions,
                        pool,
                        choices,
                        str(resolution.level),
                    )

        return TracedBatchTransitionResult(
            transition=transition,
            donor_pool_row=pool_row,
            donor_transition_row_id=transition_row,
            normalized_knn_probability=probability,
            cdf_lower=cdf_lower,
            cdf_upper=cdf_upper,
            query_state_key=query_state_key,
            pool_key=pool_key,
            pool_index_level=pool_index_level,
            source_scope=source_scope,
            source_event_key=source_event_key,
            source_player_id=source_player_id,
            source_player_name=source_player_name,
            source_round_no=source_round_no,
            source_hole_no=source_hole_no,
            source_stroke_no=source_stroke_no,
            source_shot_key=source_shot_key,
            donor_category=donor_category,
            donor_start_lie=donor_start_lie,
            donor_start_distance=donor_start_distance,
            donor_next_lie=donor_next_lie,
            donor_next_distance=donor_next_distance,
            donor_cost=donor_cost,
            donor_holed=donor_holed,
            donor_sg=donor_sg,
            donor_sg_weight_valid=donor_sg_weight_valid,
        )

    @staticmethod
    def _assert_selected_parity(
        result: batched.BatchTransitionResult,
        positions: np.ndarray,
        pool: markov.TransitionPool,
        choices: np.ndarray,
        level: str,
    ) -> None:
        comparisons = (
            ("holed", result.holed[positions], pool.holed[choices], False),
            ("next_lie", result.next_lie[positions], pool.next_lie[choices], False),
            (
                "next_distance",
                result.next_distance[positions],
                pool.next_distance[choices],
                True,
            ),
            ("cost", result.cost[positions], pool.cost[choices], False),
        )
        for label, actual, expected, allow_nan in comparisons:
            equal = (
                np.array_equal(actual, expected, equal_nan=True)
                if allow_nan
                else np.array_equal(actual, expected)
            )
            if not equal:
                raise TransitionTraceParityError(
                    f"Selected donor diverged from BatchTransitionResult: {label}"
                )
        if not np.array_equal(result.level[positions], np.full(len(positions), level)):
            raise TransitionTraceParityError(
                "Selected donor hierarchy level diverged from BatchTransitionResult"
            )
        if not result.available[positions].all():
            raise TransitionTraceParityError(
                "Mapped donor positions were unavailable in BatchTransitionResult"
            )


@dataclass(frozen=True)
class StrictPriorCourseTraceBundle:
    """A course-only Markov model and its identity-preserving transition rows."""

    model: markov.MarkovHoleModel
    transitions: pd.DataFrame
    course_scores: pd.DataFrame
    target_raw: pd.DataFrame

    def traced_sampler(
        self,
        *,
        chunk_size: int = batched.DEFAULT_CHUNK_SIZE,
        round_values: Sequence[int] = batched.DEFAULT_ROUNDS,
        hole_values: Sequence[int] = batched.DEFAULT_HOLES,
    ) -> TracedBatchedPopulationTransitionSampler:
        return TracedBatchedPopulationTransitionSampler(
            self.model,
            course_transitions=self.transitions,
            chunk_size=chunk_size,
            round_values=round_values,
            hole_values=hole_values,
        )


def build_strict_prior_course_trace_bundle(
    export_root: Path,
    prior_events: pd.DataFrame,
    target_event_key: str,
    *,
    categories: Sequence[str] = ("arg", "putt"),
    min_exact: int = 12,
    knn: int = 64,
) -> StrictPriorCourseTraceBundle:
    """Build and parity-check a strict-prior, exact-course traced model."""

    selected_categories = tuple(str(value).lower() for value in categories)
    invalid = sorted(set(selected_categories) - set(markov.CATEGORIES))
    if invalid:
        raise ValueError(f"Unsupported transition categories: {invalid}")
    canonical, course_scores, target_raw = markov.build_course_transitions(
        Path(export_root), prior_events, str(target_event_key)
    )
    traced = reconstruct_strict_prior_course_transitions(
        Path(export_root), prior_events, str(target_event_key)
    )
    assert_markov_frame_parity(canonical, traced)
    mask = canonical["category"].isin(selected_categories)
    canonical_selected = canonical.loc[mask].copy()
    traced_selected = traced.loc[mask].copy()
    if canonical_selected.empty:
        raise ValueError(f"No strict-prior transitions for categories={selected_categories}")
    empty_tour = canonical_selected.iloc[0:0].copy()
    model = markov.MarkovHoleModel(
        canonical_selected,
        empty_tour,
        min_exact=int(min_exact),
        knn=int(knn),
    )
    # Construct once here so a bundle cannot be returned with latent row-order
    # divergence.  Consumer samplers rebuild the small id(pool) registry.
    TransitionPoolTraceRegistry(model, course_transitions=traced_selected)
    return StrictPriorCourseTraceBundle(
        model=model,
        transitions=traced_selected,
        course_scores=course_scores,
        target_raw=target_raw,
    )


__all__ = [
    "MARKOV_TRANSITION_COLUMNS",
    "POOL_VALUE_COLUMNS",
    "TRACE_IDENTITY_COLUMNS",
    "PoolTrace",
    "StrictPriorCourseTraceBundle",
    "TracedBatchTransitionResult",
    "TracedBatchedPopulationTransitionSampler",
    "TransitionPoolTraceRegistry",
    "TransitionTraceParityError",
    "assert_markov_frame_parity",
    "build_strict_prior_course_trace_bundle",
    "reconstruct_strict_prior_course_transitions",
]

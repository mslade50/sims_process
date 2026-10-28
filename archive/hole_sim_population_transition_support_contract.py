"""Complete ARG/PUTT population-neighbourhood supports for bounded traces.

This archive-only module materializes the exact dynamic k-nearest-neighbour
measure used by :mod:`hole_sim_batched_population_transition` for every
ARG/PUTT query state observed in a traced ledger.  It intentionally does not
modify or participate in the streaming/full-chain execution path.

The contract retains the resolved hierarchy pool, the sorted pool-row indices
of the nearest ``min(64, n)`` donors, exact ``exp(-metric / .18)`` masses and
right-CDF intervals, physical donor identity/outcome fields, fixed-reference
whole-atom rewards, and honest strict-prior donor-ability availability.  A
separate validator reconstructs the supports from the traced sampler and
replays every selected ledger donor from its saved population uniform.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if __package__:
    from . import hole_sim_batched_population_transition as batched  # noqa: E402
    from . import hole_sim_fixed_reference_value as fixed_reference  # noqa: E402
    from . import hole_sim_traced_population_transition as traced  # noqa: E402
    from . import hole_sim_transition_mean_chain_adapter as chain_adapter  # noqa: E402
else:  # direct archive-script import, matching the neighbouring prototypes
    ARCHIVE = Path(__file__).resolve().parent
    if str(ARCHIVE) not in sys.path:
        sys.path.insert(0, str(ARCHIVE))
    import hole_sim_batched_population_transition as batched  # type: ignore[no-redef]  # noqa: E402
    import hole_sim_fixed_reference_value as fixed_reference  # type: ignore[no-redef]  # noqa: E402
    import hole_sim_traced_population_transition as traced  # type: ignore[no-redef]  # noqa: E402
    import hole_sim_transition_mean_chain_adapter as chain_adapter  # type: ignore[no-redef]  # noqa: E402


CONTRACT = "complete_population_neighbourhood_support_v1"
QUALITY_GROUP_CONTRACT = "donor_event_category_query_state_no_player_v1"
KNN_DISTANCE_SCALE = 0.18
KNN_WEIGHT_FLOOR = 1e-12
MASS_TOLERANCE = 2e-14

CATEGORY_ABILITY_COLUMN = {
    "arg": "ag_arg",
    "putt": "ag_putt",
}
BRIDGE_COLUMNS = (
    "event_key",
    "source_player_id",
    "event_start",
    "skill_date",
    "ag_arg",
    "ag_putt",
)

QUERY_COLUMNS = (
    "target_event_key",
    "target_event_start",
    "target_player_id",
    "arm",
    "round_no",
    "hole_no",
    "category",
    "start_lie",
    "start_distance_yd",
    "start_state_key",
    "population_query_state_key",
)

LEDGER_REPLAY_COLUMNS = (
    *QUERY_COLUMNS,
    "particle_id",
    "atom_sequence",
    "rng_population_uniform",
    "population_pool_key",
    "population_pool_level",
    "population_pool_index_level",
    "population_source_scope",
    "population_donor_pool_row",
    "population_donor_transition_row_id",
    "population_normalized_knn_probability",
    "population_cdf_lower",
    "population_cdf_upper",
    "population_donor_sg",
    "population_donor_sg_weight_valid",
    "atom_id",
    "donor_event_key",
    "donor_player_id",
    "donor_player_name",
    "donor_round_no",
    "donor_hole_no",
    "donor_stroke_no",
    "stroke_cost",
    "terminal",
    "end_lie",
    "end_distance_yd",
    "fixed_reference_reward",
)

SUPPORT_MEASURE_COLUMNS = (
    "contract",
    "support_key",
    "support_row_order",
    "target_event_key",
    "target_event_start",
    "target_player_id",
    "arm",
    "round_no",
    "hole_no",
    "category",
    "query_start_lie",
    "query_start_distance_yd",
    "query_start_distance_hex",
    "query_start_state_key",
    "population_query_state_key",
    "query_distance_bucket",
    "population_pool_key",
    "population_pool_level",
    "population_pool_index_level",
    "population_source_scope",
    "resolved_pool_rows",
    "knn_donors",
    "donor_pool_row",
    "donor_transition_row_id",
    "atom_id",
    "donor_event_key",
    "donor_player_id",
    "donor_player_name",
    "donor_round_no",
    "donor_hole_no",
    "donor_stroke_no",
    "donor_category",
    "donor_start_lie",
    "donor_start_distance_yd",
    "donor_next_lie_raw",
    "donor_next_distance_yd_raw",
    "donor_cost",
    "donor_holed",
    "donor_sg",
    "donor_sg_weight_valid",
    "end_lie",
    "end_distance_yd",
    "terminal",
    "base_hierarchy_mass",
    "spatial_model_weight",
    "quality_base_mass",
    "proposal_probability",
    "proposal_cdf_lower",
    "proposal_cdf_upper",
    "start_reference_value",
    "end_reference_value",
    "fixed_reference_reward",
    "reference_descriptor",
    "reference_function_sha256",
    "reference_putting_sha256",
    "donor_quality_group_key",
    "transition_input_sha256",
)

SUPPORT_ABILITY_COLUMNS = (
    "support_key",
    "support_row_order",
    "atom_id",
    "support_measure_sha256",
    "donor_quality_group_key",
    "donor_event_key",
    "donor_player_id",
    "donor_event_start",
    "donor_skill_date",
    "donor_ability_sg",
    "donor_ability_available",
    "donor_ability_snapshot_kind",
    "bridge_input_sha256",
)

TRANSITION_HASH_COLUMNS = (
    "transition_row_id",
    "event_key",
    "source_player_id",
    "player_name",
    "round_no",
    "hole_no",
    "stroke_no",
    "shot_key",
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
)


class PopulationSupportContractError(AssertionError):
    """Raised when a persisted population support cannot be reproduced."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} misses columns={missing}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_scalar(value: object) -> object:
    """Represent scalar values without platform-specific NaN/string behavior."""

    if value is None or value is pd.NA:
        return {"type": "null"}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return {"type": "null"}
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return {"type": "timestamp", "value": timestamp.isoformat()}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return {"type": "null"}
        if not np.isfinite(number):
            raise ValueError("Hashes cannot contain infinite numbers")
        if number == 0.0:
            number = 0.0
        return {"type": "float", "value": number.hex()}
    return {"type": "str", "value": str(value)}


def _stable_sha(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    """Hash ordered rows with explicit, null-stable scalar typing."""

    _require_columns(frame, columns, "frame hash")
    digest = sha256(
        json.dumps(list(columns), separators=(",", ":")).encode("utf-8")
    )
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        payload = [_canonical_scalar(value) for value in row]
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _json_key(values: Sequence[object]) -> str:
    return json.dumps(
        [value.item() if isinstance(value, np.generic) else value for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _distance_hex(value: object) -> str:
    distance = float(value)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError(f"Query distance must be finite and non-negative: {value}")
    if distance == 0.0:
        distance = 0.0
    return distance.hex()


def _support_key(row: Mapping[str, object]) -> str:
    payload = {
        "contract": CONTRACT,
        "target_event_key": str(row["target_event_key"]),
        "target_player_id": str(row["target_player_id"]),
        "arm": str(row["arm"]),
        "round_no": int(row["round_no"]),
        "hole_no": int(row["hole_no"]),
        "category": str(row["category"]),
        "query_start_lie": str(row["start_lie"]),
        "query_start_distance_hex": _distance_hex(row["start_distance_yd"]),
        "query_start_state_key": str(row["start_state_key"]),
        "population_query_state_key": str(row["population_query_state_key"]),
    }
    suffix = _stable_sha(payload)[:32]
    return (
        f"population|{payload['arm']}|R{payload['round_no']}|"
        f"H{payload['hole_no']}|{payload['category']}|{suffix}"
    )


def donor_quality_group_key(row: Mapping[str, object]) -> str:
    """Hash donor event + category + query state, intentionally no player."""

    payload = {
        "contract": QUALITY_GROUP_CONTRACT,
        "donor_event_key": str(row["donor_event_key"]),
        "category": str(row["category"]),
        "round_no": int(row["round_no"]),
        "hole_no": int(row["hole_no"]),
        "query_start_lie": str(row["query_start_lie"]),
        "query_start_distance_hex": str(row["query_start_distance_hex"]),
        "query_start_state_key": str(row["query_start_state_key"]),
        "population_query_state_key": str(row["population_query_state_key"]),
    }
    return f"quality-group|{_stable_sha(payload)}"


def _transition_input_frame(
    sampler: traced.TracedBatchedPopulationTransitionSampler,
) -> pd.DataFrame:
    registry = sampler.registry
    traces = list(registry._by_pool_id.values())  # exact traced registry contract
    if not traces:
        raise ValueError("Traced population sampler contains no registered pools")
    combined = pd.concat([item.rows for item in traces], ignore_index=True, sort=False)
    _require_columns(combined, TRANSITION_HASH_COLUMNS, "traced transition input")
    # Every physical row appears in several hierarchy pools.  Collapse exact
    # duplicate payloads in one vectorised operation before checking whether
    # any ID still has more than one distinct representation.  A per-ID
    # Python group loop is prohibitively slow for the real 85k-row registry.
    distinct = combined.loc[:, list(TRANSITION_HASH_COLUMNS)].drop_duplicates(
        ignore_index=True
    )
    if distinct["transition_row_id"].duplicated(keep=False).any():
        raise ValueError(
            "A transition_row_id has inconsistent registered pool content"
        )
    return distinct.sort_values(
        "transition_row_id", kind="mergesort"
    ).reset_index(drop=True)


def _input_hashes(
    ledger: pd.DataFrame,
    sampler: traced.TracedBatchedPopulationTransitionSampler,
    reference_model: fixed_reference.ReferenceValueModel,
    bridge: pd.DataFrame,
) -> dict[str, object]:
    population = ledger.loc[
        ledger["category"].astype(str).isin(CATEGORY_ABILITY_COLUMN)
    ].copy()
    population = population.sort_values(
        ["arm", "particle_id", "atom_sequence"], kind="mergesort"
    ).reset_index(drop=True)
    transitions = _transition_input_frame(sampler)
    bridge_work = bridge.loc[:, list(BRIDGE_COLUMNS)].copy()
    bridge_work["event_key"] = bridge_work["event_key"].astype(str)
    bridge_work["source_player_id"] = bridge_work["source_player_id"].astype(str)
    bridge_work = bridge_work.sort_values(
        ["event_key", "source_player_id"], kind="mergesort"
    ).reset_index(drop=True)
    sampler_payload = {
        "knn": int(sampler.base_sampler.model.knn),
        "min_exact": int(sampler.base_sampler.model.min_exact),
        "mapped_pools": int(sampler.registry.mapped_pool_count),
        "transition_input_sha256": frame_sha256(
            transitions, TRANSITION_HASH_COLUMNS
        ),
        "traced_source_sha256": _sha256_file(Path(traced.__file__)),
        "batched_source_sha256": _sha256_file(Path(batched.__file__)),
    }
    return {
        "ledger_population_sha256": frame_sha256(
            population, LEDGER_REPLAY_COLUMNS
        ),
        "ledger_population_rows": int(len(population)),
        "transition_input_sha256": sampler_payload["transition_input_sha256"],
        "transition_input_rows": int(len(transitions)),
        "bridge_input_sha256": frame_sha256(bridge_work, BRIDGE_COLUMNS),
        "bridge_input_rows": int(len(bridge_work)),
        "reference_function_sha256": str(reference_model.model_sha256),
        "reference_putting_sha256": str(reference_model.putting_curve_sha256),
        "reference_descriptor": str(reference_model.descriptor),
        "sampler_input_sha256": _stable_sha(sampler_payload),
        "sampler": sampler_payload,
    }


def _population_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ledger, LEDGER_REPLAY_COLUMNS, "traced population ledger")
    output = ledger.loc[
        ledger["category"].astype(str).isin(CATEGORY_ABILITY_COLUMN)
    ].copy()
    if output.empty:
        raise ValueError("The traced ledger contains no ARG/PUTT population rows")
    output["category"] = output["category"].astype(str)
    if not output["category"].isin(CATEGORY_ABILITY_COLUMN).all():
        raise ValueError("Population ledger contains an unsupported category")
    for column in ("start_distance_yd", "rng_population_uniform"):
        values = pd.to_numeric(output[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Population ledger has non-finite {column}")
    uniforms = output["rng_population_uniform"].to_numpy(float)
    if (uniforms < 0.0).any() or (uniforms >= 1.0).any():
        raise ValueError("Population uniforms must be in [0, 1)")
    return output.sort_values(
        ["arm", "particle_id", "atom_sequence"], kind="mergesort"
    ).reset_index(drop=True)


def _query_frame(
    population: pd.DataFrame,
    reference_model: fixed_reference.ReferenceValueModel,
) -> pd.DataFrame:
    queries = population.loc[:, list(QUERY_COLUMNS)].drop_duplicates().copy()
    queries["query_start_distance_hex"] = queries["start_distance_yd"].map(
        _distance_hex
    )
    queries["support_key"] = [
        _support_key(row) for row in queries.to_dict("records")
    ]
    if queries["support_key"].duplicated().any():
        raise ValueError("Population query support keys collided")
    for row in queries.itertuples(index=False):
        expected = reference_model.state_key(row.start_lie, row.start_distance_yd)
        if str(row.start_state_key) != expected:
            raise ValueError(
                f"Population ledger start-state key changed: {row.start_state_key}"
            )
    return queries.sort_values(
        [
            "arm",
            "round_no",
            "hole_no",
            "category",
            "start_lie",
            "query_start_distance_hex",
            "support_key",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _assert_query_trace_metadata(
    selected: pd.DataFrame,
    *,
    query_state_key: str,
    pool_key: str,
    pool_level: str,
    pool_index_level: str,
    source_scope: str,
) -> None:
    comparisons = {
        "population_query_state_key": query_state_key,
        "population_pool_key": pool_key,
        "population_pool_level": pool_level,
        "population_pool_index_level": pool_index_level,
        "population_source_scope": source_scope,
    }
    for column, expected in comparisons.items():
        values = selected[column].astype(str).unique().tolist()
        if values != [str(expected)]:
            raise ValueError(
                f"Ledger query metadata changed for {column}: {values} != {expected}"
            )


def _attach_abilities(
    atoms: pd.DataFrame,
    bridge: pd.DataFrame,
    *,
    target_event_start: object,
) -> tuple[pd.DataFrame, dict[str, object]]:
    _require_columns(bridge, BRIDGE_COLUMNS, "strict-prior bridge")
    skills = bridge.loc[:, list(BRIDGE_COLUMNS)].copy()
    skills["event_key"] = skills["event_key"].astype(str)
    skills["source_player_id"] = skills["source_player_id"].astype(str)
    if skills.duplicated(["event_key", "source_player_id"]).any():
        raise ValueError("Strict-prior bridge has duplicate event/player keys")
    skills = skills.rename(
        columns={
            "event_key": "donor_event_key",
            "source_player_id": "donor_player_id",
            "event_start": "donor_event_start",
            "skill_date": "donor_skill_date",
        }
    )
    output = atoms.merge(
        skills,
        on=["donor_event_key", "donor_player_id"],
        how="left",
        validate="many_to_one",
    )
    ability = np.full(len(output), np.nan, dtype=float)
    category = output["category"].astype(str).to_numpy()
    for name, column in CATEGORY_ABILITY_COLUMN.items():
        mask = category == name
        ability[mask] = pd.to_numeric(
            output.loc[mask, column], errors="coerce"
        ).to_numpy(float)
    output["donor_ability_sg"] = ability
    output["donor_event_start"] = pd.to_datetime(
        output["donor_event_start"], errors="coerce", utc=True
    )
    output["donor_skill_date"] = pd.to_datetime(
        output["donor_skill_date"], errors="coerce", utc=True
    )
    target_start = pd.Timestamp(target_event_start)
    if target_start.tzinfo is None:
        target_start = target_start.tz_localize("UTC")
    else:
        target_start = target_start.tz_convert("UTC")
    cutoff_violation = (
        output["donor_skill_date"].notna()
        & output["donor_event_start"].notna()
        & output["donor_skill_date"].ge(output["donor_event_start"])
    )
    target_violation = output["donor_event_start"].notna() & output[
        "donor_event_start"
    ].ge(target_start)
    if cutoff_violation.any() or target_violation.any():
        raise ValueError("Population donor ability violates a strict cutoff")
    available = (
        np.isfinite(ability)
        & output["donor_skill_date"].notna().to_numpy()
        & output["donor_event_start"].notna().to_numpy()
        & output["donor_skill_date"].lt(output["donor_event_start"]).to_numpy()
        & output["donor_event_start"].lt(target_start).to_numpy()
    )
    output["donor_ability_available"] = available
    output["donor_ability_snapshot_kind"] = np.where(
        available,
        chain_adapter.DONOR_SNAPSHOT_CONTRACT,
        "strict_prior_snapshot_missing_fail_closed",
    )
    output = output.drop(columns=list(CATEGORY_ABILITY_COLUMN.values()))
    coverage = {
        name: {
            "rows": int((output["category"] == name).sum()),
            "available": int(
                ((output["category"] == name) & available).sum()
            ),
        }
        for name in CATEGORY_ABILITY_COLUMN
    }
    return output, {
        "rows": int(len(output)),
        "available_rows": int(available.sum()),
        "missing_rows": int((~available).sum()),
        "coverage": float(available.mean()),
        "category": coverage,
        "strict_cutoff_violations": int(cutoff_violation.sum()),
        "target_cutoff_violations": int(target_violation.sum()),
        "missing_policy": "fail_closed_not_neutral_zero",
    }


def _materialize(
    ledger: pd.DataFrame,
    sampler: traced.TracedBatchedPopulationTransitionSampler,
    reference_model: fixed_reference.ReferenceValueModel,
    bridge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not isinstance(sampler, traced.TracedBatchedPopulationTransitionSampler):
        raise TypeError("Population support requires the identity-preserving sampler")
    population = _population_ledger(ledger)
    queries = _query_frame(population, reference_model)
    target_starts = pd.to_datetime(
        population["target_event_start"], errors="coerce", utc=True
    ).drop_duplicates()
    if len(target_starts) != 1:
        raise ValueError("Population ledger does not have one target event start")
    target_start = target_starts.iloc[0]
    input_hashes = _input_hashes(ledger, sampler, reference_model, bridge)
    transition_hash = str(input_hashes["transition_input_sha256"])
    bridge_hash = str(input_hashes["bridge_input_sha256"])
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []

    for query in queries.itertuples(index=False):
        start_distance = float(query.start_distance_yd)
        normalized_lie = batched.normalize_population_lie(query.start_lie)
        if normalized_lie == "unknown":
            raise ValueError(f"Unsupported population query lie={query.start_lie}")
        bucket = int(batched.distance_bucket_many(np.asarray([start_distance]))[0])
        query_key = (
            int(query.round_no),
            int(query.hole_no),
            str(query.category),
            normalized_lie,
            bucket,
        )
        encoded_query_key = _json_key(query_key)
        if encoded_query_key != str(query.population_query_state_key):
            raise ValueError(
                "Population query state does not match exact sampler conditioning"
            )
        resolution = sampler.base_sampler._resolution.get(query_key)
        if resolution is None:
            resolution = sampler.base_sampler._cache_resolution(query_key)
        if resolution.pool_id is None:
            raise ValueError(f"Population query is unavailable: {query_key}")
        pool_cache = sampler.base_sampler._pool_by_id[int(resolution.pool_id)]
        pool = pool_cache.pool
        pool_trace = sampler.registry.trace_for_pool(pool)
        pool_key = _json_key(pool_trace.pool_key)
        pool_rows = len(pool)
        knn = min(int(sampler.base_sampler.model.knn), pool_rows)
        if knn <= 0:
            raise ValueError(f"Resolved population pool is empty: {query_key}")

        selected_ledger = population.loc[
            population["target_event_key"].astype(str).eq(str(query.target_event_key))
            & population["target_player_id"].astype(str).eq(str(query.target_player_id))
            & population["arm"].astype(str).eq(str(query.arm))
            & population["round_no"].eq(int(query.round_no))
            & population["hole_no"].eq(int(query.hole_no))
            & population["category"].astype(str).eq(str(query.category))
            & population["start_lie"].astype(str).eq(str(query.start_lie))
            & population["start_distance_yd"].eq(start_distance)
            & population["start_state_key"].astype(str).eq(str(query.start_state_key))
            & population["population_query_state_key"].astype(str).eq(
                encoded_query_key
            )
        ]
        if selected_ledger.empty:
            raise AssertionError("A materialized population query has no ledger rows")
        _assert_query_trace_metadata(
            selected_ledger,
            query_state_key=encoded_query_key,
            pool_key=pool_key,
            pool_level=str(resolution.level),
            pool_index_level=pool_trace.pool_index_level,
            source_scope=pool_trace.source_scope,
        )

        query_log = np.log1p(max(start_distance, 0.0))
        metric = np.abs(query_log - pool_cache.start_log_distance)
        if knn < pool_rows:
            neighbours = np.argpartition(metric, knn - 1)[:knn]
        else:
            neighbours = np.arange(pool_rows, dtype=np.int64)
        neighbours = np.asarray(neighbours, dtype=np.int64)
        neighbours.sort()
        selected_metric = metric[neighbours]
        probability = np.exp(-selected_metric / KNN_DISTANCE_SCALE)
        np.maximum(probability, KNN_WEIGHT_FLOOR, out=probability)
        probability /= probability.sum()
        upper = np.cumsum(probability)
        lower = np.concatenate(([0.0], upper[:-1]))
        donors = pool_trace.rows.iloc[neighbours].copy().reset_index(drop=True)
        raw_end_lie = donors["next_lie"].astype(str).to_numpy()
        normalized_end_lie = batched.normalize_population_lies(raw_end_lie)
        donor_holed = donors["holed"].to_numpy(bool)
        terminal = donor_holed | (normalized_end_lie == "hole")
        if (normalized_end_lie == "unknown").any():
            bad = raw_end_lie[normalized_end_lie == "unknown"][:5].tolist()
            raise ValueError(f"Population donor has unsupported next lie={bad}")
        end_lie = np.where(terminal, "hole", normalized_end_lie)
        raw_end_distance = donors["next_distance"].to_numpy(float)
        if not np.isfinite(raw_end_distance).all():
            raise ValueError("Population donor next distance is non-finite")
        end_distance = np.where(terminal, 0.0, np.maximum(raw_end_distance, 0.0))
        cost = donors["cost"].to_numpy(np.int16)
        if (cost < 1).any():
            raise ValueError("Population donor cost omits the causal stroke")
        start_value = reference_model.value(query.start_lie, start_distance)
        end_value = np.asarray(
            [
                reference_model.value(lie, distance)
                for lie, distance in zip(end_lie, end_distance)
            ],
            dtype=float,
        )
        reward = start_value - cost.astype(float) - end_value

        compact = pd.DataFrame(
            {
                "contract": CONTRACT,
                "support_key": str(query.support_key),
                "support_row_order": np.arange(knn, dtype=np.int32),
                "target_event_key": str(query.target_event_key),
                "target_event_start": pd.Timestamp(query.target_event_start),
                "target_player_id": str(query.target_player_id),
                "arm": str(query.arm),
                "round_no": int(query.round_no),
                "hole_no": int(query.hole_no),
                "category": str(query.category),
                "query_start_lie": str(query.start_lie),
                "query_start_distance_yd": start_distance,
                "query_start_distance_hex": str(query.query_start_distance_hex),
                "query_start_state_key": str(query.start_state_key),
                "population_query_state_key": encoded_query_key,
                "query_distance_bucket": bucket,
                "population_pool_key": pool_key,
                "population_pool_level": str(resolution.level),
                "population_pool_index_level": pool_trace.pool_index_level,
                "population_source_scope": pool_trace.source_scope,
                "resolved_pool_rows": pool_rows,
                "knn_donors": knn,
                "donor_pool_row": neighbours,
                "donor_transition_row_id": donors["transition_row_id"].to_numpy(
                    np.int64
                ),
                "atom_id": donors["shot_key"].astype(str).to_numpy(),
                "donor_event_key": donors["event_key"].astype(str).to_numpy(),
                "donor_player_id": donors["source_player_id"].astype(str).to_numpy(),
                "donor_player_name": donors["player_name"].fillna("").astype(str).to_numpy(),
                "donor_round_no": donors["round_no"].to_numpy(np.int16),
                "donor_hole_no": donors["hole_no"].to_numpy(np.int16),
                "donor_stroke_no": donors["stroke_no"].to_numpy(np.int16),
                "donor_category": donors["category"].astype(str).to_numpy(),
                "donor_start_lie": donors["lie"].astype(str).to_numpy(),
                "donor_start_distance_yd": donors["start_distance"].to_numpy(float),
                "donor_next_lie_raw": raw_end_lie,
                "donor_next_distance_yd_raw": raw_end_distance,
                "donor_cost": cost,
                "donor_holed": donor_holed,
                "donor_sg": donors["sg"].to_numpy(float),
                "donor_sg_weight_valid": donors["sg_weight_valid"].to_numpy(bool),
                "end_lie": end_lie,
                "end_distance_yd": end_distance,
                "terminal": terminal,
                "base_hierarchy_mass": probability,
                "spatial_model_weight": 1.0,
                "quality_base_mass": probability,
                "proposal_probability": probability,
                "proposal_cdf_lower": lower,
                "proposal_cdf_upper": upper,
                "start_reference_value": start_value,
                "end_reference_value": end_value,
                "fixed_reference_reward": reward,
                "reference_descriptor": reference_model.descriptor,
                "reference_function_sha256": reference_model.model_sha256,
                "reference_putting_sha256": reference_model.putting_curve_sha256,
                "transition_input_sha256": transition_hash,
            }
        )
        compact["donor_quality_group_key"] = [
            donor_quality_group_key(row) for row in compact.to_dict("records")
        ]
        measure_sha = frame_sha256(compact, SUPPORT_MEASURE_COLUMNS)
        compact["support_measure_sha256"] = measure_sha
        frames.append(compact)
        inventory_rows.append(
            {
                "contract": CONTRACT,
                "support_key": str(query.support_key),
                "target_event_key": str(query.target_event_key),
                "target_player_id": str(query.target_player_id),
                "arm": str(query.arm),
                "round_no": int(query.round_no),
                "hole_no": int(query.hole_no),
                "category": str(query.category),
                "query_start_lie": str(query.start_lie),
                "query_start_distance_yd": start_distance,
                "query_start_distance_hex": str(query.query_start_distance_hex),
                "query_start_state_key": str(query.start_state_key),
                "population_query_state_key": encoded_query_key,
                "population_pool_key": pool_key,
                "population_pool_level": str(resolution.level),
                "population_pool_index_level": pool_trace.pool_index_level,
                "population_source_scope": pool_trace.source_scope,
                "resolved_pool_rows": pool_rows,
                "atoms": knn,
                "probability_mass": float(probability.sum()),
                "cdf_upper": float(upper[-1]),
                "selected_ledger_rows": int(len(selected_ledger)),
                "support_measure_sha256": measure_sha,
                "transition_input_sha256": transition_hash,
                "reference_function_sha256": reference_model.model_sha256,
                "reference_putting_sha256": reference_model.putting_curve_sha256,
            }
        )

    atoms = pd.concat(frames, ignore_index=True, sort=False)
    atoms, ability_audit = _attach_abilities(
        atoms, bridge, target_event_start=target_start
    )
    atoms["bridge_input_sha256"] = bridge_hash
    atoms["support_ability_sha256"] = ""
    ability_rows: list[dict[str, object]] = []
    for support_key, group in atoms.groupby("support_key", observed=True, sort=False):
        ordered = group.sort_values("support_row_order", kind="mergesort")
        ability_sha = frame_sha256(ordered, SUPPORT_ABILITY_COLUMNS)
        atoms.loc[ordered.index, "support_ability_sha256"] = ability_sha
        ability_rows.append(
            {
                "support_key": str(support_key),
                "support_ability_sha256": ability_sha,
                "ability_rows": int(len(ordered)),
                "ability_available_rows": int(
                    ordered["donor_ability_available"].sum()
                ),
                "ability_missing_rows": int(
                    (~ordered["donor_ability_available"]).sum()
                ),
                "anchor_ready": bool(ordered["donor_ability_available"].all()),
            }
        )
    inventory = pd.DataFrame(inventory_rows).merge(
        pd.DataFrame(ability_rows),
        on="support_key",
        how="left",
        validate="one_to_one",
    )
    atoms = atoms.sort_values(
        ["arm", "round_no", "hole_no", "category", "support_key", "support_row_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    inventory = inventory.sort_values(
        ["arm", "round_no", "hole_no", "category", "support_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    inventory_columns = tuple(inventory.columns)
    atoms_columns = tuple(atoms.columns)
    metadata = {
        "contract": CONTRACT,
        "supports": int(len(inventory)),
        "atoms": int(len(atoms)),
        "selected_ledger_rows": int(len(population)),
        "input_sha256": input_hashes,
        "ability_audit": ability_audit,
        "anchor_ready": bool(atoms["donor_ability_available"].all()),
        "anchor_status": (
            "ready_all_donor_abilities"
            if atoms["donor_ability_available"].all()
            else "HOLD_missing_strict_prior_donor_ability"
        ),
        "atoms_logical_sha256": frame_sha256(atoms, atoms_columns),
        "inventory_logical_sha256": frame_sha256(inventory, inventory_columns),
        "reference_function_sha256": reference_model.model_sha256,
        "reference_putting_sha256": reference_model.putting_curve_sha256,
        "transition_input_sha256": transition_hash,
        "bridge_input_sha256": bridge_hash,
    }
    metadata["contract_sha256"] = _stable_sha(
        {
            "contract": CONTRACT,
            "atoms_logical_sha256": metadata["atoms_logical_sha256"],
            "inventory_logical_sha256": metadata["inventory_logical_sha256"],
            "input_sha256": input_hashes,
            "anchor_status": metadata["anchor_status"],
        }
    )
    return atoms, inventory, metadata


def _replay_selected_rows(
    atoms: pd.DataFrame,
    ledger: pd.DataFrame,
    reference_model: fixed_reference.ReferenceValueModel,
) -> dict[str, object]:
    population = _population_ledger(ledger)
    support_lookup = {
        str(key): group.sort_values("support_row_order", kind="mergesort")
        for key, group in atoms.groupby("support_key", observed=True, sort=False)
    }
    checks = 0
    for row in population.to_dict("records"):
        support_key = _support_key(row)
        try:
            support = support_lookup[support_key]
        except KeyError as exc:
            raise PopulationSupportContractError(
                f"Selected population row lacks support={support_key}"
            ) from exc
        uniform = float(row["rng_population_uniform"])
        cdf = support["proposal_cdf_upper"].to_numpy(float)
        selected = int(np.count_nonzero(cdf <= uniform))
        selected = min(selected, len(support) - 1)
        atom = support.iloc[selected]
        exact = {
            "population_query_state_key": atom["population_query_state_key"],
            "population_pool_key": atom["population_pool_key"],
            "population_pool_level": atom["population_pool_level"],
            "population_pool_index_level": atom["population_pool_index_level"],
            "population_source_scope": atom["population_source_scope"],
            "population_donor_pool_row": atom["donor_pool_row"],
            "population_donor_transition_row_id": atom["donor_transition_row_id"],
            "population_normalized_knn_probability": atom["proposal_probability"],
            "population_cdf_lower": atom["proposal_cdf_lower"],
            "population_cdf_upper": atom["proposal_cdf_upper"],
            "population_donor_sg": atom["donor_sg"],
            "population_donor_sg_weight_valid": atom["donor_sg_weight_valid"],
            "atom_id": atom["atom_id"],
            "donor_event_key": atom["donor_event_key"],
            "donor_player_id": atom["donor_player_id"],
            "donor_player_name": atom["donor_player_name"],
            "donor_round_no": atom["donor_round_no"],
            "donor_hole_no": atom["donor_hole_no"],
            "donor_stroke_no": atom["donor_stroke_no"],
            "stroke_cost": atom["donor_cost"],
            "terminal": atom["terminal"],
            "end_lie": atom["end_lie"],
            "end_distance_yd": atom["end_distance_yd"],
            "fixed_reference_reward": atom["fixed_reference_reward"],
        }
        for column, expected in exact.items():
            observed = row[column]
            if isinstance(expected, (float, np.floating)):
                equal = float(observed) == float(expected)
            else:
                equal = str(observed) == str(expected) if isinstance(expected, str) else observed == expected
            if not bool(equal):
                raise PopulationSupportContractError(
                    f"Selected population replay changed {column}: "
                    f"observed={observed!r}, expected={expected!r}"
                )
        reward = reference_model.transition_reward(
            start_lie=row["start_lie"],
            start_distance_yd=row["start_distance_yd"],
            end_lie=atom["end_lie"],
            end_distance_yd=atom["end_distance_yd"],
            stroke_cost=int(atom["donor_cost"]),
        )
        if reward != float(atom["fixed_reference_reward"]):
            raise PopulationSupportContractError(
                "Selected population fixed-reference reward changed"
            )
        checks += 1
    return {
        "selected_rows_replayed": int(checks),
        "all_selected_donors_exact": True,
    }


def materialize_population_support_contract(
    ledger: pd.DataFrame,
    sampler: traced.TracedBatchedPopulationTransitionSampler,
    reference_model: fixed_reference.ReferenceValueModel,
    bridge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the deterministic complete support contract for ledger queries."""

    atoms, inventory, metadata = _materialize(
        ledger, sampler, reference_model, bridge
    )
    replay = _replay_selected_rows(atoms, ledger, reference_model)
    metadata = {**metadata, "selected_replay": replay}
    return atoms, inventory, metadata


def validate_population_support_contract(
    atoms: pd.DataFrame,
    inventory: pd.DataFrame,
    ledger: pd.DataFrame,
    sampler: traced.TracedBatchedPopulationTransitionSampler,
    reference_model: fixed_reference.ReferenceValueModel,
    bridge: pd.DataFrame,
    *,
    expected_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Rebuild all supports and replay every selected population ledger row."""

    _require_columns(atoms, (*SUPPORT_MEASURE_COLUMNS, *SUPPORT_ABILITY_COLUMNS, "support_ability_sha256"), "population atoms")
    if atoms.empty or inventory.empty:
        raise ValueError("Population support contract cannot be empty")
    for support_key, raw in atoms.groupby("support_key", observed=True, sort=False):
        group = raw.sort_values("support_row_order", kind="mergesort")
        order = group["support_row_order"].to_numpy(np.int32)
        if not np.array_equal(order, np.arange(len(group), dtype=np.int32)):
            raise PopulationSupportContractError(
                f"Population support row order changed: {support_key}"
            )
        if group["atom_id"].astype(str).duplicated().any():
            raise PopulationSupportContractError(
                f"Population support donor identity duplicated: {support_key}"
            )
        probability = group["proposal_probability"].to_numpy(float)
        lower = group["proposal_cdf_lower"].to_numpy(float)
        upper = group["proposal_cdf_upper"].to_numpy(float)
        if not np.isfinite(probability).all() or (probability <= 0.0).any():
            raise PopulationSupportContractError(
                f"Population support probability changed: {support_key}"
            )
        if not np.isclose(
            probability.sum(), 1.0, atol=MASS_TOLERANCE, rtol=0.0
        ):
            raise PopulationSupportContractError(
                f"Population support mass changed: {support_key}"
            )
        expected_upper = np.cumsum(probability)
        expected_lower = np.concatenate(([0.0], expected_upper[:-1]))
        if not (
            np.array_equal(lower, expected_lower)
            and np.array_equal(upper, expected_upper)
        ):
            raise PopulationSupportContractError(
                f"Population support CDF changed: {support_key}"
            )
        measure = frame_sha256(group, SUPPORT_MEASURE_COLUMNS)
        ability = frame_sha256(group, SUPPORT_ABILITY_COLUMNS)
        if group["support_measure_sha256"].astype(str).unique().tolist() != [measure]:
            raise PopulationSupportContractError(
                f"Population support measure hash changed: {support_key}"
            )
        if group["support_ability_sha256"].astype(str).unique().tolist() != [ability]:
            raise PopulationSupportContractError(
                f"Population support ability hash changed: {support_key}"
            )

    expected_atoms, expected_inventory, expected = _materialize(
        ledger, sampler, reference_model, bridge
    )
    try:
        pd.testing.assert_frame_equal(
            atoms.reset_index(drop=True),
            expected_atoms.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
    except AssertionError as exc:
        raise PopulationSupportContractError(
            "Persisted population support atoms differ from reconstructed inputs"
        ) from exc
    try:
        pd.testing.assert_frame_equal(
            inventory.reset_index(drop=True),
            expected_inventory.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
    except AssertionError as exc:
        raise PopulationSupportContractError(
            "Persisted population inventory differs from reconstructed inputs"
        ) from exc
    replay = _replay_selected_rows(atoms, ledger, reference_model)
    if expected_metadata is not None:
        for key in (
            "contract",
            "supports",
            "atoms",
            "selected_ledger_rows",
            "input_sha256",
            "ability_audit",
            "anchor_ready",
            "contract_sha256",
            "atoms_logical_sha256",
            "inventory_logical_sha256",
            "transition_input_sha256",
            "bridge_input_sha256",
            "reference_function_sha256",
            "reference_putting_sha256",
            "anchor_status",
        ):
            if expected_metadata.get(key) != expected.get(key):
                raise PopulationSupportContractError(
                    f"Population contract metadata changed: {key}"
                )
    return {
        **replay,
        "supports": int(len(inventory)),
        "atoms": int(len(atoms)),
        "measure_hashes_recomputed": True,
        "ability_hashes_recomputed": True,
        "inputs_reconstructed_exactly": True,
        "contract_sha256": expected["contract_sha256"],
        "anchor_ready": bool(expected["anchor_ready"]),
        "anchor_status": str(expected["anchor_status"]),
    }


__all__ = [
    "BRIDGE_COLUMNS",
    "CATEGORY_ABILITY_COLUMN",
    "CONTRACT",
    "KNN_DISTANCE_SCALE",
    "KNN_WEIGHT_FLOOR",
    "PopulationSupportContractError",
    "QUALITY_GROUP_CONTRACT",
    "SUPPORT_ABILITY_COLUMNS",
    "SUPPORT_MEASURE_COLUMNS",
    "donor_quality_group_key",
    "frame_sha256",
    "materialize_population_support_contract",
    "validate_population_support_contract",
]

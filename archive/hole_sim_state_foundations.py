"""Foundational state guards for the archived shot-simulator research.

This module owns two pieces that must be correct before recursive shot
simulation is meaningful:

* penalty bookkeeping is linked on the complete raw ShotLink sequence, then
  collapsed onto the causal physical stroke while retaining the exact next
  physical-stroke start state; and
* recurring-event editions are admitted as course donors only after their
  geometry verifies that they are the same physical venue as the holdout.

The R2 ``course_id`` is a setup/layout hash and commonly changes by round, so
it is useful for an exact shot-to-geometry join but is not a stable venue ID.
Event names are candidate generators only.  Geometry is the identity check.

Nothing in production imports this archived research module.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_OUTPUT = Path("permanent_data/hole_sim_state_foundations")
DEFAULT_EVENTS = (
    "Wyndham Championship",
    "Sony Open in Hawaii",
    "Travelers Championship",
)
FOUNDATION_VERSION = "full_raw_sequence_geometry_identity_v2"
FEET_PER_YARD = 3.0
PLAYED_EVENT_TYPES = frozenset({"STROKE", "PROVISIONAL"})


@dataclass(frozen=True)
class GeometryThresholds:
    """Conservative same-venue gates, calibrated against venue-change controls."""

    min_common_setups: int = 60
    max_aligned_rmse_yd: float = 50.0
    max_aligned_median_yd: float = 25.0
    max_aligned_p90_yd: float = 80.0
    max_mean_abs_yardage_delta: float = 45.0


def _require_unique_shot_keys(frame: pd.DataFrame, label: str) -> None:
    if "shot_key" not in frame:
        raise ValueError(f"{label} has no shot_key")
    missing = int(frame["shot_key"].isna().sum())
    duplicates = int(frame["shot_key"].duplicated(keep=False).sum())
    if missing or duplicates:
        raise ValueError(
            f"{label} shot_key integrity failure: missing={missing}, "
            f"duplicate_rows={duplicates}"
        )


def _map_order_columns(
    links: pd.DataFrame,
    ordered: pd.DataFrame,
    order_column: str,
    prefix: str,
    columns: Iterable[str],
) -> pd.DataFrame:
    lookup = ordered.set_index("_raw_sequence_order")
    for column in columns:
        links[f"{prefix}{column}"] = links[order_column].map(lookup[column])
    return links


def _safe_rate(mask: pd.Series | np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool)
    return float(values.mean()) if len(values) else math.nan


def _finite_gap(
    left_x: pd.Series,
    left_y: pd.Series,
    right_x: pd.Series,
    right_y: pd.Series,
) -> pd.Series:
    valid = (
        pd.to_numeric(left_x, errors="coerce").notna()
        & pd.to_numeric(left_y, errors="coerce").notna()
        & pd.to_numeric(right_x, errors="coerce").notna()
        & pd.to_numeric(right_y, errors="coerce").notna()
    )
    result = pd.Series(np.nan, index=left_x.index, dtype=float)
    result.loc[valid] = np.hypot(
        left_x.loc[valid].astype(float) - right_x.loc[valid].astype(float),
        left_y.loc[valid].astype(float) - right_y.loc[valid].astype(float),
    ) / FEET_PER_YARD
    return result


def link_app_penalties_full_sequence(
    raw: pd.DataFrame,
    app_sg: pd.DataFrame,
    shot_states: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Return physical APP strokes with sequence-safe penalty state attached.

    ``raw`` must contain every row in the requested event scope, not merely
    rows whose SG category is APP.  Export order is retained as the sequence
    tie-break because penalty and DROP rows can share ``stroke_no`` with an
    adjacent stroke.  ``shot_states`` contains the minimal all-category SG
    state used only to diagnose the next played row's SG annotation.

    The causal endpoint coordinates and original SG end state are never
    overwritten.  For a linked penalty, the public ``end_lie`` becomes the
    discrete ``penalty`` surface label. The original causal endpoint and SG
    distance remain in ``causal_end_*`` columns. The resolved recursive state
    is the raw start coordinate and raw lie from the exact next playable
    ``STROKE`` or ``PROVISIONAL`` row. Its distance must be recomputed from
    those coordinates to the forecast pin; SG start lie/distance are retained
    under explicitly diagnostic names and must not drive recursion.
    """

    raw_required = {
        "shot_key", "event_key", "source_player_id", "round_no", "hole_no",
        "stroke_no", "event_type", "course_id", "start_x", "start_y",
        "end_x", "end_y", "start_lie", "end_lie", "ball_holed",
    }
    sg_required = {
        "shot_key", "event_key", "source_player_id", "round_no", "hole_no",
        "stroke_no", "sg_category", "start_lie", "start_distance_yd",
        "end_lie", "end_distance_yd",
    }
    state_required = {
        "shot_key", "sg_category", "start_lie", "start_distance_yd",
        "end_lie", "end_distance_yd",
    }
    for frame, required, label in (
        (raw, raw_required, "raw shots"),
        (app_sg, sg_required, "APP SG"),
        (shot_states, state_required, "all-category SG states"),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
        _require_unique_shot_keys(frame, label)

    ordered = raw.reset_index(drop=True).copy()
    ordered["_raw_sequence_order"] = np.arange(len(ordered), dtype=np.int64)
    ordered["event_type"] = ordered["event_type"].fillna("").astype(str).str.upper()
    ordered["source_player_id"] = ordered["source_player_id"].astype(str)
    sequence_keys = ["event_key", "source_player_id", "round_no", "hole_no"]
    ordered = ordered.sort_values(sequence_keys + ["_raw_sequence_order"])

    # PROVISIONAL is a physically played shot, not bookkeeping. In the raw
    # feed it can be either the first playable row after a penalty or itself
    # the causal shot before a later penalty. DROP remains administrative.
    played = ordered["event_type"].isin(PLAYED_EVENT_TYPES)
    penalty = ordered["event_type"].eq("PENALTY")
    drop = ordered["event_type"].eq("DROP")
    groups = [ordered[column] for column in sequence_keys]
    ordered["_causal_order"] = ordered["_raw_sequence_order"].where(
        played
    ).groupby(groups, sort=False).ffill()
    ordered["_resolution_order"] = ordered["_raw_sequence_order"].where(
        played
    ).groupby(groups, sort=False).bfill()
    # Retain the prior STROKE-only contract strictly for an old-vs-new stroke
    # accounting audit. It is never used to construct the new state.
    stroke_only = ordered["event_type"].eq("STROKE")
    ordered["_old_stroke_only_causal_order"] = ordered[
        "_raw_sequence_order"
    ].where(stroke_only).groupby(groups, sort=False).ffill()
    ordered["_old_stroke_only_resolution_order"] = ordered[
        "_raw_sequence_order"
    ].where(stroke_only).groupby(groups, sort=False).bfill()
    ordered["_next_drop_order"] = ordered["_raw_sequence_order"].where(
        drop
    ).groupby(groups, sort=False).bfill()

    links = ordered.loc[
        penalty,
        sequence_keys
        + [
            "_raw_sequence_order", "_causal_order", "_resolution_order",
            "_old_stroke_only_causal_order",
            "_old_stroke_only_resolution_order", "_next_drop_order",
            "shot_key", "stroke_no", "course_id",
        ],
    ].rename(
        columns={
            "_raw_sequence_order": "penalty_sequence_order",
            "shot_key": "penalty_shot_key",
            "stroke_no": "penalty_stroke_no",
            "course_id": "penalty_course_id",
        }
    )
    raw_columns = [
        "shot_key", "stroke_no", "event_type", "course_id", "start_lie",
        "end_lie", "start_x", "start_y", "end_x", "end_y",
    ]
    links = _map_order_columns(
        links, ordered, "_causal_order", "causal_", raw_columns
    )
    links = _map_order_columns(
        links, ordered, "_resolution_order", "resolution_raw_", raw_columns
    )
    links = _map_order_columns(
        links,
        ordered,
        "_old_stroke_only_causal_order",
        "old_stroke_only_causal_",
        raw_columns,
    )
    links = _map_order_columns(
        links,
        ordered,
        "_old_stroke_only_resolution_order",
        "old_stroke_only_resolution_",
        raw_columns,
    )
    links = _map_order_columns(
        links, ordered, "_next_drop_order", "drop_", raw_columns
    )
    # A DROP only resolves this penalty when it lies between the bookkeeping
    # row and the next physical stroke. A later-hole DROP must never leak back.
    valid_drop_order = (
        links["_next_drop_order"].notna()
        & links["_resolution_order"].notna()
        & links["_next_drop_order"].gt(links["penalty_sequence_order"])
        & links["_next_drop_order"].lt(links["_resolution_order"])
    )
    for column in [column for column in links if column.startswith("drop_")]:
        links.loc[~valid_drop_order, column] = np.nan

    states = shot_states.set_index("shot_key")
    for column in (
        "sg_category", "start_lie", "start_distance_yd", "end_lie",
        "end_distance_yd",
    ):
        links[f"causal_sg_{column}"] = links["causal_shot_key"].map(states[column])
        links[f"penalty_sg_{column}"] = links["penalty_shot_key"].map(states[column])
        links[f"resolution_sg_{column}"] = links["resolution_raw_shot_key"].map(
            states[column]
        )
        links[f"old_stroke_only_causal_sg_{column}"] = links[
            "old_stroke_only_causal_shot_key"
        ].map(states[column])
        links[f"old_stroke_only_resolution_sg_{column}"] = links[
            "old_stroke_only_resolution_shot_key"
        ].map(states[column])

    links["causal_linked"] = links["causal_shot_key"].notna()
    links["resolution_linked"] = links["resolution_raw_shot_key"].notna()
    links["causal_category"] = (
        links["causal_sg_sg_category"].fillna("").astype(str).str.casefold()
    )
    links["causal_is_app"] = links["causal_category"].eq("app")
    links["causal_is_ott"] = links["causal_category"].eq("ott")
    links["penalty_row_is_app"] = links["penalty_sg_sg_category"].eq("app")
    links["same_course_resolution"] = (
        links["causal_course_id"].astype("string")
        == links["resolution_raw_course_id"].astype("string")
    )
    links["causal_to_resolution_gap_yd"] = _finite_gap(
        links["causal_end_x"], links["causal_end_y"],
        links["resolution_raw_start_x"], links["resolution_raw_start_y"],
    )
    links["drop_to_resolution_gap_yd"] = _finite_gap(
        links["drop_end_x"], links["drop_end_y"],
        links["resolution_raw_start_x"], links["resolution_raw_start_y"],
    )
    links["penalty_state_distance_error_yd"] = (
        pd.to_numeric(links["penalty_sg_end_distance_yd"], errors="coerce")
        - pd.to_numeric(links["resolution_sg_start_distance_yd"], errors="coerce")
    ).abs()
    links["resolution_raw_lie_matches_sg"] = (
        links["resolution_raw_start_lie"].fillna("").astype(str).str.casefold()
        == links["resolution_sg_start_lie"].fillna("").astype(str).str.casefold()
    ) & links["resolution_raw_start_lie"].notna() & links[
        "resolution_sg_start_lie"
    ].notna()

    # Reconstruct what the old APP-filtered linkage would have selected. This
    # is audit-only; it is never used to construct the transitions.
    app_keys = set(app_sg["shot_key"].astype(str))
    legacy = ordered[ordered["shot_key"].astype(str).isin(app_keys)].copy()
    legacy_groups = [legacy[column] for column in sequence_keys]
    legacy["_legacy_causal_order"] = legacy["_raw_sequence_order"].where(
        legacy["event_type"].eq("STROKE")
    ).groupby(legacy_groups, sort=False).ffill()
    legacy_penalties = legacy.loc[
        legacy["event_type"].eq("PENALTY"),
        ["shot_key", "_legacy_causal_order"],
    ].copy()
    order_to_key = ordered.set_index("_raw_sequence_order")["shot_key"]
    legacy_penalties["legacy_causal_shot_key"] = legacy_penalties[
        "_legacy_causal_order"
    ].map(order_to_key)
    legacy_map = legacy_penalties.set_index("shot_key")["legacy_causal_shot_key"]
    links["legacy_causal_shot_key"] = links["penalty_shot_key"].map(legacy_map)
    legacy_comparable = links["legacy_causal_shot_key"].notna() & links["causal_linked"]
    links["legacy_causal_mismatch"] = legacy_comparable & (
        links["legacy_causal_shot_key"].astype("string")
        != links["causal_shot_key"].astype("string")
    )

    valid_app_links = links[links["causal_linked"] & links["causal_is_app"]].copy()
    resolution_counts = valid_app_links.groupby("causal_shot_key", observed=True)[
        "resolution_raw_shot_key"
    ].nunique(dropna=False)
    ambiguous_causal = set(resolution_counts[resolution_counts.gt(1)].index.astype(str))
    attachable = valid_app_links[
        ~valid_app_links["causal_shot_key"].astype(str).isin(ambiguous_causal)
    ].copy()

    def _first(series: pd.Series) -> object:
        nonnull = series.dropna()
        return nonnull.iloc[0] if len(nonnull) else np.nan

    if len(attachable):
        attached = attachable.groupby("causal_shot_key", as_index=False).agg(
            penalty_count=("penalty_shot_key", "size"),
            penalty_shot_keys=(
                "penalty_shot_key",
                lambda values: "|".join(values.astype(str)),
            ),
            resolution_shot_key=("resolution_raw_shot_key", _first),
            causal_stroke_no=("causal_stroke_no", _first),
            resolution_stroke_no=("resolution_raw_stroke_no", _first),
            resolution_event_type=("resolution_raw_event_type", _first),
            resolution_start_x=("resolution_raw_start_x", _first),
            resolution_start_y=("resolution_raw_start_y", _first),
            resolution_start_lie_raw=("resolution_raw_start_lie", _first),
            resolution_start_lie=("resolution_raw_start_lie", _first),
            resolution_sg_start_lie_diagnostic=(
                "resolution_sg_start_lie", _first
            ),
            resolution_sg_start_distance_yd_diagnostic=(
                "resolution_sg_start_distance_yd", _first
            ),
            resolution_course_id=("resolution_raw_course_id", _first),
            drop_shot_key=("drop_shot_key", _first),
            drop_end_x=("drop_end_x", _first),
            drop_end_y=("drop_end_y", _first),
        )
    else:
        attached = pd.DataFrame(
            columns=[
                "causal_shot_key", "penalty_count", "penalty_shot_keys",
                "resolution_shot_key", "causal_stroke_no",
                "resolution_stroke_no", "resolution_start_x",
                "resolution_start_y", "resolution_start_lie_raw",
                "resolution_event_type", "resolution_start_lie",
                "resolution_sg_start_lie_diagnostic",
                "resolution_sg_start_distance_yd_diagnostic",
                "resolution_course_id", "drop_shot_key", "drop_end_x",
                "drop_end_y",
            ]
        )

    attached["penalty_stroke_cost"] = (
        pd.to_numeric(attached["resolution_stroke_no"], errors="coerce")
        - pd.to_numeric(attached["causal_stroke_no"], errors="coerce")
        - 1
    )

    raw_for_merge = raw.copy().rename(
        columns={
            "start_lie": "raw_start_lie",
            "end_lie": "raw_end_lie",
        }
    )
    raw_keep = [
        "shot_key", "event_type", "ball_holed", "start_x", "start_y",
        "end_x", "end_y", "course_id", "raw_start_lie", "raw_end_lie",
    ]
    frame = app_sg.merge(
        raw_for_merge[raw_keep], on="shot_key", how="left", validate="one_to_one"
    )
    frame["event_type"] = frame["event_type"].fillna("").astype(str).str.upper()
    frame = frame[frame["event_type"].eq("STROKE")].copy()
    frame["causal_end_x"] = frame["end_x"]
    frame["causal_end_y"] = frame["end_y"]
    frame["causal_end_lie"] = frame["end_lie"]
    frame["causal_end_distance_yd"] = frame["end_distance_yd"]
    frame = frame.merge(
        attached,
        left_on="shot_key",
        right_on="causal_shot_key",
        how="left",
        validate="one_to_one",
    ).drop(columns="causal_shot_key")
    frame["penalty_count"] = frame["penalty_count"].fillna(0).astype(int)
    frame["penalty_after"] = frame["penalty_count"].gt(0)
    frame.loc[frame["penalty_after"], "end_lie"] = "penalty"
    if "state_inferred" in frame:
        prior = frame["state_inferred"].fillna("").astype(str).str.strip(",")
        suffix = np.where(
            frame["penalty_after"], "causal_penalty_full_sequence", ""
        )
        frame["state_inferred"] = (
            prior + np.where(prior.ne("") & frame["penalty_after"], ",", "") + suffix
        ).str.strip(",")
    frame["foundation_version"] = FOUNDATION_VERSION

    hazard_frame = frame[frame["penalty_after"]]
    endpoint_preserved = (
        hazard_frame[["end_x", "end_y"]].to_numpy(float)
        == hazard_frame[["causal_end_x", "causal_end_y"]].to_numpy(float)
    ).all(axis=1) if len(hazard_frame) else np.array([], dtype=bool)
    drop_gap = links["drop_to_resolution_gap_yd"].dropna()
    causal_gap = links["causal_to_resolution_gap_yd"].dropna()
    state_error = links["penalty_state_distance_error_yd"].dropna()
    old_supported = links[
        links["old_stroke_only_causal_sg_sg_category"].isin(["ott", "app"])
        & links["old_stroke_only_causal_shot_key"].notna()
        & links["old_stroke_only_resolution_shot_key"].notna()
    ].copy()
    if len(old_supported):
        old_accounting = old_supported.groupby(
            ["event_key", "old_stroke_only_causal_shot_key"],
            as_index=False,
            observed=True,
        ).agg(
            penalty_count=("penalty_shot_key", "size"),
            causal_stroke_no=("old_stroke_only_causal_stroke_no", "first"),
            resolution_stroke_no=(
                "old_stroke_only_resolution_stroke_no", "first"
            ),
            resolution_keys=("old_stroke_only_resolution_shot_key", "nunique"),
        )
        old_accounting["stroke_span_cost"] = (
            pd.to_numeric(old_accounting["resolution_stroke_no"], errors="coerce")
            - pd.to_numeric(old_accounting["causal_stroke_no"], errors="coerce")
            - 1
        )
        old_accounting["stroke_undercount"] = (
            old_accounting["stroke_span_cost"] - old_accounting["penalty_count"]
        )
    else:
        old_accounting = pd.DataFrame(
            columns=["resolution_keys", "stroke_undercount"]
        )
    # Penalty-associated DROP coordinates are often missing in ShotLink. Audit
    # continuity on every coordinate-complete DROP in scope as an independent
    # check that bfill truly selects the next physical stroke.
    all_drops = ordered.loc[drop, ["_resolution_order", "end_x", "end_y"]].copy()
    resolution_lookup = ordered.set_index("_raw_sequence_order")
    all_drops["resolution_start_x"] = all_drops["_resolution_order"].map(
        resolution_lookup["start_x"]
    )
    all_drops["resolution_start_y"] = all_drops["_resolution_order"].map(
        resolution_lookup["start_y"]
    )
    all_drop_gap = _finite_gap(
        all_drops["end_x"], all_drops["end_y"],
        all_drops["resolution_start_x"], all_drops["resolution_start_y"],
    ).dropna()
    audit: dict[str, float | int | str] = {
        "foundation_version": FOUNDATION_VERSION,
        "season": int(season),
        "raw_rows": int(len(raw)),
        "physical_app_strokes": int(len(frame)),
        "penalty_rows": int(len(links)),
        "full_causal_link_rate": _safe_rate(links["causal_linked"]),
        "full_resolution_link_rate": _safe_rate(links["resolution_linked"]),
        "causal_provisional_penalty_rows": int(
            links["causal_event_type"].eq("PROVISIONAL").sum()
        ),
        "resolution_provisional_penalty_rows": int(
            links["resolution_raw_event_type"].eq("PROVISIONAL").sum()
        ),
        "resolution_provisional_rate_all": _safe_rate(
            links["resolution_raw_event_type"].eq("PROVISIONAL")
        ),
        "causal_app_penalty_rows": int(links["causal_is_app"].sum()),
        "attached_app_hazard_strokes": int(frame["penalty_after"].sum()),
        "ambiguous_app_causal_strokes": int(len(ambiguous_causal)),
        "resolution_sg_state_rate_app": _safe_rate(
            valid_app_links["resolution_sg_start_distance_yd"].notna()
        ),
        "resolution_raw_lie_rate_app": _safe_rate(
            valid_app_links["resolution_raw_start_lie"].notna()
        ),
        "resolution_raw_vs_sg_lie_match_rate_app": _safe_rate(
            valid_app_links["resolution_raw_lie_matches_sg"]
        ),
        "resolution_provisional_rate_app": _safe_rate(
            valid_app_links["resolution_raw_event_type"].eq("PROVISIONAL")
        ),
        "same_course_resolution_rate_app": _safe_rate(
            valid_app_links["same_course_resolution"]
        ),
        "legacy_app_penalty_rows": int(links["penalty_row_is_app"].sum()),
        "legacy_full_causal_comparable": int(legacy_comparable.sum()),
        "legacy_causal_mismatch_rate": _safe_rate(
            links.loc[legacy_comparable, "legacy_causal_mismatch"]
        ),
        "legacy_false_app_assignment_rows": int(
            (links["penalty_row_is_app"] & ~links["causal_is_app"]).sum()
        ),
        "causal_endpoint_preserved_rate_after": _safe_rate(endpoint_preserved),
        "old_stroke_only_supported_donors_before_state_gates": int(
            len(old_accounting)
        ),
        "old_stroke_only_undercount_donors_before_state_gates": int(
            (
                old_accounting["resolution_keys"].eq(1)
                & old_accounting["stroke_undercount"].gt(0)
            ).sum()
        ),
        "causal_to_resolution_median_gap_yd_before": float(causal_gap.median())
        if len(causal_gap) else math.nan,
        "drop_resolution_comparable": int(len(drop_gap)),
        "drop_to_resolution_median_gap_yd_after": float(drop_gap.median())
        if len(drop_gap) else math.nan,
        "drop_to_resolution_p99_gap_yd_after": float(drop_gap.quantile(0.99))
        if len(drop_gap) else math.nan,
        "drop_to_resolution_within_0_01yd_rate_after": _safe_rate(
            drop_gap.le(0.01)
        ),
        "all_drop_resolution_comparable": int(len(all_drop_gap)),
        "all_drop_to_next_physical_median_gap_yd": float(all_drop_gap.median())
        if len(all_drop_gap) else math.nan,
        "all_drop_to_next_physical_p99_gap_yd": float(all_drop_gap.quantile(0.99))
        if len(all_drop_gap) else math.nan,
        "all_drop_to_next_physical_within_0_01yd_rate": _safe_rate(
            all_drop_gap.le(0.01)
        ),
        "penalty_sg_vs_resolution_median_ae_yd": float(state_error.median())
        if len(state_error) else math.nan,
    }
    return frame.reset_index(drop=True), links.reset_index(drop=True), audit


def build_causal_hazard_labels(penalty_links: pd.DataFrame) -> pd.DataFrame:
    """Return one causal penalty label per physical shot key.

    This is the public bridge for incidence models. In particular, an OTT
    training frame can join on ``shot_key`` and use ``tee_hazard_after`` / 
    ``tee_penalty_count`` rather than trying to infer the causal tee shot from
    the SG category attached to a later PENALTY bookkeeping row.

    The label is valid even when a resolution is unavailable: incidence is an
    observed fact. ``resolution_coherent`` separately tells downstream state
    simulators whether all assessed penalties agree on one next playable row.
    """

    required = {
        "causal_shot_key", "penalty_shot_key", "causal_linked",
        "causal_sg_sg_category", "causal_event_type",
        "causal_stroke_no", "resolution_raw_stroke_no",
        "resolution_raw_shot_key", "resolution_raw_event_type",
    }
    missing = required.difference(penalty_links.columns)
    if missing:
        raise ValueError(f"Penalty links missing label columns: {sorted(missing)}")
    linked = penalty_links[
        penalty_links["causal_linked"].fillna(False).astype(bool)
        & penalty_links["causal_shot_key"].notna()
    ].copy()
    columns = [
        "shot_key", "causal_hazard_after", "causal_penalty_count",
        "causal_penalty_stroke_cost",
        "causal_hazard_category", "causal_event_type", "tee_hazard_after",
        "tee_penalty_count", "tee_penalty_stroke_cost", "app_hazard_after",
        "app_penalty_count", "app_penalty_stroke_cost",
        "resolution_shot_key", "resolution_event_type",
        "resolution_coherent",
    ]
    if linked.empty:
        return pd.DataFrame(columns=columns)
    linked["causal_hazard_category"] = (
        linked.get("causal_category", linked["causal_sg_sg_category"])
        .fillna("").astype(str).str.casefold()
    )

    def _first(values: pd.Series) -> object:
        nonnull = values.dropna()
        return nonnull.iloc[0] if len(nonnull) else np.nan

    labels = linked.groupby("causal_shot_key", as_index=False, observed=True).agg(
        causal_penalty_count=("penalty_shot_key", "size"),
        causal_hazard_category=("causal_hazard_category", _first),
        causal_event_type=("causal_event_type", _first),
        causal_stroke_no=("causal_stroke_no", _first),
        resolution_stroke_no=("resolution_raw_stroke_no", _first),
        resolution_shot_key=("resolution_raw_shot_key", _first),
        resolution_event_type=("resolution_raw_event_type", _first),
        _resolution_keys=("resolution_raw_shot_key", "nunique"),
        _categories=("causal_hazard_category", "nunique"),
    ).rename(columns={"causal_shot_key": "shot_key"})
    labels["causal_hazard_after"] = True
    labels["causal_penalty_stroke_cost"] = (
        pd.to_numeric(labels["resolution_stroke_no"], errors="coerce")
        - pd.to_numeric(labels["causal_stroke_no"], errors="coerce")
        - 1
    )
    labels["resolution_coherent"] = (
        labels["_resolution_keys"].eq(1) & labels["_categories"].eq(1)
    )
    labels["tee_hazard_after"] = labels["causal_hazard_category"].eq("ott")
    labels["tee_penalty_count"] = np.where(
        labels["tee_hazard_after"], labels["causal_penalty_count"], 0
    ).astype(int)
    labels["tee_penalty_stroke_cost"] = np.where(
        labels["tee_hazard_after"], labels["causal_penalty_stroke_cost"], 0
    )
    labels["app_hazard_after"] = labels["causal_hazard_category"].eq("app")
    labels["app_penalty_count"] = np.where(
        labels["app_hazard_after"], labels["causal_penalty_count"], 0
    ).astype(int)
    labels["app_penalty_stroke_cost"] = np.where(
        labels["app_hazard_after"], labels["causal_penalty_stroke_cost"], 0
    )
    labels = labels.drop(
        columns=[
            "_resolution_keys", "_categories", "causal_stroke_no",
            "resolution_stroke_no",
        ]
    )
    if labels["shot_key"].duplicated().any():
        raise AssertionError("Causal hazard labels are not unique by shot_key")
    return labels[columns].sort_values("shot_key").reset_index(drop=True)


def attach_causal_hazard_labels(
    shots: pd.DataFrame,
    penalty_links: pd.DataFrame,
) -> pd.DataFrame:
    """Attach causal hazard incidence to any shot frame by ``shot_key``.

    Non-hazard shots are explicitly filled with False/zero, making the output
    directly usable as a binary OTT or APP incidence target.
    """

    if "shot_key" not in shots:
        raise ValueError("Shot frame has no shot_key")
    labels = build_causal_hazard_labels(penalty_links)
    overlap = set(labels.columns).intersection(shots.columns).difference({"shot_key"})
    if overlap:
        raise ValueError(f"Shot frame already has hazard label columns: {sorted(overlap)}")
    out = shots.merge(labels, on="shot_key", how="left", validate="many_to_one")
    bool_columns = [
        "causal_hazard_after", "tee_hazard_after", "app_hazard_after",
        "resolution_coherent",
    ]
    count_columns = [
        "causal_penalty_count", "tee_penalty_count", "app_penalty_count",
    ]
    for column in bool_columns:
        out[column] = out[column].eq(True)
    for column in count_columns:
        out[column] = out[column].fillna(0).astype(int)
    for column in (
        "causal_penalty_stroke_cost", "tee_penalty_stroke_cost",
        "app_penalty_stroke_cost",
    ):
        out.loc[~out["causal_hazard_after"], column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    return out


def build_causal_ott_hazard_labels(penalty_links: pd.DataFrame) -> pd.DataFrame:
    """Return positive OTT incidence labels in the v2 tee-training contract.

    Join this compact table to the complete tee-shot training frame by
    ``shot_key`` and fill absent counts/costs with zero. This replaces the
    unsafe practice of treating a PENALTY row's own SG category as causal.
    """

    labels = build_causal_hazard_labels(penalty_links)
    labels = labels[labels["tee_hazard_after"]].copy()
    if "season" in penalty_links:
        season_by_key = (
            penalty_links[
                penalty_links["causal_linked"].fillna(False).astype(bool)
            ]
            .groupby("causal_shot_key", observed=True)["season"]
            .first()
        )
        labels["season"] = labels["shot_key"].map(season_by_key)
    else:
        labels["season"] = pd.NA
    keep = [
        "shot_key", "season", "tee_penalty_count",
        "tee_penalty_stroke_cost", "resolution_shot_key",
        "resolution_event_type", "resolution_coherent",
    ]
    return labels[keep].reset_index(drop=True)


def _event_structure(geometry: pd.DataFrame) -> dict[str, float | int | bool]:
    frame = geometry[geometry["hole_no"].between(1, 18)].copy()
    if frame.empty:
        return {
            "geometry_rows": 0,
            "rounds": 0,
            "layout_ids": 0,
            "max_layouts_per_round": 0,
            "max_rows_per_round": 0,
            "duplicate_round_hole_rows": 0,
            "single_course_structure": False,
        }
    by_round = frame.groupby("round_no", observed=True).agg(
        rows=("hole_no", "size"),
        holes=("hole_no", "nunique"),
        layout_ids=("course_id", "nunique"),
    )
    duplicates = int(frame.duplicated(["round_no", "hole_no"], keep=False).sum())
    single = bool(
        len(by_round) >= 3
        and by_round["rows"].eq(18).all()
        and by_round["holes"].eq(18).all()
        and by_round["layout_ids"].eq(1).all()
        and duplicates == 0
    )
    return {
        "geometry_rows": int(len(frame)),
        "rounds": int(frame["round_no"].nunique()),
        "layout_ids": int(frame["course_id"].nunique()),
        "max_layouts_per_round": int(by_round["layout_ids"].max()),
        "max_rows_per_round": int(by_round["rows"].max()),
        "duplicate_round_hole_rows": duplicates,
        "single_course_structure": single,
    }


def _rigid_alignment(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    u, _, vt = np.linalg.svd(left_centered.T @ right_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    fitted = left_centered @ rotation
    translation = right.mean(axis=0) - left.mean(axis=0) @ rotation
    distances = np.linalg.norm(fitted - right_centered, axis=1) / FEET_PER_YARD
    return rotation, translation, distances


def compare_course_geometry(
    candidate: pd.DataFrame,
    target: pd.DataFrame,
) -> dict[str, float | int]:
    """Compare full round/hole tee+pin geometry after proper rigid alignment."""

    keys = ["round_no", "hole_no"]
    values = ["tee_x", "tee_y", "pin_x", "pin_y", "yardage"]
    left = candidate.groupby(keys, as_index=False)[values].mean()
    right = target.groupby(keys, as_index=False)[values].mean()
    merged = left.merge(right, on=keys, suffixes=("_candidate", "_target"))
    if merged.empty:
        return {
            "common_setups": 0,
            "aligned_rmse_yd": math.inf,
            "aligned_median_yd": math.inf,
            "aligned_p90_yd": math.inf,
            "mean_abs_yardage_delta": math.inf,
        }
    left_points = np.vstack(
        [
            merged[["tee_x_candidate", "tee_y_candidate"]].to_numpy(float),
            merged[["pin_x_candidate", "pin_y_candidate"]].to_numpy(float),
        ]
    )
    right_points = np.vstack(
        [
            merged[["tee_x_target", "tee_y_target"]].to_numpy(float),
            merged[["pin_x_target", "pin_y_target"]].to_numpy(float),
        ]
    )
    finite = np.isfinite(left_points).all(axis=1) & np.isfinite(right_points).all(axis=1)
    if int(finite.sum()) < 2:
        rotation = np.full((2, 2), np.nan)
        translation = np.full(2, np.nan)
        aligned = np.array([math.inf])
    else:
        rotation, translation, aligned = _rigid_alignment(
            left_points[finite], right_points[finite]
        )
    yardage_delta = (
        merged["yardage_candidate"] - merged["yardage_target"]
    ).abs()
    return {
        "common_setups": int(len(merged)),
        "aligned_rmse_yd": float(np.sqrt(np.mean(aligned**2))),
        "aligned_median_yd": float(np.median(aligned)),
        "aligned_p90_yd": float(np.quantile(aligned, 0.90)),
        "mean_abs_yardage_delta": float(yardage_delta.mean()),
        "rotation_00": float(rotation[0, 0]),
        "rotation_01": float(rotation[0, 1]),
        "rotation_10": float(rotation[1, 0]),
        "rotation_11": float(rotation[1, 1]),
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
    }


def align_coordinates_to_target(
    frame: pd.DataFrame,
    identity_row: pd.Series | object,
    coordinate_prefixes: Iterable[str],
) -> pd.DataFrame:
    """Map one accepted edition into the audited reference coordinate frame.

    With ``reference_mode='latest_prior'`` the reference is the latest edition
    strictly before the holdout, so historical donor coordinates do not use
    heldout tee/pin geometry.
    """

    getter = (
        identity_row.get
        if isinstance(identity_row, pd.Series)
        else lambda name: getattr(identity_row, name)
    )
    accepted = bool(getter("accepted"))
    if not accepted:
        raise ValueError("Refusing to align coordinates for a rejected edition")
    rotation = np.array(
        [
            [getter("rotation_00"), getter("rotation_01")],
            [getter("rotation_10"), getter("rotation_11")],
        ],
        dtype=float,
    )
    translation = np.array(
        [getter("translation_x"), getter("translation_y")], dtype=float
    )
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Accepted edition has no finite coordinate transform")
    out = frame.copy()
    for prefix in coordinate_prefixes:
        x_column, y_column = f"{prefix}_x", f"{prefix}_y"
        if x_column not in out or y_column not in out:
            continue
        values = out[[x_column, y_column]].to_numpy(float)
        finite = np.isfinite(values).all(axis=1)
        transformed = values[finite] @ rotation + translation
        out.loc[finite, x_column] = transformed[:, 0]
        out.loc[finite, y_column] = transformed[:, 1]
    out["physical_course_id"] = str(getter("physical_course_id"))
    out["geometry_alignment_rmse_yd"] = float(getter("aligned_rmse_yd"))
    return out


def physical_course_fingerprint(geometry: pd.DataFrame) -> str:
    """Build a rigid-invariant target fingerprint for audit/provenance."""

    holes = geometry.groupby("hole_no", as_index=True).agg(
        tee_x=("tee_x", "mean"),
        tee_y=("tee_y", "mean"),
        yardage=("yardage", "mean"),
    ).reindex(range(1, 19))
    if holes[["tee_x", "tee_y", "yardage"]].isna().any().any():
        raise ValueError("Cannot fingerprint course: incomplete 18-hole geometry")
    points = holes[["tee_x", "tee_y"]].to_numpy(float) / FEET_PER_YARD
    pairwise = np.hypot(
        points[:, None, 0] - points[None, :, 0],
        points[:, None, 1] - points[None, :, 1],
    )
    upper = pairwise[np.triu_indices(18, 1)]
    payload = np.concatenate([upper, holes["yardage"].to_numpy(float)])
    # Five-yard quantization prevents false precision from setup movement.
    quantized = np.rint(payload / 5.0).astype("<i4")
    digest = hashlib.sha256(quantized.tobytes()).hexdigest()[:16]
    return f"geometry:{digest}"


def verify_physical_course_identity(
    editions: pd.DataFrame,
    geometry: pd.DataFrame,
    holdout_year: int,
    thresholds: GeometryThresholds = GeometryThresholds(),
    reference_mode: str = "holdout",
) -> pd.DataFrame:
    """Audit exact-name editions and admit only verified same-course donors.

    The target must be a structurally single-course edition.  A multicourse
    target raises immediately.  Multicourse or geometry-mismatched historical
    editions remain in the audit as rejected rows but are never returned as
    accepted donors.
    """

    required_editions = {"event_key", "season", "event_name"}
    required_geometry = {
        "event_key", "round_no", "hole_no", "course_id", "yardage",
        "tee_x", "tee_y", "pin_x", "pin_y",
    }
    if missing := required_editions.difference(editions.columns):
        raise ValueError(f"Edition catalog missing columns: {sorted(missing)}")
    if missing := required_geometry.difference(geometry.columns):
        raise ValueError(f"Geometry missing columns: {sorted(missing)}")
    if editions.groupby("season", observed=True)["event_key"].nunique().gt(1).any():
        duplicate = editions.groupby("season")["event_key"].nunique()
        raise ValueError(
            "Multiple exact-name event keys in a season; physical identity is "
            f"ambiguous: {duplicate[duplicate.gt(1)].to_dict()}"
        )
    target_rows = editions[editions["season"].astype(int).eq(int(holdout_year))]
    if len(target_rows) != 1:
        raise ValueError(
            f"Expected exactly one {holdout_year} target edition, got "
            f"{target_rows['event_key'].tolist()}"
        )
    target_key = str(target_rows.iloc[0]["event_key"])
    if reference_mode == "holdout":
        reference_rows = target_rows
    elif reference_mode == "latest_prior":
        prior = editions[editions["season"].astype(int).lt(int(holdout_year))]
        if prior.empty:
            raise ValueError(
                f"No prior edition can define the {holdout_year} forecast frame"
            )
        reference_rows = prior[
            prior["season"].astype(int).eq(int(prior["season"].max()))
        ]
    else:
        raise ValueError(f"Unknown reference_mode: {reference_mode!r}")
    if len(reference_rows) != 1:
        raise ValueError(
            f"Expected one {reference_mode} reference edition, got "
            f"{reference_rows['event_key'].tolist()}"
        )
    reference_key = str(reference_rows.iloc[0]["event_key"])
    reference_geometry = geometry[
        geometry["event_key"].astype(str).eq(reference_key)
    ].copy()
    reference_structure = _event_structure(reference_geometry)
    if not reference_structure["single_course_structure"]:
        raise ValueError(
            f"Reference {reference_key} is not a verified single-course edition: "
            f"{reference_structure}. Multicourse targets require an explicit "
            "course assignment and are intentionally unsupported."
        )
    physical_id = physical_course_fingerprint(reference_geometry)

    rows: list[dict[str, object]] = []
    for edition in editions.sort_values("season").itertuples(index=False):
        event_key = str(edition.event_key)
        sample = geometry[geometry["event_key"].astype(str).eq(event_key)].copy()
        structure = _event_structure(sample)
        row: dict[str, object] = {
            "event_name": str(edition.event_name),
            "season": int(edition.season),
            "event_key": event_key,
            "target_event_key": target_key,
            "reference_event_key": reference_key,
            "reference_mode": reference_mode,
            "physical_course_id": physical_id,
            **structure,
        }
        if event_key == reference_key:
            metrics = compare_course_geometry(sample, reference_geometry)
            accepted = True
            status = (
                "target_verified" if reference_mode == "holdout"
                else "latest_prior_reference"
            )
        elif not structure["single_course_structure"]:
            metrics = {
                "common_setups": 0,
                "aligned_rmse_yd": math.inf,
                "aligned_median_yd": math.inf,
                "aligned_p90_yd": math.inf,
                "mean_abs_yardage_delta": math.inf,
            }
            accepted = False
            status = "rejected_multicourse_or_incomplete"
        else:
            metrics = compare_course_geometry(sample, reference_geometry)
            accepted = bool(
                metrics["common_setups"] >= thresholds.min_common_setups
                and metrics["aligned_rmse_yd"] <= thresholds.max_aligned_rmse_yd
                and metrics["aligned_median_yd"] <= thresholds.max_aligned_median_yd
                and metrics["aligned_p90_yd"] <= thresholds.max_aligned_p90_yd
                and metrics["mean_abs_yardage_delta"]
                <= thresholds.max_mean_abs_yardage_delta
            )
            if accepted and event_key == target_key and reference_mode == "latest_prior":
                status = "heldout_verified_against_prior"
            else:
                status = "verified_same_course" if accepted else "rejected_venue_mismatch"
        row.update(metrics)
        row["accepted"] = accepted
        row["status"] = status
        rows.append(row)
    audit = pd.DataFrame(rows)
    if not audit.loc[audit["accepted"], "event_key"].eq(reference_key).any():
        raise AssertionError("Reference course identity was not retained")
    return audit


def _read_event_geometry(
    export_root: Path,
    editions: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "event_key", "round_no", "hole_no", "course_id", "par", "yardage",
        "tee_x", "tee_y", "pin_x", "pin_y",
    ]
    parts: list[pd.DataFrame] = []
    for season, group in editions.groupby("season", observed=True):
        part = pd.read_parquet(
            export_root / f"pga_{int(season)}_hole_geometry.parquet",
            columns=columns,
            filters=[("event_key", "in", group["event_key"].tolist())],
        )
        if len(part):
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def load_verified_recurring_event(
    export_root: Path,
    event_name: str,
    holdout_year: int,
    reference_mode: str = "holdout",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(
        export_root / "events.parquet",
        columns=["event_key", "tour", "season", "event_name"],
    )
    editions = events[
        events["tour"].str.casefold().eq("pga")
        & events["event_name"].str.casefold().eq(event_name.casefold())
        & events["season"].le(holdout_year)
    ].copy()
    if editions.empty:
        raise ValueError(f"No PGA editions named {event_name!r}")
    geometry = _read_event_geometry(export_root, editions)
    audit = verify_physical_course_identity(
        editions, geometry, holdout_year, reference_mode=reference_mode
    )
    accepted_keys = set(audit.loc[audit["accepted"], "event_key"].astype(str))
    verified = editions[editions["event_key"].astype(str).isin(accepted_keys)].copy()
    return verified, geometry, audit


def _read_penalty_inputs(
    export_root: Path,
    season: int,
    event_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_columns = [
        "shot_key", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "event_type", "start_lie", "end_lie",
        "distance_remaining_yd", "course_id", "start_x", "start_y", "end_x",
        "end_y", "ball_holed",
    ]
    sg_columns = [
        "shot_key", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "sg_category", "start_lie",
        "start_distance_yd", "end_lie", "end_distance_yd", "sg_valid",
        "sg_issue", "complete_hole", "state_inferred",
    ]
    state_columns = [
        "shot_key", "sg_category", "start_lie", "start_distance_yd", "end_lie",
        "end_distance_yd",
    ]
    filters = [("event_key", "in", event_keys)]
    raw = pd.read_parquet(
        export_root / f"pga_{season}_shots.parquet",
        columns=raw_columns,
        filters=filters,
    )
    app_filters = filters + [("sg_category", "==", "app")]
    app = pd.read_parquet(
        export_root / f"pga_{season}_shot_sg.parquet",
        columns=sg_columns,
        filters=app_filters,
    )
    states = pd.read_parquet(
        export_root / f"pga_{season}_shot_sg.parquet",
        columns=state_columns,
        filters=filters,
    )
    return raw, app, states


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    printable = frame.copy()
    for column in printable.select_dtypes(include=["float"]).columns:
        printable[column] = printable[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    headers = [str(column) for column in printable.columns]
    rows = [[str(value) for value in row] for row in printable.itertuples(index=False, name=None)]
    widths = [
        max([len(headers[index]), *[len(row[index]) for row in rows]])
        for index in range(len(headers))
    ]
    line = lambda row: "| " + " | ".join(
        str(value).ljust(widths[index]) for index, value in enumerate(row)
    ) + " |"
    return "\n".join(
        [line(headers), line(["-" * width for width in widths]), *[line(row) for row in rows]]
    )


def run_foundation_audit(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_parts: list[pd.DataFrame] = []
    verified_by_year: dict[int, list[str]] = {}
    for event_name in args.event_name:
        verified, _, audit = load_verified_recurring_event(
            args.export_root, event_name, args.holdout_year,
            reference_mode="latest_prior",
        )
        audit["audit_role"] = "primary"
        identity_parts.append(audit)
        for season, group in verified.groupby("season", observed=True):
            verified_by_year.setdefault(int(season), []).extend(group["event_key"].tolist())

    # The Genesis 2025 relocation is a labelled negative control. It should be
    # rejected against the 2026 Riviera target while the Riviera editions pass.
    if args.venue_change_control:
        control_holdout_year = args.holdout_year - 1
        _, _, control = load_verified_recurring_event(
            args.export_root, args.venue_change_control, control_holdout_year,
            reference_mode="latest_prior",
        )
        control["audit_role"] = "venue_change_control"
        identity_parts.append(control)

    multicourse_error = "not_run"
    if args.multicourse_control:
        try:
            load_verified_recurring_event(
                args.export_root, args.multicourse_control, args.holdout_year,
                reference_mode="latest_prior",
            )
        except ValueError as exc:
            multicourse_error = str(exc)
        else:
            raise AssertionError(
                f"Multicourse control {args.multicourse_control!r} did not fail closed"
            )

    identity = pd.concat(identity_parts, ignore_index=True)
    identity.to_csv(args.output_dir / "physical_course_identity_audit.csv", index=False)

    penalty_audits: list[dict[str, float | int | str]] = []
    link_parts: list[pd.DataFrame] = []
    seasons = sorted(verified_by_year)
    if args.smoke_test:
        seasons = seasons[-2:]
    for season in seasons:
        keys = sorted(set(verified_by_year[season]))
        raw, app, states = _read_penalty_inputs(args.export_root, season, keys)
        _, links, audit = link_app_penalties_full_sequence(raw, app, states, season)
        penalty_audits.append(audit)
        if len(links):
            links["season"] = season
            link_parts.append(links)
        print(
            f"{season}: penalties={audit['penalty_rows']}, "
            f"linked={audit['full_causal_link_rate']:.3%}, "
            f"APP hazards={audit['attached_app_hazard_strokes']}"
        )
    penalty = pd.DataFrame(penalty_audits)
    penalty.to_csv(args.output_dir / "penalty_sequence_audit.csv", index=False)
    if link_parts:
        links = pd.concat(link_parts, ignore_index=True)
        labels = build_causal_hazard_labels(links)
        labels.to_csv(
            args.output_dir / "causal_hazard_labels_by_shot_key.csv", index=False
        )
        build_causal_ott_hazard_labels(links).to_csv(
            args.output_dir / "causal_ott_hazard_labels_by_shot_key.csv",
            index=False,
        )
        link_columns = [
            "season", "event_key", "round_no", "hole_no",
            "penalty_shot_key", "causal_shot_key", "causal_event_type",
            "causal_sg_sg_category", "resolution_raw_shot_key",
            "resolution_raw_event_type", "resolution_raw_start_lie",
            "resolution_sg_start_lie",
            "resolution_sg_start_distance_yd", "drop_shot_key",
            "causal_to_resolution_gap_yd", "drop_to_resolution_gap_yd",
            "legacy_causal_shot_key", "legacy_causal_mismatch",
        ]
        links[link_columns].to_csv(
            args.output_dir / "penalty_links_full_sequence.csv", index=False
        )

    identity_display = identity[
        [
            "audit_role", "event_name", "season", "event_key", "status",
            "reference_mode", "reference_event_key", "accepted",
            "aligned_rmse_yd", "aligned_p90_yd",
            "mean_abs_yardage_delta", "max_layouts_per_round",
        ]
    ]
    penalty_display = penalty[
        [
            "season", "penalty_rows", "full_causal_link_rate",
            "full_resolution_link_rate", "causal_provisional_penalty_rows",
            "resolution_provisional_penalty_rows", "causal_app_penalty_rows",
            "attached_app_hazard_strokes", "legacy_causal_mismatch_rate",
            "legacy_false_app_assignment_rows", "resolution_sg_state_rate_app",
            "resolution_raw_vs_sg_lie_match_rate_app",
            "causal_endpoint_preserved_rate_after",
            "old_stroke_only_supported_donors_before_state_gates",
            "old_stroke_only_undercount_donors_before_state_gates",
            "all_drop_resolution_comparable",
            "all_drop_to_next_physical_median_gap_yd",
            "all_drop_to_next_physical_within_0_01yd_rate",
        ]
    ]
    report = f"""# Shot-simulator state foundations audit

Version: `{FOUNDATION_VERSION}`. This is archived research; production is not
importing it.

## Full-sequence penalty and resolution linkage

{_markdown_table(penalty_display)}

Every penalty is sequenced against all raw ShotLink rows in the same
event/player/round/hole. The causal observation is the latest preceding
played `STROKE` or `PROVISIONAL`; `DROP` and other bookkeeping rows remain in
the path but can never become causal shots. The resolution is the exact next
played row's raw start coordinates and raw start lie. Its recursive distance
is intentionally not copied from SG: callers must recompute distance from
those coordinates to the forecast pin. The all-category SG start lie/distance
are joined by `shot_key` only as diagnostics because post-penalty SG state is
sometimes stale (especially on replay/provisional sequences).

`causal_to_resolution_*_before` measures the real hazard/drop displacement;
it is not expected to be continuous. `all_drop_to_next_physical_*` verifies
that coordinate-complete DROP endpoints agree with the next played row's
start. Penalty-associated DROP coordinates are frequently absent. Rows without
an exact raw resolution row remain visibly unresolved and cannot silently
enter a recursive state transition. `causal_hazard_labels_by_shot_key.csv`
exposes the causal OTT/APP incidence labels for downstream training, while
`causal_ott_hazard_labels_by_shot_key.csv` is the compact positive-label API
for joining to a complete tee-shot training frame.

`penalty_count` is the number of explicit PENALTY rows. The companion
`penalty_stroke_cost` is derived from
`resolution_stroke_no - causal_stroke_no - 1` and is the value a simulator
must add after it has already charged the causal shot. It then simulates the
returned STROKE/PROVISIONAL as the next played stroke. The old STROKE-only
audit columns quantify replay paths where skipping a provisional would have
undercharged the official stroke span.

## Physical-course identity

{_markdown_table(identity_display)}

Event name generates candidates only. The strict forecast reference is the
latest edition before the holdout; heldout tee/pin geometry never defines the
donor coordinate frame. A donor is accepted only when it is a
structurally single-course edition and its full round/hole tee+pin geometry,
after proper rigid alignment, passes every conservative venue gate. Layout
`course_id` remains in shot-to-geometry joins but is not treated as a stable
cross-edition venue identifier. The accepted rigid transform is retained and
the direct APP evaluator maps each donor edition into that latest-prior frame
before pooling historical pins or absolute endpoint-to-surface maps. Heldout
geometry is used only to express realized scoring observations in the already
fixed prior frame.

The labelled venue-change control is **{args.venue_change_control or 'not run'}**.
The multicourse control failed closed with:

> {multicourse_error}

## Recursive-simulator contract

1. Draw/transport the causal physical endpoint in stable course coordinates.
2. Reclassify its surface/hazard from that transported endpoint.
3. If penalized, add `penalty_count` and resume only from the linked/resampled
   raw resolution lie/coordinates; compute distance to the forecast pin and
   never use either the bookkeeping penalty row or stale SG resolution state.
4. Refuse exact-course donors that are not accepted by the physical identity
   table. Multicourse simulation requires an explicit course assignment first.
"""
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--event-name", action="append", default=[])
    parser.add_argument(
        "--venue-change-control", default="The Genesis Invitational"
    )
    parser.add_argument(
        "--multicourse-control", default="AT&T Pebble Beach Pro-Am"
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.event_name:
        args.event_name = list(DEFAULT_EVENTS)
    return args


def main() -> None:
    args = parse_args()
    report = run_foundation_audit(args)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

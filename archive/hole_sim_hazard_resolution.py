"""Endpoint-linked hazard resolution for the archived shot-simulator chain.

The caller has already simulated an endpoint and classified it as a penalty
hazard. This module resolves that hazard into the next playable state using
only historical causal penalty outcomes from the verified physical course::

    simulated endpoint + round/hole/category + common uniform
        -> penalty count + exact linked next-playable-row state

Donors must come from ``hole_sim_state_foundations`` and be normalized into
the latest-prior forecast course coordinate frame. Selection uses a spatial
kernel inside a strong hierarchy (round/hole/category -> hole/category ->
category). An optional radius extension is permitted only when the primary
same-category radius has no support, and it is restricted to the same
physical hole and category. It never silently invents a drop or substitutes a
bookkeeping penalty row for the next playable STROKE/PROVISIONAL. Unsupported
states fail closed with ``HazardResolutionUnavailable``.

Coordinates are ShotLink feet. Distances used by kernels and diagnostics are
converted to yards; returned coordinates remain feet so they can re-enter the
spatial chain without another conversion. Nothing in production imports this
research module.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from archive.hole_sim_state_foundations import (
        DEFAULT_EXPORT,
        FOUNDATION_VERSION,
        align_coordinates_to_target,
        link_app_penalties_full_sequence,
        load_verified_recurring_event,
    )
except ModuleNotFoundError:  # Direct execution: python archive/<script>.py
    from hole_sim_state_foundations import (  # type: ignore
        DEFAULT_EXPORT,
        FOUNDATION_VERSION,
        align_coordinates_to_target,
        link_app_penalties_full_sequence,
        load_verified_recurring_event,
    )


DEFAULT_OUTPUT = Path("permanent_data/hole_sim_hazard_resolution_smoke")
RESOLVER_VERSION = "endpoint_linked_hazard_v3"
SUPPORTED_CATEGORIES = ("ott", "app")
FEET_PER_YARD = 3.0
PLAYED_EVENT_TYPES = frozenset({"STROKE", "PROVISIONAL"})
RESOLUTION_LIES = frozenset({"tee", "fairway", "rough", "sand", "recovery"})


class HazardResolutionUnavailable(RuntimeError):
    """Raised when no defensible same-category resolution donor is available."""


@dataclass(frozen=True)
class HazardResolution:
    donor_key: str
    category: str
    penalty_count: int
    penalty_stroke_cost: int
    resolution_x: float
    resolution_y: float
    resolution_lie: str
    resolution_lie_raw: str
    resolution_distance_yd: float
    resolution_shot_key: str
    resolution_event_type: str
    forecast_pin_x: float
    forecast_pin_y: float
    donor_geometry_resolution_distance_yd: float
    donor_sg_resolution_distance_yd: float
    donor_sg_resolution_lie: str
    donor_endpoint_x: float
    donor_endpoint_y: float
    endpoint_distance_yd: float
    selection_probability: float
    effective_n: float
    support_setup: int
    support_hole: int
    support_category: int
    backoff_weights: str
    physical_course_id: str
    coordinate_reference_year: int
    coordinate_reference_event_key: str
    coordinate_reference_mode: str = "latest_prior"
    foundation_version: str = FOUNDATION_VERSION
    resolver_version: str = RESOLVER_VERSION
    fallback_used: bool = False
    support_source: str = "same_category_primary_radius"
    nearest_category_endpoint_distance_yd: float = math.nan
    nearest_same_hole_endpoint_distance_yd: float = math.nan
    support_radius_limit_yd: float = math.nan


@dataclass(frozen=True)
class HazardSupportBatch:
    """Vectorized resolver-support diagnostics for one hole/category query."""

    supported: np.ndarray
    primary_supported: np.ndarray
    fallback_used: np.ndarray
    support_source: np.ndarray
    nearest_category_endpoint_distance_yd: np.ndarray
    nearest_same_hole_endpoint_distance_yd: np.ndarray


REQUIRED_LINK_COLUMNS = {
    "event_key", "round_no", "hole_no", "causal_shot_key",
    "causal_sg_sg_category", "causal_end_x", "causal_end_y",
    "causal_event_type", "causal_stroke_no", "causal_start_lie",
    "causal_course_id",
    "resolution_raw_shot_key", "resolution_raw_start_x",
    "resolution_raw_start_y", "resolution_raw_start_lie",
    "resolution_raw_event_type", "resolution_raw_stroke_no",
    "resolution_raw_course_id",
    "resolution_sg_start_lie",
    "resolution_sg_start_distance_yd", "causal_linked",
    "donor_tee_x", "donor_tee_y", "donor_pin_x", "donor_pin_y",
    "donor_par",
}
REQUIRED_DONOR_COLUMNS = {
    "donor_key", "event_key", "season", "round_no", "hole_no", "category",
    "causal_endpoint_x", "causal_endpoint_y", "penalty_count",
    "penalty_stroke_cost",
    "resolution_shot_key", "resolution_event_type", "resolution_x",
    "resolution_y", "resolution_lie", "resolution_lie_raw",
    "donor_geometry_resolution_distance_yd",
    "donor_sg_resolution_distance_yd", "donor_sg_resolution_lie",
    "donor_pin_x", "donor_pin_y", "physical_course_id", "foundation_version",
    "coordinate_reference_year", "coordinate_reference_event_key",
    "coordinate_reference_mode", "data_role",
}


def _normalize_category(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "sg_ott": "ott",
        "tee": "ott",
        "drive": "ott",
        "driving": "ott",
        "sg_app": "app",
        "approach": "app",
    }
    return aliases.get(text, text)


def prepare_hazard_donors(
    penalty_links: pd.DataFrame,
    categories: tuple[str, ...] = SUPPORTED_CATEGORIES,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Collapse one-row-per-penalty links into one causal hazard outcome.

    A causal shot may incur more than one assessed penalty. Those rows become
    one donor with ``penalty_count > 1`` only if they agree on the exact next
    playable-row resolution. Ambiguous or incomplete outcomes are excluded
    and counted in diagnostics; they can never enter the resolver silently.
    """

    missing = REQUIRED_LINK_COLUMNS.difference(penalty_links.columns)
    if missing:
        raise ValueError(f"Foundation penalty links missing: {sorted(missing)}")
    frame = penalty_links.copy()
    if "season" not in frame:
        frame["season"] = pd.NA
    if "physical_course_id" not in frame:
        raise ValueError(
            "Penalty links are not normalized to a verified physical course; "
            "physical_course_id is required"
        )
    if "foundation_version" not in frame:
        frame["foundation_version"] = FOUNDATION_VERSION
    provenance = {
        "coordinate_reference_year", "coordinate_reference_event_key",
        "coordinate_reference_mode", "data_role",
    }
    if missing_provenance := provenance.difference(frame.columns):
        raise ValueError(
            "Hazard links lack strict-prior provenance: "
            f"{sorted(missing_provenance)}"
        )
    frame["category"] = frame["causal_sg_sg_category"].map(_normalize_category)
    # A provisional replay from a par-4/5 tee is physically an OTT shot even
    # when the derived SG row retained the prior penalty's APP/recovery state.
    # Geometry supplies the par needed to avoid relabelling par-3 tee shots.
    causal_provisional_tee = (
        frame["causal_event_type"].fillna("").astype(str).str.upper()
        .eq("PROVISIONAL")
        & frame["causal_start_lie"].fillna("").astype(str).str.casefold().eq("tee")
        & pd.to_numeric(frame["donor_par"], errors="coerce").ge(4)
    )
    frame.loc[causal_provisional_tee, "category"] = "ott"
    frame["causal_shot_key"] = frame["causal_shot_key"].astype("string")
    requested = {_normalize_category(value) for value in categories}
    initial_rows = len(frame)
    linked = frame["causal_linked"].fillna(False).astype(bool)
    category_ok = frame["category"].isin(requested)
    coordinate_columns = [
        "causal_end_x", "causal_end_y", "resolution_raw_start_x",
        "resolution_raw_start_y", "donor_pin_x", "donor_pin_y",
    ]
    coordinates_ok = np.isfinite(
        frame[coordinate_columns].apply(pd.to_numeric, errors="coerce")
    ).all(axis=1)
    frame["resolution_lie_raw"] = (
        frame["resolution_raw_start_lie"].fillna("").astype(str).str.strip()
    )
    frame["resolution_lie"] = frame["resolution_lie_raw"].str.casefold()
    lie_ok = frame["resolution_lie"].isin(RESOLUTION_LIES)
    played_ok = (
        frame["resolution_raw_event_type"].fillna("").astype(str).str.upper()
        .isin(PLAYED_EVENT_TYPES)
    )
    key_ok = frame["causal_shot_key"].notna() & frame[
        "resolution_raw_shot_key"
    ].notna()
    same_course = (
        frame["causal_course_id"].astype("string")
        == frame["resolution_raw_course_id"].astype("string")
    ) & frame["causal_course_id"].notna() & frame[
        "resolution_raw_course_id"
    ].notna()
    frame["resolution_geometry_distance_yd"] = np.hypot(
        pd.to_numeric(frame["resolution_raw_start_x"], errors="coerce")
        - pd.to_numeric(frame["donor_pin_x"], errors="coerce"),
        pd.to_numeric(frame["resolution_raw_start_y"], errors="coerce")
        - pd.to_numeric(frame["donor_pin_y"], errors="coerce"),
    ) / FEET_PER_YARD
    geometry_distance_ok = frame["resolution_geometry_distance_yd"].between(0, 700)
    frame["resolution_tee_gap_yd"] = np.hypot(
        pd.to_numeric(frame["resolution_raw_start_x"], errors="coerce")
        - pd.to_numeric(frame["donor_tee_x"], errors="coerce"),
        pd.to_numeric(frame["resolution_raw_start_y"], errors="coerce")
        - pd.to_numeric(frame["donor_tee_y"], errors="coerce"),
    ) / FEET_PER_YARD
    tee_coherent = ~frame["resolution_lie"].eq("tee") | frame[
        "resolution_tee_gap_yd"
    ].le(1.0)
    frame["resolution_sg_distance_abs_error_yd"] = (
        frame["resolution_geometry_distance_yd"]
        - pd.to_numeric(
            frame["resolution_sg_start_distance_yd"], errors="coerce"
        )
    ).abs()
    frame["resolution_raw_vs_sg_lie_match"] = (
        frame["resolution_lie"]
        == frame["resolution_sg_start_lie"].fillna("").astype(str).str.casefold()
    ) & frame["resolution_sg_start_lie"].notna()
    complete = coordinates_ok & lie_ok & played_ok & key_ok
    coherent = same_course & geometry_distance_ok & tee_coherent
    eligible = frame[linked & category_ok & complete & coherent].copy()

    # Reconstruct the retired STROKE-only resolver's support/accounting using
    # its original state gates. This quantifies how often it jumped over a
    # physical PROVISIONAL without charging that stroke before resuming.
    old_columns = {
        "old_stroke_only_causal_shot_key",
        "old_stroke_only_causal_stroke_no",
        "old_stroke_only_causal_sg_sg_category",
        "old_stroke_only_causal_end_x", "old_stroke_only_causal_end_y",
        "old_stroke_only_resolution_shot_key",
        "old_stroke_only_resolution_stroke_no",
        "old_stroke_only_resolution_start_x",
        "old_stroke_only_resolution_start_y",
        "old_stroke_only_resolution_sg_start_lie",
        "old_stroke_only_resolution_sg_start_distance_yd",
    }
    old_donors = pd.DataFrame()
    if old_columns.issubset(frame.columns):
        old = frame.copy()
        old["_old_category"] = old[
            "old_stroke_only_causal_sg_sg_category"
        ].map(_normalize_category)
        old_coordinates = [
            "old_stroke_only_causal_end_x",
            "old_stroke_only_causal_end_y",
            "old_stroke_only_resolution_start_x",
            "old_stroke_only_resolution_start_y",
            "old_stroke_only_resolution_sg_start_distance_yd",
        ]
        old_complete = (
            np.isfinite(
                old[old_coordinates].apply(pd.to_numeric, errors="coerce")
            ).all(axis=1)
            & old["old_stroke_only_causal_shot_key"].notna()
            & old["old_stroke_only_resolution_shot_key"].notna()
            & old["old_stroke_only_resolution_sg_start_lie"].notna()
            & old["_old_category"].isin(requested)
        )
        old = old[old_complete].copy()
        old_group_keys = ["event_key", "old_stroke_only_causal_shot_key"]
        old_uniqueness = old.groupby(old_group_keys, observed=True).agg(
            resolution_keys=("old_stroke_only_resolution_shot_key", "nunique"),
            resolution_x=("old_stroke_only_resolution_start_x", "nunique"),
            resolution_y=("old_stroke_only_resolution_start_y", "nunique"),
            resolution_distance=(
                "old_stroke_only_resolution_sg_start_distance_yd", "nunique"
            ),
            categories=("_old_category", "nunique"),
        )
        old_ambiguous = old_uniqueness[old_uniqueness.max(axis=1).gt(1)].index
        if len(old_ambiguous):
            old_index = pd.MultiIndex.from_frame(old[old_group_keys])
            old = old[~old_index.isin(old_ambiguous)].copy()
        old_donors = old.groupby(
            old_group_keys, as_index=False, observed=True
        ).agg(
            penalty_count=("penalty_shot_key", "size"),
            causal_stroke_no=("old_stroke_only_causal_stroke_no", "first"),
            resolution_stroke_no=(
                "old_stroke_only_resolution_stroke_no", "first"
            ),
        )
        old_donors["stroke_span_cost"] = (
            pd.to_numeric(old_donors["resolution_stroke_no"], errors="coerce")
            - pd.to_numeric(old_donors["causal_stroke_no"], errors="coerce")
            - 1
        )
        old_donors["stroke_undercount"] = (
            old_donors["stroke_span_cost"] - old_donors["penalty_count"]
        )

    group_keys = ["event_key", "causal_shot_key"]
    resolution_uniqueness = eligible.groupby(group_keys, observed=True).agg(
        n_resolution_key=("resolution_raw_shot_key", "nunique"),
        n_resolution_event=("resolution_raw_event_type", "nunique"),
        n_resolution_x=("resolution_raw_start_x", "nunique"),
        n_resolution_y=("resolution_raw_start_y", "nunique"),
        n_resolution_lie=("resolution_lie", "nunique"),
        n_resolution_distance=("resolution_geometry_distance_yd", "nunique"),
        n_category=("category", "nunique"),
    )
    ambiguous_index = resolution_uniqueness[
        resolution_uniqueness.max(axis=1).gt(1)
    ].index
    if len(ambiguous_index):
        index = pd.MultiIndex.from_frame(eligible[group_keys])
        eligible = eligible[~index.isin(ambiguous_index)].copy()

    def _first(values: pd.Series) -> object:
        nonnull = values.dropna()
        return nonnull.iloc[0] if len(nonnull) else np.nan

    donors = eligible.groupby(group_keys, as_index=False, observed=True).agg(
        season=("season", _first),
        round_no=("round_no", _first),
        hole_no=("hole_no", _first),
        category=("category", _first),
        causal_endpoint_x=("causal_end_x", _first),
        causal_endpoint_y=("causal_end_y", _first),
        penalty_count=("penalty_shot_key", "size"),
        causal_stroke_no=("causal_stroke_no", _first),
        resolution_stroke_no=("resolution_raw_stroke_no", _first),
        resolution_shot_key=("resolution_raw_shot_key", _first),
        resolution_event_type=("resolution_raw_event_type", _first),
        resolution_x=("resolution_raw_start_x", _first),
        resolution_y=("resolution_raw_start_y", _first),
        resolution_lie=("resolution_lie", _first),
        resolution_lie_raw=("resolution_lie_raw", _first),
        donor_geometry_resolution_distance_yd=(
            "resolution_geometry_distance_yd", _first
        ),
        donor_sg_resolution_distance_yd=(
            "resolution_sg_start_distance_yd", _first
        ),
        donor_sg_resolution_lie=("resolution_sg_start_lie", _first),
        donor_sg_distance_abs_error_yd=(
            "resolution_sg_distance_abs_error_yd", _first
        ),
        donor_raw_vs_sg_lie_match=("resolution_raw_vs_sg_lie_match", _first),
        donor_tee_x=("donor_tee_x", _first),
        donor_tee_y=("donor_tee_y", _first),
        donor_pin_x=("donor_pin_x", _first),
        donor_pin_y=("donor_pin_y", _first),
        donor_par=("donor_par", _first),
        physical_course_id=("physical_course_id", _first),
        foundation_version=("foundation_version", _first),
        coordinate_reference_year=("coordinate_reference_year", _first),
        coordinate_reference_event_key=("coordinate_reference_event_key", _first),
        coordinate_reference_mode=("coordinate_reference_mode", _first),
        data_role=("data_role", _first),
    )
    donors = donors.rename(columns={"causal_shot_key": "donor_key"})
    donors["season"] = pd.to_numeric(donors["season"], errors="raise").astype(int)
    donors["round_no"] = pd.to_numeric(donors["round_no"], errors="raise").astype(int)
    donors["hole_no"] = pd.to_numeric(donors["hole_no"], errors="raise").astype(int)
    donors["penalty_count"] = donors["penalty_count"].astype(int)
    donors["penalty_stroke_cost"] = (
        pd.to_numeric(donors["resolution_stroke_no"], errors="coerce")
        - pd.to_numeric(donors["causal_stroke_no"], errors="coerce")
        - 1
    )
    accounting_valid = (
        donors["penalty_stroke_cost"].notna()
        & donors["penalty_stroke_cost"].between(1, 4)
        & donors["penalty_stroke_cost"].ge(donors["penalty_count"])
        & np.isclose(donors["penalty_stroke_cost"] % 1, 0)
    )
    excluded_invalid_accounting = int((~accounting_valid).sum())
    donors = donors[accounting_valid].copy()
    donors["penalty_stroke_cost"] = donors["penalty_stroke_cost"].astype(int)
    donors["coordinate_reference_year"] = pd.to_numeric(
        donors["coordinate_reference_year"], errors="raise"
    ).astype(int)
    donors["resolution_event_type"] = (
        donors["resolution_event_type"].astype(str).str.upper()
    )
    donors = donors.sort_values("donor_key").reset_index(drop=True)
    sg_error = pd.to_numeric(
        donors["donor_sg_distance_abs_error_yd"], errors="coerce"
    ).dropna()
    diagnostics = {
        "input_penalty_rows": int(initial_rows),
        "excluded_unlinked": int((~linked).sum()),
        "excluded_category": int((linked & ~category_ok).sum()),
        "excluded_incomplete_resolution": int(
            (linked & category_ok & ~complete).sum()
        ),
        "excluded_incoherent_resolution": int(
            (linked & category_ok & complete & ~coherent).sum()
        ),
        "excluded_cross_course_resolution": int(
            (linked & category_ok & complete & ~same_course).sum()
        ),
        "excluded_tee_not_at_tee": int(
            (linked & category_ok & complete & same_course & ~tee_coherent).sum()
        ),
        "excluded_ambiguous_causal": int(len(ambiguous_index)),
        "excluded_invalid_stroke_accounting": excluded_invalid_accounting,
        "causal_hazard_donors": int(len(donors)),
        "ott_donors": int(donors["category"].eq("ott").sum()),
        "app_donors": int(donors["category"].eq("app").sum()),
        "causal_provisional_relabelled_ott_rows": int(
            (linked & causal_provisional_tee).sum()
        ),
        "resolution_provisional_donors": int(
            donors["resolution_event_type"].eq("PROVISIONAL").sum()
        ),
        "resolution_raw_vs_sg_lie_mismatch_donors": int(
            (~donors["donor_raw_vs_sg_lie_match"].fillna(False).astype(bool)).sum()
        ),
        "resolution_sg_distance_gt50yd_mismatch_donors": int(
            pd.to_numeric(
                donors["donor_sg_distance_abs_error_yd"], errors="coerce"
            ).gt(50).sum()
        ),
        "resolution_sg_distance_max_abs_error_yd": float(sg_error.max())
        if len(sg_error) else math.nan,
        "penalty_cost_exceeds_observed_rows_donors": int(
            donors["penalty_stroke_cost"].gt(donors["penalty_count"]).sum()
        ),
        "stroke_span_reconciled_donors": int(
            (
                donors["causal_stroke_no"]
                + donors["penalty_stroke_cost"] + 1
                == donors["resolution_stroke_no"]
            ).sum()
        ),
        "old_stroke_only_donors": int(len(old_donors)),
        "old_stroke_only_undercount_donors": int(
            old_donors.get("stroke_undercount", pd.Series(dtype=float)).gt(0).sum()
        ),
        "old_stroke_only_missing_stroke_cost": int(
            old_donors.get("stroke_undercount", pd.Series(dtype=float))
            .clip(lower=0).sum()
        ),
    }
    for category in sorted(requested):
        subset = donors[donors["category"].eq(category)]
        errors = pd.to_numeric(
            subset["donor_sg_distance_abs_error_yd"], errors="coerce"
        )
        diagnostics[f"{category}_resolution_sg_distance_gt50yd_mismatch"] = int(
            errors.gt(50).sum()
        )
        diagnostics[f"{category}_resolution_sg_distance_comparable"] = int(
            errors.notna().sum()
        )
        diagnostics[f"{category}_resolution_sg_distance_max_abs_error_yd"] = (
            float(errors.max()) if errors.notna().any() else math.nan
        )
    return donors, diagnostics


class EndpointLinkedHazardResolver:
    """Spatial, hierarchical resolver with caller-supplied common uniforms."""

    def __init__(
        self,
        donors: pd.DataFrame,
        *,
        setup_prior: float = 12.0,
        hole_prior: float = 40.0,
        spatial_bandwidth_yd: float = 24.0,
        max_nearest_distance_yd: float = 120.0,
        same_hole_fallback_max_distance_yd: float | None = None,
        minimum_category_support: int = 12,
    ) -> None:
        missing = REQUIRED_DONOR_COLUMNS.difference(donors.columns)
        if missing:
            raise ValueError(f"Hazard donors missing columns: {sorted(missing)}")
        self.donors = donors.copy().sort_values("donor_key").reset_index(drop=True)
        self.donors["category"] = self.donors["category"].map(_normalize_category)
        physical_ids = set(self.donors["physical_course_id"].dropna().astype(str))
        foundation_versions = set(
            self.donors["foundation_version"].dropna().astype(str)
        )
        reference_modes = set(
            self.donors["coordinate_reference_mode"].dropna().astype(str)
        )
        reference_years = set(
            self.donors["coordinate_reference_year"].dropna().astype(int)
        )
        reference_keys = set(
            self.donors["coordinate_reference_event_key"].dropna().astype(str)
        )
        data_roles = set(self.donors["data_role"].dropna().astype(str))
        if len(physical_ids) != 1:
            raise ValueError(
                f"Hazard donors span physical courses: {sorted(physical_ids)}"
            )
        if foundation_versions != {FOUNDATION_VERSION}:
            raise ValueError(
                "Hazard donors do not have the current full-sequence foundation: "
                f"{sorted(foundation_versions)}"
            )
        if reference_modes != {"latest_prior"}:
            raise ValueError(
                "Hazard resolver requires latest-prior coordinates, got "
                f"{sorted(reference_modes)}"
            )
        if len(reference_years) != 1 or len(reference_keys) != 1:
            raise ValueError(
                "Hazard donors do not share one prior coordinate reference: "
                f"years={sorted(reference_years)}, keys={sorted(reference_keys)}"
            )
        if data_roles != {"predictor_donor"}:
            raise ValueError(
                "Heldout scoring rows cannot enter the predictor resolver: "
                f"roles={sorted(data_roles)}"
            )
        self.coordinate_reference_year = next(iter(reference_years))
        self.coordinate_reference_event_key = next(iter(reference_keys))
        if self.donors["season"].max() > self.coordinate_reference_year:
            raise ValueError(
                "Donor season exceeds its latest-prior coordinate reference"
            )
        if self.donors["donor_key"].duplicated().any():
            raise ValueError("Hazard donor_key must be unique after causal collapse")
        if (self.donors["penalty_count"] < 1).any():
            raise ValueError("Every hazard donor must carry at least one penalty")
        if (
            (self.donors["penalty_stroke_cost"] < self.donors["penalty_count"])
            | (self.donors["penalty_stroke_cost"] < 1)
        ).any():
            raise ValueError("Hazard donor stroke accounting is inconsistent")
        finite_state = np.isfinite(
            self.donors[
                [
                    "resolution_x", "resolution_y", "donor_pin_x",
                    "donor_pin_y", "donor_geometry_resolution_distance_yd",
                ]
            ].apply(pd.to_numeric, errors="coerce")
        ).all(axis=1)
        played_state = self.donors["resolution_event_type"].isin(
            PLAYED_EVENT_TYPES
        )
        lie_state = self.donors["resolution_lie"].isin(RESOLUTION_LIES)
        if not (finite_state & played_state & lie_state).all():
            raise ValueError("Hazard donors contain an incoherent recursive state")
        self.physical_course_id = next(iter(physical_ids))
        self.setup_prior = float(setup_prior)
        self.hole_prior = float(hole_prior)
        self.spatial_bandwidth_yd = float(spatial_bandwidth_yd)
        self.max_nearest_distance_yd = float(max_nearest_distance_yd)
        self.same_hole_fallback_max_distance_yd = (
            None
            if same_hole_fallback_max_distance_yd is None
            else float(same_hole_fallback_max_distance_yd)
        )
        if (
            self.same_hole_fallback_max_distance_yd is not None
            and self.same_hole_fallback_max_distance_yd
            <= self.max_nearest_distance_yd
        ):
            raise ValueError(
                "same_hole_fallback_max_distance_yd must exceed the primary "
                "same-category radius"
            )
        self.minimum_category_support = int(minimum_category_support)
        self._category_trees: dict[str, cKDTree] = {}
        self._hole_category_trees: dict[tuple[str, int], cKDTree] = {}
        for category, frame in self.donors.groupby("category", observed=True):
            points_yd = frame[
                ["causal_endpoint_x", "causal_endpoint_y"]
            ].to_numpy(float) / FEET_PER_YARD
            self._category_trees[str(category)] = cKDTree(points_yd)
        for (category, hole_no), frame in self.donors.groupby(
            ["category", "hole_no"], observed=True
        ):
            points_yd = frame[
                ["causal_endpoint_x", "causal_endpoint_y"]
            ].to_numpy(float) / FEET_PER_YARD
            self._hole_category_trees[(str(category), int(hole_no))] = cKDTree(
                points_yd
            )

    def support_diagnostics_many(
        self,
        endpoint_x: np.ndarray | float,
        endpoint_y: np.ndarray | float,
        *,
        hole_no: int,
        round_no: int,
        category: str,
    ) -> HazardSupportBatch:
        """Return vectorized support under the exact resolver radius hierarchy.

        The primary pool may use any donor on the same physical course and in
        the same shot category within ``max_nearest_distance_yd``.  Only when
        that pool is empty may the optional extension activate, and that
        extension is restricted to the same physical hole and category.
        ``round_no`` is accepted to keep this predicate's public state contract
        identical to :meth:`resolve`; support radii themselves are not
        round-conditioned.
        """

        del round_no
        normalized = _normalize_category(category)
        x, y = np.broadcast_arrays(
            np.asarray(endpoint_x, dtype=float),
            np.asarray(endpoint_y, dtype=float),
        )
        original_shape = x.shape
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        finite = np.isfinite(flat_x) & np.isfinite(flat_y)
        nearest_category = np.full(len(flat_x), np.inf, dtype=float)
        nearest_same_hole = np.full(len(flat_x), np.inf, dtype=float)
        category_count = int(self.donors["category"].eq(normalized).sum())
        category_tree = self._category_trees.get(normalized)
        if category_tree is not None and finite.any():
            query_yd = np.column_stack(
                [flat_x[finite], flat_y[finite]]
            ) / FEET_PER_YARD
            nearest_category[finite] = np.asarray(
                category_tree.query(query_yd, k=1)[0], dtype=float
            )
        hole_tree = self._hole_category_trees.get((normalized, int(hole_no)))
        if hole_tree is not None and finite.any():
            query_yd = np.column_stack(
                [flat_x[finite], flat_y[finite]]
            ) / FEET_PER_YARD
            nearest_same_hole[finite] = np.asarray(
                hole_tree.query(query_yd, k=1)[0], dtype=float
            )
        enough_category_support = category_count >= self.minimum_category_support
        primary = (
            finite
            & enough_category_support
            & (nearest_category <= self.max_nearest_distance_yd)
        )
        fallback = np.zeros(len(flat_x), dtype=bool)
        if self.same_hole_fallback_max_distance_yd is not None:
            fallback = (
                finite
                & enough_category_support
                & ~primary
                & (
                    nearest_same_hole
                    <= self.same_hole_fallback_max_distance_yd
                )
            )
        supported = primary | fallback
        source = np.full(len(flat_x), "unavailable", dtype=object)
        source[primary] = "same_category_primary_radius"
        source[fallback] = "same_hole_category_fallback_radius"
        return HazardSupportBatch(
            supported=supported.reshape(original_shape),
            primary_supported=primary.reshape(original_shape),
            fallback_used=fallback.reshape(original_shape),
            support_source=source.reshape(original_shape),
            nearest_category_endpoint_distance_yd=nearest_category.reshape(
                original_shape
            ),
            nearest_same_hole_endpoint_distance_yd=nearest_same_hole.reshape(
                original_shape
            ),
        )

    def has_support_many(
        self,
        endpoint_x: np.ndarray | float,
        endpoint_y: np.ndarray | float,
        *,
        hole_no: int,
        round_no: int,
        category: str,
    ) -> np.ndarray:
        """Vectorized boolean form of :meth:`support_diagnostics_many`."""

        return np.asarray(
            self.support_diagnostics_many(
                endpoint_x,
                endpoint_y,
                hole_no=hole_no,
                round_no=round_no,
                category=category,
            ).supported,
            dtype=bool,
        )

    def has_support(
        self,
        endpoint_x: float,
        endpoint_y: float,
        *,
        hole_no: int,
        round_no: int,
        category: str,
    ) -> bool:
        """Scalar resolver-support predicate using the same hierarchy."""

        result = self.has_support_many(
            endpoint_x,
            endpoint_y,
            hole_no=hole_no,
            round_no=round_no,
            category=category,
        )
        return bool(np.asarray(result).item())

    def _distribution(
        self,
        endpoint_x: float,
        endpoint_y: float,
        hole_no: int,
        round_no: int,
        category: str,
    ) -> tuple[
        pd.DataFrame,
        np.ndarray,
        dict[str, int],
        dict[str, float],
        float,
        float,
        bool,
        str,
        float,
    ]:
        category = _normalize_category(category)
        category_frame = self.donors[self.donors["category"].eq(category)].copy()
        if len(category_frame) < self.minimum_category_support:
            raise HazardResolutionUnavailable(
                f"Only {len(category_frame)} {category!r} hazard donors; "
                f"minimum is {self.minimum_category_support}"
            )
        support_batch = self.support_diagnostics_many(
            endpoint_x,
            endpoint_y,
            hole_no=hole_no,
            round_no=round_no,
            category=category,
        )
        supported = bool(np.asarray(support_batch.supported).item())
        fallback_used = bool(np.asarray(support_batch.fallback_used).item())
        nearest_category = float(
            np.asarray(
                support_batch.nearest_category_endpoint_distance_yd
            ).item()
        )
        nearest_same_hole = float(
            np.asarray(
                support_batch.nearest_same_hole_endpoint_distance_yd
            ).item()
        )
        support_source = str(np.asarray(support_batch.support_source).item())
        if not supported:
            same_hole_text = (
                f"{nearest_same_hole:.1f} yd"
                if np.isfinite(nearest_same_hole)
                else "unavailable"
            )
            fallback_text = (
                f"; same-hole fallback limit is "
                f"{self.same_hole_fallback_max_distance_yd:.1f} yd"
                if self.same_hole_fallback_max_distance_yd is not None
                else ""
            )
            raise HazardResolutionUnavailable(
                f"Nearest {category} hazard endpoint is "
                f"{nearest_category:.1f} yd away; primary limit is "
                f"{self.max_nearest_distance_yd:.1f} yd; nearest same-hole "
                f"endpoint is {same_hole_text}{fallback_text}"
            )
        distance = np.hypot(
            category_frame["causal_endpoint_x"].to_numpy(float) - float(endpoint_x),
            category_frame["causal_endpoint_y"].to_numpy(float) - float(endpoint_y),
        ) / FEET_PER_YARD
        category_frame["_endpoint_distance_yd"] = distance
        if fallback_used:
            radius_limit = float(self.same_hole_fallback_max_distance_yd)
            within = (
                category_frame["hole_no"].eq(int(hole_no))
                & category_frame["_endpoint_distance_yd"].le(radius_limit)
            )
        else:
            radius_limit = self.max_nearest_distance_yd
            within = category_frame["_endpoint_distance_yd"].le(radius_limit)
        category_frame = category_frame[within].copy()
        setup_mask = (
            category_frame["round_no"].eq(int(round_no))
            & category_frame["hole_no"].eq(int(hole_no))
        )
        hole_mask = category_frame["hole_no"].eq(int(hole_no))
        pools = [
            ("setup", setup_mask, self.setup_prior),
            ("hole", hole_mask, self.hole_prior),
            ("category", pd.Series(True, index=category_frame.index), math.inf),
        ]
        mass = pd.Series(0.0, index=category_frame.index)
        remaining = 1.0
        support: dict[str, int] = {}
        component_masses: dict[str, float] = {}
        for name, mask, prior in pools:
            pool = category_frame.loc[mask]
            support[name] = int(len(pool))
            if pool.empty:
                component_masses[name] = 0.0
                continue
            distances = pool["_endpoint_distance_yd"].to_numpy(float)
            # The bandwidth floor prevents one donor from becoming a point
            # mass; expanding to the local p40 maintains support in sparse
            # hole cells without borrowing another category.
            local_bandwidth = max(
                self.spatial_bandwidth_yd,
                float(np.quantile(distances, 0.40)) if len(distances) > 2 else 0.0,
            )
            kernel = np.exp(-0.5 * (distances / local_bandwidth) ** 2)
            kernel = np.maximum(kernel, 1e-12)
            kernel /= kernel.sum()
            fraction = 1.0 if not np.isfinite(prior) else len(pool) / (len(pool) + prior)
            component = remaining * fraction
            mass.loc[pool.index] += component * kernel
            remaining -= component
            component_masses[name] = float(component)
        positive = mass[mass.gt(0)]
        if positive.empty or not np.isclose(positive.sum(), 1.0, atol=1e-8):
            raise HazardResolutionUnavailable("Hierarchical hazard mass is incomplete")
        selected = category_frame.loc[positive.index].copy()
        weights = positive.to_numpy(float)
        weights /= weights.sum()
        return (
            selected,
            weights,
            support,
            component_masses,
            nearest_category,
            nearest_same_hole,
            fallback_used,
            support_source,
            float(radius_limit),
        )

    def resolve(
        self,
        *,
        endpoint_x: float,
        endpoint_y: float,
        hole_no: int,
        round_no: int,
        category: str,
        common_uniform: float,
        forecast_pin_x: float,
        forecast_pin_y: float,
        physical_course_id: str | None = None,
    ) -> HazardResolution:
        """Resolve one hazard using a common uniform in ``[0, 1)``.

        Passing the same uniform for alternative player/endpoint arms provides
        common random numbers. The donor ordering is stable by ``donor_key``.
        ``forecast_pin_x/y`` must share the resolver's absolute latest-prior
        feet frame; the returned recursive distance is derived from it.
        """

        if physical_course_id is not None and str(physical_course_id) != self.physical_course_id:
            raise HazardResolutionUnavailable(
                f"Query course {physical_course_id!r} != donor course "
                f"{self.physical_course_id!r}"
            )
        values = np.asarray(
            [endpoint_x, endpoint_y, common_uniform, forecast_pin_x, forecast_pin_y],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise HazardResolutionUnavailable(
                "Endpoint, forecast pin and common uniform must be finite"
            )
        uniform = float(common_uniform)
        if not 0.0 <= uniform < 1.0:
            raise ValueError("common_uniform must satisfy 0 <= u < 1")
        (
            candidates,
            weights,
            support,
            masses,
            nearest_category,
            nearest_same_hole,
            fallback_used,
            support_source,
            support_radius_limit,
        ) = self._distribution(
            endpoint_x, endpoint_y, hole_no, round_no, category
        )
        order = np.argsort(candidates["donor_key"].astype(str).to_numpy())
        candidates = candidates.iloc[order].reset_index(drop=True)
        weights = weights[order]
        cumulative = np.cumsum(weights)
        index = min(int(np.searchsorted(cumulative, uniform, side="right")), len(candidates) - 1)
        donor = candidates.iloc[index]
        resolution_distance = float(
            np.hypot(
                float(donor["resolution_x"]) - float(forecast_pin_x),
                float(donor["resolution_y"]) - float(forecast_pin_y),
            ) / FEET_PER_YARD
        )
        if not 0.0 <= resolution_distance <= 700.0:
            raise HazardResolutionUnavailable(
                "Selected raw resolution is outside the playable distance range "
                f"for the supplied forecast pin: {resolution_distance:.1f} yd"
            )
        effective_n = float(1.0 / np.sum(weights * weights))
        backoff = ",".join(f"{name}={mass:.4f}" for name, mass in masses.items())
        return HazardResolution(
            donor_key=str(donor["donor_key"]),
            category=_normalize_category(category),
            penalty_count=int(donor["penalty_count"]),
            penalty_stroke_cost=int(donor["penalty_stroke_cost"]),
            resolution_x=float(donor["resolution_x"]),
            resolution_y=float(donor["resolution_y"]),
            resolution_lie=str(donor["resolution_lie"]),
            resolution_lie_raw=str(donor["resolution_lie_raw"]),
            resolution_distance_yd=resolution_distance,
            resolution_shot_key=str(donor["resolution_shot_key"]),
            resolution_event_type=str(donor["resolution_event_type"]),
            forecast_pin_x=float(forecast_pin_x),
            forecast_pin_y=float(forecast_pin_y),
            donor_geometry_resolution_distance_yd=float(
                donor["donor_geometry_resolution_distance_yd"]
            ),
            donor_sg_resolution_distance_yd=float(
                donor["donor_sg_resolution_distance_yd"]
            ),
            donor_sg_resolution_lie=str(donor["donor_sg_resolution_lie"]),
            donor_endpoint_x=float(donor["causal_endpoint_x"]),
            donor_endpoint_y=float(donor["causal_endpoint_y"]),
            endpoint_distance_yd=float(donor["_endpoint_distance_yd"]),
            selection_probability=float(weights[index]),
            effective_n=effective_n,
            support_setup=int(support["setup"]),
            support_hole=int(support["hole"]),
            support_category=int(support["category"]),
            backoff_weights=backoff,
            physical_course_id=self.physical_course_id,
            coordinate_reference_year=self.coordinate_reference_year,
            coordinate_reference_event_key=self.coordinate_reference_event_key,
            fallback_used=fallback_used,
            support_source=support_source,
            nearest_category_endpoint_distance_yd=nearest_category,
            nearest_same_hole_endpoint_distance_yd=nearest_same_hole,
            support_radius_limit_yd=support_radius_limit,
        )

    def resolve_state(
        self,
        state: Mapping[str, object],
        common_uniform: float,
    ) -> HazardResolution:
        """Mapping adapter for full-chain callers."""

        required = {
            "endpoint_x", "endpoint_y", "hole_no", "round_no", "category",
            "forecast_pin_x", "forecast_pin_y",
        }
        missing = required.difference(state)
        if missing:
            raise HazardResolutionUnavailable(
                f"Hazard state missing fields: {sorted(missing)}"
            )
        return self.resolve(
            endpoint_x=float(state["endpoint_x"]),
            endpoint_y=float(state["endpoint_y"]),
            hole_no=int(state["hole_no"]),
            round_no=int(state["round_no"]),
            category=str(state["category"]),
            common_uniform=common_uniform,
            forecast_pin_x=float(state["forecast_pin_x"]),
            forecast_pin_y=float(state["forecast_pin_y"]),
            physical_course_id=(
                str(state["physical_course_id"])
                if state.get("physical_course_id") is not None else None
            ),
        )

    def donor_diagnostics(self) -> dict[str, object]:
        by_category = self.donors["category"].value_counts().sort_index().to_dict()
        by_hole_category = (
            self.donors.groupby(["category", "hole_no"], observed=True)
            .size()
            .rename("donors")
            .reset_index()
        )
        return {
            "resolver_version": RESOLVER_VERSION,
            "foundation_version": FOUNDATION_VERSION,
            "physical_course_id": self.physical_course_id,
            "coordinate_reference_year": self.coordinate_reference_year,
            "coordinate_reference_event_key": self.coordinate_reference_event_key,
            "coordinate_reference_mode": "latest_prior",
            "donors": int(len(self.donors)),
            "by_category": {str(key): int(value) for key, value in by_category.items()},
            "hole_category_min": int(by_hole_category["donors"].min()),
            "hole_category_median": float(by_hole_category["donors"].median()),
            "hole_category_max": int(by_hole_category["donors"].max()),
            "setup_prior": self.setup_prior,
            "hole_prior": self.hole_prior,
            "spatial_bandwidth_yd": self.spatial_bandwidth_yd,
            "max_nearest_distance_yd": self.max_nearest_distance_yd,
            "same_hole_fallback_max_distance_yd": (
                self.same_hole_fallback_max_distance_yd
            ),
            "minimum_category_support": self.minimum_category_support,
        }


def _read_penalty_inputs(
    export_root: Path,
    season: int,
    event_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filters = [("event_key", "==", event_key)]
    raw_columns = [
        "shot_key", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "event_type", "start_lie", "end_lie",
        "distance_remaining_yd", "course_id", "start_x", "start_y", "end_x",
        "end_y", "ball_holed",
    ]
    app_columns = [
        "shot_key", "event_key", "source_player_id", "player_name", "round_no",
        "hole_no", "stroke_no", "sg_category", "start_lie",
        "start_distance_yd", "end_lie", "end_distance_yd", "sg_valid",
        "sg_issue", "complete_hole", "state_inferred",
    ]
    state_columns = [
        "shot_key", "sg_category", "start_lie", "start_distance_yd", "end_lie",
        "end_distance_yd",
    ]
    raw = pd.read_parquet(
        export_root / f"pga_{season}_shots.parquet",
        columns=raw_columns,
        filters=filters,
    )
    app = pd.read_parquet(
        export_root / f"pga_{season}_shot_sg.parquet",
        columns=app_columns,
        filters=filters + [("sg_category", "==", "app")],
    )
    states = pd.read_parquet(
        export_root / f"pga_{season}_shot_sg.parquet",
        columns=state_columns,
        filters=filters,
    )
    return raw, app, states


def attach_resolution_geometry(
    penalty_links: pd.DataFrame,
    edition_geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the exact causal layout's tee/pin to raw resolution rows.

    The join retains ``course_id`` as a within-edition layout gate. A missing
    or duplicated setup cannot silently borrow another course's pin.
    """

    geometry_columns = {
        "event_key", "round_no", "hole_no", "course_id", "par", "yardage",
        "tee_x", "tee_y", "pin_x", "pin_y",
    }
    missing = geometry_columns.difference(edition_geometry.columns)
    if missing:
        raise ValueError(f"Edition geometry missing: {sorted(missing)}")
    keys = ["event_key", "round_no", "hole_no", "course_id"]
    geometry = edition_geometry[list(geometry_columns)].copy()
    if geometry.duplicated(keys).any():
        raise ValueError("Edition geometry is duplicated by exact course setup")
    geometry = geometry.rename(
        columns={
            "course_id": "causal_course_id",
            "par": "donor_par",
            "yardage": "donor_yardage",
            "tee_x": "donor_tee_x",
            "tee_y": "donor_tee_y",
            "pin_x": "donor_pin_x",
            "pin_y": "donor_pin_y",
        }
    )
    merge_keys = ["event_key", "round_no", "hole_no", "causal_course_id"]
    return penalty_links.merge(
        geometry,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )


def load_normalized_hazard_panel(
    export_root: Path,
    event_name: str,
    holdout_year: int,
    course_start_year: int = 2020,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Load strict-prior donors and heldout hazards in one verified course frame."""

    editions, all_geometry, identity = load_verified_recurring_event(
        export_root, event_name, holdout_year, reference_mode="latest_prior"
    )
    reference_modes = set(identity["reference_mode"].dropna().astype(str))
    reference_keys = set(identity["reference_event_key"].dropna().astype(str))
    if reference_modes != {"latest_prior"} or len(reference_keys) != 1:
        raise AssertionError(
            "Hazard panel did not receive one latest-prior coordinate reference"
        )
    reference_event_key = next(iter(reference_keys))
    reference_catalog = editions[
        editions["event_key"].astype(str).eq(reference_event_key)
    ]
    if len(reference_catalog) != 1:
        raise AssertionError("Reference event key is absent or duplicated in catalog")
    reference_year = int(reference_catalog.iloc[0]["season"])
    if reference_year >= holdout_year:
        raise AssertionError(
            f"Coordinate reference {reference_year} is not strictly pre-holdout"
        )
    if reference_year != int(
        editions.loc[editions["season"].lt(holdout_year), "season"].max()
    ):
        raise AssertionError("Coordinate reference is not the latest prior edition")
    identity_by_key = identity.set_index("event_key")
    donor_links: list[pd.DataFrame] = []
    holdout_links: list[pd.DataFrame] = []
    prior_setup_geometry: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for edition in editions.sort_values("season").itertuples(index=False):
        season = int(edition.season)
        if season < course_start_year:
            continue
        raw, app, states = _read_penalty_inputs(
            export_root, season, str(edition.event_key)
        )
        _, links, audit = link_app_penalties_full_sequence(raw, app, states, season)
        links["season"] = season
        edition_geometry = all_geometry[
            all_geometry["event_key"].astype(str).eq(str(edition.event_key))
        ].copy()
        aligned_geometry = align_coordinates_to_target(
            edition_geometry,
            identity_by_key.loc[str(edition.event_key)],
            ("tee", "pin"),
        )
        if season < holdout_year:
            prior_setup_geometry.append(
                aligned_geometry[
                    ["round_no", "hole_no", "pin_x", "pin_y"]
                ].assign(season=season)
            )
        links = attach_resolution_geometry(links, edition_geometry)
        links = align_coordinates_to_target(
            links,
            identity_by_key.loc[str(edition.event_key)],
            (
                "causal_start", "causal_end", "resolution_raw_start",
                "old_stroke_only_causal_start",
                "old_stroke_only_causal_end",
                "old_stroke_only_resolution_start",
                "drop_end", "donor_tee", "donor_pin",
            ),
        )
        links["coordinate_reference_mode"] = "latest_prior"
        links["coordinate_reference_year"] = reference_year
        links["coordinate_reference_event_key"] = reference_event_key
        links["data_role"] = (
            "heldout_scoring_only" if season == holdout_year
            else "predictor_donor"
        )
        audits.append(audit)
        (holdout_links if season == holdout_year else donor_links).append(links)
    if not donor_links or not holdout_links:
        raise ValueError("Hazard panel requires prior and heldout penalty links")
    if not prior_setup_geometry:
        raise ValueError("Hazard panel has no strict-prior forecast pin geometry")
    donors, donor_diagnostics = prepare_hazard_donors(
        pd.concat(donor_links, ignore_index=True)
    )
    heldout, heldout_diagnostics = prepare_hazard_donors(
        pd.concat(holdout_links, ignore_index=True)
    )
    forecast_pins = (
        pd.concat(prior_setup_geometry, ignore_index=True)
        .groupby(["round_no", "hole_no"], as_index=False, observed=True)
        .agg(forecast_pin_x=("pin_x", "median"), forecast_pin_y=("pin_y", "median"))
    )
    heldout = heldout.merge(
        forecast_pins,
        on=["round_no", "hole_no"],
        how="left",
        validate="many_to_one",
    )
    if heldout[["forecast_pin_x", "forecast_pin_y"]].isna().any().any():
        raise AssertionError("Strict-prior forecast pin coverage is incomplete")
    if set(donors["data_role"]) != {"predictor_donor"}:
        raise AssertionError("Non-donor rows entered hazard predictor table")
    if set(heldout["data_role"]) != {"heldout_scoring_only"}:
        raise AssertionError("Heldout scoring table provenance is invalid")
    if donors["season"].max() > reference_year or heldout["season"].min() != holdout_year:
        raise AssertionError("Strict temporal split failed in hazard panel")
    diagnostics = {
        **{f"donor_{key}": value for key, value in donor_diagnostics.items()},
        **{f"heldout_{key}": value for key, value in heldout_diagnostics.items()},
    }
    return donors, heldout, identity, diagnostics


def _common_uniform(key: str) -> float:
    return zlib.crc32(str(key).encode("utf-8")) / float(2**32)


def _markdown_table(frame: pd.DataFrame) -> str:
    printable = frame.copy()
    for column in printable.select_dtypes(include=["float"]).columns:
        printable[column] = printable[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
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


def run_wyndham_smoke(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    donors, heldout, identity, load_diagnostics = load_normalized_hazard_panel(
        args.export_root,
        args.event_name,
        args.holdout_year,
        args.course_start_year,
    )
    resolver = EndpointLinkedHazardResolver(
        donors,
        max_nearest_distance_yd=args.max_nearest_distance_yd,
        same_hole_fallback_max_distance_yd=(
            args.same_hole_fallback_max_distance_yd
        ),
    )
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for record in heldout.itertuples(index=False):
        try:
            result = resolver.resolve(
                endpoint_x=float(record.causal_endpoint_x),
                endpoint_y=float(record.causal_endpoint_y),
                hole_no=int(record.hole_no),
                round_no=int(record.round_no),
                category=str(record.category),
                common_uniform=_common_uniform(str(record.donor_key)),
                forecast_pin_x=float(record.forecast_pin_x),
                forecast_pin_y=float(record.forecast_pin_y),
                physical_course_id=str(record.physical_course_id),
            )
        except HazardResolutionUnavailable as exc:
            failures.append(
                {
                    "donor_key": record.donor_key,
                    "category": record.category,
                    "reason": str(exc),
                }
            )
            continue
        row = asdict(result)
        actual_resolution_distance = float(
            np.hypot(
                float(record.resolution_x) - float(record.forecast_pin_x),
                float(record.resolution_y) - float(record.forecast_pin_y),
            ) / FEET_PER_YARD
        )
        row.update(
            {
                "query_key": str(record.donor_key),
                "query_round_no": int(record.round_no),
                "query_hole_no": int(record.hole_no),
                "actual_penalty_count": int(record.penalty_count),
                "actual_resolution_distance_yd": actual_resolution_distance,
                "actual_resolution_lie": str(record.resolution_lie),
                "actual_resolution_event_type": str(
                    record.resolution_event_type
                ),
                "resolution_distance_abs_error_yd": abs(
                    result.resolution_distance_yd
                    - actual_resolution_distance
                ),
                "resolution_lie_match": (
                    result.resolution_lie == str(record.resolution_lie)
                ),
            }
        )
        rows.append(row)
    scores = pd.DataFrame(rows)
    failures_frame = pd.DataFrame(failures)
    scores.to_csv(args.output_dir / "wyndham_hazard_resolutions.csv", index=False)
    failures_frame.to_csv(args.output_dir / "wyndham_hazard_failures.csv", index=False)
    donors.to_parquet(args.output_dir / "wyndham_hazard_donors.parquet", index=False)
    identity.to_csv(args.output_dir / "wyndham_course_identity.csv", index=False)

    by_category = []
    for category in SUPPORTED_CATEGORIES:
        expected = int(heldout["category"].eq(category).sum())
        group = scores[scores["category"].eq(category)] if len(scores) else scores
        by_category.append(
            {
                "category": category,
                "heldout_hazards": expected,
                "resolved": int(len(group)),
                "resolution_rate": len(group) / expected if expected else math.nan,
                "median_selected_endpoint_distance_yd": float(
                    group["endpoint_distance_yd"].median()
                ) if len(group) else math.nan,
                "p90_selected_endpoint_distance_yd": float(
                    group["endpoint_distance_yd"].quantile(0.90)
                ) if len(group) else math.nan,
                "median_effective_n": float(group["effective_n"].median())
                if len(group) else math.nan,
                "resolution_distance_mae_yd": float(
                    group["resolution_distance_abs_error_yd"].mean()
                ) if len(group) else math.nan,
                "resolution_lie_match_rate": float(
                    group["resolution_lie_match"].mean()
                ) if len(group) else math.nan,
            }
        )
    category_metrics = pd.DataFrame(by_category)
    category_metrics.to_csv(args.output_dir / "category_smoke_metrics.csv", index=False)
    state_audit = (
        donors.groupby("category", as_index=False, observed=True)
        .agg(
            retained_donors=("donor_key", "size"),
            observed_penalty_rows=("penalty_count", "sum"),
            penalty_stroke_cost=("penalty_stroke_cost", "sum"),
            provisional_resolutions=(
                "resolution_event_type", lambda values: values.eq("PROVISIONAL").sum()
            ),
            raw_vs_sg_lie_mismatches=(
                "donor_raw_vs_sg_lie_match", lambda values: (~values).sum()
            ),
            sg_distance_gt50yd_mismatches=(
                "donor_sg_distance_abs_error_yd", lambda values: values.gt(50).sum()
            ),
            sg_distance_max_abs_error_yd=(
                "donor_sg_distance_abs_error_yd", "max"
            ),
        )
    )
    state_audit.to_csv(args.output_dir / "donor_state_audit.csv", index=False)
    diagnostics = {
        **load_diagnostics,
        **resolver.donor_diagnostics(),
        "heldout_hazards": int(len(heldout)),
        "resolved_hazards": int(len(scores)),
        "failed_hazards": int(len(failures_frame)),
        "resolution_rate": len(scores) / len(heldout),
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    table = _markdown_table(category_metrics)
    state_table = _markdown_table(state_audit)
    report = f"""# Endpoint-linked hazard resolver: Wyndham smoke

- Strict-prior donors: **{len(donors):,}** causal penalty outcomes from editions before {args.holdout_year}
- Heldout hazards: **{len(heldout):,}**; resolved: **{len(scores):,}**; fail-closed: **{len(failures_frame):,}**
- Physical course: **{resolver.physical_course_id}**
- Coordinate reference: **latest prior edition**; heldout setup geometry does not define the donor frame
- Foundation: **{FOUNDATION_VERSION}**; resolver: **{RESOLVER_VERSION}**
- Recursive state: **raw next STROKE/PROVISIONAL lie + XY**; distance is recomputed to the strict-prior forecast pin

{table}

## Donor state audit

{state_table}

The SG mismatches above reproduce the source problem as a diagnostic; they are
not recursive inputs. After requiring an exact raw played row, finite raw XY,
a recognized raw lie, exact course continuity and geometry-valid distance,
**{len(donors):,}** donors remain. The loader excluded
**{load_diagnostics['donor_excluded_incomplete_resolution']:,}** supported-category
penalty rows with incomplete raw/geometry state and
**{load_diagnostics['donor_excluded_incoherent_resolution']:,}** with incoherent
raw state. It relabelled
**{load_diagnostics['donor_causal_provisional_relabelled_ott_rows']:,}** causal
par-4/5 tee provisionals as OTT using raw state plus verified geometry.

## Stroke accounting

The retired STROKE-only contract undercounted the official path by one stroke
for **{load_diagnostics['donor_old_stroke_only_undercount_donors']:,} / {load_diagnostics['donor_old_stroke_only_donors']:,}**
otherwise eligible donors. Those are replay/provisional paths that jumped to a
later STROKE but charged only the explicit PENALTY rows. The current resolver
returns both semantic `penalty_count` and official `penalty_stroke_cost`.
After the caller has charged the causal shot, adding `penalty_stroke_cost`
lands at `resolution_stroke_no - 1`; simulating the returned STROKE or
PROVISIONAL then charges the next played stroke. This identity holds for
**{load_diagnostics['donor_stroke_span_reconciled_donors']:,} / {len(donors):,}**
retained donors. In
**{load_diagnostics['donor_penalty_cost_exceeds_observed_rows_donors']:,}**
case the official stroke-number span records an extra penalty stroke not
represented by a distinct raw PENALTY row, so the span-derived cost is used.

The smoke begins from actual heldout causal hazard endpoints solely to test
support and state-chain mechanics. It is not an accuracy claim about hazard
incidence. Each query uses a deterministic common uniform, borrows only within
its SG category, and returns one historical donor's exact linked penalty count
and next playable-row coordinates/lie. Distance is derived from that returned
XY and the caller-supplied forecast pin. SG resolution lie/distance remain
diagnostics and never enter the recursive state. The endpoint hierarchy is
round/hole/category -> hole/category -> category. A query outside the primary
same-category radius and the optional same-hole extension raises
`HazardResolutionUnavailable`.

OTT donors provide the tee-hazard path and APP donors the approach-hazard
path. No Tour, other-course or cross-category fallback is permitted. The
optional extended radius, when enabled, is same-hole and same-category only.
Returned
coordinates remain in the verified forecast frame and can be fed directly to
the next transition.
"""
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(category_metrics.to_string(index=False))
    print(f"Wrote {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-name", default="Wyndham Championship")
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument("--max-nearest-distance-yd", type=float, default=120.0)
    parser.add_argument(
        "--same-hole-fallback-max-distance-yd", type=float, default=None
    )
    return parser.parse_args()


def main() -> None:
    run_wyndham_smoke(parse_args())


if __name__ == "__main__":
    main()

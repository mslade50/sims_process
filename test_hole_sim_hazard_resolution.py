from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive.hole_sim_hazard_resolution import (
    EndpointLinkedHazardResolver,
    HazardResolutionUnavailable,
    prepare_hazard_donors,
)
from archive.hole_sim_state_foundations import FOUNDATION_VERSION


def _link(
    key: str,
    category: str,
    hole: int,
    endpoint_x: float,
    *,
    penalty_suffix: str = "p",
    complete: bool = True,
    resolution_lie: str = "rough",
    resolution_event_type: str = "STROKE",
    sg_resolution_lie: str = "rough",
    sg_resolution_distance: float | None = None,
    donor_pin_x: float = 4500.0,
    donor_pin_y: float = 1010.0,
    donor_tee_x: float = 2500.0,
    donor_tee_y: float = 1010.0,
    causal_event_type: str = "STROKE",
    causal_start_lie: str = "fairway",
    donor_par: int = 4,
    causal_stroke_no: int = 1,
    resolution_stroke_no: int = 3,
) -> dict:
    resolution_x = endpoint_x + 30.0 if complete else None
    return {
        "event_key": "pga:R2025999",
        "season": 2025,
        "round_no": 1,
        "hole_no": hole,
        "causal_shot_key": key,
        "penalty_shot_key": f"{key}-{penalty_suffix}",
        "causal_sg_sg_category": category,
        "causal_event_type": causal_event_type,
        "causal_stroke_no": causal_stroke_no,
        "causal_start_lie": causal_start_lie,
        "causal_course_id": "layout:a",
        "causal_end_x": endpoint_x,
        "causal_end_y": 1000.0 + hole * 10,
        "resolution_raw_shot_key": f"{key}-resolution",
        "resolution_raw_event_type": resolution_event_type,
        "resolution_raw_stroke_no": resolution_stroke_no,
        "resolution_raw_course_id": "layout:a",
        "resolution_raw_start_x": resolution_x,
        "resolution_raw_start_y": 1015.0 + hole * 10,
        "resolution_raw_start_lie": resolution_lie,
        "resolution_sg_start_lie": sg_resolution_lie,
        "resolution_sg_start_distance_yd": (
            120.0 + hole
            if sg_resolution_distance is None else sg_resolution_distance
        ),
        "donor_tee_x": donor_tee_x,
        "donor_tee_y": donor_tee_y,
        "donor_pin_x": donor_pin_x,
        "donor_pin_y": donor_pin_y,
        "donor_par": donor_par,
        "causal_linked": True,
        "physical_course_id": "geometry:test",
        "foundation_version": FOUNDATION_VERSION,
        "coordinate_reference_year": 2025,
        "coordinate_reference_event_key": "pga:R2025999",
        "coordinate_reference_mode": "latest_prior",
        "data_role": "predictor_donor",
    }


def _donors() -> pd.DataFrame:
    links = []
    for category, offset in (("ott", 0.0), ("app", 900.0)):
        for index in range(12):
            links.append(
                _link(
                    f"{category}-{index}",
                    category,
                    1 + index % 2,
                    3000.0 + offset + index * 8.0,
                )
            )
    donors, diagnostics = prepare_hazard_donors(pd.DataFrame(links))
    assert diagnostics["causal_hazard_donors"] == 24
    return donors


def test_prepare_collapses_multiple_penalties_and_excludes_incomplete() -> None:
    rows = [
        _link(
            "multi", "app", 1, 3000.0, penalty_suffix="p1",
            resolution_stroke_no=4,
        ),
        _link(
            "multi", "app", 1, 3000.0, penalty_suffix="p2",
            resolution_stroke_no=4,
        ),
        _link("bad", "app", 1, 3010.0, complete=False),
    ]
    donors, diagnostics = prepare_hazard_donors(pd.DataFrame(rows))
    assert len(donors) == 1
    assert donors.iloc[0]["donor_key"] == "multi"
    assert donors.iloc[0]["penalty_count"] == 2
    assert donors.iloc[0]["penalty_stroke_cost"] == 2
    assert donors.iloc[0]["resolution_x"] == 3030.0
    assert diagnostics["excluded_incomplete_resolution"] == 1


def test_common_uniform_resolution_is_deterministic_and_exact() -> None:
    resolver = EndpointLinkedHazardResolver(_donors())
    kwargs = dict(
        endpoint_x=3030.0,
        endpoint_y=1010.0,
        hole_no=1,
        round_no=1,
        category="ott",
        common_uniform=0.371,
        forecast_pin_x=4500.0,
        forecast_pin_y=1010.0,
        physical_course_id="geometry:test",
    )
    first = resolver.resolve(**kwargs)
    second = resolver.resolve(**kwargs)
    assert first == second
    selected = resolver.donors.set_index("donor_key").loc[first.donor_key]
    assert first.penalty_count == selected["penalty_count"]
    assert first.penalty_stroke_cost == selected["penalty_stroke_cost"]
    assert first.resolution_x == selected["resolution_x"]
    assert first.resolution_y == selected["resolution_y"]
    assert first.resolution_lie == selected["resolution_lie"]
    expected_distance = (
        ((selected["resolution_x"] - 4500.0) ** 2
         + (selected["resolution_y"] - 1010.0) ** 2) ** 0.5 / 3.0
    )
    assert first.resolution_distance_yd == pytest.approx(expected_distance)
    assert first.resolution_lie == selected["resolution_lie"]
    assert first.support_setup > 0
    assert first.support_hole > 0
    assert first.support_category == 12
    assert first.coordinate_reference_year == 2025


def test_resolver_fails_closed_for_wrong_category_course_distance_or_role() -> None:
    donors = _donors()
    resolver = EndpointLinkedHazardResolver(donors)
    base = dict(
        endpoint_x=3030.0,
        endpoint_y=1010.0,
        hole_no=1,
        round_no=1,
        common_uniform=0.2,
        forecast_pin_x=4500.0,
        forecast_pin_y=1010.0,
    )
    with pytest.raises(HazardResolutionUnavailable, match="Only 0"):
        resolver.resolve(**base, category="arg")
    with pytest.raises(HazardResolutionUnavailable, match="Query course"):
        resolver.resolve(
            **base, category="ott", physical_course_id="geometry:wrong"
        )
    with pytest.raises(HazardResolutionUnavailable, match="Nearest"):
        resolver.resolve(
            **{**base, "endpoint_x": 100000.0}, category="ott"
        )
    heldout = donors.copy()
    heldout["data_role"] = "heldout_scoring_only"
    with pytest.raises(ValueError, match="Heldout scoring"):
        EndpointLinkedHazardResolver(heldout)


def test_raw_replay_state_overrides_stale_sg_and_distance_uses_forecast_pin() -> None:
    row = _link(
        "replay",
        "ott",
        11,
        100.0,
        resolution_lie="tee",
        resolution_event_type="PROVISIONAL",
        sg_resolution_lie="recovery",
        sg_resolution_distance=145.0,
        donor_tee_x=130.0,
        donor_tee_y=1125.0,
        donor_pin_x=1558.0,
        donor_pin_y=1125.0,
        causal_start_lie="tee",
    )
    donors, diagnostics = prepare_hazard_donors(pd.DataFrame([row]))
    assert diagnostics["resolution_provisional_donors"] == 1
    assert diagnostics["resolution_sg_distance_gt50yd_mismatch_donors"] == 1
    resolver = EndpointLinkedHazardResolver(donors, minimum_category_support=1)
    result = resolver.resolve(
        endpoint_x=100.0,
        endpoint_y=1110.0,
        hole_no=11,
        round_no=1,
        category="ott",
        common_uniform=0.5,
        forecast_pin_x=1558.0,
        forecast_pin_y=1125.0,
    )
    assert result.resolution_event_type == "PROVISIONAL"
    assert result.penalty_count == 1
    assert result.penalty_stroke_cost == 1
    assert result.resolution_lie == "tee"
    assert result.resolution_x == 130.0
    assert result.resolution_distance_yd == pytest.approx(476.0)
    assert result.donor_sg_resolution_distance_yd == 145.0
    assert result.donor_sg_resolution_lie == "recovery"


def test_prepare_fails_closed_on_incoherent_raw_tee_state() -> None:
    row = _link(
        "bad-tee",
        "ott",
        1,
        3000.0,
        resolution_lie="tee",
        donor_tee_x=0.0,
        donor_tee_y=0.0,
    )
    donors, diagnostics = prepare_hazard_donors(pd.DataFrame([row]))
    assert donors.empty
    assert diagnostics["excluded_incoherent_resolution"] == 1
    assert diagnostics["excluded_tee_not_at_tee"] == 1


def test_official_stroke_span_can_exceed_observed_penalty_rows() -> None:
    row = _link(
        "implicit-cost", "ott", 1, 3000.0, resolution_stroke_no=4
    )
    donors, diagnostics = prepare_hazard_donors(pd.DataFrame([row]))
    assert len(donors) == 1
    assert donors.iloc[0]["penalty_count"] == 1
    assert donors.iloc[0]["penalty_stroke_cost"] == 2
    assert diagnostics["penalty_cost_exceeds_observed_rows_donors"] == 1
    assert diagnostics["stroke_span_reconciled_donors"] == 1


def test_resolve_state_requires_forecast_pin() -> None:
    resolver = EndpointLinkedHazardResolver(_donors())
    state = {
        "endpoint_x": 3030.0,
        "endpoint_y": 1010.0,
        "hole_no": 1,
        "round_no": 1,
        "category": "ott",
    }
    with pytest.raises(HazardResolutionUnavailable, match="forecast_pin"):
        resolver.resolve_state(state, 0.4)


def test_vectorized_support_and_same_hole_only_fallback() -> None:
    rows = [
        # Query (2616, 1010) is 128 yd from this same-hole donor.
        _link("same-hole", "app", 1, 3000.0),
        # This different-hole donor is closer (about 124.7 yd), but remains
        # outside the 120 yd primary radius and must not enter the extension.
        _link("cross-hole", "app", 2, 2990.0),
    ]
    donors, _ = prepare_hazard_donors(pd.DataFrame(rows))
    resolver = EndpointLinkedHazardResolver(
        donors,
        minimum_category_support=1,
        same_hole_fallback_max_distance_yd=150.0,
    )
    endpoint_x = np.asarray([2616.0, 2998.0])
    endpoint_y = np.asarray([1010.0, 1010.0])
    support = resolver.has_support_many(
        endpoint_x,
        endpoint_y,
        hole_no=1,
        round_no=1,
        category="app",
    )
    diagnostics = resolver.support_diagnostics_many(
        endpoint_x,
        endpoint_y,
        hole_no=1,
        round_no=1,
        category="app",
    )
    assert support.tolist() == [True, True]
    assert diagnostics.primary_supported.tolist() == [False, True]
    assert diagnostics.fallback_used.tolist() == [True, False]
    assert diagnostics.support_source.tolist() == [
        "same_hole_category_fallback_radius",
        "same_category_primary_radius",
    ]
    assert diagnostics.nearest_category_endpoint_distance_yd[0] < 128.0
    assert diagnostics.nearest_category_endpoint_distance_yd[0] > 120.0
    assert diagnostics.nearest_same_hole_endpoint_distance_yd[0] == pytest.approx(
        128.0
    )
    assert resolver.has_support(
        2616.0,
        1010.0,
        hole_no=1,
        round_no=1,
        category="app",
    )

    resolved = resolver.resolve(
        endpoint_x=2616.0,
        endpoint_y=1010.0,
        hole_no=1,
        round_no=1,
        category="app",
        common_uniform=0.73,
        forecast_pin_x=4500.0,
        forecast_pin_y=1010.0,
    )
    assert resolved.donor_key == "same-hole"
    assert resolved.fallback_used
    assert resolved.support_source == "same_hole_category_fallback_radius"
    assert resolved.support_radius_limit_yd == 150.0
    assert resolved.nearest_same_hole_endpoint_distance_yd == pytest.approx(128.0)


def test_same_hole_extension_never_borrows_cross_hole_donor() -> None:
    rows = [
        # Same-hole support is outside both the 120 yd primary and 150 yd
        # extension radii.
        _link("same-hole-too-far", "app", 1, 3100.0),
        # A different-hole donor lies within 150 yd but outside 120 yd.
        _link("cross-hole-125yd", "app", 2, 2990.0),
    ]
    donors, _ = prepare_hazard_donors(pd.DataFrame(rows))
    resolver = EndpointLinkedHazardResolver(
        donors,
        minimum_category_support=1,
        same_hole_fallback_max_distance_yd=150.0,
    )
    diagnostics = resolver.support_diagnostics_many(
        np.asarray([2616.0]),
        np.asarray([1010.0]),
        hole_no=1,
        round_no=1,
        category="app",
    )
    assert not diagnostics.supported.item()
    assert not diagnostics.fallback_used.item()
    assert 120.0 < diagnostics.nearest_category_endpoint_distance_yd.item() < 150.0
    assert diagnostics.nearest_same_hole_endpoint_distance_yd.item() > 150.0
    with pytest.raises(HazardResolutionUnavailable, match="nearest same-hole"):
        resolver.resolve(
            endpoint_x=2616.0,
            endpoint_y=1010.0,
            hole_no=1,
            round_no=1,
            category="app",
            common_uniform=0.5,
            forecast_pin_x=4500.0,
            forecast_pin_y=1010.0,
        )

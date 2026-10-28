from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive.hole_sim_state_foundations import (
    align_coordinates_to_target,
    attach_causal_hazard_labels,
    build_causal_hazard_labels,
    build_causal_ott_hazard_labels,
    link_app_penalties_full_sequence,
    verify_physical_course_identity,
)


def _raw_row(
    key: str,
    stroke_no: int,
    event_type: str,
    start_x: float,
    end_x: float,
    *,
    player: str = "1",
    course_id: str = "layout:a",
    start_lie: str = "fairway",
    end_lie: str = "rough",
) -> dict:
    return {
        "shot_key": key,
        "event_key": "pga:R2026999",
        "source_player_id": player,
        "player_name": f"player {player}",
        "round_no": 1,
        "hole_no": 1,
        "stroke_no": stroke_no,
        "event_type": event_type,
        "start_lie": start_lie,
        "end_lie": end_lie,
        "distance_remaining_yd": 100.0,
        "course_id": course_id,
        "start_x": start_x,
        "start_y": 0.0,
        "end_x": end_x,
        "end_y": 0.0,
        "ball_holed": 0,
    }


def _sg_row(
    key: str,
    stroke_no: int,
    category: str,
    start_distance: float,
    end_distance: float,
    *,
    player: str = "1",
    start_lie: str = "fairway",
    end_lie: str = "rough",
) -> dict:
    return {
        "shot_key": key,
        "event_key": "pga:R2026999",
        "source_player_id": player,
        "player_name": f"player {player}",
        "round_no": 1,
        "hole_no": 1,
        "stroke_no": stroke_no,
        "sg_category": category,
        "start_lie": start_lie,
        "start_distance_yd": start_distance,
        "end_lie": end_lie,
        "end_distance_yd": end_distance,
        "sg_valid": 1,
        "sg_issue": "",
        "complete_hole": 1,
        "state_inferred": "",
    }


def test_full_sequence_penalties_keep_endpoint_and_exact_resolution() -> None:
    raw = pd.DataFrame(
        [
            _raw_row("a", 1, "STROKE", 0.0, 90.0),
            _raw_row("p1", 2, "PENALTY", 90.0, 90.0),
            _raw_row("p2", 2, "PENALTY", 90.0, 90.0),
            _raw_row("d", 2, "DROP", 90.0, 120.0),
            _raw_row("r", 4, "STROKE", 120.0, 180.0, start_lie="tee"),
            # The second sequence demonstrates the filtered-stream bug: an
            # APP stroke is followed by a non-APP physical stroke and then an
            # APP-labelled penalty bookkeeping row.
            _raw_row("a2", 1, "STROKE", 0.0, 60.0, player="2"),
            _raw_row("n2", 2, "STROKE", 60.0, 75.0, player="2"),
            _raw_row("p3", 3, "PENALTY", 75.0, 75.0, player="2"),
            _raw_row("r2", 4, "STROKE", 105.0, 150.0, player="2"),
        ]
    )
    states = pd.DataFrame(
        [
            _sg_row("a", 1, "app", 150.0, 70.0),
            _sg_row("p1", 2, "app", 70.0, 42.0),
            _sg_row("p2", 2, "app", 70.0, 42.0),
            _sg_row("d", 2, "app", 70.0, 42.0),
            _sg_row("r", 4, "app", 42.0, 10.0, start_lie="recovery"),
            _sg_row("a2", 1, "app", 160.0, 100.0, player="2"),
            _sg_row("n2", 2, "arg", 100.0, 90.0, player="2"),
            _sg_row("p3", 3, "app", 90.0, 55.0, player="2"),
            _sg_row("r2", 4, "arg", 55.0, 20.0, player="2"),
        ]
    )
    app = states[states["sg_category"].eq("app")].copy()

    frame, links, audit = link_app_penalties_full_sequence(raw, app, states, 2026)

    causal = frame.set_index("shot_key").loc["a"]
    assert causal["penalty_after"]
    assert causal["penalty_count"] == 2
    assert causal["end_x"] == 90.0
    assert causal["causal_end_x"] == 90.0
    assert causal["causal_end_distance_yd"] == 70.0
    assert causal["end_lie"] == "penalty"
    # The causal SG endpoint remains diagnostic; the recursive distance is not
    # copied from the stale SG annotation and will be computed to a forecast pin.
    assert causal["end_distance_yd"] == 70.0
    assert causal["resolution_shot_key"] == "r"
    assert causal["resolution_event_type"] == "STROKE"
    assert causal["resolution_start_x"] == 120.0
    assert causal["resolution_start_lie"] == "tee"
    assert causal["resolution_sg_start_lie_diagnostic"] == "recovery"
    assert causal["resolution_sg_start_distance_yd_diagnostic"] == 42.0
    assert not frame.set_index("shot_key").loc["a2", "penalty_after"]

    false_app = links.set_index("penalty_shot_key").loc["p3"]
    assert false_app["causal_shot_key"] == "n2"
    assert false_app["causal_sg_sg_category"] == "arg"
    assert false_app["legacy_causal_shot_key"] == "a2"
    assert false_app["legacy_causal_mismatch"]
    assert audit["legacy_false_app_assignment_rows"] == 1
    assert audit["attached_app_hazard_strokes"] == 1
    assert audit["causal_endpoint_preserved_rate_after"] == 1.0
    assert audit["resolution_raw_vs_sg_lie_match_rate_app"] == 0.0
    assert audit["drop_to_resolution_median_gap_yd_after"] == 0.0
    assert audit["all_drop_to_next_physical_median_gap_yd"] == 0.0


def test_provisional_is_next_playable_and_can_be_causal() -> None:
    raw = pd.DataFrame(
        [
            _raw_row("tee", 1, "STROKE", 0.0, 80.0, start_lie="tee"),
            _raw_row("p1", 2, "PENALTY", 80.0, 80.0),
            _raw_row(
                "prov", 3, "PROVISIONAL", 0.0, 90.0, start_lie="tee"
            ),
            _raw_row("p2", 4, "PENALTY", 90.0, 90.0),
            _raw_row("next", 5, "STROKE", 95.0, 160.0, start_lie="rough"),
        ]
    )
    states = pd.DataFrame(
        [
            _sg_row("tee", 1, "ott", 450.0, 200.0, start_lie="tee"),
            _sg_row("p1", 2, "app", 200.0, 145.0),
            # This intentionally mimics stale SG state on a raw tee replay.
            _sg_row("prov", 3, "app", 145.0, 110.0, start_lie="recovery"),
            _sg_row("p2", 4, "app", 110.0, 100.0),
            _sg_row("next", 5, "app", 100.0, 40.0, start_lie="rough"),
        ]
    )
    app = states[states["sg_category"].eq("app")].copy()

    _, links, audit = link_app_penalties_full_sequence(raw, app, states, 2026)
    by_penalty = links.set_index("penalty_shot_key")
    assert by_penalty.loc["p1", "causal_shot_key"] == "tee"
    assert by_penalty.loc["p1", "resolution_raw_shot_key"] == "prov"
    assert by_penalty.loc["p1", "resolution_raw_event_type"] == "PROVISIONAL"
    assert by_penalty.loc["p1", "resolution_raw_start_lie"] == "tee"
    assert by_penalty.loc["p2", "causal_shot_key"] == "prov"
    assert by_penalty.loc["p2", "causal_event_type"] == "PROVISIONAL"
    assert by_penalty.loc["p2", "resolution_raw_shot_key"] == "next"
    assert audit["causal_provisional_penalty_rows"] == 1
    assert audit["resolution_provisional_penalty_rows"] == 1
    assert audit["old_stroke_only_supported_donors_before_state_gates"] == 1
    assert audit["old_stroke_only_undercount_donors_before_state_gates"] == 1

    labels = build_causal_hazard_labels(links).set_index("shot_key")
    assert labels.loc["tee", "tee_hazard_after"]
    assert labels.loc["tee", "tee_penalty_count"] == 1
    assert labels.loc["tee", "tee_penalty_stroke_cost"] == 1
    assert labels.loc["prov", "app_hazard_after"]
    assert labels.loc["prov", "app_penalty_stroke_cost"] == 1
    ott_labels = build_causal_ott_hazard_labels(links).set_index("shot_key")
    assert list(ott_labels.index) == ["tee"]
    assert ott_labels.loc["tee", "tee_penalty_stroke_cost"] == 1
    labelled = attach_causal_hazard_labels(
        pd.DataFrame({"shot_key": ["tee", "prov", "next"]}), links
    ).set_index("shot_key")
    assert labelled.loc["tee", "tee_hazard_after"]
    assert not labelled.loc["next", "causal_hazard_after"]
    assert labelled.loc["next", "causal_penalty_count"] == 0


def _geometry(event_key: str, transform: str = "same") -> pd.DataFrame:
    rows: list[dict] = []
    for round_no in range(1, 5):
        for hole_no in range(1, 19):
            theta = 2 * np.pi * (hole_no - 1) / 18
            tee = np.array(
                [1800 * np.cos(theta) + hole_no * 31, 1200 * np.sin(theta) + hole_no * 17]
            )
            direction = np.array([500 + hole_no * 8, 120 - hole_no * 3])
            pin = tee + direction + np.array([round_no * 2.0, -round_no * 1.5])
            if transform == "rigid":
                angle = 0.37
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                tee = tee @ rotation + np.array([5000.0, -900.0])
                pin = pin @ rotation + np.array([5000.0, -900.0])
            elif transform == "different":
                tee = np.array([hole_no * hole_no * 80.0, hole_no * 430.0])
                pin = tee + np.array([900.0 - hole_no * 15.0, hole_no * 21.0])
            rows.append(
                {
                    "event_key": event_key,
                    "round_no": round_no,
                    "hole_no": hole_no,
                    "course_id": f"{event_key}:layout:{round_no}",
                    "par": 4,
                    "yardage": float(np.hypot(*(pin - tee)) / 3.0),
                    "tee_x": tee[0],
                    "tee_y": tee[1],
                    "pin_x": pin[0],
                    "pin_y": pin[1],
                }
            )
    return pd.DataFrame(rows)


def test_physical_identity_accepts_rigid_match_and_rejects_venue_change() -> None:
    editions = pd.DataFrame(
        [
            {"event_key": "old-multi", "season": 2023, "event_name": "Test"},
            {"event_key": "old-other", "season": 2024, "event_name": "Test"},
            {"event_key": "old-same", "season": 2025, "event_name": "Test"},
            {"event_key": "target", "season": 2026, "event_name": "Test"},
        ]
    )
    target = _geometry("target")
    same = _geometry("old-same", "rigid")
    other = _geometry("old-other", "different")
    multi = _geometry("old-multi")
    duplicate = multi.copy()
    duplicate["course_id"] = duplicate["course_id"] + ":second"
    multi = pd.concat([multi, duplicate], ignore_index=True)
    geometry = pd.concat([multi, other, same, target], ignore_index=True)

    audit = verify_physical_course_identity(editions, geometry, 2026).set_index(
        "event_key"
    )
    assert audit.loc["target", "status"] == "target_verified"
    assert audit.loc["old-same", "status"] == "verified_same_course"
    assert bool(audit.loc["old-same", "accepted"])
    assert audit.loc["old-other", "status"] == "rejected_venue_mismatch"
    assert not bool(audit.loc["old-other", "accepted"])
    assert audit.loc["old-multi", "status"] == "rejected_multicourse_or_incomplete"
    assert not bool(audit.loc["old-multi", "accepted"])
    aligned = align_coordinates_to_target(
        same,
        audit.loc["old-same"],
        ("tee", "pin"),
    )
    target_sorted = target.sort_values(["round_no", "hole_no"])
    aligned_sorted = aligned.sort_values(["round_no", "hole_no"])
    assert np.allclose(
        aligned_sorted[["tee_x", "tee_y", "pin_x", "pin_y"]],
        target_sorted[["tee_x", "tee_y", "pin_x", "pin_y"]],
        atol=1e-8,
    )


def test_multicourse_target_fails_closed() -> None:
    editions = pd.DataFrame(
        [{"event_key": "target", "season": 2026, "event_name": "Test"}]
    )
    target = _geometry("target")
    duplicate = target.copy()
    duplicate["course_id"] = duplicate["course_id"] + ":second"
    with pytest.raises(ValueError, match="Multicourse targets"):
        verify_physical_course_identity(
            editions, pd.concat([target, duplicate], ignore_index=True), 2026
        )


def test_latest_prior_reference_is_invariant_to_holdout_coordinate_origin() -> None:
    editions = pd.DataFrame(
        [
            {"event_key": "prior-2024", "season": 2024, "event_name": "Test"},
            {"event_key": "prior-2025", "season": 2025, "event_name": "Test"},
            {"event_key": "holdout", "season": 2026, "event_name": "Test"},
        ]
    )
    prior_2024 = _geometry("prior-2024")
    prior_2025 = _geometry("prior-2025", "rigid")
    holdout_a = _geometry("holdout")
    holdout_b = holdout_a.copy()
    holdout_b[["tee_x", "pin_x"]] += 9000.0
    holdout_b[["tee_y", "pin_y"]] -= 4000.0
    first = verify_physical_course_identity(
        editions,
        pd.concat([prior_2024, prior_2025, holdout_a], ignore_index=True),
        2026,
        reference_mode="latest_prior",
    ).set_index("event_key")
    second = verify_physical_course_identity(
        editions,
        pd.concat([prior_2024, prior_2025, holdout_b], ignore_index=True),
        2026,
        reference_mode="latest_prior",
    ).set_index("event_key")
    assert first.loc["prior-2025", "status"] == "latest_prior_reference"
    assert first.loc["holdout", "status"] == "heldout_verified_against_prior"
    assert first.loc["prior-2025", "reference_event_key"] == "prior-2025"
    assert first.loc["prior-2025", "reference_mode"] == "latest_prior"
    assert first.loc["prior-2025", "physical_course_id"] == second.loc[
        "prior-2025", "physical_course_id"
    ]
    for column in (
        "rotation_00", "rotation_01", "rotation_10", "rotation_11",
        "translation_x", "translation_y",
    ):
        assert first.loc["prior-2024", column] == pytest.approx(
            second.loc["prior-2024", column]
        )

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parent / "archive" / "calibrate_live_scoring_setup_yardage.py"
SPEC = importlib.util.spec_from_file_location("yardage_calibrator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
calibrator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibrator)


def _geometry(varying_holes: int = 18) -> dict[int, dict[int, tuple[int, float]]]:
    return {
        rnd: {
            hole: (
                4,
                400.0 + hole + (rnd - 1 if hole <= varying_holes else 0),
            )
            for hole in range(1, 19)
        }
        for rnd in range(1, 5)
    }


def _hole_means(event_keys: list[str]) -> dict[tuple[str, int, int], dict]:
    return {
        (event_key, rnd, hole): {
            "mean_score": 4.0 + 0.01 * rnd + 0.001 * hole,
            "completed_scores": 20,
        }
        for event_key in event_keys
        for rnd in range(1, 5)
        for hole in range(1, 19)
    }


def test_exact_course_mapping_never_aliases_name_date_or_layout() -> None:
    panel = [
        {
            "event_key": "pga:R2025001",
            "source_calendar_year": 2025,
            "event_number": 1,
        },
        {
            "event_key": "pga:R2025002",
            "source_calendar_year": 2025,
            "event_number": 2,
        },
        {
            "event_key": "pga:R2025003",
            "source_calendar_year": 2025,
            "event_number": 3,
        },
    ]
    dg_editions = [
        {
            "year": 2025,
            "event_id": 1,
            "course_id": 101,
            "course_name": "Stable Course",
        },
        {
            "year": 2024,
            "event_id": 1,
            "course_id": 999,
            "course_name": "Wrong Year",
        },
        {
            "year": 2025,
            "event_id": 2,
            "course_id": 201,
            "course_name": "Multi A",
        },
        {
            "year": 2025,
            "event_id": 2,
            "course_id": 202,
            "course_name": "Multi B",
        },
    ]

    mapped, rejected = calibrator._map_exact_datagolf_courses(panel, dg_editions)

    assert [row["event_key"] for row in mapped] == ["pga:R2025001"]
    assert mapped[0]["coefficient_course_id"] == 101
    assert [row["event_key"] for row in rejected] == [
        "pga:R2025002",
        "pga:R2025003",
    ]
    assert rejected[0]["candidate_course_nums"] == [201, 202]
    assert rejected[1]["candidate_course_nums"] == []


def test_informative_panel_requires_eight_varying_holes_and_complete_scores() -> None:
    mapped = [
        {
            "event_key": "include",
            "coefficient_calendar_year": 2025,
            "coefficient_course_id": 1,
            "coefficient_course_names": ["Course"],
            "holes_by_round": _geometry(8),
        },
        {
            "event_key": "too_few_varying",
            "coefficient_calendar_year": 2025,
            "coefficient_course_id": 2,
            "coefficient_course_names": ["Other"],
            "holes_by_round": _geometry(7),
        },
        {
            "event_key": "incomplete",
            "coefficient_calendar_year": 2025,
            "coefficient_course_id": 3,
            "coefficient_course_names": ["Incomplete"],
            "holes_by_round": _geometry(18),
        },
    ]
    means = _hole_means([row["event_key"] for row in mapped])
    means[("incomplete", 4, 18)]["completed_scores"] = 19

    observations, editions = calibrator._build_coefficient_panel(mapped, means)

    assert list(editions) == ["include"]
    assert editions["include"]["varying_holes"] == 8
    assert len(observations) == 72


def test_hole_score_loader_counts_penalties_and_requires_holed_stroke(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shots.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE archive_shots (
                event_key TEXT,
                source TEXT,
                round_no INTEGER,
                hole_no INTEGER,
                source_player_id TEXT,
                source_team_id TEXT,
                stroke_no INTEGER,
                event_type TEXT,
                ball_holed INTEGER
            )
            """
        )
        rows = [
            ("event", "pga_tourcast", 1, 1, "complete", None, 1, "STROKE", 0),
            ("event", "pga_tourcast", 1, 1, "complete", None, 2, "STROKE", 0),
            ("event", "pga_tourcast", 1, 1, "complete", None, None, "PENALTY", 0),
            ("event", "pga_tourcast", 1, 1, "complete", None, 3, "STROKE", 1),
            ("event", "pga_tourcast", 1, 1, "unfinished", None, 1, "STROKE", 0),
        ]
        connection.executemany(
            "INSERT INTO archive_shots VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )

    result = calibrator._load_completed_hole_means(database, ["event"])

    # The completed player made three strokes plus a penalty.  The unfinished
    # player is not allowed into the field mean.
    assert result[("event", 1, 1)] == {
        "mean_score": 4.0,
        "completed_scores": 1,
    }


def test_event_hole_and_event_round_fe_recover_pooled_slope() -> None:
    beta = 0.03
    event_deviations = [-0.006, -0.002, 0.002, 0.006]
    observations = []
    for event_index, deviation in enumerate(event_deviations):
        event_key = f"edition-{event_index}"
        for rnd in range(1, 5):
            for hole in range(1, 19):
                yardage_per_10 = (
                    40.0
                    + 0.5 * hole
                    + 0.2 * rnd
                    + 0.4 * (((hole * rnd) % 5) - 2)
                )
                observations.append({
                    "event_key": event_key,
                    "round_no": rnd,
                    "hole_no": hole,
                    "yardage_per_10": yardage_per_10,
                    "mean_hole_score": (
                        (beta + deviation) * yardage_per_10
                        + 0.1 * event_index * hole
                        + 0.2 * event_index * rnd
                    ),
                })

    fit = calibrator._fit_fixed_effect_slope(observations)

    assert fit["beta"] == pytest.approx(beta, abs=1e-12)
    assert fit["cluster_se"] > 0.0
    assert fit["n_events"] == 4
    assert fit["n_hole_rounds"] == 288


def test_frozen_v3_coefficient_model_is_reproducible_and_eb_is_fixed_center() -> None:
    artifact_path = (
        Path(__file__).parent
        / "permanent_data"
        / "live_scoring_setup_yardage_shadow.json"
    )
    calibration = json.loads(artifact_path.read_text(encoding="utf-8"))
    model = calibration["yardage_coefficient_model"]

    assert calibration["schema_version"] == "live-scoring-setup-yardage/v3"
    assert calibration["selected_adjustment_mode"] == "broadie"
    assert calibration["max_abs_adjustment"] == 0.5
    assert model["schema_version"] == "course-yardage-coefficient/v1"
    assert model["units"] == "strokes_per_10_yards"
    assert model["max_abs_adjustment"] == 0.35
    assert model["training_cutoff_calendar_year"] == 2025
    assert model["eligibility"] == {
        "min_informative_years": 3,
        "min_varying_holes_per_edition": 8,
    }
    assert model["training_event_keys"] == sorted(model["training_event_keys"])
    canonical = json.dumps(
        model["training_event_keys"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert model["training_event_keys_sha256"] == hashlib.sha256(
        canonical
    ).hexdigest()
    assert model["global"]["n_events"] == 192
    assert model["global"]["n_hole_rounds"] == 13_824
    assert model["training_event_keys_sha256"] == (
        "e255e3c42f409bcd3fbc505d50b0f3664e4bedf116c0dcd47e07a845834ba291"
    )
    assert model["global"]["beta"] == pytest.approx(0.028169308642838602)
    assert len(model["courses"]) == 28

    heterogeneity = model["heterogeneity"]
    assert heterogeneity["method"] == "fixed_center_marginal_ml"
    assert heterogeneity["mu"] == model["global"]["beta"]
    assert heterogeneity["tau"] ** 2 == pytest.approx(
        heterogeneity["tau2"], rel=1e-12
    )
    for course in model["courses"].values():
        assert course["n_informative_years"] >= 3
        assert course["informative_years"] == sorted(course["informative_years"])
        assert max(course["informative_years"]) <= 2025
        assert course["event_keys"] == sorted(course["event_keys"])
        assert set(course["event_keys"]) <= set(model["training_event_keys"])
        expected_reliability = heterogeneity["tau2"] / (
            heterogeneity["tau2"] + course["cluster_se"] ** 2
        )
        assert course["reliability"] == pytest.approx(expected_reliability)
        assert course["shrunk_beta"] == pytest.approx(
            expected_reliability * course["raw_beta"]
            + (1.0 - expected_reliability) * model["global"]["beta"]
        )

    references = calibration["yardage_delta_references"]
    assert "raw yards" in references["definition"]
    assert references["round_reference_mean"] == {
        "2": -17.591836734693878,
        "3": -25.616326530612245,
        "4": 2.795918367346946,
    }
    assert references["round_reference_mode"] == {
        "2": "global",
        "3": "course_eb",
        "4": "course_eb",
    }
    assert references["round_course_eb_pseudocount"] == {
        "2": None,
        "3": 5.0,
        "4": 8.0,
    }
    assert references["course_references"]["688"]["rounds"]["4"][
        "mean_delta"
    ] == -14.4
    assert references["course_references"] != calibration["course_references"]
    assert calibration["round_reference_mean"] == {
        "2": -0.0543551020408156,
        "3": -0.06686122448979614,
        "4": 0.021309523809523685,
    }

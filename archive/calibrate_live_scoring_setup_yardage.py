"""Rebuild the frozen daily-setup scoring-shadow calibration.

This is an analysis utility, not part of the weekly pipeline.  It reads the
collector's archived PGA hole geometry and DataGolf's historical round store,
maps each geometry edition to an exact DataGolf physical ``course_num``, and
prints either an audit summary or the deterministic calibration JSON.

Example::

    python archive/calibrate_live_scoring_setup_yardage.py \
      --geometry-db C:/path/to/imgarena_shots_lean.sqlite3 \
      --datagolf-db C:/path/to/dg_historical.db --emit-json

The script never writes either source database or the repository artifact.
Redirect ``--emit-json`` to a scratch file and review the diff before freezing
an updated artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scipy.optimize import minimize_scalar


SOURCE_SEASONS = range(2020, 2026)
PSEUDOCOUNT_GRID = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 100.0)
REQUIRED_HOLES = 18
REQUIRED_ROUNDS = (1, 2, 3, 4)
MIN_INFORMATIVE_YEARS = 3
MIN_VARYING_HOLES_PER_EDITION = 8
MIN_COMPLETED_HOLE_SCORES = 20
MAX_ROUND_YARDAGE_DELTA = 300.0
PAR_RANGES = {
    3: (70.0, 350.0),
    4: (180.0, 600.0),
    5: (350.0, 750.0),
    6: (500.0, 900.0),
}

# PGA TOUR benchmark tee expectations from Broadie (2012), Table B.1.
TEE_DISTANCES_YD = (
    100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340,
    360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
)
TEE_EXPECTED_STROKES = (
    2.92, 2.99, 2.97, 2.99, 3.05, 3.12, 3.17, 3.25, 3.45,
    3.65, 3.71, 3.79, 3.86, 3.92, 3.96, 3.99, 4.02, 4.08,
    4.17, 4.28, 4.41, 4.54, 4.65, 4.74, 4.79, 4.82,
)


def _read_only(path):
    resolved = Path(path).resolve().as_posix()
    # These archived inputs are immutable snapshots for calibration.  The URI
    # flag also prevents SQLite from attempting a journal sidecar next to a
    # read-only/OneDrive-managed source file.
    return sqlite3.connect(f"file:{resolved}?mode=ro&immutable=1", uri=True)


def _normalise_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _tee_expected_strokes(distance):
    upper = next(
        (i for i, threshold in enumerate(TEE_DISTANCES_YD) if distance <= threshold),
        len(TEE_DISTANCES_YD),
    )
    if upper == 0:
        lower, upper = 0, 1
    elif upper == len(TEE_DISTANCES_YD):
        lower, upper = upper - 2, upper - 1
    else:
        lower = upper - 1
    x0, x1 = TEE_DISTANCES_YD[lower], TEE_DISTANCES_YD[upper]
    y0, y1 = TEE_EXPECTED_STROKES[lower], TEE_EXPECTED_STROKES[upper]
    return y0 + (distance - x0) * (y1 - y0) / (x1 - x0)


def _date_range(metadata):
    """Return scheduled start/end month-days from the PGA display date."""
    display = str(metadata.get("displayDate") or "").strip()
    match = re.fullmatch(
        r"([A-Za-z]{3})\s+(\d{1,2})\s*-\s*(?:([A-Za-z]{3})\s+)?(\d{1,2})",
        display,
    )
    if not match:
        return None
    first_month, first_day, second_month, second_day = match.groups()
    second_month = second_month or first_month
    try:
        start = datetime.strptime(f"{first_month} {first_day}", "%b %d")
        end = datetime.strptime(f"{second_month} {second_day}", "%b %d")
    except ValueError:
        return None
    return (start.month, start.day), (end.month, end.day)


def _event_number(event_key, source_event_id):
    token = str(source_event_id or event_key.split(":", 1)[-1])
    match = re.fullmatch(r"R\d{4}(\d+)", token)
    return int(match.group(1)) if match else None


def _load_geometry_panel(path):
    """Return audited four-round PGA editions plus rejection counts."""
    with _read_only(path) as connection:
        events = connection.execute(
            """
            SELECT event_key, season, event_name, metadata_json
            FROM archive_events
            WHERE tour = 'pga' AND season BETWEEN 2020 AND 2025
            ORDER BY event_key
            """
        ).fetchall()
        rows = connection.execute(
            """
            SELECT event_key, source_event_id, course_id, round_no, hole_no,
                   par, yardage
            FROM archive_hole_geometry
            WHERE source = 'pga_tourcast' AND event_key IN (
                SELECT event_key FROM archive_events
                WHERE tour = 'pga' AND season BETWEEN 2020 AND 2025
            )
            ORDER BY event_key, round_no, hole_no
            """
        ).fetchall()

    geometry = defaultdict(list)
    source_ids = {}
    for event_key, source_id, layout_id, rnd, hole, par, yardage in rows:
        source_ids[event_key] = source_id
        geometry[event_key].append((layout_id, rnd, hole, par, yardage))

    rejected = defaultdict(int)
    rejection_events = defaultdict(list)
    rejection_details = {}
    panel = []
    for event_key, season, event_name, metadata_json in events:
        event_rows = geometry.get(event_key, ())
        by_round = {rnd: {} for rnd in range(1, 5)}
        layouts = {rnd: set() for rnd in range(1, 5)}
        bad_identity = False
        for layout_id, rnd, hole, par, yardage in event_rows:
            try:
                rnd, hole, par = int(rnd), int(hole), int(par)
                yardage = float(yardage)
            except (TypeError, ValueError):
                bad_identity = True
                break
            if rnd not in by_round:
                continue
            if hole in by_round[rnd]:
                bad_identity = True
                break
            by_round[rnd][hole] = (par, yardage)
            if layout_id:
                layouts[rnd].add(str(layout_id))
        if bad_identity:
            rejected["invalid_or_duplicate_geometry"] += 1
            rejection_events["invalid_or_duplicate_geometry"].append(event_key)
            continue
        if any(set(holes) != set(range(1, REQUIRED_HOLES + 1)) for holes in by_round.values()):
            rejected["not_four_complete_18_hole_rounds"] += 1
            rejection_events["not_four_complete_18_hole_rounds"].append(event_key)
            continue
        if any(len(layouts[rnd]) > 1 for rnd in layouts):
            rejected["multiple_layouts_within_round"] += 1
            rejection_events["multiple_layouts_within_round"].append(event_key)
            continue
        if any(
            len({by_round[rnd][hole][0] for rnd in by_round}) != 1
            for hole in range(1, REQUIRED_HOLES + 1)
        ):
            rejected["par_changed_across_rounds"] += 1
            rejection_events["par_changed_across_rounds"].append(event_key)
            continue
        yardage_violations = [
            {
                "round_no": rnd,
                "hole_no": hole,
                "par": par,
                "yardage": yardage,
                "allowed_range": list(PAR_RANGES.get(par, ())),
            }
            for rnd, holes in by_round.items()
            for hole, (par, yardage) in holes.items()
            if par not in PAR_RANGES
            or not PAR_RANGES[par][0] <= yardage <= PAR_RANGES[par][1]
        ]
        if yardage_violations:
            rejected["yardage_outside_par_guard"] += 1
            rejection_events["yardage_outside_par_guard"].append(event_key)
            rejection_details[event_key] = {
                "reason": "yardage_outside_runtime_par_guard",
                "violations": yardage_violations,
            }
            continue

        indices = {
            rnd: sum(_tee_expected_strokes(yardage) for _, yardage in holes.values())
            for rnd, holes in by_round.items()
        }
        yardages = {
            rnd: sum(yardage for _, yardage in holes.values())
            for rnd, holes in by_round.items()
        }
        deltas = {}
        yardage_deltas = {}
        outside_delta_guard = False
        for target in (2, 3, 4):
            prior = range(1, target)
            yardage_delta = yardages[target] - sum(yardages[rnd] for rnd in prior) / (target - 1)
            if abs(yardage_delta) > MAX_ROUND_YARDAGE_DELTA:
                outside_delta_guard = True
                break
            yardage_deltas[target] = yardage_delta
            deltas[target] = indices[target] - sum(indices[rnd] for rnd in prior) / (target - 1)
        if outside_delta_guard:
            rejected["round_yardage_delta_guard"] += 1
            rejection_events["round_yardage_delta_guard"].append(event_key)
            continue

        metadata = json.loads(metadata_json or "{}")
        date_range = _date_range(metadata)
        panel.append({
            "event_key": event_key,
            "source_season": int(season),
            "source_calendar_year": (
                int(metadata["year"])
                if str(metadata.get("year") or "").isdigit()
                else None
            ),
            "source_event_id": source_ids.get(event_key),
            "event_number": _event_number(event_key, source_ids.get(event_key)),
            "event_name": event_name,
            "date_range": date_range,
            "deltas": deltas,
            "yardage_deltas": yardage_deltas,
            # Kept internal to the rebuild process.  Only aggregate references
            # and coefficient provenance are serialized in the frozen file.
            "holes_by_round": by_round,
        })
    return (
        panel,
        dict(sorted(rejected.items())),
        {key: sorted(values) for key, values in sorted(rejection_events.items())},
        dict(sorted(rejection_details.items())),
    )


def _load_datagolf_editions(path):
    with _read_only(path) as connection:
        rows = connection.execute(
            """
            SELECT year, event_id, event_name, course_num, course_name,
                   MIN(round_date), MAX(round_date)
            FROM player_rounds
            WHERE lower(tour) = 'pga' AND year BETWEEN 2019 AND 2025
                  AND course_num IS NOT NULL
            GROUP BY year, event_id, event_name, course_num, course_name
            """
        ).fetchall()
    editions = []
    for year, event_id, event_name, course_num, course_name, start, end in rows:
        start_dt = datetime.fromisoformat(str(start))
        end_dt = datetime.fromisoformat(str(end))
        editions.append({
            "year": int(year),
            "event_id": int(event_id),
            "event_name": event_name,
            "normalised_name": _normalise_name(event_name),
            "course_id": int(course_num),
            "course_name": course_name,
            "date_range": ((start_dt.month, start_dt.day), (end_dt.month, end_dt.day)),
        })
    return editions


def _map_courses(panel, dg_editions):
    """Map with event identity and dates; never alias by course/layout name."""
    mapped = []
    unmapped = []
    methods = defaultdict(int)
    for row in panel:
        source_year = row["source_season"]
        plausible_years = {source_year}
        if source_year <= 2023:
            plausible_years.add(source_year - 1)
        candidates = [
            item for item in dg_editions
            if item["year"] in plausible_years
            and item["event_id"] == row["event_number"]
        ]
        method = "event_id"
        if row["date_range"]:
            dated = [item for item in candidates if item["date_range"] == row["date_range"]]
            if dated:
                candidates = dated
                method = "event_id_exact_date"
            else:
                start_matched = [
                    item for item in candidates
                    if item["date_range"][0] == row["date_range"][0]
                ]
                if start_matched:
                    candidates = start_matched
                    method = "event_id_start_date"

        course_ids = {item["course_id"] for item in candidates}
        if len(course_ids) != 1:
            name_candidates = [
                item for item in dg_editions
                if item["year"] in plausible_years
                and item["normalised_name"] == _normalise_name(row["event_name"])
            ]
            if row["date_range"]:
                dated = [item for item in name_candidates if item["date_range"] == row["date_range"]]
                if dated:
                    name_candidates = dated
                    method = "event_name_exact_date"
                else:
                    start_matched = [
                        item for item in name_candidates
                        if item["date_range"][0] == row["date_range"][0]
                    ]
                    if start_matched:
                        name_candidates = start_matched
                        method = "event_name_start_date"
            candidates = name_candidates
            course_ids = {item["course_id"] for item in candidates}

        years = {item["year"] for item in candidates}
        if len(course_ids) != 1 or len(years) != 1:
            unmapped.append({
                "event_key": row["event_key"],
                "event_name": row["event_name"],
                "candidate_course_ids": sorted(course_ids),
                "candidate_years": sorted(years),
            })
            continue
        course_id = next(iter(course_ids))
        year = next(iter(years))
        names = sorted({
            str(item["course_name"]) for item in candidates
            if item["course_id"] == course_id and item["year"] == year
        })
        output = dict(row)
        output.update({
            "datagolf_calendar_year": year,
            "course_id": course_id,
            "course_names": names,
            "mapping_method": method,
        })
        mapped.append(output)
        methods[method] += 1
    return mapped, unmapped, dict(sorted(methods.items()))


def _map_exact_datagolf_courses(panel, dg_editions):
    """Map coefficient editions by an exact calendar-year/event-ID key.

    Course names, dates, source seasons, and TOURCAST layout IDs are never
    aliases here.  A DataGolf edition is usable only when every row under the
    exact ``(year, event_id)`` key has one stable physical ``course_num``.
    This deliberately excludes multi-course events and missing exact keys.
    """
    by_edition = defaultdict(list)
    for edition in dg_editions:
        by_edition[(edition["year"], edition["event_id"])].append(edition)

    mapped = []
    rejected = []
    for row in panel:
        year = row.get("source_calendar_year")
        event_id = row.get("event_number")
        candidates = by_edition.get((year, event_id), ())
        course_ids = {item["course_id"] for item in candidates}
        if year is None or year > 2025 or len(course_ids) != 1:
            rejected.append({
                "event_key": row["event_key"],
                "calendar_year": year,
                "event_id": event_id,
                "candidate_course_nums": sorted(course_ids),
            })
            continue
        course_id = next(iter(course_ids))
        output = dict(row)
        output.update({
            "coefficient_calendar_year": int(year),
            "coefficient_course_id": int(course_id),
            "coefficient_course_names": sorted({
                str(item["course_name"])
                for item in candidates
                if item["course_id"] == course_id and item["course_name"]
            }),
            # Conventional aliases let the leakage-safe reference CV reuse the
            # same strict-prior implementation as the frozen Broadie audit.
            "datagolf_calendar_year": int(year),
            "course_id": int(course_id),
            "course_names": sorted({
                str(item["course_name"])
                for item in candidates
                if item["course_id"] == course_id and item["course_name"]
            }),
            "mapping_method": "exact_year_event_id",
        })
        mapped.append(output)
    return mapped, rejected


def _load_completed_hole_means(path, event_keys):
    """Return field scoring means for completed player-hole observations."""
    event_keys = sorted(set(event_keys))
    if not event_keys:
        return {}
    placeholders = ",".join("?" for _ in event_keys)
    with _read_only(path) as connection:
        rows = connection.execute(
            f"""
            WITH completed_player_holes AS (
                SELECT event_key, round_no, hole_no, source_player_id,
                       COALESCE(source_team_id, '') AS source_team_id,
                       SUM(
                           CASE WHEN event_type IN ('STROKE', 'PENALTY')
                                THEN 1 ELSE 0 END
                       ) AS hole_score,
                       MAX(
                           CASE WHEN event_type = 'STROKE'
                                     AND COALESCE(ball_holed, 0) = 1
                                THEN 1 ELSE 0 END
                       ) AS complete
                FROM archive_shots
                WHERE source = 'pga_tourcast'
                      AND event_key IN ({placeholders})
                      AND round_no BETWEEN 1 AND 4
                      AND hole_no BETWEEN 1 AND 18
                      AND (
                          NULLIF(source_player_id, '') IS NOT NULL
                          OR NULLIF(source_team_id, '') IS NOT NULL
                      )
                GROUP BY event_key, round_no, hole_no, source_player_id,
                         COALESCE(source_team_id, '')
            ), valid_player_holes AS (
                SELECT *
                FROM completed_player_holes
                WHERE complete = 1 AND hole_score BETWEEN 1 AND 15
            )
            SELECT event_key, round_no, hole_no, AVG(hole_score), COUNT(*)
            FROM valid_player_holes
            GROUP BY event_key, round_no, hole_no
            ORDER BY event_key, round_no, hole_no
            """,
            event_keys,
        ).fetchall()
    return {
        (event_key, int(rnd), int(hole)): {
            "mean_score": float(mean_score),
            "completed_scores": int(completed_scores),
        }
        for event_key, rnd, hole, mean_score, completed_scores in rows
    }


def _build_coefficient_panel(mapped, hole_means):
    """Build the balanced, informative event-round-hole coefficient panel."""
    observations = []
    editions = {}
    for edition in sorted(mapped, key=lambda item: item["event_key"]):
        event_key = edition["event_key"]
        geometry = edition["holes_by_round"]
        complete_keys = [
            (event_key, rnd, hole)
            for rnd in REQUIRED_ROUNDS
            for hole in range(1, REQUIRED_HOLES + 1)
        ]
        if any(
            key not in hole_means
            or hole_means[key]["completed_scores"] < MIN_COMPLETED_HOLE_SCORES
            for key in complete_keys
        ):
            continue
        varying_holes = sum(
            len({geometry[rnd][hole][1] for rnd in REQUIRED_ROUNDS}) > 1
            for hole in range(1, REQUIRED_HOLES + 1)
        )
        if varying_holes < MIN_VARYING_HOLES_PER_EDITION:
            continue

        editions[event_key] = {
            "calendar_year": edition["coefficient_calendar_year"],
            "course_id": edition["coefficient_course_id"],
            "course_names": edition["coefficient_course_names"],
            "varying_holes": varying_holes,
        }
        observations.extend(
            {
                "event_key": event_key,
                "round_no": rnd,
                "hole_no": hole,
                "yardage_per_10": geometry[rnd][hole][1] / 10.0,
                "mean_hole_score": hole_means[(event_key, rnd, hole)][
                    "mean_score"
                ],
            }
            for rnd in REQUIRED_ROUNDS
            for hole in range(1, REQUIRED_HOLES + 1)
        )
    return observations, editions


def _absorb_event_hole_and_round_effects(observations):
    """Two-way demean a balanced 4x18 panel separately within each edition."""
    by_event = defaultdict(list)
    for row in observations:
        by_event[row["event_key"]].append(row)

    absorbed = []
    expected_cells = {
        (rnd, hole)
        for rnd in REQUIRED_ROUNDS
        for hole in range(1, REQUIRED_HOLES + 1)
    }
    for event_key in sorted(by_event):
        rows = by_event[event_key]
        cells = {(row["round_no"], row["hole_no"]) for row in rows}
        if len(rows) != len(expected_cells) or cells != expected_cells:
            raise ValueError(
                f"{event_key} coefficient panel must contain one complete "
                "4-round x 18-hole grid"
            )
        x_grand = sum(row["yardage_per_10"] for row in rows) / len(rows)
        y_grand = sum(row["mean_hole_score"] for row in rows) / len(rows)
        x_by_hole = {
            hole: sum(
                row["yardage_per_10"]
                for row in rows
                if row["hole_no"] == hole
            ) / len(REQUIRED_ROUNDS)
            for hole in range(1, REQUIRED_HOLES + 1)
        }
        y_by_hole = {
            hole: sum(
                row["mean_hole_score"]
                for row in rows
                if row["hole_no"] == hole
            ) / len(REQUIRED_ROUNDS)
            for hole in range(1, REQUIRED_HOLES + 1)
        }
        x_by_round = {
            rnd: sum(
                row["yardage_per_10"]
                for row in rows
                if row["round_no"] == rnd
            ) / REQUIRED_HOLES
            for rnd in REQUIRED_ROUNDS
        }
        y_by_round = {
            rnd: sum(
                row["mean_hole_score"]
                for row in rows
                if row["round_no"] == rnd
            ) / REQUIRED_HOLES
            for rnd in REQUIRED_ROUNDS
        }
        absorbed.extend({
            "event_key": event_key,
            "x": (
                row["yardage_per_10"]
                - x_by_hole[row["hole_no"]]
                - x_by_round[row["round_no"]]
                + x_grand
            ),
            "y": (
                row["mean_hole_score"]
                - y_by_hole[row["hole_no"]]
                - y_by_round[row["round_no"]]
                + y_grand
            ),
        } for row in rows)
    return absorbed


def _fit_fixed_effect_slope(observations):
    """Fit the absorbed slope and a CR1 edition-clustered standard error."""
    absorbed = _absorb_event_hole_and_round_effects(observations)
    event_keys = sorted({row["event_key"] for row in absorbed})
    if len(event_keys) < 2:
        raise ValueError("clustered coefficient fit requires at least two editions")
    sxx = sum(row["x"] * row["x"] for row in absorbed)
    if not math.isfinite(sxx) or sxx <= 0.0:
        raise ValueError("coefficient panel has no identifying yardage variation")
    beta = sum(row["x"] * row["y"] for row in absorbed) / sxx
    cluster_scores = defaultdict(float)
    for row in absorbed:
        cluster_scores[row["event_key"]] += row["x"] * (
            row["y"] - beta * row["x"]
        )
    cluster_count = len(event_keys)
    variance = (
        cluster_count
        / (cluster_count - 1)
        * sum(value * value for value in cluster_scores.values())
        / (sxx * sxx)
    )
    return {
        "beta": beta,
        "cluster_se": math.sqrt(max(variance, 0.0)),
        "n_events": cluster_count,
        "n_hole_rounds": len(absorbed),
    }


def _fixed_center_tau2(course_fits, center):
    """Estimate normal-normal heterogeneity with its mean fixed at ``center``."""
    inputs = [
        (fit["beta"], fit["cluster_se"] ** 2)
        for fit in course_fits
        if math.isfinite(fit["beta"])
        and math.isfinite(fit["cluster_se"])
        and fit["cluster_se"] > 0.0
    ]
    if len(inputs) < 2:
        return 0.0

    def objective(tau2):
        return 0.5 * sum(
            math.log(variance + tau2)
            + (estimate - center) ** 2 / (variance + tau2)
            for estimate, variance in inputs
        )

    upper = max(
        max((estimate - center) ** 2 for estimate, _ in inputs),
        max(variance for _, variance in inputs),
        1e-8,
    ) * 100.0
    result = minimize_scalar(
        objective,
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-15},
    )
    candidates = (0.0, float(result.x), upper)
    return min(candidates, key=lambda value: (objective(value), value))


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_yardage_coefficient_model(geometry_db, mapped):
    event_keys = [row["event_key"] for row in mapped]
    hole_means = _load_completed_hole_means(geometry_db, event_keys)
    observations, editions = _build_coefficient_panel(mapped, hole_means)
    training_event_keys = sorted(editions)
    global_fit = _fit_fixed_effect_slope(observations)

    by_course = defaultdict(list)
    for event_key, edition in editions.items():
        by_course[edition["course_id"]].append(event_key)
    course_fits = {}
    for course_id in sorted(by_course):
        keys = sorted(by_course[course_id])
        years = sorted({editions[key]["calendar_year"] for key in keys})
        if len(years) < MIN_INFORMATIVE_YEARS:
            continue
        key_set = set(keys)
        fit = _fit_fixed_effect_slope([
            row for row in observations if row["event_key"] in key_set
        ])
        fit.update({
            "course_id": course_id,
            "course_names": sorted({
                name
                for key in keys
                for name in editions[key]["course_names"]
            }),
            "informative_years": years,
            "event_keys": keys,
        })
        course_fits[course_id] = fit

    center = global_fit["beta"]
    tau2 = _fixed_center_tau2(list(course_fits.values()), center)
    courses = {}
    for course_id, fit in sorted(course_fits.items()):
        sampling_variance = fit["cluster_se"] ** 2
        reliability = (
            tau2 / (tau2 + sampling_variance)
            if tau2 + sampling_variance > 0.0
            else 0.0
        )
        courses[str(course_id)] = {
            "course_names": fit["course_names"],
            "informative_years": fit["informative_years"],
            "n_informative_years": len(fit["informative_years"]),
            "n_events": fit["n_events"],
            "event_keys": fit["event_keys"],
            "raw_beta": fit["beta"],
            "cluster_se": fit["cluster_se"],
            "reliability": reliability,
            "shrunk_beta": (
                reliability * fit["beta"]
                + (1.0 - reliability) * center
            ),
        }

    return {
        "schema_version": "course-yardage-coefficient/v1",
        "model_version": "course-yardage-coefficient-v1-2026-08-31",
        "units": "strokes_per_10_yards",
        "max_abs_adjustment": 0.35,
        "design": {
            "observation": (
                "Event-round-hole field mean completed-hole strokes; valid "
                "hole scores are 1-15 and each cell requires at least 20."
            ),
            "formula": (
                "mean_hole_strokes ~ yardage_per_10 + event_key:hole_no FE "
                "+ event_key:round_no FE"
            ),
            "fixed_effects": [
                "event_key_x_hole_no",
                "event_key_x_round_no",
            ],
            "within_transform": (
                "Balanced 4-round x 18-hole two-way demeaning within edition."
            ),
            "standard_errors": (
                "CR1 cluster-robust by event_key after FE absorption, using "
                "the G/(G-1) finite-cluster correction."
            ),
            "course_identity": (
                "Exact archive metadata calendar year plus PGA event number "
                "join to DataGolf (year,event_id), requiring one stable "
                "physical course_num; no name, date, or layout aliases."
            ),
            "informative_edition": (
                "Complete 72-cell score/setup panel with at least 8 holes "
                "whose yardage varies across rounds."
            ),
            "heterogeneity": (
                "Normal-normal marginal ML with mu fixed at the all-edition "
                "global beta; reliability=tau2/(tau2+cluster_se^2)."
            ),
            "shrinkage": (
                "shrunk_beta = reliability*raw_beta + "
                "(1-reliability)*global.beta"
            ),
        },
        "training_cutoff_calendar_year": 2025,
        "training_event_keys": training_event_keys,
        "training_event_keys_sha256": _canonical_sha256(training_event_keys),
        "eligibility": {
            "min_informative_years": MIN_INFORMATIVE_YEARS,
            "min_varying_holes_per_edition": MIN_VARYING_HOLES_PER_EDITION,
        },
        "global": global_fit,
        "heterogeneity": {
            "method": "fixed_center_marginal_ml",
            "mu": center,
            "tau2": tau2,
            "tau": math.sqrt(tau2),
        },
        "courses": courses,
        "fallback": "global",
    }


def _strict_prior_cv(panel, mapped, *, value_key="deltas"):
    results = {}
    for target in (2, 3, 4):
        squared_errors = {pseudo: [] for pseudo in PSEUDOCOUNT_GRID}
        global_errors = []
        for holdout in mapped:
            year = holdout["datagolf_calendar_year"]
            global_prior = [
                row[value_key][target]
                for row in panel
                if row["calendar_year"] < year
            ]
            course_prior = [
                row[value_key][target]
                for row in mapped
                if row["datagolf_calendar_year"] < year
                and row["course_id"] == holdout["course_id"]
            ]
            if not global_prior or not course_prior:
                continue
            global_mean = sum(global_prior) / len(global_prior)
            course_mean = sum(course_prior) / len(course_prior)
            actual = holdout[value_key][target]
            global_errors.append((actual - global_mean) ** 2)
            for pseudo in PSEUDOCOUNT_GRID:
                reference = (
                    len(course_prior) * course_mean + pseudo * global_mean
                ) / (len(course_prior) + pseudo)
                squared_errors[pseudo].append((actual - reference) ** 2)

        rmse = {
            str(int(pseudo)): math.sqrt(sum(errors) / len(errors))
            for pseudo, errors in squared_errors.items()
        }
        selected = min(PSEUDOCOUNT_GRID, key=lambda pseudo: (rmse[str(int(pseudo))], pseudo))
        global_rmse = math.sqrt(sum(global_errors) / len(global_errors))
        reference_mode = (
            "global"
            if global_rmse <= rmse[str(int(selected))]
            else "course_eb"
        )
        results[str(target)] = {
            "selected_reference_mode": reference_mode,
            "selected_pseudocount": (
                selected if reference_mode == "course_eb" else None
            ),
            "best_finite_pseudocount": selected,
            "eligible_holdouts": len(global_errors),
            "rmse_by_pseudocount": rmse,
            "global_only_rmse": global_rmse,
        }
    return results


def _ols_audit(pairs, *, fitted_parameters=2):
    """Return a compact unweighted OLS slope audit with an intercept."""
    n = len(pairs)
    x_mean = sum(x for x, _ in pairs) / n
    y_mean = sum(y for _, y in pairs) / n
    sxx = sum((x - x_mean) ** 2 for x, _ in pairs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / sxx
    intercept = y_mean - slope * x_mean
    residuals = [y - intercept - slope * x for x, y in pairs]
    residual_variance = sum(value * value for value in residuals) / (
        n - fitted_parameters
    )
    standard_error = math.sqrt(residual_variance / sxx)
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "slope_standard_error": standard_error,
        "slope_95pct_normal_approx": [
            slope - 1.96 * standard_error,
            slope + 1.96 * standard_error,
        ],
        "t_statistic": slope / standard_error,
    }


def _score_response_audit(path, mapped):
    """Audit setup-index deltas against paired active-cohort score changes."""
    with _read_only(path) as connection:
        rows = connection.execute(
            """
            SELECT year, event_id, course_num, dg_id, round_num, score
            FROM player_rounds
            WHERE lower(tour) = 'pga' AND year BETWEEN 2019 AND 2025
                  AND course_num IS NOT NULL AND score IS NOT NULL
                  AND round_num BETWEEN 1 AND 4
            """
        ).fetchall()
    scores = defaultdict(lambda: defaultdict(dict))
    for year, event_id, course_id, player_id, rnd, score in rows:
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and 50.0 <= score <= 110.0:
            scores[(int(year), int(event_id), int(course_id))][int(player_id)][
                int(rnd)
            ] = score

    pairs_by_round = {target: [] for target in (2, 3, 4)}
    for edition in mapped:
        edition_scores = scores.get((
            edition["datagolf_calendar_year"],
            edition["event_number"],
            edition["course_id"],
        ), {})
        for target in (2, 3, 4):
            eligible = [
                rounds for rounds in edition_scores.values()
                if all(rnd in rounds for rnd in range(1, target + 1))
            ]
            if not eligible:
                continue
            target_mean = sum(rounds[target] for rounds in eligible) / len(eligible)
            prior_mean = sum(
                sum(rounds[rnd] for rounds in eligible) / len(eligible)
                for rnd in range(1, target)
            ) / (target - 1)
            pairs_by_round[target].append(
                (edition["deltas"][target], target_mean - prior_mean)
            )

    per_round = {
        str(target): _ols_audit(pairs) for target, pairs in pairs_by_round.items()
    }
    # Demeaning within transition is equivalent to fitting R2/R3/R4 fixed
    # effects and isolates the common setup slope from normal round drift.
    pooled = []
    for pairs in pairs_by_round.values():
        x_mean = sum(x for x, _ in pairs) / len(pairs)
        y_mean = sum(y for _, y in pairs) / len(pairs)
        pooled.extend((x - x_mean, y - y_mean) for x, y in pairs)
    pooled_audit = _ols_audit(pooled, fitted_parameters=4)
    pooled_audit["intercept"] = 0.0
    return {
        "outcome_definition": (
            "For each target round, absolute field score mean minus the mean "
            "of strictly prior round score means, using only golfers with "
            "scores in every round through the target (paired active cohort)."
        ),
        "regression_definition": (
            "Unweighted edition-level OLS; pooled result demeans setup and "
            "score deltas within R2/R3/R4 (transition fixed effects)."
        ),
        "per_round": per_round,
        "pooled_round_fixed_effects": pooled_audit,
    }


def build_calibration(geometry_db, datagolf_db):
    panel, rejected, rejection_events, rejection_details = _load_geometry_panel(
        geometry_db
    )
    dg_editions = _load_datagolf_editions(datagolf_db)
    mapped, unmapped, methods = _map_courses(panel, dg_editions)
    coefficient_mapped, _coefficient_mapping_rejections = (
        _map_exact_datagolf_courses(panel, dg_editions)
    )

    # Give every geometry row a leakage-safe calendar year for global CV.  An
    # exact DG mapping is preferred; the source season/date convention is used
    # only for the two rows that cannot be assigned a physical course.
    mapped_years = {row["event_key"]: row["datagolf_calendar_year"] for row in mapped}
    for row in panel:
        if row["event_key"] in mapped_years:
            row["calendar_year"] = mapped_years[row["event_key"]]
        else:
            start_month = row["date_range"][0][0] if row["date_range"] else None
            row["calendar_year"] = (
                row["source_season"] - 1
                if row["source_season"] <= 2023 and start_month in (9, 10, 11, 12)
                else row["source_season"]
            )

    global_references = {
        str(target): sum(row["deltas"][target] for row in panel) / len(panel)
        for target in (2, 3, 4)
    }
    cv = _strict_prior_cv(panel, mapped)
    score_response = _score_response_audit(datagolf_db, mapped)
    yardage_coefficient_model = _build_yardage_coefficient_model(
        geometry_db,
        coefficient_mapped,
    )

    by_course = defaultdict(list)
    for row in mapped:
        by_course[row["course_id"]].append(row)
    course_references = {}
    for course_id in sorted(by_course):
        rows = by_course[course_id]
        entry = {
            "n": len(rows),
            "course_names": sorted({name for row in rows for name in row["course_names"]}),
            "rounds": {},
        }
        for target in (2, 3, 4):
            values = [row["deltas"][target] for row in rows]
            total = sum(values)
            mode = cv[str(target)]["selected_reference_mode"]
            pseudo = cv[str(target)]["selected_pseudocount"]
            global_mean = global_references[str(target)]
            entry["rounds"][str(target)] = {
                "sum_delta": total,
                "mean_delta": total / len(values),
                "eb_reference": (
                    (total + pseudo * global_mean) / (len(values) + pseudo)
                    if mode == "course_eb"
                    else None
                ),
            }
        course_references[str(course_id)] = entry

    yardage_panel = [
        dict(row, calendar_year=row["source_calendar_year"])
        for row in panel
        if row["source_calendar_year"] is not None
        and row["source_calendar_year"] <= 2025
    ]
    yardage_global_references = {
        str(target): sum(row["yardage_deltas"][target] for row in yardage_panel)
        / len(yardage_panel)
        for target in (2, 3, 4)
    }
    yardage_cv = _strict_prior_cv(
        yardage_panel,
        coefficient_mapped,
        value_key="yardage_deltas",
    )
    yardage_by_course = defaultdict(list)
    for row in coefficient_mapped:
        yardage_by_course[row["course_id"]].append(row)
    yardage_course_references = {}
    for course_id in sorted(yardage_by_course):
        rows = yardage_by_course[course_id]
        entry = {
            "n": len(rows),
            "course_names": sorted({
                name for row in rows for name in row["course_names"]
            }),
            "rounds": {},
        }
        for target in (2, 3, 4):
            values = [row["yardage_deltas"][target] for row in rows]
            total = sum(values)
            mode = yardage_cv[str(target)]["selected_reference_mode"]
            pseudo = yardage_cv[str(target)]["selected_pseudocount"]
            global_mean = yardage_global_references[str(target)]
            entry["rounds"][str(target)] = {
                "sum_delta": total,
                "mean_delta": total / len(values),
                "eb_reference": (
                    (total + pseudo * global_mean) / (len(values) + pseudo)
                    if mode == "course_eb"
                    else None
                ),
            }
        yardage_course_references[str(course_id)] = entry

    yardage_delta_references = {
        "definition": (
            "Target-round total setup yardage minus the mean total setup "
            "yardage of all strictly prior rounds, in raw yards."
        ),
        "round_reference_mean": yardage_global_references,
        "round_reference_mode": {
            target: values["selected_reference_mode"]
            for target, values in yardage_cv.items()
        },
        "round_course_eb_pseudocount": {
            target: values["selected_pseudocount"]
            for target, values in yardage_cv.items()
        },
        "course_references": yardage_course_references,
    }

    return {
        "calibration_version": (
            "setup-yardage-shadow-v3-course-yardage-coefficient-2026-08-31"
        ),
        "schema_version": "live-scoring-setup-yardage/v3",
        "status": "shadow_only",
        "frozen": True,
        "method": "broadie_2012_tee_expected_strokes_round_setup_delta",
        "selected_adjustment_mode": "broadie",
        "response_weight": 1.0,
        "response_weight_provenance": {
            "selection": (
                "Prospective shadow-only physical tee-state pass-through, not "
                "a fitted historical score-outcome coefficient."
            ),
            "historical_score_outcome_audit": score_response,
        },
        "max_abs_adjustment": 0.5,
        "max_abs_round_yardage_delta": MAX_ROUND_YARDAGE_DELTA,
        "max_abs_actual_official_hole_delta": 100.0,
        "required_holes": REQUIRED_HOLES,
        "round_reference_mean": global_references,
        "round_reference_mode": {
            target: values["selected_reference_mode"]
            for target, values in cv.items()
        },
        "round_course_eb_pseudocount": {
            target: values["selected_pseudocount"] for target, values in cv.items()
        },
        "training_panel": {
            "tour": "pga",
            "source_pga_season_min": min(SOURCE_SEASONS),
            "source_pga_season_max": max(SOURCE_SEASONS),
            "eligible_single_course_four_round_editions": len(panel),
            "exact_datagolf_course_mapped_editions": len(mapped),
            "unmapped_course_editions": unmapped,
            "geometry_rejections": rejected,
            "geometry_rejection_events": rejection_events,
            "geometry_rejection_details": rejection_details,
            "mapping_method_counts": methods,
            "mapped_edition_fields": [
                "datagolf_calendar_year",
                "datagolf_course_id",
                "mapping_method",
            ],
            "mapped_editions": {
                row["event_key"]: [
                    row["datagolf_calendar_year"],
                    row["course_id"],
                    row["mapping_method"],
                ]
                for row in sorted(mapped, key=lambda item: item["event_key"])
            },
            "mapping_definition": (
                "Within source season (and season-1 for the 2020-23 wrap era), "
                "match PGA source event ID to a unique DataGolf player_rounds "
                "edition and physical course_num, preferring exact scheduled "
                "start/end month-day and then start month-day. If event ID is "
                "ambiguous, use exact normalized event name under the same date "
                "rules. A single calendar year and course_num are required. No "
                "course-name or TOURCAST layout-ID aliases are used."
            ),
            "eligibility_definition": (
                "Exactly one geometry row per round/hole for R1-R4, 18 holes "
                "per round, one layout within each round, stable hole pars, "
                "runtime-equivalent par yardage bounds, and no target-round "
                "yardage delta beyond 300 yards versus prior-round mean."
            ),
            "reference_definition": (
                "target round tee expected-strokes index minus the mean index "
                "of all strictly prior rounds"
            ),
            "yardage_precedence": (
                "actual_yardage, then canonical yardage, then official_yardage; "
                "the audited source schema stores canonical yardage"
            ),
            "source_databases": {
                "geometry": "golf_scraping/output/imgarena_shots_lean.sqlite3: archive_events + archive_hole_geometry",
                "datagolf": "OneDrive/dg_historical.db: player_rounds",
            },
            "rebuild_script": "archive/calibrate_live_scoring_setup_yardage.py",
        },
        "course_eb_cv": {
            "method": (
                "leave-calendar-year-out strict-prior RMSE; training rows have "
                "DataGolf calendar year strictly less than the held-out edition; "
                "same physical course_num needs at least one prior edition"
            ),
            "global_prior_panel": "all eligible geometry editions strictly before holdout calendar year",
            "course_prior_panel": "exact DataGolf course_num mappings strictly before holdout calendar year",
            "pseudocount_grid": list(PSEUDOCOUNT_GRID),
            "rounds": cv,
        },
        "course_references": course_references,
        "yardage_delta_references": yardage_delta_references,
        "yardage_coefficient_model": yardage_coefficient_model,
        "guards": {
            "par_3_yardage": list(PAR_RANGES[3]),
            "par_4_yardage": list(PAR_RANGES[4]),
            "par_5_yardage": list(PAR_RANGES[5]),
            "par_6_yardage": list(PAR_RANGES[6]),
            "same_hole_par_across_rounds": True,
            "unique_event_round_hole": True,
            "single_course_only": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-db", required=True)
    parser.add_argument("--datagolf-db", required=True)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    calibration = build_calibration(args.geometry_db, args.datagolf_db)
    if args.emit_json:
        print(json.dumps(calibration, indent=2, sort_keys=False))
        return
    panel = calibration["training_panel"]
    print(
        f"eligible={panel['eligible_single_course_four_round_editions']} "
        f"mapped={panel['exact_datagolf_course_mapped_editions']} "
        f"unmapped={len(panel['unmapped_course_editions'])}"
    )
    print("mapping_methods", panel["mapping_method_counts"])
    print("unmapped", panel["unmapped_course_editions"])
    print("global_references", calibration["round_reference_mean"])
    print(
        "score_response",
        calibration["response_weight_provenance"][
            "historical_score_outcome_audit"
        ]["pooled_round_fixed_effects"],
    )
    for rnd, result in calibration["course_eb_cv"]["rounds"].items():
        print(
            f"R{rnd} mode={result['selected_reference_mode']} "
            f"k={result['selected_pseudocount']} "
            f"holdouts={result['eligible_holdouts']} "
            f"best_finite_rmse={result['rmse_by_pseudocount'][str(int(result['best_finite_pseudocount']))]:.9f} "
            f"global={result['global_only_rmse']:.9f}"
        )
    coefficient = calibration["yardage_coefficient_model"]
    print(
        "yardage_coefficient_global",
        coefficient["global"],
        f"courses={len(coefficient['courses'])}",
    )
    print("yardage_coefficient_heterogeneity", coefficient["heterogeneity"])
    print(
        "yardage_delta_references",
        calibration["yardage_delta_references"]["round_reference_mean"],
        calibration["yardage_delta_references"]["round_reference_mode"],
        calibration["yardage_delta_references"][
            "round_course_eb_pseudocount"
        ],
    )
    print("course_688", calibration["course_references"].get("688"))
    print("course_933", calibration["course_references"].get("933"))


if __name__ == "__main__":
    main()

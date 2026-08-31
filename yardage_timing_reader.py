"""Fail-closed reader for prospective PGA daily-yardage observations.

The timing probe appends immutable JSONL observations while it is polling.  A
live scoring run may use the newest complete numbered-round setup as a fallback
when the rich shot archive has not collected that round yet.  Nominal
``All Rounds`` scorecards are deliberately ignored.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA_VERSION = "pga-yardage-availability/v2"
PLAUSIBLE_YARDAGE = {
    3: (75.0, 350.0),
    4: (200.0, 650.0),
    5: (350.0, 800.0),
    6: (450.0, 900.0),
}


class YardageTimingUnavailable(ValueError):
    """No recent, internally consistent daily setup is available."""


def _parse_utc(value, label):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise YardageTimingUnavailable(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise YardageTimingUnavailable(f"{label} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _read_observations(path):
    """Read one file snapshot, tolerating only a partial final append."""
    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise YardageTimingUnavailable(f"yardage timing log is unavailable: {path}") from exc
    if not payload:
        raise YardageTimingUnavailable(f"yardage timing log is empty: {path}")

    observations = []
    lines = payload.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        final = index == len(lines) - 1
        complete_line = raw.endswith((b"\n", b"\r"))
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if final and not complete_line:
                continue
            raise YardageTimingUnavailable(
                f"yardage timing log has malformed interior row {index + 1}"
            ) from exc
        if not isinstance(row, dict):
            raise YardageTimingUnavailable(
                f"yardage timing row {index + 1} is not an object"
            )
        observations.append(row)
    if not observations:
        raise YardageTimingUnavailable("yardage timing log has no complete rows")
    return observations


def _canonical_setup_hash(holes):
    canonical = [
        {"hole": row["hole"], "par": row["par"], "yards": row["yards"]}
        for row in holes
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validated_setup(setup, *, round_no, course_id):
    if not isinstance(setup, Mapping) or setup.get("complete") is not True:
        raise YardageTimingUnavailable(f"R{round_no} setup is incomplete")
    try:
        reported_round = int(setup.get("round"))
    except (TypeError, ValueError) as exc:
        raise YardageTimingUnavailable(f"R{round_no} setup has invalid identity") from exc
    if reported_round != round_no:
        raise YardageTimingUnavailable(f"R{round_no} setup identity changed")

    holes = []
    seen = set()
    for source in setup.get("holes") or []:
        if not isinstance(source, Mapping):
            raise YardageTimingUnavailable(f"R{round_no} has malformed hole rows")
        try:
            hole = int(source.get("hole"))
            par = int(source.get("par"))
            yards_number = float(source.get("yards"))
        except (TypeError, ValueError) as exc:
            raise YardageTimingUnavailable(
                f"R{round_no} has non-numeric hole geometry"
            ) from exc
        if hole in seen or not 1 <= hole <= 18:
            raise YardageTimingUnavailable(f"R{round_no} has duplicate/invalid H{hole}")
        if (
            par not in PLAUSIBLE_YARDAGE
            or not math.isfinite(yards_number)
            or not yards_number.is_integer()
        ):
            raise YardageTimingUnavailable(f"R{round_no}H{hole} is implausible")
        yards = int(yards_number)
        lower, upper = PLAUSIBLE_YARDAGE[par]
        if not lower <= yards <= upper:
            raise YardageTimingUnavailable(
                f"R{round_no}H{hole} yardage is outside [{lower:.0f}, {upper:.0f}]"
            )
        seen.add(hole)
        holes.append({"hole": hole, "par": par, "yards": yards})
    holes.sort(key=lambda row: row["hole"])
    if seen != set(range(1, 19)):
        raise YardageTimingUnavailable(f"R{round_no} does not contain holes 1-18")

    total = sum(row["yards"] for row in holes)
    try:
        reported_total = float(setup.get("total_yards"))
    except (TypeError, ValueError) as exc:
        raise YardageTimingUnavailable(f"R{round_no} total yardage is invalid") from exc
    if not math.isclose(total, reported_total, rel_tol=0.0, abs_tol=1e-9):
        raise YardageTimingUnavailable(f"R{round_no} total yardage is inconsistent")
    expected_hash = _canonical_setup_hash(holes)
    if setup.get("setup_sha256") != expected_hash:
        raise YardageTimingUnavailable(f"R{round_no} setup hash is inconsistent")

    return [
        {
            "course_id": course_id,
            "round_no": round_no,
            "hole_no": row["hole"],
            "par": row["par"],
            "yardage": row["yards"],
            "setup_sha256": expected_hash,
        }
        for row in holes
    ]


def load_latest_round_geometry(
    path,
    *,
    event_id,
    target_round,
    expected_pga_course_id=None,
    max_age_minutes=15.0,
    now=None,
):
    """Return ``(rows, provenance)`` for the newest safe v2 observation.

    ``rows`` contains every numbered setup from R1 through ``target_round``.
    The caller still supplies the separate DataGolf physical ``course_num`` to
    the scoring model; PGA course IDs here only prevent layout mixing.
    """
    event_id = str(event_id).strip()
    if not event_id:
        raise YardageTimingUnavailable("expected PGA event_id is empty")
    try:
        target_round = int(target_round)
        max_age_minutes = float(max_age_minutes)
    except (TypeError, ValueError) as exc:
        raise YardageTimingUnavailable("target round/freshness is invalid") from exc
    if target_round not in (2, 3, 4) or max_age_minutes <= 0:
        raise YardageTimingUnavailable("target round/freshness is outside bounds")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    candidates = []
    for row in _read_observations(path):
        if row.get("schema_version") != SCHEMA_VERSION:
            continue
        if str(row.get("event_id") or "") != event_id:
            continue
        source = (row.get("sources") or {}).get("course_stats") or {}
        if source.get("error") or source.get("http_status") != 200:
            continue
        data = source.get("data") or {}
        if str(data.get("tournament_id") or "") != event_id:
            continue
        received = _parse_utc(source.get("received_at_utc"), "course_stats receipt")
        candidates.append((received, row))
    if not candidates:
        raise YardageTimingUnavailable(
            f"no successful {SCHEMA_VERSION} observation matches {event_id}"
        )
    received, observation = max(candidates, key=lambda item: item[0])
    age = now - received
    if age < timedelta(minutes=-1) or age > timedelta(minutes=max_age_minutes):
        raise YardageTimingUnavailable(
            f"latest course-stats observation is stale ({age.total_seconds() / 60:.1f} minutes)"
        )

    summary = observation.get("summary") or {}
    completeness = (summary.get("round_course_completeness") or {}).get(
        str(target_round)
    )
    if not isinstance(completeness, Mapping) or completeness.get("complete") is not True:
        raise YardageTimingUnavailable(f"R{target_round} daily setup is not complete")
    used_course_ids = [
        str(value) for value in completeness.get("used_course_ids") or []
    ]
    if len(set(used_course_ids)) != 1:
        raise YardageTimingUnavailable(
            f"R{target_round} does not resolve to one PGA course"
        )
    course_id = used_course_ids[0]
    if (
        expected_pga_course_id not in (None, "")
        and course_id != str(expected_pga_course_id)
    ):
        raise YardageTimingUnavailable(
            f"R{target_round} PGA course {course_id} does not match "
            f"{expected_pga_course_id}"
        )

    course_data = ((observation.get("sources") or {}).get("course_stats") or {}).get(
        "data"
    ) or {}
    courses = [
        item for item in course_data.get("courses") or []
        if isinstance(item, Mapping) and str(item.get("course_id") or "") == course_id
    ]
    if len(courses) != 1:
        raise YardageTimingUnavailable(f"PGA course {course_id} is missing/ambiguous")
    setups = courses[0].get("setups") or []

    rows = []
    pars_by_hole = {}
    setup_hashes = {}
    for round_no in range(1, target_round + 1):
        matches = [
            setup for setup in setups
            if isinstance(setup, Mapping) and setup.get("round") == round_no
        ]
        if len(matches) != 1:
            raise YardageTimingUnavailable(
                f"R{round_no} numbered setup is missing/ambiguous"
            )
        round_rows = _validated_setup(
            matches[0], round_no=round_no, course_id=course_id
        )
        for item in round_rows:
            previous = pars_by_hole.setdefault(item["hole_no"], item["par"])
            if previous != item["par"]:
                raise YardageTimingUnavailable(
                    f"H{item['hole_no']} par changes across rounds"
                )
        setup_hashes[str(round_no)] = round_rows[0]["setup_sha256"]
        rows.extend(round_rows)

    for item in rows:
        item["source"] = "pga_course_stats_timing_v2"
        item["observed_at_utc"] = received.isoformat()
    return rows, {
        "source": "pga_course_stats_timing_v2",
        "event_key": f"pga:{event_id}",
        "pga_course_id": course_id,
        "observed_at_utc": received.isoformat(),
        "age_minutes": age.total_seconds() / 60,
        "setup_hashes": setup_hashes,
        "log_path": str(Path(path)),
    }

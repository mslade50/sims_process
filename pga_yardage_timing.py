"""Record when PGA TOUR publishes round-specific hole yardages and tee times.

The public course-stats page exposes a nominal ``All Rounds`` scorecard well
before an event.  That is not a played-round setup.  This probe records each
poll and only calls a setup complete when a numbered round contains all 18
valid ``CourseHoleStats`` rows.  The accompanying tee-times page lets the
result be measured against the first tee time instead of the collection time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

SCHEMA_VERSION = "pga-yardage-availability/v2"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)
PLAUSIBLE_YARDAGE = {
    3: (75, 350),
    4: (200, 650),
    5: (350, 800),
    6: (450, 900),
}


class _NextDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_next_data = False
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag.lower() == "script" and values.get("id") == "__NEXT_DATA__":
            self.in_next_data = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self.in_next_data:
            self.in_next_data = False

    def handle_data(self, data: str) -> None:
        if self.in_next_data:
            self.parts.append(data)


def parse_next_data(html: str) -> dict[str, Any]:
    parser = _NextDataParser()
    parser.feed(html)
    if not parser.parts:
        raise ValueError("PGA page omitted __NEXT_DATA__")
    value = json.loads("".join(parser.parts))
    if not isinstance(value, dict):
        raise TypeError("PGA __NEXT_DATA__ was not an object")
    return value


def _query_entry(document: dict[str, Any], name: str) -> dict[str, Any]:
    page_props = document.get("props", {}).get("pageProps", {})
    dehydrated = page_props.get("dehydratedState", {})
    for entry in dehydrated.get("queries", []):
        key = entry.get("queryKey") if isinstance(entry, dict) else None
        if isinstance(key, list) and key and key[0] == name:
            return entry
    raise ValueError(f"PGA page omitted {name!r} query data")


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pin_is_published(hole: dict[str, Any]) -> bool:
    pin = hole.get("pinGreen")
    if not isinstance(pin, dict):
        return False
    for orientation in ("leftToRightCoords", "bottomToTopCoords"):
        coords = pin.get(orientation)
        if not isinstance(coords, dict):
            continue
        for axes in (("x", "y"), ("enhancedX", "enhancedY")):
            values = [_as_float(coords.get(axis)) for axis in axes]
            if all(value is not None and value >= 0 for value in values):
                return True
    return False


def parse_course_stats(document: dict[str, Any]) -> dict[str, Any]:
    entry = _query_entry(document, "courseStats")
    state = entry.get("state", {})
    data = state.get("data") or {}
    courses: list[dict[str, Any]] = []
    for course in data.get("courses") or []:
        if not isinstance(course, dict):
            continue
        setups: list[dict[str, Any]] = []
        for panel in course.get("roundHoleStats") or []:
            if not isinstance(panel, dict):
                continue
            holes: list[dict[str, Any]] = []
            for row in panel.get("holeStats") or []:
                if not isinstance(row, dict):
                    continue
                hole_no = _as_int(row.get("courseHoleNum"))
                if row.get("__typename") != "CourseHoleStats" or not hole_no:
                    continue
                if not 1 <= hole_no <= 18:
                    continue
                holes.append(
                    {
                        "hole": hole_no,
                        "par": _as_int(row.get("parValue")),
                        "yards": _as_int(row.get("yards")),
                        "live": bool(row.get("live")),
                        "pin_published": _pin_is_published(row),
                    }
                )
            holes.sort(key=lambda row: row["hole"])
            unique_holes = {row["hole"] for row in holes}
            valid_holes = [
                row
                for row in holes
                if row["par"] in PLAUSIBLE_YARDAGE
                and row["yards"] is not None
                and PLAUSIBLE_YARDAGE[row["par"]][0]
                <= row["yards"]
                <= PLAUSIBLE_YARDAGE[row["par"]][1]
            ]
            round_no = _as_int(panel.get("roundNum"))
            complete = (
                len(holes) == 18
                and len(unique_holes) == 18
                and len(valid_holes) == 18
            )
            canonical_setup = [
                {"hole": row["hole"], "par": row["par"], "yards": row["yards"]}
                for row in holes
            ]
            setup_sha256 = hashlib.sha256(
                json.dumps(
                    canonical_setup, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            setups.append(
                {
                    "round": round_no,
                    "round_header": panel.get("roundHeader"),
                    "live": bool(panel.get("live")),
                    "complete": complete,
                    "hole_count": len(unique_holes),
                    "pin_count": sum(row["pin_published"] for row in holes),
                    "total_yards": (
                        sum(row["yards"] for row in valid_holes) if complete else None
                    ),
                    "setup_sha256": setup_sha256,
                    "holes": holes,
                }
            )
        courses.append(
            {
                "course_id": str(course.get("courseId") or ""),
                "course_code": course.get("courseCode"),
                "course_name": course.get("courseName"),
                "host_course": bool(course.get("hostCourse")),
                "nominal_par": _as_int(course.get("par")),
                "nominal_yardage": _as_int(course.get("yardage")),
                "setups": setups,
            }
        )
    complete_courses_by_round: dict[str, list[str]] = {}
    for course in courses:
        course_id = str(course["course_id"])
        for setup in course["setups"]:
            if setup["round"] is None or not setup["complete"]:
                continue
            key = str(int(setup["round"]))
            complete_courses_by_round.setdefault(key, []).append(course_id)
    complete_courses_by_round = {
        round_no: sorted(set(course_ids))
        for round_no, course_ids in complete_courses_by_round.items()
    }
    return {
        "page_data_updated_at_ms": _as_int(state.get("dataUpdatedAt")),
        "tournament_id": data.get("tournamentId"),
        "complete_courses_by_round": complete_courses_by_round,
        "courses": courses,
    }


def _iso_from_epoch_ms(value: Any) -> str | None:
    epoch_ms = _as_int(value)
    if epoch_ms is None or epoch_ms <= 0:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000, timezone.utc).isoformat()


def parse_tee_times(document: dict[str, Any]) -> dict[str, Any]:
    entry = _query_entry(document, "teeTimes")
    state = entry.get("state", {})
    data = state.get("data") or {}
    rounds: list[dict[str, Any]] = []
    for round_data in data.get("rounds") or []:
        if not isinstance(round_data, dict):
            continue
        groups = [
            group
            for group in round_data.get("groups") or []
            if isinstance(group, dict)
        ]
        tee_times = [
            _as_int(group.get("teeTime"))
            for group in groups
        ]
        tee_times = [value for value in tee_times if value is not None and value > 0]
        course_ids = sorted(
            {
                str(group["courseId"])
                for group in groups
                if group.get("courseId") not in (None, "")
            }
        )
        rounds.append(
            {
                "round": _as_int(round_data.get("roundInt")),
                "status": round_data.get("roundStatus"),
                "group_count": len(tee_times),
                "course_ids": course_ids,
                "first_tee_at_utc": _iso_from_epoch_ms(min(tee_times)) if tee_times else None,
                "last_tee_at_utc": _iso_from_epoch_ms(max(tee_times)) if tee_times else None,
            }
        )
    return {
        "page_data_updated_at_ms": _as_int(state.get("dataUpdatedAt")),
        "timezone": data.get("timezone"),
        "rounds": rounds,
    }


def fetch_page(url: str, timeout: float, attempts: int) -> tuple[dict[str, Any], dict[str, Any]]:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request_started_at = datetime.now(timezone.utc).isoformat()
            response = requests.get(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.content
            metadata = {
                "url": response.url,
                "request_started_at_utc": request_started_at,
                "received_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code,
                "http_date": response.headers.get("Date"),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_length": len(body),
                "payload_sha256": hashlib.sha256(body).hexdigest(),
            }
            return parse_next_data(response.text), metadata
        except Exception as exc:  # noqa: BLE001 - each failed poll is recorded
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {attempts} attempts: {error}")


def collect_observation(
    *,
    event_id: str,
    season: int,
    slug: str,
    event_name: str | None,
    timeout: float,
    attempts: int,
    fetcher: Callable[[str, float, int], tuple[dict[str, Any], dict[str, Any]]] = fetch_page,
) -> tuple[dict[str, Any], bool]:
    base = f"https://www.pgatour.com/tournaments/{season}/{slug}/{event_id}"
    observation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "event_name": event_name,
        "season": season,
        "slug": slug,
        "poll_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }
    ok = True
    for label, suffix, parser in (
        ("course_stats", "course-stats", parse_course_stats),
        ("tee_times", "tee-times", parse_tee_times),
    ):
        url = f"{base}/{suffix}"
        try:
            document, metadata = fetcher(url, timeout, attempts)
            observation["sources"][label] = {
                **metadata,
                "data": parser(document),
            }
        except Exception as exc:  # noqa: BLE001 - preserve the other source
            ok = False
            observation["sources"][label] = {"url": url, "error": str(exc)[:1000]}
    observation["poll_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    course_data = observation["sources"].get("course_stats", {}).get("data", {})
    tee_data = observation["sources"].get("tee_times", {}).get("data", {})
    complete_courses = {
        int(round_no): set(course_ids)
        for round_no, course_ids in course_data.get("complete_courses_by_round", {}).items()
    }
    course_count = len(course_data.get("courses", []))
    round_completeness: dict[str, dict[str, Any]] = {}
    tee_rounds = {
        int(row["round"]): row
        for row in tee_data.get("rounds", [])
        if row.get("round") is not None
    }
    candidate_rounds = sorted(set(complete_courses) | set(tee_rounds))
    for round_no in candidate_rounds:
        available = complete_courses.get(round_no, set())
        used = set(tee_rounds.get(round_no, {}).get("course_ids", []))
        if course_count == 1 and not used:
            required = {
                str(course_data["courses"][0]["course_id"])
            }
        else:
            required = used
        complete = bool(required) and required.issubset(available)
        round_completeness[str(round_no)] = {
            "used_course_ids": sorted(required),
            "complete_course_ids": sorted(available),
            "missing_course_ids": sorted(required - available),
            "complete": complete,
        }
    daily_complete_rounds = [
        int(round_no)
        for round_no, detail in round_completeness.items()
        if detail["complete"]
    ]
    observation["summary"] = {
        "daily_complete_rounds": sorted(daily_complete_rounds),
        "round_course_completeness": round_completeness,
        "first_tee_by_round_utc": {
            str(row["round"]): row["first_tee_at_utc"]
            for row in tee_data.get("rounds", [])
            if row.get("round") is not None and row.get("first_tee_at_utc")
        },
    }
    return observation, ok


def append_observation(path: Path, observation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    locked = False
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("JSONL append returned zero bytes written")
            offset += written
        os.fsync(descriptor)
    finally:
        if locked:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root.add_argument("--event-id", required=True)
    root.add_argument("--season", type=int, required=True)
    root.add_argument("--slug", required=True)
    root.add_argument("--event-name")
    root.add_argument("--output", type=Path, required=True)
    root.add_argument("--timeout", type=float, default=30.0)
    root.add_argument("--attempts", type=int, default=3)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.attempts < 1:
        raise SystemExit("--attempts must be positive")
    observation, ok = collect_observation(
        event_id=args.event_id,
        season=args.season,
        slug=args.slug,
        event_name=args.event_name,
        timeout=args.timeout,
        attempts=args.attempts,
    )
    append_observation(args.output, observation)
    summary = observation["summary"]
    print(
        f"[yardage-timing] {args.event_id} at {observation['poll_started_at_utc']}: "
        f"daily_rounds={summary['daily_complete_rounds']} "
        f"first_tees={summary['first_tee_by_round_utc']} output={args.output}"
    )
    if not ok:
        errors = {
            name: source.get("error")
            for name, source in observation["sources"].items()
            if source.get("error")
        }
        print(f"[yardage-timing] source errors: {errors}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

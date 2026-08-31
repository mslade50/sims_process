from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pga_yardage_timing as probe


def _course_document(*, include_daily: bool = True) -> dict:
    def rows(yards: int, *, pin: bool = False) -> list[dict]:
        result = [
            {
                "__typename": "CourseHoleStats",
                "courseHoleNum": hole,
                "parValue": "4",
                "yards": yards + hole,
                "live": False,
                "pinGreen": {
                    "leftToRightCoords": {
                        "x": -1,
                        "y": -1,
                        "enhancedX": 0.5 if pin else -1,
                        "enhancedY": 0.7 if pin else -1,
                    }
                },
            }
            for hole in range(1, 19)
        ]
        result.extend(
            [
                {"__typename": "SummaryRow", "rowType": "OUT", "yardage": 3600},
                {"__typename": "SummaryRow", "rowType": "IN", "yardage": 3600},
                {"__typename": "SummaryRow", "rowType": "TOTAL", "yardage": 7200},
            ]
        )
        return result

    panels = [
        {
            "roundHeader": "All Rounds",
            "roundNum": None,
            "live": False,
            "holeStats": rows(390),
        }
    ]
    if include_daily:
        panels.insert(
            0,
            {
                "roundHeader": "Round 1",
                "roundNum": 1,
                "live": False,
                "holeStats": rows(400, pin=True),
            },
        )
    return {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": ["courseStats", {"tournamentId": "R1"}],
                            "state": {
                                "dataUpdatedAt": 1234,
                                "data": {
                                    "tournamentId": "R1",
                                    "courses": [
                                        {
                                            "courseId": "942",
                                            "courseCode": "TC",
                                            "courseName": "Test Course",
                                            "hostCourse": True,
                                            "par": 71,
                                            "yardage": "7,249",
                                            "roundHoleStats": panels,
                                        }
                                    ],
                                },
                            },
                        }
                    ]
                }
            }
        }
    }


def _tee_document() -> dict:
    return {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": ["teeTimes", {"teeTimesCompressedV2Id": "R1"}],
                            "state": {
                                "dataUpdatedAt": 5678,
                                "data": {
                                    "timezone": "America/New_York",
                                    "rounds": [
                                        {
                                            "roundInt": 1,
                                            "roundStatus": "NOT_STARTED",
                                            "groups": [
                                                {"teeTime": 1_800_003_600_000},
                                                {"teeTime": 1_800_000_000_000},
                                            ],
                                        }
                                    ],
                                },
                            },
                        }
                    ]
                }
            }
        }
    }


def test_parse_course_stats_separates_nominal_from_daily_setup() -> None:
    result = probe.parse_course_stats(_course_document())

    assert result["complete_courses_by_round"] == {"1": ["942"]}
    course = result["courses"][0]
    assert course["nominal_yardage"] == 7249
    assert [setup["round"] for setup in course["setups"]] == [1, None]
    assert course["setups"][0]["hole_count"] == 18
    assert course["setups"][0]["pin_count"] == 18
    assert course["setups"][0]["total_yards"] == sum(400 + hole for hole in range(1, 19))
    assert len(course["setups"][0]["setup_sha256"]) == 64


def test_nominal_layout_alone_is_not_a_daily_round() -> None:
    result = probe.parse_course_stats(_course_document(include_daily=False))

    assert result["complete_courses_by_round"] == {}
    assert result["courses"][0]["setups"][0]["complete"] is True
    assert result["courses"][0]["setups"][0]["round"] is None


def test_daily_round_rejects_implausible_par_yardage() -> None:
    document = _course_document()
    course = document["props"]["pageProps"]["dehydratedState"]["queries"][0]["state"][
        "data"
    ]["courses"][0]
    course["roundHoleStats"][0]["holeStats"][0]["yards"] = 90

    result = probe.parse_course_stats(document)

    assert result["complete_courses_by_round"] == {}
    assert result["courses"][0]["setups"][0]["complete"] is False


def test_parse_tee_times_uses_earliest_group() -> None:
    result = probe.parse_tee_times(_tee_document())

    assert result["timezone"] == "America/New_York"
    assert result["rounds"][0]["group_count"] == 2
    assert result["rounds"][0]["course_ids"] == []
    assert result["rounds"][0]["first_tee_at_utc"] == "2027-01-15T08:00:00+00:00"


def test_parse_next_data_and_append_jsonl(tmp_path) -> None:
    source = {"props": {"pageProps": {"dehydratedState": {"queries": []}}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(source)}</script></html>'

    assert probe.parse_next_data(html) == source
    output = tmp_path / "observations.jsonl"
    probe.append_observation(output, {"poll": 1})
    probe.append_observation(output, {"poll": 2})
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"poll": 1},
        {"poll": 2},
    ]


def test_append_observation_handles_partial_writes(tmp_path, monkeypatch) -> None:
    output = tmp_path / "partial.jsonl"
    real_write = probe.os.write

    def partial_write(descriptor: int, payload: bytes) -> int:
        return real_write(descriptor, payload[:3])

    monkeypatch.setattr(probe.os, "write", partial_write)
    probe.append_observation(output, {"complete": True, "round": 4})

    assert json.loads(output.read_text()) == {"complete": True, "round": 4}


def test_append_observation_serializes_concurrent_writers(tmp_path) -> None:
    output = tmp_path / "concurrent.jsonl"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda poll: probe.append_observation(output, {"poll": poll}), range(40)))

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert sorted(row["poll"] for row in rows) == list(range(40))


def test_collect_observation_keeps_both_sources() -> None:
    def fetcher(url: str, _timeout: float, _attempts: int):
        document = _tee_document() if url.endswith("tee-times") else _course_document()
        return document, {"url": url, "http_status": 200, "payload_sha256": "abc"}

    observation, ok = probe.collect_observation(
        event_id="R1",
        season=2026,
        slug="test-event",
        event_name="Test Event",
        timeout=1,
        attempts=1,
        fetcher=fetcher,
    )

    assert ok is True
    assert observation["summary"]["daily_complete_rounds"] == [1]
    assert observation["summary"]["first_tee_by_round_utc"]["1"]
    assert set(observation["sources"]) == {"course_stats", "tee_times"}


def test_multicourse_post_cut_round_requires_only_courses_in_tee_times() -> None:
    course_document = _course_document(include_daily=False)
    courses = course_document["props"]["pageProps"]["dehydratedState"]["queries"][0][
        "state"
    ]["data"]["courses"]
    host = courses[0]
    host_daily = deepcopy(_course_document()["props"]["pageProps"]["dehydratedState"]["queries"][0]["state"]["data"]["courses"][0]["roundHoleStats"][0])
    host_daily["roundNum"] = 3
    host_daily["roundHeader"] = "Round 3"
    host["roundHoleStats"].insert(0, host_daily)
    alternate = deepcopy(host)
    alternate["courseId"] = "943"
    alternate["courseName"] = "Alternate Course"
    alternate["hostCourse"] = False
    alternate["roundHoleStats"] = [
        panel for panel in alternate["roundHoleStats"] if panel["roundNum"] is None
    ]
    courses.append(alternate)

    tee_document = _tee_document()
    tee_round = tee_document["props"]["pageProps"]["dehydratedState"]["queries"][0][
        "state"
    ]["data"]["rounds"][0]
    tee_round["roundInt"] = 3
    for group in tee_round["groups"]:
        group["courseId"] = "942"

    def fetcher(url: str, _timeout: float, _attempts: int):
        document = tee_document if url.endswith("tee-times") else course_document
        return document, {"url": url, "http_status": 200, "payload_sha256": "abc"}

    observation, ok = probe.collect_observation(
        event_id="R1",
        season=2026,
        slug="test-event",
        event_name="Test Event",
        timeout=1,
        attempts=1,
        fetcher=fetcher,
    )

    assert ok is True
    assert observation["summary"]["daily_complete_rounds"] == [3]
    assert observation["summary"]["round_course_completeness"]["3"] == {
        "used_course_ids": ["942"],
        "complete_course_ids": ["942"],
        "missing_course_ids": [],
        "complete": True,
    }

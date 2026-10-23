"""Fail when sim_inputs.py drifts from the week recorded below.

round_sim blocks a run when the Sheet tourney/event_id and sim_inputs
disagree. This check is the offline half of that guard: it compares the
file with an expected week and does not read the Sheet, DataGolf, or
credentials. Advance EXPECTED_WEEK in the same commit that advances
sim_inputs.py.
"""

import sim_inputs

# Butterfield Bermuda Championship, 2026-10-22 through 2026-10-25.
# Sheet identity from the blocked round-sim run: butterfield / 528 / course 765.
EXPECTED_WEEK = {
    "tour": "pga",
    "tourney": "butterfield",
    "event_ids": [528],
    "course_id": 765,
    "course_id_1": 765,
    "course_id_2": 0,
    "PAR": 71,
    "course_par": 71,
    "CUT_LINE": 65,
    "USE_10_SHOT_RULE": False,
    "shot_rule": 0,
    "lat_override": 32.35,
    "lon_override": -64.8,
    "manual_venue_profile": None,
    "course_adjustment_scale_overrides": {},
}


def test_sim_inputs_matches_expected_week():
    mismatches = []
    for name, expected in EXPECTED_WEEK.items():
        actual = getattr(sim_inputs, name)
        if actual != expected:
            mismatches.append(f"{name}: sim_inputs={actual!r} expected={expected!r}")
    assert not mismatches, "sim_inputs week drifted:\n" + "\n".join(mismatches)
    assert sim_inputs.cutline == sim_inputs.CUT_LINE
    assert sim_inputs.PAR == sim_inputs.course_par
    assert sim_inputs.course_id == sim_inputs.course_id_1
    assert sim_inputs.tourney == EXPECTED_WEEK["tourney"]
    assert list(sim_inputs.event_ids) == list(EXPECTED_WEEK["event_ids"])


def test_course_adjustment_overrides_match_the_active_event():
    event_id = int(sim_inputs.event_ids[0])
    tour = str(sim_inputs.tour).strip().lower()
    course_id = int(sim_inputs.course_id)
    overrides = getattr(sim_inputs, "course_adjustment_scale_overrides", {}) or {}
    for key in overrides:
        tour_part, event_part, course_part, _date = key.split(":")
        assert tour_part == tour
        assert int(event_part) == event_id
        assert int(course_part) == course_id

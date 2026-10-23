"""Fail when sim_inputs.py drifts from the week recorded below.

round_sim blocks a run when the Sheet tourney/event_id and sim_inputs
disagree. This check is the offline half of that guard: it compares the
file with the expected week and does not read the Sheet, DataGolf, or
credentials. Advance EXPECTED_WEEK in the same commit that advances
sim_inputs.py.
"""

import sim_inputs

# Butterfield Bermuda Championship. Synced from the sim_prep master
# written by tools/etr_week_config.py on 2026-10-20.
EXPECTED_WEEK = {
    "tourney": "butterfield",
    "event_ids": [528],
    "course_id": 765,
    "course_id_1": 765,
    "CUT_LINE": 65,
    "lat_override": None,
    "lon_override": None,
}


def test_sim_inputs_matches_expected_week():
    mismatches = []
    for name, expected in EXPECTED_WEEK.items():
        actual = getattr(sim_inputs, name)
        if actual != expected:
            mismatches.append(f"{name}: sim_inputs={actual!r} expected={expected!r}")
    assert not mismatches, "sim_inputs week drifted:\n" + "\n".join(mismatches)
    assert sim_inputs.tourney == EXPECTED_WEEK["tourney"]
    assert list(sim_inputs.event_ids) == list(EXPECTED_WEEK["event_ids"])
    assert sim_inputs.cutline == sim_inputs.CUT_LINE
    assert sim_inputs.PAR == sim_inputs.course_par

"""Reset Google Sheet round_config for a new tournament week.

First step of the weekly pipeline. Pushes the tournament-identity fields
from sim_inputs.py to the round_config tab:
    round       -> 0
    tourney     -> sim_inputs.tourney
    event_id    -> sim_inputs.event_ids[0]
    course_id   -> sim_inputs.course_id
    cut_line    -> sim_inputs.CUT_LINE   (if defined)
    course_pars -> sim_inputs.course_par (if defined)

Downstream scripts (new_sim, scoring_baseline, live_stats_engine, round_sim)
read tourney + round from the Sheet, so without this they keep using last
week's values until the Sheet is hand-edited from a phone.

Per-round arrays (wind_r1..r4, dew_r1..r4, expected_score_r1..r4) are set
later in the pipeline by humidity.py and scoring_baseline.py — not this step.
"""
from sheet_config import reset_for_new_week


if __name__ == "__main__":
    reset_for_new_week()

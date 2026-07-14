"""Weekly event configuration ONLY.

Every fitted coefficient dict now lives in the golf_sims sheet's 'coefficients'
tab (single source of truth) and is materialized into this module's namespace
by coeff_loader at the bottom of this file - consumers keep importing names
from sim_inputs unchanged. Edit the sheet to change the model; edit THIS FILE
only for the week's event/course/cut/wind and manual player knobs.
"""

from datetime import datetime

##New sim inputs
SIMULATIONS   = 100000
STD_DEV       = 2.8
PAR           = 70
CUT_LINE      = 70
USE_10_SHOT_RULE = False
WIND_FACTOR_SIM  = 0.12  # must match your main script
TOP_K = 20

#number of simulations you want to run. Applied to rd lvl and hole lvl scripts
num_sims=50000

###basic information for the week
wind_override = 0.2
baseline_wind = 0.12

baseline_dew = -0.018
dewpoint_wave = -0.035
dew_calculation = .6*baseline_dew + .4*dewpoint_wave
wind_speed_base=12.2

start_yr=2019 #first year of data you want to consider in your course baslines
tour='pga'
event_ids = [100]
course_id = 541
tourney = 'the_open'
course_par = 70
course_name = "" #this is for the multi course showdown sims to id proper course
# course_name = "Arnold Palmer's Bay Hill Club & Lodge"

# --- No-data-course overrides (The Open @ Royal Birkdale 2026) ---
# We have no ShotLink/scoring history for this specific venue, so:
#  1) Pull category variance from the EVENT (Open Champ, all venues 2019-25),
#     not the empty Birkdale course filter — see scoring_baseline.py.
variance_event_level_only = True
#  2) Hard-code per-round field scoring averages instead of the mixed-venue
#     computed baseline. Absolute strokes (par 70). Consumed by scoring_baseline
#     when writing expected_score_r1..r4 to the round_config sheet.
expected_score_override = {1: 71.3, 2: 71.3, 3: 71.0, 4: 71.0}

#for multiple course setups in the showdown sim
course_id_1=541
course_id_2=0

#cut rules. Line is inclusive of ties, shot rule should be 0 as a default
cutline = CUT_LINE
shot_rule=0

#for players who we don't have a birthday (monday q guys etc.)
default_birthday = datetime(1995, 1, 1)

#expected tee time range on the weekend to forecast weather in sims
tee_time_start="8:30"
tee_time_end="1:00"

#any names that cause trouble, want to ensure consistency
name_replacements = {
    'echavarria, nico': 'echavarria, nicolas',
    'norgaard, niklas': 'norgaard moller, niklas',
    'moller, niklas norgaard': 'norgaard moller, niklas',
    'stevens, sam': 'stevens, samuel',
    # DK salary CSV -> our canonical (Zurich 2026)
    'brown, daniel': 'brown, dan',
    'l. smith, jordan': 'smith, jordan',
    'smith, jordan l.': 'smith, jordan',
    'li, hao-tong': 'li, haotong',
    'mccarty, matthew': 'mccarty, matt',
    'skov olesen, jacob': 'olesen, jacob skov',
    'davis, cam': 'davis, cameron',
    'lee, k.h.': 'lee, kyounghoon',
    'ventura, kris': 'ventura, kristoffer',
    'schmid, matti': 'schmid, matthias',
    'dumont de chassart, adrien': 'dumont de chassart, adrien',
    'nesmith, matthew': 'nesmith, matt',
    'ayora, angel': 'ayora fanegas, angel',
    'capan, frankie': 'capan iii, frankie',
    'stallings, stephen jr': 'stallings jr., stephen',
    'bauchou, zachary': 'bauchou, zach',
    'kim, seonghyeon': 'kim, s.h.',
    'chassart, adrien dumont de': 'dumont de chassart, adrien',
    'keefer, john': 'keefer, johnny',
    'kim, sh': 'kim, s.h.',
    'rooyen, erik van': 'van rooyen, erik',
    'spaun, jj': 'spaun, j.j.'
}

##manual adjustments for players which we do not have requisite data on.
##number here is a replacement for the skill prediction pre course fit etc
overrides = {
}

overrides_sd = {
}

manual_boosts={
}

# Field replacements: WD'd player -> replacement (used by skill_imports.py
# when DataGolf hasn't updated the field yet). Both names lowercase
# "lastname, firstname". Replacement just needs to exist in DG
# decompositions or player_rounds; remaining fields fall through fallbacks.
field_replacements = {
    # 'cantlay, patrick': 'thorbjornsen, michael',
}


#for etr export to sheet
dk_naming_convention= {
    'frankie capan iii': 'frankie capan',
    'kyounghoon lee' : 'kyoung-hoon lee',
    'willie mack iii': 'willie mack',
    'matthias schmid' : 'matti schmid',
    'smith, jordan': 'smith, jordan l.',
    'nesmith, matt': 'nesmith, matthew'
}

# ── coefficients: loaded from the golf_sims sheet (single source of truth) ───
from coeff_loader import load_sheet_coefficients as _load_sheet_coefficients
globals().update(_load_sheet_coefficients())

# majors scalar comes from the sheet (scalars/major_adjustment); the event
# lists are event identity, not tunable values, so they stay here
major_adjustment = major_adjustment if any(eid in [33, 14, 100, 26] for eid in event_ids) else 0  # noqa: F821
links_adjustment = 1 if any(eid in [100, 541] for eid in event_ids) else 0

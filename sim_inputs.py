"""Weekly event configuration ONLY.

Every fitted coefficient dict now lives in the golf_sims sheet's 'coefficients'
tab (single source of truth) and is materialized into this module's namespace
by coeff_loader at the bottom of this file - consumers keep importing names
from sim_inputs unchanged. Edit the sheet to change the model; edit THIS FILE
only for the week's event/course/cut/wind and manual player knobs.

sim_prep holds the master copy; push_sim_inputs.py syncs this file (and
coeff_loader.py) into sims_process. Keep it in this slim format on both
sides — a fat copy with inline coefficient dicts clobbers the sheet
architecture (2026-07-20 bug: weekly push from a stale sim_prep clone).
"""

from datetime import datetime

##New sim inputs
SIMULATIONS   = 100000
STD_DEV       = 2.8
PAR           = 70
CUT_LINE      = 100  # no-cut playoff event; >= field size triggers no-cut handling in new_sim
USE_10_SHOT_RULE = False
WIND_FACTOR_SIM  = 0.155  # must match your main script
TOP_K = 20

# One simulation-count contract for both round- and hole-level consumers.
num_sims = SIMULATIONS

###basic information for the week
wind_override = 0.0
# 0.155 re-estimated 2026-08 (scratch_weather_reanalysis/REPORT.md section 2)
baseline_wind = 0.155

baseline_dew = -0.018
dewpoint_wave = -0.035
dew_calculation = .6*baseline_dew + .4*dewpoint_wave
wind_speed_base=12.2

start_yr=2019 #first year of data you want to consider in your course baslines
tour='pga'
event_ids = [28]
# DataGolf's 2026 Bellerive identity; the 2018 PGA is stored as course 534.
course_id = 679
tourney = 'bmw'

# Betting validation did not support allowing the 0.65 category-profile
# calibration to change production prices yet. False writes the original,
# unshrunk category means (factor 1.0); flip only for an intentional shadow run.
# Supported course-category history remains active independently below this
# layer in cat_dists_player.py.
category_profile_shrinkage_enabled = False

# Completed events to ingest before the weekly model run. Keep these separate
# from event_ids, which identifies the upcoming simulation slate.
#
# Precedence in db_updates.py: a fresh work/ingest_manifest.json written by
# tools/discover_ingest_events.py (the guarded pipeline's discover stage) wins;
# otherwise a non-empty ingest_events below; otherwise the legacy single-tour
# trio. Populate ingest_events only for deliberate manual/backfill ingests.
ingest_events = []
# ingest_events = [
#     {"tour": "pga", "event_ids": [524], "years": [2026]},
#     {"tour": "kft", "event_ids": [26], "years": [2026]},
# ]
ingest_event_ids = [27]
ingest_tour = 'pga'
ingest_years = [2026]
course_par = 70
course_name = "" #this is for the multi course showdown sims to id proper course
# course_name = "Arnold Palmer's Bay Hill Club & Lodge"

#for multiple course setups in the showdown sim
course_id_1=876
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

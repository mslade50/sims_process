from datetime import datetime
import numpy as np

##New sim inputs
SIMULATIONS   = 100000
STD_DEV       = 3.02
PAR           = 72
CUT_LINE      = 65
USE_10_SHOT_RULE = False
WIND_FACTOR_SIM  = 0.15  # must match your main script
TOP_K = 20 

#number of simulations you want to run. Applied to rd lvl and hole lvl scripts
num_sims=50000

###basic information for the week
wind_override = 0.0
baseline_wind = 0.12

baseline_dew = -0.018
dewpoint_wave = -0.035
dew_calculation = .6*baseline_dew + .4*dewpoint_wave
wind_speed_base=12.2

start_yr=2019 #first year of data you want to consider in your course baslines
tour='pga'
event_ids = [3]
course_id = 510
tourney = 'wm_phoenix'
course_par = 72
course_name = "" #this is for the multi course showdown sims to id proper course
# course_name = "Spyglass Hill Golf Course"

major_adjustment = 0.0022 if any(eid in [33, 14, 100, 26] for eid in event_ids) else 0
links_adjustment = 1 if any(eid in [100,541] for eid in event_ids) else 0

#for multiple course setups in the showdown sim
course_id_1=510
course_id_2=0

#cut rules. Line is inclusive of ties, shot rule should be 0 as a default
cutline = 65
shot_rule=0

#for players who we don't have a birthday (monday q guys etc.)
default_birthday = datetime(1995, 1, 1)

#wind speeds expected on a pre-tournament basis. Start at 6 am end 8 pm

dewpoint_1 = [23.2, 23.7, 24.4, 24.8, 25.7, 25.6, 26.1, 26.4, 26.2, 26.5, 27.2, 28.6, 30.3, 31.8, 31.2]
dewpoint_2 = [44.0, 45.3, 45.2, 44.1, 42.6, 40.8, 39.4, 38.7, 37.3, 36.8, 35.8, 35.4, 35.4, 36.1, 36.9]
dewpoint_3 = [37.5, 37.4, 37.4, 37.5, 37.6, 37.5, 37.4, 37.5, 37.5, 38.0, 37.6, 36.3, 34.8, 33.8, 33.4]
dewpoint_4 = [36.0, 36.1, 37.2, 38.7, 39.6, 39.4, 38.8, 37.8, 37.1, 36.9, 36.5, 36.4, 36.0, 36.1, 36.6]

wind_1 = [7.1, 6.8, 6.6, 7.0, 6.0, 5.2, 5.7, 5.2, 6.2, 8.1, 9.0, 9.5, 9.5, 5.4, 1.9]
wind_2 = [3.7, 3.7, 5.1, 5.7, 4.6, 4.2, 3.9, 4.2, 4.6, 5.0, 5.7, 6.0, 5.0, 4.6, 3.6]
wind_3 = [2.9, 3.8, 4.0, 4.3, 4.1, 3.5, 3.4, 3.4, 3.0, 3.2, 3.4, 3.4, 3.7, 4.5, 5.0]
wind_4 = [2.9, 2.8, 4.2, 5.1, 5.4, 4.8, 3.4, 1.9, 0.6, 3.4, 5.8, 7.6, 8.8, 9.1, 8.1]
# wind = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] to gauge length of list
# wind = [1,1,1,1,1,1,1,1,10,10,10,10,10,10,10] 8 single 7 double
# wind = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10] all double


#if you have reason to believe the course will play easier than it has in the past
#postiive value here indicates increased difficulty
score_adj_r1 = 0
score_adj_r2 = 0
score_adj_r3 = 0
score_adj_r4 = 0

###for showdown sims
score_adj_r1_sd= 0
score_adj_r2_sd= 0 
score_adj_r3_sd= 0
score_adj_r4_sd= 0 

#how much variance do we want the sim to price in? Higher = more variance
player_var= 2

#expected tee time range on the weekend to forecast weather in sims
tee_time_start="8:30" 
tee_time_end="1:00"

#any names that cause trouble, want to ensure consistency
name_replacements = {
    'echavarria, nico': 'echavarria, nicolas',
    'norgaard, niklas': 'norgaard moller, niklas',
    'moller, niklas norgaard': 'norgaard moller, niklas',
    'stevens, sam': 'stevens, samuel',
    'stevens, sam': 'stevens, samuel',
    'davis, cam': 'davis, cameron',
    'lee, k.h.': 'lee, kyounghoon',
    'ventura, kris': 'ventura, kristoffer',
    'schmid, matti': 'schmid, matthias',
    'dumont de chassart, adrien': 'dumont de chassart, adrien',
    'nesmith, matthew' : 'nesmith, matt',
    'ayora, angel': 'ayora fanegas, angel',
    'capan, frankie' : 'capan iii, frankie',
    'stallings, stephen jr' : 'stallings jr., stephen'
}

#regression informed coefficients for base model
coefficients = {
    'adj_skill_est': 1.0757,
    'player_age': 0.010859,
    'age_squared': -0.000347,
    'delta_adj_skill_est':0.2226,
    'good_shot_ema':-0.0201,
    'relative_dsle': -0.0134,
    'consec_wks': 0.0,
    'delta_putt': -0.13,
    'delta_ott': 0.24
}

# # regression informed coefficients for base model
coefficients_2 = {
    'adj_skill_est': 0.6968,
    'ema_sg_adj': 0.1874,
    'player_age': 0.013838,
    'age_squared': -0.000394,
    'good_shot_ema': -0.0652,
    'delta_putt': -0.23,
    'delta_ott': 0.18,
    'delta_adj_skill_est': 0.0,
    'relative_dsle': -0.0117,
    'max_skill': 0.1167,
    'max_skill_50': 0.0,
    'c_exp': 0.0055,
    'course_adjustment': 0.95,
    'course_history': 0.010
}

####change the course history impacts each week, default low skill is 0.0088

coefficients_3 = {
    'adj_skill_est': 0.6118,
    'ema_sg_adj': 0.0957,
    'player_age': 0.012253,
    'age_squared': -0.000382,
    'good_shot_ema': 0.0,
    'delta_putt': 0.0,
    'delta_ott': 0.0,
    'delta_adj_skill_est': 0.0,
    'relative_dsle': -0.013,
    'max_skill': 0.0,
    'max_skill_50': 0.2514,
    'c_exp':0.0064,
    'course_adjustment': 1.05,
    'course_history': 0.005
}

##manual adjustments for players which we do not have requisite data on.
##number here is a replacement for the skill prediction pre course fit etc
overrides = {
}

overrides_sd = {
}

manual_boosts={
}
# manual_boosts = { }


#for etr export to sheet
dk_naming_convention= {
    'frankie capan iii': 'frankie capan',
    'kyounghoon lee' : 'kyoung-hoon lee',
    'willie mack iii': 'willie mack',
    'matthias schmid' : 'matti schmid',
    'smith, jordan': 'smith, jordan l.',
    'nesmith, matt': 'nesmith, matthew'
}




coefficients_r1_high = {
    'residual': 0.0460,
    'residual2': 0.0126,
    'ott': 0,
    'putt': 0,
}

coefficients_r1_midh = {
    'residual': 0.0415,
    'residual2': -0.0057,
    'ott': 0,
    'putt': 0,

}
coefficients_r1_midl = {
    'residual': 0.0571,
    'residual2': -0.0052,
    'ott': 0.11,
    'putt': -0.03,

}
coefficients_r1_low = {
    'residual': 0.0578,
    'residual2': -0.0092,
    'ott': 0.09,
    'putt': -0.07,
}

coefficients_r2 = {
    'residual': 0.1641,
    'residual2': -0.0809,
    'residual3': 0.0076,
    'avg_ott': 0.11,
    'avg_putt': -0.03,
    'avg_app': -0.0128,
    'avg_arg': -0.15,
    'delta_app': 0.06
}

coefficients_r2_6_30 = {
    'residual': 0.0029,
    'residual2': -0.0209,
    'residual3': 0.0038,
    'avg_ott': 0.07,
    'avg_putt': -0.01,
    'avg_app': -0.03,
    'avg_arg': -0.03,
    'delta_app': 0.01
}

coefficients_r2_30_up = {
    'residual': 0.0603,
    'residual2': -0.0010,
    'residual3': 0.0015,
    'avg_ott': 0.166,
    'avg_putt': 0.00,
    'avg_app': 0.015,
    'avg_arg': -0.06,
    'delta_app': 0.004
}

coefficients_r3 = {
    'sg_ott_avg': 0.035,
    'sg_putt_avg': -0.14,
    'sg_app_avg': -0.02,
    'sg_arg_avg': -0.3,
}

coefficients_r3_mid = {
    'sg_ott_avg': 0.089,
    'sg_putt_avg': -0.02,
    'sg_app_avg': -0.07,
    'sg_arg_avg': -0.13,
}

coefficients_r3_high = {
    'sg_ott_avg': 0.15,
    'sg_putt_avg': -0.00,
    'sg_app_avg': 0.05,
    'sg_arg_avg': -0.01,
}

pressure_curves_r3 = {
    'skill_1': {
        1: -0.8, 2: -0.45, 3: -0.31, 4: -0.35, 5: -0.38,
        6: -0.49, 11: -0.34, 21: -0.25, 31: -0.25, 41: -0.1,
        51: -0.1, 61: -0.01, 70: 0.0,160:0
    },
    'skill_2': {
        1: -0.47, 2: -0.35, 3: -0.31, 4: -0.3, 5: -0.25,
        6: -0.225, 11: -0.2, 21: -0.175, 31: -0.15, 41: -0.1,
        51: -0.1, 61: -0.01, 70: 0.0,160:0
    },
    'skill_3': {
        1: -0.425, 2: -0.35, 3: -0.25, 4: -0.225, 5: -0.2,
        6: -0.18, 11: -0.1, 21: -0.05, 31: -0.02, 41: -0.01,
        51: 0, 61: 0.01, 70: 0.0,160:0
    },
    'skill_4': {
        1: -0.38, 2: -0.3, 3: -0.21, 4: -0.21, 5: -0.2,
        6: -0.13, 11: -0.03, 21: -0.15, 31: 0.0, 41: 0.015,
        51: 0.05, 61: 0.00, 70: 0.0, 160:0
    },
    'skill_5': {
        1: -0.33, 2: -0.21, 3: -0.15, 4: -0.125, 5: -0.1,
        6: -0.08, 11: -0.03, 21: -0.01, 31: 0.01, 41: 0.015,
        51: 0.07, 61: 0.00, 70: 0.0, 160:0
    }
}

pressure_curves_r4 = {
    'skill_1': {
        1: -1, 2: -0.7, 3: -0.31, 4: -0.35, 5: -0.38,
        6: -0.49, 11: -0.34, 21: -0.25, 31: -0.25, 41: -0.1,
        51: -0.1, 61: -0.01, 70: 0.0,160:0
    },
    'skill_2': {
        1: -0.9, 2: -0.6, 3: -0.31, 4: -0.3, 5: -0.25,
        6: -0.225, 11: -0.2, 21: -0.175, 31: -0.15, 41: -0.1,
        51: -0.1, 61: -0.01, 70: 0.0,160:0
    },
    'skill_3': {
        1: -0.8, 2: -0.5, 3: -0.25, 4: -0.225, 5: -0.2,
        6: -0.18, 11: -0.1, 21: -0.05, 31: -0.02, 41: -0.01,
        51: 0, 61: 0.01, 70: 0.0,160:0
    },
    'skill_4': {
        1: -0.75, 2: -0.4, 3: -0.21, 4: -0.21, 5: -0.2,
        6: -0.13, 11: -0.03, 21: -0.15, 31: 0.0, 41: 0.015,
        51: 0.05, 61: 0.00, 70: 0.0, 160:0
    },
    'skill_5': {
        1: -0.7, 2: -0.3, 3: -0.15, 4: -0.125, 5: -0.1,
        6: -0.08, 11: -0.03, 21: -0.01, 31: 0.01, 41: 0.015,
        51: 0.07, 61: 0.00, 70: 0.0, 160:0
    }
}

pressure_curves_wins = {
    'over_5': {
        1: 0.12, 2: 0.1, 3: 0.09, 4: 0.08, 5: 0.05,
        6: 0.02, 11: 0
    },
    'two_to_4': {
        1: 0.05, 2: 0.05, 3: 0.03, 4: 0.02, 5: 0.01,
        6: 0, 11: 0
    },
    'exactly_1': {
        1: -0.0, 2: -0.0, 3: -0.0, 4: -0.0, 5: -0.0,
        6: -0.0, 11: 0
    },
    'none': {
        1: -0.25, 2: -0.2, 3: -0.15, 4: -0.08, 5: -0.05,
        6: -0.02, 11: 0
    }
}
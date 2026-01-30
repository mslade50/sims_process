# ============================
# PGA SIM: Integer Scoring + Realistic Ties
# Weather + Category-aware R1→R2, R2→R3, R3→R4 (avg-SG only) + Markets + MATCHUPS
# ============================

import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from numpy.linalg import cholesky

from sim_inputs import (
    tourney, dew_calculation, wind_1, wind_2, dewpoint_1, dewpoint_2,
    name_replacements,
    # R1 update sets
    coefficients_r1_high, coefficients_r1_midh, coefficients_r1_midl, coefficients_r1_low,
    # R2 update sets (pos buckets)
    coefficients_r2, coefficients_r2_6_30, coefficients_r2_30_up,
    # R3 update sets (avg SG only; no residual terms)
    coefficients_r3, coefficients_r3_mid, coefficients_r3_high,
    SIMULATIONS, STD_DEV, PAR, CUT_LINE, USE_10_SHOT_RULE,
    WIND_FACTOR_SIM, TOP_K
)


# Matchup weather-impact report settings (doesn't affect sim)
wind_calculation_report = WIND_FACTOR_SIM

# Dump the full leaderboard for the FIRST simulation iteration
DUMP_FIRST_SIM = True
DUMP_FILENAME  = f"sim_iter_0001_leaderboard_{tourney}.csv"

# Master toggle: use OTT-based in-tournament adjustments or not
USE_IN_TOURN_OTT = True  # set to False to zero all in-tournament OTT adjustments

# Input predictions
primary_path  = f"pre_course_fit_{tourney}.csv"
fallback_path = f"pre_course_fit_{tourney}.csv"

# Per-player category distribution file (course-shaped)
DISTS_FILE = "this_week_dists_adjusted.csv"

# Tour-level category correlation fallbacks
CORR_PREFS = [
    "sg_cat_corr_tour_within_player_pearson.csv",
    "sg_cat_corr_tour_spearman.csv",
    "sg_cat_corr_tour_pearson.csv",
]

CAT_ORDER = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
CLIP_CAT = (-8.0, 8.0)
RNG = np.random.default_rng(123)

# DataGolf API
API_KEY = 'c05ee5fd8f2f3b14baab409bd83c'
MATCHUPS_URL = "https://feeds.datagolf.com/betting-tools/matchups"
OUTRIGHTS_URL = "https://feeds.datagolf.com/betting-tools/outrights"

# --- Helpers ---
def parse_time(teetime):
    if pd.isnull(teetime):
        return None
    if isinstance(teetime, (int, float)) and (pd.isna(teetime) or teetime == 0):
        return None
    s = str(teetime).strip()
    if s == "":
        return None
    for fmt in ["%Y-%m-%d %H:%M", "%I:%M%p", "%m/%d/%Y %H:%M"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def calculate_avg_wind(teetime, wind_data):
    parsed = parse_time(teetime)
    if parsed is None:
        return 0.0
    dec_hour = parsed.hour + parsed.minute / 60.0
    start_idx = dec_hour - 6
    end_idx   = start_idx + 5
    minutes = np.arange(start_idx, end_idx, 1/60.0)
    return float(np.mean(np.interp(minutes, np.arange(len(wind_data)), wind_data)))

def prob_to_american(p):
    if p <= 0: return None
    if p >= 1: return -100
    return int(round(-100 * p / (1 - p))) if p > 0.5 else int(round(100 * (1 - p) / p))

def american_to_implied_probability(american_odds):
    if pd.isna(american_odds):
        return np.nan
    if american_odds > 0:
        return 100 / (american_odds + 100) * 100
    elif american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100) * 100
    return np.nan

def implied_prob_to_american_odds(prob):
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(np.floor(-100 * prob / (1 - prob)))
    else:
        return int(np.floor(100 * (1 - prob) / prob))

def load_corr_matrix(cat_order):
    for fn in CORR_PREFS:
        if os.path.exists(fn):
            R = pd.read_csv(fn, index_col=0)
            R = R.loc[cat_order, cat_order]
            return R.values
    return np.eye(len(cat_order))

def categories_given_total_for_player(mu, L, v, denom, S_vec):
    """
    Draw X ~ N(mu, Sigma) via Cholesky, then project so sum(X)=S for each S in S_vec.
    Sigma is implied by L (Cholesky), v = Sigma * 1, denom = 1' Sigma 1.
    """
    Z = RNG.standard_normal(size=(S_vec.shape[0], 4))
    X = mu + (Z @ L.T)
    sum_x = X @ np.ones(4)
    k = (S_vec - sum_x) / denom
    Xc = X + k[:, None] * v
    return np.clip(Xc, CLIP_CAT[0], CLIP_CAT[1])

def rank_positions_from_strokes(strokes_asc_int):
    s = pd.Series(strokes_asc_int)
    return s.rank(method='min').astype(int).to_numpy()

def coeff_vec_r1(cdict):
    # order: ott, (app unused), (arg unused), putt, residual, residual2
    return np.array([cdict['ott'], 0.0, 0.0, cdict['putt'], cdict['residual'], cdict['residual2']], dtype=float)

def ensure_array(x, shape):
    return x if isinstance(x, np.ndarray) else np.zeros(shape, dtype=float)

# --- Load predictions ---
if os.path.exists(primary_path):
    model_preds = pd.read_csv(primary_path).drop(columns=['my_pred'], errors='ignore')
    model_preds['my_pred'] = pd.read_csv(fallback_path)['pred']
else:
    model_preds = pd.read_csv(fallback_path).rename(columns={'pred': 'my_pred'})

model_preds['player_name'] = (
    model_preds['player_name'].astype(str).str.lower().str.strip()
    .replace(name_replacements)
)
model_preds = model_preds.drop_duplicates(subset=['player_name']).reset_index(drop=True)
model_preds.to_csv('test.csv')
# --- Weather for SIM (R1/R2 only; sim waves centered) ---
wind_r1_sim, wind_r2_sim, dew_r1_sim, dew_r2_sim = [], [], [], []
for _, row in model_preds.iterrows():
    r1 = row.get('r1_teetime', None)
    r2 = row.get('r2_teetime', None)
    wind_r1_sim.append(calculate_avg_wind(r1, wind_1))
    dew_r1_sim.append(calculate_avg_wind(r1, dewpoint_1))
    wind_r2_sim.append(calculate_avg_wind(r2, wind_2))
    dew_r2_sim.append(calculate_avg_wind(r2, dewpoint_2))

model_preds['wind_adj_r1_sim'] = WIND_FACTOR_SIM * np.array(wind_r1_sim, dtype=float)
model_preds['wind_adj_r2_sim'] = WIND_FACTOR_SIM * np.array(wind_r2_sim, dtype=float)
model_preds['dew_adj_r1_sim']  = dew_calculation * np.array(dew_r1_sim, dtype=float)
model_preds['dew_adj_r2_sim']  = dew_calculation * np.array(dew_r2_sim, dtype=float)

for col in ['wind_adj_r1_sim', 'wind_adj_r2_sim', 'dew_adj_r1_sim', 'dew_adj_r2_sim']:
    model_preds[col] = model_preds[col].mean() - model_preds[col]

# Round-level means for SIM
model_preds['r1_pred'] = model_preds['my_pred'] + model_preds['wind_adj_r1_sim'] + model_preds['dew_adj_r1_sim']
model_preds['r2_pred'] = model_preds['my_pred'] + model_preds['wind_adj_r2_sim'] + model_preds['dew_adj_r2_sim']
model_preds['r3_pred'] = model_preds['my_pred']   # no wave
model_preds['r4_pred'] = model_preds['my_pred']   # no wave

# Per-player round stdev (blend global & player)
preds = model_preds[['player_name', 'my_pred', 'std_dev', 'r1_pred', 'r2_pred', 'r3_pred', 'r4_pred']].copy()
preds['std'] = (preds['std_dev'] + STD_DEV) / 2.0

player_names = preds['player_name'].tolist()
n_players = len(player_names)

# --- Load per-player category distributions ---
if not os.path.exists(DISTS_FILE):
    raise FileNotFoundError(f"Missing {DISTS_FILE}. Build it earlier.")

dists = pd.read_csv(DISTS_FILE)
dists['player_name'] = (
    dists['player_name'].astype(str).str.lower().str.strip()
    .replace(name_replacements)
)

need_cols = {'player_name', 'category_clean', 'mean_adj', 'std_adj'}
missing = need_cols - set(dists.columns)
if missing:
    raise ValueError(f"{DISTS_FILE} missing columns: {missing}")

mu_w  = dists.pivot(index='player_name', columns='category_clean', values='mean_adj')
std_w = dists.pivot(index='player_name', columns='category_clean', values='std_adj')
global_mu  = dists.groupby('category_clean')['mean_adj'].mean()
global_std = dists.groupby('category_clean')['std_adj'].median()

R = load_corr_matrix(CAT_ORDER)
try:
    _ = cholesky(R)
except np.linalg.LinAlgError:
    R = 0.95*R + 0.05*np.eye(4)

player_params = []
ones4 = np.ones(4)
for p in player_names:
    mu_row  = mu_w.loc[p].reindex(CAT_ORDER) if p in mu_w.index else pd.Series(index=CAT_ORDER, dtype=float)
    std_row = std_w.loc[p].reindex(CAT_ORDER) if p in std_w.index else pd.Series(index=CAT_ORDER, dtype=float)

    mu  = mu_row.fillna(global_mu.reindex(CAT_ORDER)).to_numpy(dtype=float)
    std = std_row.fillna(global_std.reindex(CAT_ORDER)).to_numpy(dtype=float).clip(1e-6)

    D = np.diag(std)
    Sigma = D @ R @ D
    L = cholesky(Sigma)
    v = Sigma @ ones4
    denom = float(ones4 @ v)
    player_params.append((mu, std, Sigma, L, v, denom))

# Align arrays
indexer = preds.set_index('player_name')
r1_mu = indexer['r1_pred'].reindex(player_names).to_numpy(dtype=float)
r2_mu = indexer['r2_pred'].reindex(player_names).to_numpy(dtype=float)
r3_mu = indexer['r3_pred'].reindex(player_names).to_numpy(dtype=float)
r4_mu = indexer['r4_pred'].reindex(player_names).to_numpy(dtype=float)
my_pred_base = indexer['my_pred'].reindex(player_names).to_numpy(dtype=float)
round_std = indexer['std'].reindex(player_names).to_numpy(dtype=float)

# ======================
# R1: sample totals and decompose; integer strokes
# ======================
sg_r1 = RNG.normal(loc=r1_mu[:, None], scale=round_std[:, None], size=(n_players, SIMULATIONS))
cats_r1 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
for i, (mu, std, Sigma, L, v, denom) in enumerate(player_params):
    cats_r1[i] = categories_given_total_for_player(mu, L, v, denom, sg_r1[i])
max_err_r1 = float(np.abs(cats_r1.sum(axis=2) - sg_r1).max())
print(f"[check] R1 category sum error max: {max_err_r1:.6f}")

strokes_r1 = np.rint(PAR - sg_r1).astype(int)

# ======================
# R1 → R2 skill update
# ======================
resid_r1  = sg_r1 - my_pred_base[:, None]
resid2_r1 = resid_r1**2
ott_r1    = cats_r1[:, :, 0]
putt_r1   = cats_r1[:, :, 3]

high_m  = (my_pred_base >  1.0)
midh_m  = (my_pred_base >  0.5) & (my_pred_base <= 1.0)
midl_m  = (my_pred_base > -0.5) & (my_pred_base <= 0.5)
low_m   = (my_pred_base <= -0.5)

C_high = coeff_vec_r1(coefficients_r1_high)
C_midh = coeff_vec_r1(coefficients_r1_midh)
C_midl = coeff_vec_r1(coefficients_r1_midl)
C_low  = coeff_vec_r1(coefficients_r1_low)

C = np.zeros((n_players, 6), dtype=float)
C[high_m] = C_high
C[midh_m] = C_midh
C[midl_m] = C_midl
C[low_m]  = C_low

tot_resid_adj_r1 = resid_r1 * C[:, [4]] + resid2_r1 * C[:, [5]]
mask_bad = (resid_r1 < 0) & (tot_resid_adj_r1 > 0.2)
tot_resid_adj_r1 = np.minimum(np.where(mask_bad, 0.2, tot_resid_adj_r1), 0.5)

ott_adj_r1  = ott_r1  * C[:, [0]]
putt_adj_r1 = putt_r1 * C[:, [3]]
sg_adj_r1   = ott_adj_r1 + putt_adj_r1
total_adjustment_r1 = tot_resid_adj_r1 + sg_adj_r1

updated_skill_r2 = my_pred_base[:, None] + total_adjustment_r1
sg_r2_mean = updated_skill_r2 + (r2_mu - my_pred_base)[:, None]

# ======================
# R2: sample totals and decompose; integer strokes
# ======================
sg_r2 = RNG.normal(loc=sg_r2_mean, scale=round_std[:, None], size=(n_players, SIMULATIONS))
cats_r2 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
for i, (mu, std, Sigma, L, v, denom) in enumerate(player_params):
    cats_r2[i] = categories_given_total_for_player(mu, L, v, denom, sg_r2[i])
max_err_r2 = float(np.abs(cats_r2.sum(axis=2) - sg_r2).max())
print(f"[check] R2 category sum error max: {max_err_r2:.6f}")

strokes_r2 = np.rint(PAR - sg_r2).astype(int)
r1_r2_scores = strokes_r1 + strokes_r2

# ======================
# CUT LOGIC after 36 (Top-N and ties)
# ======================
made_cut_mask = np.ones_like(r1_r2_scores, dtype=bool)
for j in range(SIMULATIONS):
    sc = r1_r2_scores[:, j]
    cut_score = np.sort(sc)[CUT_LINE - 1]
    top_cut = sc <= cut_score
    if USE_10_SHOT_RULE:
        within_10 = sc <= (sc.min() + 10)
        made_cut_mask[:, j] = top_cut | within_10
    else:
        made_cut_mask[:, j] = top_cut

# ======================
# R2 → R3 skill update (position buckets; uses R1+R2 stats)
# ======================
resid_r2  = sg_r2 - sg_r2_mean
resid2_r2 = resid_r2**2
resid3_r2 = resid_r2**3

avg_ott_r2  = 0.5 * (cats_r1[:, :, 0] + cats_r2[:, :, 0])
avg_app_r2  = 0.5 * (cats_r1[:, :, 1] + cats_r2[:, :, 1])
avg_arg_r2  = 0.5 * (cats_r1[:, :, 2] + cats_r2[:, :, 2])
avg_putt_r2 = 0.5 * (cats_r1[:, :, 3] + cats_r2[:, :, 3])
delta_app_r2 = cats_r2[:, :, 1] - cats_r1[:, :, 1]

pos_lt_6_mask  = np.zeros((n_players, SIMULATIONS), dtype=bool)
pos_6_30_mask  = np.zeros((n_players, SIMULATIONS), dtype=bool)
pos_gt_30_mask = np.zeros((n_players, SIMULATIONS), dtype=bool)
for j in range(SIMULATIONS):
    pos = rank_positions_from_strokes(r1_r2_scores[:, j])
    pos_lt_6_mask[:, j]  = (pos < 6)
    pos_6_30_mask[:, j]  = (pos >= 6) & (pos <= 30)
    pos_gt_30_mask[:, j] = (pos > 30)

def apply_block(adj_dict, mask):
    out = {}
    for key, coeff in adj_dict.items():
        if key == 'residual':
            base = resid_r2
        elif key == 'residual2':
            base = resid2_r2
        elif key == 'residual3':
            base = resid3_r2
        elif key == 'avg_ott':
            base = avg_ott_r2
        elif key == 'avg_putt':
            base = avg_putt_r2
        elif key == 'avg_app':
            base = avg_app_r2
        elif key == 'avg_arg':
            base = avg_arg_r2
        elif key == 'delta_app':
            base = delta_app_r2
        else:
            continue
        out[f"{key}_adj"] = np.where(mask, base * coeff, 0.0)
    return out

adj_lt6  = apply_block(coefficients_r2,          pos_lt_6_mask)
adj_6_30 = apply_block(coefficients_r2_6_30,     pos_6_30_mask)
adj_30up = apply_block(coefficients_r2_30_up,    pos_gt_30_mask)

all_keys = set(adj_lt6) | set(adj_6_30) | set(adj_30up)
adj_sum = {}
for k in all_keys:
    adj_sum[k] = adj_lt6.get(k, 0.0) + adj_6_30.get(k, 0.0) + adj_30up.get(k, 0.0)

shape2 = resid_r2.shape
tot_resid_adj_r2 = (
    ensure_array(adj_sum.get('residual_adj', 0.0),  shape2) +
    ensure_array(adj_sum.get('residual2_adj', 0.0), shape2) +
    ensure_array(adj_sum.get('residual3_adj', 0.0), shape2)
)
tot_sg_adj_r2 = (
    ensure_array(adj_sum.get('avg_ott_adj', 0.0),   shape2) +
    ensure_array(adj_sum.get('avg_putt_adj', 0.0),  shape2) +
    ensure_array(adj_sum.get('avg_app_adj', 0.0),   shape2) +
    ensure_array(adj_sum.get('avg_arg_adj', 0.0),   shape2) +
    ensure_array(adj_sum.get('delta_app_adj', 0.0), shape2)
)
sg_adj_r1 = ensure_array(sg_adj_r1, shape2)  # avoid double counting R1 part
total_adjustment_r2 = (tot_resid_adj_r2 + tot_sg_adj_r2) - sg_adj_r1

updated_skill_r3 = updated_skill_r2 + total_adjustment_r2
sg_r3_mean = updated_skill_r3 + (r3_mu - my_pred_base)[:, None]

# ======================
# R3: sample totals & decompose; integer strokes
# ======================
sg_r3 = RNG.normal(loc=sg_r3_mean, scale=round_std[:, None], size=(n_players, SIMULATIONS))
cats_r3 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
for i, (mu, std, Sigma, L, v, denom) in enumerate(player_params):
    cats_r3[i] = categories_given_total_for_player(mu, L, v, denom, sg_r3[i])

strokes_r3 = np.rint(PAR - sg_r3).astype(int)
r1_r3_scores = r1_r2_scores + strokes_r3

# ======================
# R3 → R4 (AVG-SG ONLY; position buckets)
# ======================
avg_ott_r3  = 0.66 * (0.5 * (cats_r1[:, :, 0] + cats_r2[:, :, 0])) + 0.34 * cats_r3[:, :, 0]
avg_app_r3  = 0.66 * ( 0.5 * (cats_r1[:, :, 1] + cats_r2[:, :, 1])) + 0.34 * cats_r3[:, :, 1]
avg_arg_r3  = 0.66 * (0.5 * (cats_r1[:, :, 2] + cats_r2[:, :, 2])) + 0.34 * cats_r3[:, :, 2]
avg_putt_r3 = 0.66 * (0.5 * (cats_r1[:, :, 3] + cats_r2[:, :, 3])) + 0.34 * cats_r3[:, :, 3]

pos_lt_6_mask_r3  = np.zeros((n_players, SIMULATIONS), dtype=bool)
pos_6_20_mask_r3  = np.zeros((n_players, SIMULATIONS), dtype=bool)
pos_gt_20_mask_r3 = np.zeros((n_players, SIMULATIONS), dtype=bool)
for j in range(SIMULATIONS):
    pos = rank_positions_from_strokes(r1_r3_scores[:, j])
    pos_lt_6_mask_r3[:, j]  = (pos < 6)
    pos_6_20_mask_r3[:, j]  = (pos >= 6) & (pos <= 20)
    pos_gt_20_mask_r3[:, j] = (pos > 20)

def apply_block_r3_avg(adj_dict, mask):
    out = {}
    for key, coeff in adj_dict.items():
        if key == 'sg_ott_avg':
            base = avg_ott_r3
        elif key == 'sg_putt_avg':
            base = avg_putt_r3
        elif key == 'sg_app_avg':
            base = avg_app_r3
        elif key == 'sg_arg_avg':
            base = avg_arg_r3
        else:
            continue
        out[f"{key}_adj_r3"] = np.where(mask, base * coeff, 0.0)
    return out

adj_lt6_r3  = apply_block_r3_avg(coefficients_r3,       pos_lt_6_mask_r3)
adj_6_20_r3 = apply_block_r3_avg(coefficients_r3_mid,   pos_6_20_mask_r3)
adj_20up_r3 = apply_block_r3_avg(coefficients_r3_high,  pos_gt_20_mask_r3)

all_keys_r3 = set(adj_lt6_r3) | set(adj_6_20_r3) | set(adj_20up_r3)
adj_sum_r3 = {}
for k in all_keys_r3:
    adj_sum_r3[k] = adj_lt6_r3.get(k, 0.0) + adj_6_20_r3.get(k, 0.0) + adj_20up_r3.get(k, 0.0)

shape3 = (n_players, SIMULATIONS)
tot_resid_adj_r3 = np.zeros(shape3, dtype=float)  # no residual terms at R3
tot_sg_adj_r3 = (
    ensure_array(adj_sum_r3.get('sg_ott_avg_adj_r3', 0.0),  shape3) +
    ensure_array(adj_sum_r3.get('sg_putt_avg_adj_r3', 0.0), shape3) +
    ensure_array(adj_sum_r3.get('sg_app_avg_adj_r3', 0.0),  shape3) +
    ensure_array(adj_sum_r3.get('sg_arg_avg_adj_r3', 0.0),  shape3)
)
total_adjustment_r3 = tot_sg_adj_r3

updated_skill_r4 = updated_skill_r3 - (tot_sg_adj_r2 + tot_resid_adj_r2) + total_adjustment_r3
sg_r4_mean = updated_skill_r4 + (r4_mu - my_pred_base)[:, None]

# ======================
# R4: sample totals & decompose; integer strokes
# ======================
sg_r4 = RNG.normal(loc=sg_r4_mean, scale=round_std[:, None], size=(n_players, SIMULATIONS))
cats_r4 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
for i, (mu, std, Sigma, L, v, denom) in enumerate(player_params):
    cats_r4[i] = categories_given_total_for_player(mu, L, v, denom, sg_r4[i])

strokes_r4 = np.rint(PAR - sg_r4).astype(int)

# Missed-cut penalty for R3+R4
r3_r4 = strokes_r3 + strokes_r4
r3_r4[~made_cut_mask] = 200

# Final integer 72-hole totals
final_scores = r1_r2_scores + r3_r4

# ======================
# Markets (WIN; Top-5/10/20 with dead-heat)
# ======================
simulated_winners = []
for j in range(SIMULATIONS):
    sc = final_scores[:, j]
    min_score = sc.min()
    tied = np.where(sc == min_score)[0]
    winner_idx = RNG.choice(tied)
    simulated_winners.append(player_names[winner_idx])

win_counts = pd.Series(simulated_winners).value_counts(normalize=True)
sim_win_probs = win_counts.rename_axis('player_name').reset_index(name='simulated_win_prob')
sim_win_probs.to_csv("simulated_probs.csv", index=False)

df_long = pd.DataFrame(final_scores, index=player_names).T
df_long['simulation_id'] = np.arange(SIMULATIONS)
long_df = df_long.melt(id_vars='simulation_id', var_name='player_name', value_name='score')
long_df['rank'] = long_df.groupby('simulation_id')['score'].rank(method='min')

def dead_heat_factor(position, tie_count, threshold):
    start = position
    end = position + tie_count - 1
    overlap_start = max(start, 1)
    overlap_end = min(end, threshold)
    overlap_count = max(0, overlap_end - overlap_start + 1)
    return overlap_count / tie_count

player_stats = {p: {"top_5": 0.0, "top_10": 0.0, "top_20": 0.0} for p in player_names}
for sim_id, group in long_df.groupby("simulation_id", sort=False):
    pos_counts = group['rank'].value_counts().to_dict()
    for _, row in group.iterrows():
        p = row['player_name']
        pos = int(row['rank'])
        tie_ct = pos_counts[pos]
        player_stats[p]["top_5"]  += dead_heat_factor(pos, tie_ct, 5)
        player_stats[p]["top_10"] += dead_heat_factor(pos, tie_ct, 10)
        player_stats[p]["top_20"] += dead_heat_factor(pos, tie_ct, 20)

topn_df = pd.DataFrame.from_dict(player_stats, orient='index')
topn_df = topn_df.div(SIMULATIONS).reset_index().rename(columns={'index': 'player_name'})
topn_df.to_csv(f"top_finish_probs_{tourney}.csv", index=False)

finish_equity_df = pd.merge(sim_win_probs, topn_df, on="player_name", how="outer").fillna(0)
for col in ['simulated_win_prob', 'top_5', 'top_10', 'top_20']:
    finish_equity_df[f"{col}_a"] = finish_equity_df[col].apply(prob_to_american)
finish_equity_df.to_csv(f"finish_equity_{tourney}.csv", index=False)

# ======================
# Dump the full leaderboard for the FIRST simulation iteration (j=0) with adjustments
# ======================
if DUMP_FIRST_SIM:
    j = 0
    pos_r1 = rank_positions_from_strokes(strokes_r1[:, j])
    pos_r2 = rank_positions_from_strokes(r1_r2_scores[:, j])
    pos_r3 = rank_positions_from_strokes(r1_r3_scores[:, j])
    pos_r4 = rank_positions_from_strokes(final_scores[:, j])

    r2_wave = (r2_mu - my_pred_base)
    r3_wave = (r3_mu - my_pred_base)
    r4_wave = (r4_mu - my_pred_base)

    skill_base = my_pred_base
    skill_r2   = updated_skill_r2[:, j]
    skill_r3   = updated_skill_r3[:, j]
    skill_r4   = updated_skill_r4[:, j]

    adj_r1_resid = tot_resid_adj_r1[:, j]
    adj_r1_ott   = ott_adj_r1[:, j]
    adj_r1_putt  = putt_adj_r1[:, j]
    adj_r1_total = adj_r1_resid + adj_r1_ott + adj_r1_putt

    adj_r2_resid_total = tot_resid_adj_r2[:, j]
    adj_r2_sg_total    = tot_sg_adj_r2[:, j]
    adj_r2_total       = total_adjustment_r2[:, j]

    adj_r3_resid_total = np.zeros_like(adj_r2_resid_total)
    adj_r3_sg_total    = tot_sg_adj_r3[:, j]
    adj_r3_total       = adj_r3_sg_total
    adj_r4_increment   = -(adj_r2_resid_total + adj_r2_sg_total) + (adj_r3_sg_total)

    r1_to_par = (strokes_r1[:, j] - PAR)
    r2_to_par = (r1_r2_scores[:, j] - 2*PAR)
    r3_to_par = (r1_r3_scores[:, j] - 3*PAR)
    r4_to_par = (final_scores[:, j] - 4*PAR)

    df_dump = pd.DataFrame({
        "player_name": player_names,
        # integer strokes and to-par
        "r1_strokes":  strokes_r1[:, j], "r1_to_par": r1_to_par,
        "r2_strokes_36": r1_r2_scores[:, j], "r2_to_par": r2_to_par,
        "r3_strokes_54": r1_r3_scores[:, j], "r3_to_par": r3_to_par,
        "r4_strokes_72": final_scores[:, j], "r4_to_par": r4_to_par,
        # SG totals & cats
        "r1_sg_total": sg_r1[:, j], "r1_sg_ott": cats_r1[:, j, 0], "r1_sg_app": cats_r1[:, j, 1],
        "r1_sg_arg": cats_r1[:, j, 2], "r1_sg_putt": cats_r1[:, j, 3], "r1_pos": pos_r1,
        "r2_sg_total": sg_r2[:, j], "r2_sg_ott": cats_r2[:, j, 0], "r2_sg_app": cats_r2[:, j, 1],
        "r2_sg_arg": cats_r2[:, j, 2], "r2_sg_putt": cats_r2[:, j, 3], "r2_pos": pos_r2,
        "r3_sg_total": sg_r3[:, j], "r3_sg_ott": cats_r3[:, j, 0], "r3_sg_app": cats_r3[:, j, 1],
        "r3_sg_arg": cats_r3[:, j, 2], "r3_sg_putt": cats_r3[:, j, 3], "r3_pos": pos_r3,
        "r4_sg_total": sg_r4[:, j], "r4_sg_ott": cats_r4[:, j, 0], "r4_sg_app": cats_r4[:, j, 1],
        "r4_sg_arg": cats_r4[:, j, 2], "r4_sg_putt": cats_r4[:, j, 3], "r4_pos": pos_r4,
        # running avg cats
        "avg_ott_r1": cats_r1[:, j, 0], "avg_app_r1": cats_r1[:, j, 1],
        "avg_arg_r1": cats_r1[:, j, 2], "avg_putt_r1": cats_r1[:, j, 3],
        "avg_ott_r2": 0.5*(cats_r1[:, j, 0]+cats_r2[:, j, 0]),
        "avg_app_r2": 0.5*(cats_r1[:, j, 1]+cats_r2[:, j, 1]),
        "avg_arg_r2": 0.5*(cats_r1[:, j, 2]+cats_r2[:, j, 2]),
        "avg_putt_r2": 0.5*(cats_r1[:, j, 3]+cats_r2[:, j, 3]),
        "avg_ott_r3": avg_ott_r3[:, j],
        "avg_app_r3": avg_app_r3[:, j],
        "avg_arg_r3": avg_arg_r3[:, j],
        "avg_putt_r3": avg_putt_r3[:, j],
        # skill & waves
        "skill_base": skill_base,
        "adj_r1_resid": adj_r1_resid, "adj_r1_ott": adj_r1_ott, "adj_r1_putt": adj_r1_putt,
        "adj_r1_total": adj_r1_total, "skill_r2": skill_r2, "r2_wave": r2_wave, "r2_mean": sg_r2_mean[:, j],
        "adj_r2_resid_total": adj_r2_resid_total, "adj_r2_sg_total": adj_r2_sg_total, "adj_r2_total": adj_r2_total,
        "skill_r3": skill_r3, "r3_wave": r3_wave, "r3_mean": sg_r3_mean[:, j],
        "adj_r3_resid_total": adj_r3_resid_total, "adj_r3_sg_total": adj_r3_sg_total, "adj_r3_total": adj_r3_total,
        "adj_r4_increment": adj_r4_increment, "skill_r4": skill_r4, "r4_wave": r4_wave, "r4_mean": sg_r4_mean[:, j],
        # totals
        "total_sg": (sg_r1[:, j] + sg_r2[:, j] + sg_r3[:, j] + sg_r4[:, j]),
        "total_strokes": final_scores[:, j],
        "final_pos": pos_r4,
    })
    df_dump.sort_values(["final_pos", "total_strokes", "player_name"], inplace=True)
    df_dump.to_csv(DUMP_FILENAME, index=False)
    print(f"[dump] wrote {DUMP_FILENAME}")

print(f"[ok] Sim complete for {tourney}.")
print(f"  Players: {n_players}, Sims: {SIMULATIONS}")
print(f"  Outputs: simulated_probs.csv, top_finish_probs_{tourney}.csv, finish_equity_{tourney}.csv")

# ======================
# Per-player expected SG summaries (optional; same as before)
# ======================
COLS = ["ott", "app", "arg", "putt"]
r1m = cats_r1.mean(axis=1); r2m = cats_r2.mean(axis=1); r3m = cats_r3.mean(axis=1); r4m = cats_r4.mean(axis=1)
r1_total_mean = sg_r1.mean(axis=1); r2_total_mean = sg_r2.mean(axis=1)
r3_total_mean = sg_r3.mean(axis=1); r4_total_mean = sg_r4.mean(axis=1)
per_round_avg_cat = ((cats_r1 + cats_r2 + cats_r3 + cats_r4) / 4.0).mean(axis=1)
tourn_total_per_round_mean = ((sg_r1 + sg_r2 + sg_r3 + sg_r4) / 4.0).mean(axis=1)

rows = []
for i, p in enumerate(player_names):
    row = {"player_name": p}
    for k, col in enumerate(COLS):
        row[f"r1_{col}_mean"] = float(r1m[i, k])
        row[f"r2_{col}_mean"] = float(r2m[i, k])
        row[f"r3_{col}_mean"] = float(r3m[i, k])
        row[f"r4_{col}_mean"] = float(r4m[i, k])
        row[f"tourn_{col}_per_round_mean"] = float(per_round_avg_cat[i, k])
    row["r1_total_mean"] = float(r1_total_mean[i])
    row["r2_total_mean"] = float(r2_total_mean[i])
    row["r3_total_mean"] = float(r3_total_mean[i])
    row["r4_total_mean"] = float(r4_total_mean[i])
    row["tourn_total_sg_per_round_mean"] = float(tourn_total_per_round_mean[i])
    rows.append(row)

df_avg = pd.DataFrame(rows)
out_avg_file = f"avg_expected_cat_sg_{tourney}.csv"
df_avg.to_csv(out_avg_file, index=False)
print(f"[ok] wrote {out_avg_file}")

import plotly.graph_objects as go

rank_probs = (
    long_df.groupby(['player_name', 'rank'])
            .size()
            .div(SIMULATIONS)
            .rename('prob')
            .reset_index()
)

# === SAVE UPDATED-SIM RANK DISTRIBUTIONS (for later overlay) ===
rank_probs_updated = rank_probs.rename(columns={'prob': 'prob_u'}).copy()
rank_probs_updated['rank'] = rank_probs_updated['rank'].astype(int)
rank_probs_updated.to_parquet(f"rank_probs_updated_{tourney}.parquet", index=False)
print(f"[ok] wrote rank_probs_updated_{tourney}.parquet")


if TOP_K is not None and not sim_win_probs.empty:
    top_players = (
        sim_win_probs.sort_values('simulated_win_prob', ascending=False)
                      .head(TOP_K)['player_name'].tolist()
    )
    rank_probs_plot = rank_probs[rank_probs['player_name'].isin(top_players)].copy()
else:
    top_players = rank_probs['player_name'].unique().tolist()
    rank_probs_plot = rank_probs.copy()

n_positions = len(player_names)
full_index = pd.MultiIndex.from_product(
    [top_players, np.arange(1, n_positions + 1)],
    names=['player_name', 'rank']
)
rank_probs_plot = (
    rank_probs_plot.set_index(['player_name', 'rank'])
                    .reindex(full_index, fill_value=0.0)
                    .reset_index()
)

players = list(dict.fromkeys(top_players))
traces = []
for i, p in enumerate(players):
    dfp = rank_probs_plot[rank_probs_plot['player_name'] == p]
    traces.append(
        go.Bar(
            x=dfp['rank'].astype(int),
            y=(dfp['prob'] * 100.0),
            name=p,
            visible=(i == 0),
            hovertemplate="Pos %{x}<br>Prob %{y:.2f}%<extra></extra>"
        )
    )

buttons = []
for i, p in enumerate(players):
    vis = [False] * len(players)
    vis[i] = True
    buttons.append(dict(
        label=p,
        method="update",
        args=[{"visible": vis},
              {"title": f"Finish-position distribution: {p}"}]
    ))

fig = go.Figure(data=traces)
fig.update_layout(
    title=f"Finish-position distribution: {players[0]}",
    xaxis_title="Finish position (1 = winner)",
    yaxis_title="Probability (%)",
    template="plotly_white",
    bargap=0,
    updatemenus=[dict(
        type="dropdown",
        x=1.02, y=1.0, xanchor="left", yanchor="top",
        showactive=True,
        buttons=buttons
    )],
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0)
)
fig.update_xaxes(rangemode='tozero', dtick=1, tickmode='auto')
fig.update_yaxes(rangemode='tozero', ticksuffix="%")
fig.show()


# ============================================================
# MATCHUPS PRICING (weather impact CSV only; sim already done)
# ============================================================
wx = model_preds[['player_name', 'r1_teetime', 'r2_teetime', 'my_pred']].copy()
wx['r1_teetime'] = pd.to_datetime(wx['r1_teetime'], errors='coerce')
wx['r2_teetime'] = pd.to_datetime(wx['r2_teetime'], errors='coerce')

wind_r1_rep, wind_r2_rep, dew_r1_rep, dew_r2_rep = [], [], [], []
for _, row in wx.iterrows():
    wind_r1_rep.append(calculate_avg_wind(row['r1_teetime'], wind_1))
    dew_r1_rep.append(calculate_avg_wind(row['r1_teetime'], dewpoint_1))
    wind_r2_rep.append(calculate_avg_wind(row['r2_teetime'], wind_2))
    dew_r2_rep.append(calculate_avg_wind(row['r2_teetime'], dewpoint_2))

wx['wind_adj_r1'] = np.array(wind_r1_rep) * wind_calculation_report
wx['wind_adj_r2'] = np.array(wind_r2_rep) * wind_calculation_report
wx['dew_adj_r1']  = np.array(dew_r1_rep) * dew_calculation
wx['dew_adj_r2']  = np.array(dew_r2_rep) * dew_calculation

for c in ['wind_adj_r1','wind_adj_r2','dew_adj_r1','dew_adj_r2']:
    wx[c] = wx[c].mean() - wx[c]

wx['wind_adv_r1_2'] = wx['wind_adj_r1'] + wx['wind_adj_r2']
wx['dew_adv_r1_2']  = wx['dew_adj_r1']  + wx['dew_adj_r2']

wx_out = wx[['player_name','dew_adj_r1','wind_adj_r1','dew_adj_r2','wind_adj_r2','dew_adv_r1_2','wind_adv_r1_2']].copy()
wx_out.to_csv(f'weather_impact_{tourney}.csv', index=False)
print(f"[ok] wrote weather_impact_{tourney}.csv")

# ============================================================
# OUTRIGHTS & TOP-N PRICING VS MARKET + BET SHEETS (same outputs as old flow)
# ============================================================

# --- Config for betting outputs ---
EDGE_THRESHOLD_WIN   = 3.0     # minimum edge (%) for WIN market
EDGE_THRESHOLD_TOPN  = 3.0     # minimum edge (%) for Top-N markets
BANKROLL             = 10000.0
KELLY_FRACTION       = 0.25
RETAIL_BOOKS         = ['draftkings','fanduel','betmgm','caesars','barstool','espn','pointsbet','wynnbet','unibet','betway','betfred','betrivers']
BANKROLLS            = {'pinnacle': 10000, 'betcris': 10000, 'betonline': 8000, 'retail': 4000, 'bovada': 3000}

books_to_use = ['betcris', 'betmgm', 'betonline', 'bovada', 'caesars', 'draftkings', 'fanduel', 'pinnacle', 'unibet']

def decimal_to_american(decimal_odds):
    if pd.isna(decimal_odds):
        return np.nan
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(round(-100 / (decimal_odds - 1)))

def fetch_market_data(market_name):
    params = {
        'tour': 'pga',
        'market': market_name,
        'odds_format': 'decimal',
        'file_format': 'json',
        'key': API_KEY
    }
    try:
        r = requests.get(OUTRIGHTS_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[warn] Failed to fetch {market_name}: {e}")
        return {}

def extract_market_rows(json_obj, odds_key='odds'):
    if not isinstance(json_obj, dict):
        return pd.DataFrame()
    
    entries = json_obj.get(odds_key, [])
    if not isinstance(entries, list):
        return pd.DataFrame()

    rows = []
    for entry in entries:
        # Check if entry is actually a dictionary before accessing .get
        if not isinstance(entry, dict):
            continue
            
        player = entry.get('player_name', '')
        if not player:
            continue
        player = player.lower().strip()
        for book in books_to_use:
            odds = entry.get(book)
            if odds is not None:
                rows.append({'player_name': player, 'bookmaker': book, 'decimal_odds': float(odds)})
    return pd.DataFrame(rows)

model_preds['player_name'] = model_preds['player_name'].str.lower().str.strip()
# (Do the same for any other model tables if needed, e.g., topn_df, sim_win_probs)

# Rebuild helpers
sample_df = model_preds[['player_name','sample']].copy() if 'sample' in model_preds.columns else \
            pd.DataFrame({'player_name': model_preds['player_name'], 'sample': np.nan})

pred_join = model_preds[['player_name','r3_pred']].rename(columns={'r3_pred':'my_pred'})
if pred_join['my_pred'].isna().all() and 'my_pred' in model_preds.columns:
    pred_join = model_preds[['player_name','my_pred']].copy()

# --- WIN market ---
data_win = fetch_market_data('win')
win_df = extract_market_rows(data_win)
if not win_df.empty:
    win_merged = pd.merge(win_df, sim_win_probs, on='player_name', how='inner')
    win_merged['implied_prob'] = 1.0 / win_merged['decimal_odds']
    p = win_merged['simulated_win_prob']
    b = win_merged['decimal_odds'] - 1.0
    q = 1.0 - p
    win_merged['edge'] = ((p * b) - q) * 100.0

    # Filter by edge
    win_filtered = win_merged[win_merged['edge'] > EDGE_THRESHOLD_WIN].copy()
    f_star = (b * p - q) / b
    win_filtered['stake'] = (BANKROLL * KELLY_FRACTION * f_star.clip(lower=0)).astype(float)
    win_filtered['eg'] = f_star * win_filtered['edge'] / 2.0
    win_filtered['market_type'] = 'win'
else:
    win_filtered = pd.DataFrame()

# --- Top-N helper ---
def process_topn_market(market, prob_col):
    data = fetch_market_data(market)
    if not data:
        return pd.DataFrame()
        
    df = extract_market_rows(data, odds_key='odds')
    if df.empty:
        return pd.DataFrame()

    # merge model probs
    if prob_col not in topn_df.columns:
        print(f"[warn] {prob_col} not found in topn_df")
        return pd.DataFrame()

    df = df.merge(topn_df[['player_name', prob_col]], on='player_name', how='inner')

    # implieds from market price
    df['implied_prob'] = 1.0 / df['decimal_odds']                # market implied (no-vig)
    df['american_odds'] = df['decimal_odds'].apply(decimal_to_american)

    # ensure p is 0–1
    p = df[prob_col].astype(float)
    if p.max() > 1.0:
        p = p / 100.0

    # edge & sizing
    b = df['decimal_odds'] - 1.0
    q = 1.0 - p
    df['edge'] = ((p * b) - q) * 100.0
    df = df[df['edge'] > EDGE_THRESHOLD_TOPN].copy()
    if df.empty:
        return df

    f_star = (b * p - q) / b
    df['stake'] = (BANKROLL * KELLY_FRACTION * f_star.clip(lower=0)).astype(float)
    df['eg'] = f_star * df['edge'] / 2.0
    df['market_type'] = market

    # your model fair (for display)
    df['my_fair'] = p.apply(prob_to_american)
    return df


top5_bets  = process_topn_market('top_5',  'top_5')
top10_bets = process_topn_market('top_10', 'top_10')
top20_bets = process_topn_market('top_20', 'top_20')

# Combine all candidate bets (Filter empty DFs first)
frames_to_concat = [df for df in [win_filtered, top5_bets, top10_bets, top20_bets] if not df.empty]
if frames_to_concat:
    combined_finish_df = pd.concat(frames_to_concat, ignore_index=True)
else:
    combined_finish_df = pd.DataFrame()

combined_finish_df.to_csv('finish_test.csv')

# Keep only best price per player/market (highest decimal odds)
if not combined_finish_df.empty:
    combined_finish_df = (
        combined_finish_df.sort_values(['player_name','market_type','decimal_odds'], ascending=[True, True, False])
        .drop_duplicates(subset=['player_name','market_type'], keep='first')
    )

    # Add metadata: sample + my_pred
    combined_finish_df = (
        combined_finish_df
        .merge(sample_df, on='player_name', how='left')
        .merge(pred_join, on='player_name', how='left')
    )

    # Light filter like old flow (tweak as needed)
    combined_finish_df = combined_finish_df[
        (combined_finish_df['sample'].fillna(0) >= 0) &
        (combined_finish_df['my_pred'].fillna(0) >= -1.0)
    ].copy()

    # Present odds shapes
    combined_finish_df['american_odds'] = combined_finish_df['decimal_odds'].apply(decimal_to_american)
    def pick_my_fair(row):
        if row['market_type'] == 'win':
            return prob_to_american(row['simulated_win_prob'])
        elif row['market_type'] == 'top_5':
            return prob_to_american(row['top_5'])
        elif row['market_type'] == 'top_10':
            return prob_to_american(row['top_10'])
        elif row['market_type'] == 'top_20':
            return prob_to_american(row['top_20'])
        return np.nan
    combined_finish_df['my_fair'] = combined_finish_df.apply(pick_my_fair, axis=1)

    # Collapse books to one row per player/market/price (aggregate identical best prices)
    output_df = (
        combined_finish_df
        .groupby(['player_name','market_type','decimal_odds'])
        .agg({
            'bookmaker': lambda x: ', '.join(sorted(set(x))),
            'american_odds': 'first',
            'my_fair': 'first',
            'stake': lambda x: round(float(x.iloc[0]), 2),
            'sample': lambda s: next((v for v in s if pd.notna(v) and str(v).strip()!=''), np.nan),
            'my_pred': 'first',
            'edge': 'first',
            'eg': 'first'
        })
        .reset_index()
    )

    output_df.rename(columns={'market_type':'market','bookmaker':'book'}, inplace=True)

    # Book groups & bankroll scaling
    def classify_book(book_str):
        books = [b.strip().lower() for b in str(book_str).split(',')]
        if any(b == 'betonline' for b in books): return 'betonline'
        if any(b == 'betcris'   for b in books): return 'betcris'
        if any(b == 'bovada'    for b in books): return 'bovada'
        if any(b == 'pinnacle'  for b in books): return 'pinnacle'
        if any(b in RETAIL_BOOKS for b in books): return 'retail'
        return None

    output_df['book_group'] = output_df['book'].apply(classify_book)
    output_df['bookroll']   = output_df['book_group'].map(BANKROLLS).fillna(BANKROLL)
    # Share by edge gain (eg)
    eg_total = output_df['eg'].sum()
    output_df['eg_share'] = np.where(eg_total > 0, output_df['eg'] / eg_total, 0.0)

    # Scale stake by book group bankroll; cap at 15% of bucket
    output_df['size_grouped'] = output_df['stake'] * (output_df['bookroll'] / BANKROLL)
    output_df['size_grouped'] = np.minimum(output_df['size_grouped'], 0.15 * output_df['bookroll'])
    output_df['size'] = output_df['size_grouped'].round(2)

    # Rank within player/market by size
    output_df['rank'] = output_df.groupby(['player_name','market'])['size'].rank(method='max', ascending=False)
    # After: output_df['rank'] = ...
    fb = pd.read_csv(fallback_path)
    fb['player_name'] = (
        fb['player_name'].astype(str).str.lower().str.strip().replace(name_replacements)
    )
    sample_map = fb.set_index('player_name')['sample']

    # map by a normalized key, but keep original names intact
    _key = output_df['player_name'].astype(str).str.lower().str.strip().replace(name_replacements)
    output_df['sample'] = _key.map(sample_map).combine_first(output_df['sample'])
    del _key


    # Keep pre-filter copy for ALL workbook
    output_df_all = output_df.copy()

    # Optional filter: like old flow, drop non-win bets when my_pred < 1
    sharp_books = ['betcris','pinnacle','betonline']
    pattern = '|'.join(sharp_books)
    sharp_df = output_df[output_df['book'].str.contains(pattern, case=False, na=False)].copy()
    # sharp_df = sharp_df[~((sharp_df['my_pred'] < 1) & (sharp_df['market'] != 'win'))].copy()
    # output_df = output_df[~((output_df['my_pred'] < 1) & (output_df['market'] != 'win'))].copy()

    # --- Save outputs (same structure/names as old flow) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_dir  = os.path.join(r"C:\Users\mckin\OneDrive", tourney)
    finish_dir = os.path.join(base_dir, "finish_pos")
    os.makedirs(finish_dir, exist_ok=True)

    if not output_df.empty:
        output_df.to_csv(os.path.join(finish_dir, f"positions_{tourney}_{timestamp}.csv"), index=False)
    if not sharp_df.empty:
        sharp_df.sort_values("eg", ascending=False).assign(size=lambda d: d['size'].round(2)) \
                .to_csv(os.path.join(finish_dir, f"sharp_pos_{tourney}.csv"), index=False)

    # Excel workbooks (grouped by book group)
    def write_grouped_workbook(df, path):
        grouped = df.drop(columns=['decimal_odds'], errors='ignore').copy()
        combined = grouped.assign(bet_key=lambda d: d['player_name'] + ' | ' + d['market'])
        combined['eg_rank'] = combined.groupby('bet_key')['eg'].rank(method='first', ascending=False)
        top_edge = combined[combined['eg_rank'] == 1].drop(columns='eg_rank')
        with pd.ExcelWriter(path) as writer:
            for book_group, g in grouped.groupby('book_group'):
                sheet = str(book_group)[:31] if book_group is not None else 'Unknown'
                g.sort_values("size", ascending=False).to_excel(writer, sheet_name=sheet, index=False)
            top_edge.sort_values("size", ascending=False).to_excel(writer, sheet_name="Best_price", index=False)

    write_grouped_workbook(output_df,     os.path.join(finish_dir, f"grouped_bankroll_{tourney}.xlsx"))
    write_grouped_workbook(output_df_all, os.path.join(finish_dir, f"grouped_bankroll_ALL_{tourney}.xlsx"))

else:
    print("[warn] No valid finish positions bets found to process.")


# ======================
# Per-player expected SG summaries (categories + total, per-round averages only)
# ======================
COLS = ["ott", "app", "arg", "putt"]
r1m = cats_r1.mean(axis=1); r2m = cats_r2.mean(axis=1); r3m = cats_r3.mean(axis=1); r4m = cats_r4.mean(axis=1)
r1_total_mean = sg_r1.mean(axis=1); r2_total_mean = sg_r2.mean(axis=1)
r3_total_mean = sg_r3.mean(axis=1); r4_total_mean = sg_r4.mean(axis=1)
per_round_avg_cat = ((cats_r1 + cats_r2 + cats_r3 + cats_r4) / 4.0).mean(axis=1)
tourn_total_per_round_mean = ((sg_r1 + sg_r2 + sg_r3 + sg_r4) / 4.0).mean(axis=1)

rows = []
for i, p in enumerate(player_names):
    row = {"player_name": p}
    for k, col in enumerate(COLS):
        row[f"r1_{col}_mean"] = float(r1m[i, k])
        row[f"r2_{col}_mean"] = float(r2m[i, k])
        row[f"r3_{col}_mean"] = float(r3m[i, k])
        row[f"r4_{col}_mean"] = float(r4m[i, k])
        row[f"tourn_{col}_per_round_mean"] = float(per_round_avg_cat[i, k])
    row["r1_total_mean"] = float(r1_total_mean[i])
    row["r2_total_mean"] = float(r2_total_mean[i])
    row["r3_total_mean"] = float(r3_total_mean[i])
    row["r4_total_mean"] = float(r4_total_mean[i])
    row["tourn_total_sg_per_round_mean"] = float(tourn_total_per_round_mean[i])
    rows.append(row)

df_avg = pd.DataFrame(rows)
out_avg_file = f"avg_expected_cat_sg_{tourney}.csv"
df_avg.to_csv(out_avg_file, index=False)
print(f"[ok] wrote {out_avg_file}")

# ============================================================
# MATCHUPS PRICING (tournament matchups) + weather impact CSV
# ============================================================

# --- Weather impact report (independent of sim waves) ---
# We recompute wind/dew with report factors (wind_calculation_report, dew_calculation)
wx = model_preds[['player_name', 'r1_teetime', 'r2_teetime', 'my_pred']].copy()
wx['r1_teetime'] = pd.to_datetime(wx['r1_teetime'], errors='coerce')
wx['r2_teetime'] = pd.to_datetime(wx['r2_teetime'], errors='coerce')

wind_r1_rep, wind_r2_rep, dew_r1_rep, dew_r2_rep = [], [], [], []
for _, row in wx.iterrows():
    wind_r1_rep.append(calculate_avg_wind(row['r1_teetime'], wind_1))
    dew_r1_rep.append(calculate_avg_wind(row['r1_teetime'], dewpoint_1))
    wind_r2_rep.append(calculate_avg_wind(row['r2_teetime'], wind_2))
    dew_r2_rep.append(calculate_avg_wind(row['r2_teetime'], dewpoint_2))

wx['wind_adj_r1'] = np.array(wind_r1_rep) * wind_calculation_report
wx['wind_adj_r2'] = np.array(wind_r2_rep) * wind_calculation_report
wx['dew_adj_r1']  = np.array(dew_r1_rep) * dew_calculation
wx['dew_adj_r2']  = np.array(dew_r2_rep) * dew_calculation

# center field means (so waves sum to ~0)
for c in ['wind_adj_r1','wind_adj_r2','dew_adj_r1','dew_adj_r2']:
    wx[c] = wx[c].mean() - wx[c]

wx['wind_adv_r1_2'] = wx['wind_adj_r1'] + wx['wind_adj_r2']
wx['dew_adv_r1_2']  = wx['dew_adj_r1']  + wx['dew_adj_r2']

wx_out = wx[['player_name','dew_adj_r1','wind_adj_r1','dew_adj_r2','wind_adj_r2','dew_adv_r1_2','wind_adv_r1_2']].copy()
wx_out.to_csv(f'weather_impact_{tourney}.csv', index=False)
print(f"[ok] wrote weather_impact_{tourney}.csv")

# --- Pull DataGolf tournament matchups ---
params = {
    'tour': 'pga',
    'market': 'tournament_matchups',
    'odds_format': 'american',
    'file_format': 'json',
    'key': API_KEY
}
try:
    resp = requests.get(MATCHUPS_URL, params=params)
    resp.raise_for_status()
    data_tournament = resp.json()
except Exception as e:
    print(f"[warn] Failed to fetch matchups: {e}")
    data_tournament = {}

# Normalize a helper to canonical player key
def norm_player(s: str) -> str:
    if s is None: return None
    x = str(s).lower().strip()
    return name_replacements.get(x, x)

rows = []
for m in data_tournament.get('match_list', []):
    p1 = norm_player(m.get('p1_player_name'))
    p2 = norm_player(m.get('p2_player_name'))
    ties_handling = m.get('ties', 'unknown')
    for book, odds in m.get('odds', {}).items():
        if book == 'datagolf':
            continue
        rows.append({
            'Player 1': p1,
            'Player 2': p2,
            'Bookmaker': book,
            'P1 Odds': odds.get('p1'),
            'P2 Odds': odds.get('p2'),
            'Datagolf Odds (P1)': m.get('odds', {}).get('datagolf', {}).get('p1'),
            'Datagolf Odds (P2)': m.get('odds', {}).get('datagolf', {}).get('p2'),
            'Ties': ties_handling
        })

df_match = pd.DataFrame(rows).drop_duplicates(subset=['Player 1','Player 2','Bookmaker'], keep='first')
# remove rows with missing names
if not df_match.empty:
    df_match = df_match.dropna(subset=['Player 1','Player 2'])

    # --- Compute matchup probabilities from the SIM integer totals ---
    # Build fast access: player -> index and their final score vectors (int)
    name_to_idx = {p: i for i, p in enumerate(player_names)}
    final_scores_np = final_scores  # shape (n_players, SIMULATIONS), dtype=int

    def get_score_vec(pname):
        idx = name_to_idx.get(pname)
        if idx is None:
            return None
        return final_scores_np[idx]

    p1_probs, p2_probs = [], []
    p1_probs_tl, p2_probs_tl = [], []   # ties-as-loss
    for _, r in df_match.iterrows():
        p1 = r['Player 1']; p2 = r['Player 2']
        s1 = get_score_vec(p1); s2 = get_score_vec(p2)
        if s1 is None or s2 is None:
            p1_probs.append(None); p2_probs.append(None)
            p1_probs_tl.append(None); p2_probs_tl.append(None)
            continue
        wins_p1 = np.sum(s1 < s2)
        wins_p2 = np.sum(s1 > s2)
        ties    = np.sum(s1 == s2)
        total   = float(SIMULATIONS)
        # "ties push" (i.e., condition on no tie)
        denom = max(total - ties, 1.0)
        p1_probs.append(wins_p1 / denom)
        p2_probs.append(wins_p2 / denom)
        # "ties as loss"
        p1_probs_tl.append(wins_p1 / total)
        p2_probs_tl.append(wins_p2 / total)

    df_match['my_odds_p1'] = p1_probs
    df_match['my_odds_p2'] = p2_probs
    df_match['my_odds_p1_ties_loss'] = p1_probs_tl
    df_match['my_odds_p2_ties_loss'] = p2_probs_tl

    # Convert book odds to implied, decimals, edges
    df_match['P1 Odds'] = pd.to_numeric(df_match['P1 Odds'], errors='coerce')
    df_match['P2 Odds'] = pd.to_numeric(df_match['P2 Odds'], errors='coerce')

    dfs_by_book = {bk: df for bk, df in df_match.groupby('Bookmaker', dropna=True)}
    round_var = 'tourn'  # label column header

    # optional: sample & my_pred lookups (for later filters)
    pre_sim_path = f'pre_sim_summary_{tourney}.csv'
    sample_lookup = {}
    if os.path.exists(pre_sim_path):
        try:
            smpl = pd.read_csv(pre_sim_path)
            sample_lookup = dict(zip(smpl['player_name'].str.lower(), smpl['sample']))
        except Exception as e:
            print(f"[warn] could not read {pre_sim_path}: {e}")

    my_pred_lookup = dict(zip(model_preds['player_name'].str.lower(), model_preds['my_pred']))

    # write each bookmaker file similar to your previous script
    for bookmaker, dfb in dfs_by_book.items():
        dfb = dfb.copy()
        dfb['p1_implied'] = dfb['P1 Odds'].apply(american_to_implied_probability).round(1)
        dfb['p2_implied'] = dfb['P2 Odds'].apply(american_to_implied_probability).round(1)

        dfb['use_ties_loss'] = (dfb['Ties'] == "separate bet offered")

        # decimal odds
        dfb['p1_decimal_odds'] = np.where(dfb['P1 Odds'] > 0, dfb['P1 Odds'] / 100 + 1, 100 / dfb['P1 Odds'].abs() + 1)
        dfb['p2_decimal_odds'] = np.where(dfb['P2 Odds'] > 0, dfb['P2 Odds'] / 100 + 1, 100 / dfb['P2 Odds'].abs() + 1)

        # expected return edges (%)
        dfb['edge_p1'] = np.where(
            dfb['use_ties_loss'],
            ((dfb['my_odds_p1_ties_loss'] * (dfb['p1_decimal_odds'] - 1)) - (1 - dfb['my_odds_p1_ties_loss'])) * 100,
            ((dfb['my_odds_p1'] * (dfb['p1_decimal_odds'] - 1)) - (1 - dfb['my_odds_p1'])) * 100
        )
        dfb['edge_p2'] = np.where(
            dfb['use_ties_loss'],
            ((dfb['my_odds_p2_ties_loss'] * (dfb['p2_decimal_odds'] - 1)) - (1 - dfb['my_odds_p2_ties_loss'])) * 100,
            ((dfb['my_odds_p2'] * (dfb['p2_decimal_odds'] - 1)) - (1 - dfb['my_odds_p2'])) * 100
        )

        # Fairs (ties push)
        # handle NaNs safely when converting to American odds
        dfb['Fair_p1'] = dfb['my_odds_p1'].apply(lambda p: implied_prob_to_american_odds(p) if pd.notna(p) else None)
        dfb['Fair_p2'] = dfb['my_odds_p2'].apply(lambda p: implied_prob_to_american_odds(p) if pd.notna(p) else None)


        dfb['Round'] = round_var

        # Final column order
        final_cols = ['Player 1','Player 2','Bookmaker','Ties','P1 Odds','P2 Odds','Fair_p1','Fair_p2','edge_p1','edge_p2','Round']
        use_cols = [c for c in final_cols if c in dfb.columns]
        dfb = dfb[use_cols].dropna(subset=['Fair_p1','Fair_p2'])

        out_name = f"{bookmaker}_odds_with_my_odds_tu.csv"
        dfb.drop_duplicates(subset=['Player 1','Player 2','Bookmaker'], keep='first').to_csv(out_name, index=False)
        print(f"[ok] wrote {out_name}")

    # -------- Combine + filter into {tourney}/matchups_ftsimp_*.csv --------
    timestamp = datetime.now().strftime('%H%M')
    tourney_folder = f"./{tourney}"
    os.makedirs(tourney_folder, exist_ok=True)

    csv_files = [f"{bk}_odds_with_my_odds_tu.csv" for bk in dfs_by_book.keys()]
    dfs_list = []
    for fpath in csv_files:
        if os.path.exists(fpath):
            dfs_list.append(pd.read_csv(fpath))
        else:
            print(f"[miss] {fpath}")

    if dfs_list:
        combined_df = pd.concat(dfs_list, ignore_index=True)
        combined_df_raw = combined_df.copy()
        # Add sample sizes (if available)
        combined_df['Sample_P1'] = combined_df['Player 1'].str.lower().map(sample_lookup)
        combined_df['Sample_P2'] = combined_df['Player 2'].str.lower().map(sample_lookup)
        combined_df['sample_on'] = combined_df.apply(
            lambda r: r['Sample_P1'] if r.get('edge_p1', 0) > r.get('edge_p2', 0) else r['Sample_P2'], axis=1
        )

        # Keep decent samples (>=30)
        if 'sample_on' in combined_df.columns:
            combined_df = combined_df[combined_df['sample_on'].fillna(0) >= 30]

        # Add my_pred lookup & edge_on, pred_on, bet_on
        combined_df['my_pred_p1'] = combined_df['Player 1'].str.lower().map(my_pred_lookup)
        combined_df['my_pred_p2'] = combined_df['Player 2'].str.lower().map(my_pred_lookup)
        combined_df['edge_on'] = combined_df[['edge_p1','edge_p2']].max(axis=1)
        # Require the greater of edge_p1/edge_p2 to be > 3 (add right after you set 'edge_on')
        combined_df = combined_df[combined_df['edge_on'] > 3]

        combined_df['pred_on'] = combined_df.apply(
            lambda r: r['my_pred_p1'] if r['edge_p1'] > r['edge_p2'] else r['my_pred_p2'], axis=1
        )
        # add predicted value for the player we are betting against
        combined_df['pred_against'] = combined_df.apply(
            lambda r: r['my_pred_p2'] if r['edge_p1'] > r['edge_p2'] else r['my_pred_p1'],
            axis=1
        )

        combined_df = combined_df[
            ((combined_df['pred_on'] > 0) & (combined_df['edge_on'] > 7)) |
            (combined_df['pred_on'] > 1)
        ]
        combined_df['bet_on'] = combined_df.apply(
            lambda r: r['Player 1'] if r['edge_p1'] > r['edge_p2'] else r['Player 2'], axis=1
        )

        combined_csv_name = f"{tourney_folder}/matchups_ftsimp_{tourney}_{timestamp}.csv"
        combined_df.to_csv(combined_csv_name, index=False)
        print(f"[ok] combined matchups -> {combined_csv_name}")

        # Sharp filter + wind diffs
        sharp_books = ['betonline', 'betcris', 'pinnacle']
        sharp_df = combined_df[combined_df['Bookmaker'].str.lower().isin(sharp_books)].copy()

        # Weather advantages for sides
        wind_lookup = dict(zip(wx['player_name'].str.lower(), wx['wind_adv_r1_2']))
        sharp_df['wind_on'] = sharp_df['bet_on'].str.lower().map(wind_lookup)
        sharp_df['bet_against'] = sharp_df.apply(
            lambda r: r['Player 2'] if r['bet_on'] == r['Player 1'] else r['Player 1'], axis=1
        )
        sharp_df['wind_against'] = sharp_df['bet_against'].str.lower().map(wind_lookup)
        sharp_df['wind_diff'] = sharp_df['wind_on'] - sharp_df['wind_against']

        # Keep only highest edge per matchup_key (order-independent)
        sharp_df['matchup_key'] = sharp_df.apply(
            lambda r: '-'.join(sorted([r['Player 1'].lower(), r['Player 2'].lower()])),
            axis=1
        )
        sharp_df = sharp_df.sort_values('edge_on', ascending=False).drop_duplicates('matchup_key', keep='first')
        sharp_df = sharp_df.drop(columns=['matchup_key', 'Sample_P1', 'Sample_P2', 'my_pred_p1', 'my_pred_p2'], errors='ignore')

        sharp_filename = f"{tourney_folder}/sharp_filtered_{tourney}_{timestamp}.csv"
        sharp_df.to_csv(sharp_filename, index=False)
        print(f"[ok] sharp filtered -> {sharp_filename}")

        # rename a couple files to {book}_{tourney}.csv (compat)
        for bk in ['betcris','betonline']:
            oldf = f"{bk}_odds_with_my_odds_tu.csv"
            newf = f"{bk}_{tourney}.csv"
            if os.path.exists(oldf):
                try:
                    if os.path.exists(newf):
                        os.remove(newf)
                    os.rename(oldf, newf)
                    print(f"[ok] renamed {oldf} -> {newf}")
                except Exception as e:
                    print(f"[warn] rename {oldf} -> {newf}: {e}")
    else:
        print("[note] no bookmaker CSVs found to combine; skipping combined/sharp outputs.")
else:
    print("[warn] No valid tournament matchups found.")

# ============== (Optional) Finish-position distribution widget =============
# You can reuse the Plotly widget from earlier if you want a UI toggle.
# ==========================================================================

# ============================================================
# SENSITIVITY ANALYSIS: Edge per unit SG
# Drop this at the end of your sim script after matchups are computed
# ============================================================

from scipy.stats import linregress

print("\n" + "="*60)
print("SENSITIVITY ANALYSIS: How much does edge move per unit pred?")
print("="*60)

# --- MATCHUP SENSITIVITY ---
if 'combined_df_raw' in dir() and not combined_df_raw.empty:
    
    # Get average edge per player (across all their matchups)
    # We want edge for the player, not just "edge_on" which is the bet side
    p1_edges = combined_df_raw[['Player 1', 'edge_p1']].rename(
        columns={'Player 1': 'player_name', 'edge_p1': 'edge'}
    )
    p2_edges = combined_df_raw[['Player 2', 'edge_p2']].rename(
        columns={'Player 2': 'player_name', 'edge_p2': 'edge'}
    )
    all_mu_edges = pd.concat([p1_edges, p2_edges], ignore_index=True)
    all_mu_edges['player_name'] = all_mu_edges['player_name'].str.lower().str.strip()
    
    avg_mu_edge = all_mu_edges.groupby('player_name')['edge'].mean().reset_index()
    avg_mu_edge.columns = ['player_name', 'avg_mu_edge']
    
    # Merge with predictions
    sens_mu = avg_mu_edge.merge(
        model_preds[['player_name', 'my_pred']], 
        on='player_name', 
        how='inner'
    )
    
    if len(sens_mu) >= 10:
        slope_mu, intercept_mu, r_mu, p_mu, se_mu = linregress(
            sens_mu['my_pred'], 
            sens_mu['avg_mu_edge']
        )
        
        print(f"\n[MATCHUPS]")
        print(f"  Players analyzed: {len(sens_mu)}")
        print(f"  Sensitivity: {slope_mu:.2f}% edge per 1.0 SG")
        print(f"  Intercept: {intercept_mu:.2f}% (edge when pred=0)")
        print(f"  R-squared: {r_mu**2:.3f}")
        print(f"  To zero a 5% edge, adjust pred by: {-5/slope_mu:.2f} SG")
    else:
        print(f"\n[MATCHUPS] Not enough data ({len(sens_mu)} players)")
        slope_mu = None

# --- FINISH POSITION SENSITIVITY (win/top5/top10/top20) ---
if 'combined_finish_df' in dir() and not combined_finish_df.empty:
    
    # Use the pre-filtered combined_finish_df which has edges
    finish_sens = combined_finish_df.copy()
    finish_sens['player_name'] = finish_sens['player_name'].str.lower().str.strip()
    
    # Average edge across all markets per player
    avg_finish_edge = finish_sens.groupby('player_name')['edge'].mean().reset_index()
    avg_finish_edge.columns = ['player_name', 'avg_finish_edge']
    
    sens_fin = avg_finish_edge.merge(
        model_preds[['player_name', 'my_pred']], 
        on='player_name', 
        how='inner'
    )
    
    if len(sens_fin) >= 10:
        slope_fin, intercept_fin, r_fin, p_fin, se_fin = linregress(
            sens_fin['my_pred'], 
            sens_fin['avg_finish_edge']
        )
        
        print(f"\n[FINISH POSITIONS]")
        print(f"  Players analyzed: {len(sens_fin)}")
        print(f"  Sensitivity: {slope_fin:.2f}% edge per 1.0 SG")
        print(f"  Intercept: {intercept_fin:.2f}% (edge when pred=0)")
        print(f"  R-squared: {r_fin**2:.3f}")
        print(f"  To zero a 5% edge, adjust pred by: {-5/slope_fin:.2f} SG")
    else:
        print(f"\n[FINISH POSITIONS] Not enough data ({len(sens_fin)} players)")
        slope_fin = None

# --- BUCKET ANALYSIS (by pred tier) ---
print(f"\n" + "-"*60)
print("BUCKET ANALYSIS: Edge by prediction tier")
print("-"*60)

if 'sens_mu' in dir() and len(sens_mu) >= 10:
    sens_mu['pred_bucket'] = pd.cut(
        sens_mu['my_pred'],
        bins=[-3, -0.5, 0, 0.5, 1.0, 1.5, 4],
        labels=['< -0.5', '-0.5 to 0', '0 to 0.5', '0.5 to 1.0', '1.0 to 1.5', '> 1.5']
    )
    
    bucket_mu = sens_mu.groupby('pred_bucket', observed=True).agg({
        'my_pred': ['mean', 'count'],
        'avg_mu_edge': ['mean', 'std']
    }).round(2)
    bucket_mu.columns = ['avg_pred', 'n', 'avg_edge', 'edge_std']
    
    print(f"\n[MATCHUPS by tier]")
    print(bucket_mu.to_string())
    
    # Flag buckets where you're showing negative avg edge (market is better)
    bad_buckets = bucket_mu[bucket_mu['avg_edge'] < 0]
    if not bad_buckets.empty:
        print(f"\n  ⚠️  Negative avg edge in: {list(bad_buckets.index)}")
        print(f"      Consider heavier regression in these tiers.")

# --- SUGGESTED REGRESSION MULTIPLIERS ---
print(f"\n" + "-"*60)
print("SUGGESTED REGRESSION MULTIPLIERS")
print("-"*60)

if 'slope_mu' in dir() and slope_mu is not None and slope_mu != 0:
    # Base idea: if you want to take X% of the "zero edge" adjustment
    # And you know sensitivity, you can back out multipliers
    
    print(f"\nBased on matchup sensitivity of {slope_mu:.2f}% per SG:")
    print(f"  - To regress 25% toward market: multiply tot_rgrs by {0.25 * (8/slope_mu):.2f}")
    print(f"  - To regress 50% toward market: multiply tot_rgrs by {0.50 * (8/slope_mu):.2f}")
    print(f"  - To regress 75% toward market: multiply tot_rgrs by {0.75 * (8/slope_mu):.2f}")
    
    print(f"\nPractical tier multipliers (adjust based on bucket analysis above):")
    print(f"  pred >= 0.75:  0.75  (trust your model)")
    print(f"  pred >= 0.5:   1.00  (neutral)")
    print(f"  pred >= 0:     1.50  (regress harder)")
    print(f"  pred < 0:      2.00  (regress heavily)")

# --- SAVE SENSITIVITY DATA ---
sensitivity_output = {
    'tourney': tourney,
    'matchup_sensitivity': slope_mu if 'slope_mu' in dir() else None,
    'matchup_r2': r_mu**2 if 'r_mu' in dir() else None,
    'finish_sensitivity': slope_fin if 'slope_fin' in dir() else None,
    'finish_r2': r_fin**2 if 'r_fin' in dir() else None,
    'n_matchup_players': len(sens_mu) if 'sens_mu' in dir() else 0,
    'n_finish_players': len(sens_fin) if 'sens_fin' in dir() else 0,
}

sens_out_df = pd.DataFrame([sensitivity_output])
sens_out_df.to_csv(f'sensitivity_analysis_{tourney}.csv', index=False)
print(f"\n[ok] wrote sensitivity_analysis_{tourney}.csv")

print("\n" + "="*60)
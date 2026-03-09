# ============================
# PGA SIM: Category-First Draws with Course Variance Multipliers
# Draws SG categories from course-adjusted multivariate normal, sums to total.
# ============================

import os
import numpy as np
import pandas as pd
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from numpy.linalg import cholesky

# --- Weekly-changing config from Google Sheet ---
from sheet_config import load_config
_cfg = load_config()
tourney          = _cfg["tourney"]
SIMULATIONS      = _cfg["simulations"]
STD_DEV          = _cfg["std_dev"]
PAR              = 72  # constant — doesn't affect relative rankings
CUT_LINE         = _cfg["cut_line"]
USE_10_SHOT_RULE = _cfg["use_10_shot_rule"]
WIND_FACTOR_SIM  = _cfg["wind_override"]  # course-specific wind coefficient (auto-set by scoring_baseline)
wind_1           = _cfg["wind_r1"]
wind_2           = _cfg["wind_r2"]
dewpoint_1       = _cfg["dew_r1"]
dewpoint_2       = _cfg["dew_r2"]
dew_calculation  = _cfg["dew_calculation"]
COURSE_CAT_MULTS = _cfg["course_cat_mults"]
COURSE_CAT_SKEW  = _cfg["course_cat_skew"]
TOP_K            = 20  # hardcode — never changes
_event_id        = _cfg["event_id"]

# --- Stable model params from sim_inputs ---
from sim_inputs import (
    name_replacements,
    # R1 update sets
    coefficients_r1_high, coefficients_r1_midh, coefficients_r1_midl, coefficients_r1_low,
    # R2 update sets (pos buckets)
    coefficients_r2, coefficients_r2_6_30, coefficients_r2_30_up,
    # R3 update sets (avg SG only; no residual terms)
    coefficients_r3, coefficients_r3_mid, coefficients_r3_high,
)

# Tour-wide baseline skewness (PGA, 2019-2025 — stable, update rarely)
BASELINE_CAT_SKEW = {
    'sg_ott': -0.93,
    'sg_app': -0.21,
    'sg_arg': -0.18,
    'sg_putt': -0.05,
}

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("  Tournament Sim — Category-First")
print("=" * 60)

# Matchup weather-impact report settings (doesn't affect sim)
wind_calculation_report = WIND_FACTOR_SIM

# Weather delta distribution across categories [OTT, APP, ARG, PUTT]
WEATHER_CAT_SPLIT = np.array([0.35, 0.35, 0.15, 0.15])

CAT_ORDER = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
CLIP_CAT = (-8.0, 8.0)
RNG = np.random.default_rng(456)  # different seed from v1 for independent draws

# DataGolf API
API_KEY = os.getenv('DATAGOLF_API_KEY', 'c05ee5fd8f2f3b14baab409bd83c')
MATCHUPS_URL = "https://feeds.datagolf.com/betting-tools/matchups"
OUTRIGHTS_URL = "https://feeds.datagolf.com/betting-tools/outrights"

# Email config
EMAIL_FROM = os.getenv("EMAIL_USER")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Tournament sim email filter thresholds
EMAIL_MU_MIN_PRED = 0.75
EMAIL_MU_MIN_SAMPLE = 20
EMAIL_FP_MIN_PRED = 0.0

# Sharp books for matchup filtering
SHARP_BOOKS = ["pinnacle", "betonline", "betcris"]
HALF_SHOT_ADJ = {"betonline": 25, "betcris": 30}

# Input predictions
_final_pred_path = f"final_predictions_{tourney}.csv"
_pre_course_path = f"pre_course_fit_{tourney}.csv"
PRED_PATH = _final_pred_path if os.path.exists(_final_pred_path) else _pre_course_path
print(f"[info] Using predictions from: {PRED_PATH}")

# Per-player category distribution file (course-shaped)
DISTS_FILE = "this_week_dists_v2.csv"

# Tour-level category correlation fallbacks
CORR_PREFS = [
    "permanent_data/sg_cat_corr_tour_within_player_pearson.csv",
    "permanent_data/sg_cat_corr_tour_spearman.csv",
    "permanent_data/sg_cat_corr_tour_pearson.csv",
]

# Course multipliers
course_mults = np.array([COURSE_CAT_MULTS.get(c, 1.0) for c in CAT_ORDER])
print(f"[info] Course category multipliers: OTT={course_mults[0]:.3f}, APP={course_mults[1]:.3f}, "
      f"ARG={course_mults[2]:.3f}, PUTT={course_mults[3]:.3f}")

# Course skewness — use course-specific if available, else tour-wide baseline
course_skew = np.array([
    COURSE_CAT_SKEW.get(c, BASELINE_CAT_SKEW.get(c, 0.0)) for c in CAT_ORDER
])
print(f"[info] Course category skewness:    OTT={course_skew[0]:+.2f}, APP={course_skew[1]:+.2f}, "
      f"ARG={course_skew[2]:+.2f}, PUTT={course_skew[3]:+.2f}")


def _cf_calibration_multiplier(gamma):
    """
    Correction multiplier for first-order Cornish-Fisher saturation.

    The CF expansion underdelivers skewness for |gamma| > ~0.5.
    This polynomial fit (R²>0.999 on grid from 0 to -2.0) maps
    target skewness to the input gamma needed to realize it exactly.
    """
    ag = abs(gamma)
    if ag < 0.2:
        return 1.0
    return 1.0 + 0.0234 * ag**2 + 0.0125 * ag**3


def _apply_skew(z, gamma):
    """
    Apply Cornish-Fisher skewness to standard-normal draws.

    Uses first-order CF expansion with calibration correction so that
    realized skewness matches the target (compensates for CF saturation
    at |gamma| > ~0.5).

    Args:
        z: array of draws (any shape, transformation applied elementwise)
        gamma: target skewness (negative = left-skewed)

    Returns:
        Transformed draws with mean ~0, variance ~1, skewness ~gamma.
    """
    if abs(gamma) < 0.01:
        return z
    # Overshoot input to compensate for CF first-order saturation
    gamma_adj = gamma * _cf_calibration_multiplier(gamma)
    z_skewed = z + (gamma_adj / 6.0) * (z ** 2 - 1.0)
    # Variance correction: Var(z') = 1 + gamma_adj^2/18
    z_skewed /= np.sqrt(1.0 + gamma_adj ** 2 / 18.0)
    return z_skewed

# Master toggle: use OTT-based in-tournament adjustments or not
USE_IN_TOURN_OTT = True


# --- Helpers ---
def parse_time(teetime):
    if pd.isnull(teetime):
        return None
    if isinstance(teetime, pd.Timestamp):
        return teetime.to_pydatetime()
    if isinstance(teetime, datetime):
        return teetime
    if isinstance(teetime, (int, float)) and (pd.isna(teetime) or teetime == 0):
        return None
    s = str(teetime).strip()
    if s == "":
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%I:%M%p", "%m/%d/%Y %H:%M"]:
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
    if pd.isna(p) or p <= 0: return None
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

def rank_positions_from_strokes(strokes_asc_int):
    s = pd.Series(strokes_asc_int)
    return s.rank(method='min').astype(int).to_numpy()

def coeff_vec_r1(cdict):
    return np.array([cdict['ott'], 0.0, 0.0, cdict['putt'], cdict['residual'], cdict['residual2']], dtype=float)

def ensure_array(x, shape):
    return x if isinstance(x, np.ndarray) else np.zeros(shape, dtype=float)


# --- Load predictions ---
model_preds = pd.read_csv(PRED_PATH).rename(columns={'pred': 'my_pred'})

model_preds['player_name'] = (
    model_preds['player_name'].astype(str).str.lower().str.strip()
    .replace(name_replacements)
)
model_preds = model_preds.drop_duplicates(subset=['player_name']).reset_index(drop=True)

# Pull sample sizes from pre_course_fit if missing
if 'sample' not in model_preds.columns and os.path.exists(_pre_course_path):
    pcf = pd.read_csv(_pre_course_path, usecols=['player_name', 'sample'])
    pcf['player_name'] = pcf['player_name'].astype(str).str.lower().str.strip().replace(name_replacements)
    model_preds = model_preds.merge(pcf, on='player_name', how='left')

# Replace low-confidence predictions with DG decomposition
from api_utils import fetch_player_decompositions
dg_decomp = fetch_player_decompositions(API_KEY)
if not dg_decomp.empty and 'dg_final_pred' in dg_decomp.columns:
    model_preds = model_preds.merge(dg_decomp[['player_name', 'dg_final_pred']], on='player_name', how='left')
    mask = (model_preds['my_pred'].abs() < 0.5) & model_preds['dg_final_pred'].notna()
    n_replaced = mask.sum()
    if n_replaced > 0:
        replaced = model_preds.loc[mask, ['player_name', 'my_pred', 'dg_final_pred']].copy()
        model_preds.loc[mask, 'my_pred'] = model_preds.loc[mask, 'dg_final_pred']
        print(f"[DG decomp] Replaced {n_replaced} predictions (|my_pred| < 0.5) with DG decomposition:")
        for _, r in replaced.iterrows():
            print(f"  {r['player_name']}: {r['my_pred']:.3f} -> {r['dg_final_pred']:.3f}")
    else:
        print("[DG decomp] No predictions below threshold needed replacement")
    model_preds = model_preds.drop(columns=['dg_final_pred'])

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
    model_preds[col] = model_preds[col] - model_preds[col].mean()

# Weather deltas per player (scalar, already mean-centered)
weather_delta_r1 = (model_preds['wind_adj_r1_sim'] + model_preds['dew_adj_r1_sim']).to_numpy(dtype=float)
weather_delta_r2 = (model_preds['wind_adj_r2_sim'] + model_preds['dew_adj_r2_sim']).to_numpy(dtype=float)

# Round-level means for SIM (used for skill update math — same as v1)
model_preds['r1_pred'] = model_preds['my_pred'] + model_preds['wind_adj_r1_sim'] + model_preds['dew_adj_r1_sim']
model_preds['r2_pred'] = model_preds['my_pred'] + model_preds['wind_adj_r2_sim'] + model_preds['dew_adj_r2_sim']
model_preds['r3_pred'] = model_preds['my_pred']   # no wave
model_preds['r4_pred'] = model_preds['my_pred']   # no wave

# Per-player round stdev (for validation comparison only — not used for draws)
preds = model_preds[['player_name', 'my_pred', 'std_dev', 'r1_pred', 'r2_pred', 'r3_pred', 'r4_pred']].copy()
preds['std'] = (preds['std_dev'] + STD_DEV) / 2.0

player_names = preds['player_name'].tolist()
n_players = len(player_names)

# Sample sizes and pred lookups for email filtering
sample_lookup = dict(zip(
    model_preds['player_name'],
    model_preds['sample'].fillna(0).astype(int)
)) if 'sample' in model_preds.columns else {}

my_pred_lookup = dict(zip(model_preds['player_name'], model_preds['my_pred']))

# --- Load per-player category distributions ---
if not os.path.exists(DISTS_FILE):
    raise FileNotFoundError(f"Missing {DISTS_FILE}. Build it earlier.")

dists = pd.read_csv(DISTS_FILE)
dists['player_name'] = (
    dists['player_name'].astype(str).str.lower().str.strip()
    .replace(name_replacements)
)

need_cols = {'player_name', 'category_clean', 'mean', 'std', 'skew', 'n_eff'}
missing = need_cols - set(dists.columns)
if missing:
    raise ValueError(f"{DISTS_FILE} missing columns: {missing}")

# V2 reads RAW (un-scaled) distributions — course variance scaling is applied
# via COURSE_CAT_MULTS below, so using 'mean'/'std' avoids double-counting.
mu_w   = dists.pivot(index='player_name', columns='category_clean', values='mean')
std_w  = dists.pivot(index='player_name', columns='category_clean', values='std')
skew_w = dists.pivot(index='player_name', columns='category_clean', values='skew')
neff_w = dists.pivot(index='player_name', columns='category_clean', values='n_eff')
global_mu  = dists.groupby('category_clean')['mean'].mean()
global_std = dists.groupby('category_clean')['std'].median()

R = load_corr_matrix(CAT_ORDER)
try:
    L_corr = cholesky(R)
except np.linalg.LinAlgError:
    R = 0.95*R + 0.05*np.eye(4)
    L_corr = cholesky(R)

# --- Build course-adjusted player params ---
# Re-center category means so they sum to my_pred (preserves relative proportions,
# changes only the variance structure — not the base predictions)
my_pred_series = preds.set_index('player_name')['my_pred']

player_params_v2 = []
recenter_diffs = []
for p in player_names:
    mu_row  = mu_w.loc[p].reindex(CAT_ORDER) if p in mu_w.index else pd.Series(index=CAT_ORDER, dtype=float)
    std_row = std_w.loc[p].reindex(CAT_ORDER) if p in std_w.index else pd.Series(index=CAT_ORDER, dtype=float)

    mu  = mu_row.fillna(global_mu.reindex(CAT_ORDER)).to_numpy(dtype=float)
    std = std_row.fillna(global_std.reindex(CAT_ORDER)).to_numpy(dtype=float).clip(1e-6)

    # Re-center: shift each category equally so sum(mu) == my_pred
    cat_sum = mu.sum()
    target = my_pred_series.loc[p]
    shift = (target - cat_sum) / 4.0
    mu = mu + shift
    recenter_diffs.append(cat_sum - target)

    # Apply course variance multipliers
    std_course = std * course_mults
    player_params_v2.append((mu, std_course))

recenter_diffs = np.array(recenter_diffs)
print(f"[recenter] Category means re-centered to my_pred: "
      f"mean shift={recenter_diffs.mean():.3f}, max |shift|={np.abs(recenter_diffs).max():.3f}, "
      f"std={recenter_diffs.std():.3f}")

# --- Per-player effective skewness (confidence-weighted blend) ---
SKEW_BLEND_MAX = 0.5       # max weight on player skew (at full confidence)
SKEW_CONFIDENCE_N = 100.0   # n_eff at which player reaches full confidence

effective_skew = np.zeros((n_players, 4), dtype=float)
for i, p in enumerate(player_names):
    for j, cat in enumerate(CAT_ORDER):
        # Player-specific skew (fall back to 0 if missing)
        p_skew = (skew_w.at[p, cat]
                  if p in skew_w.index and cat in skew_w.columns
                     and pd.notna(skew_w.at[p, cat])
                  else 0.0)
        # Effective sample size
        p_neff = (neff_w.at[p, cat]
                  if p in neff_w.index and cat in neff_w.columns
                     and pd.notna(neff_w.at[p, cat])
                  else 0.0)
        # Confidence ramp: 0 → SKEW_BLEND_MAX as n_eff → SKEW_CONFIDENCE_N
        confidence = min(p_neff / SKEW_CONFIDENCE_N, 1.0)
        blend_w = SKEW_BLEND_MAX * confidence
        effective_skew[i, j] = (1 - blend_w) * course_skew[j] + blend_w * p_skew

print(f"[info] Per-player skew blending: max_weight={SKEW_BLEND_MAX}, "
      f"full_confidence_at_n_eff={SKEW_CONFIDENCE_N}")
print(f"[info] Effective skew range: "
      f"min={effective_skew.min():.3f}, max={effective_skew.max():.3f}, "
      f"mean={effective_skew.mean():.3f}")

# Align arrays (same as v1 — needed for skill updates)
indexer = preds.set_index('player_name')
r1_mu = indexer['r1_pred'].reindex(player_names).to_numpy(dtype=float)
r2_mu = indexer['r2_pred'].reindex(player_names).to_numpy(dtype=float)
r3_mu = indexer['r3_pred'].reindex(player_names).to_numpy(dtype=float)
r4_mu = indexer['r4_pred'].reindex(player_names).to_numpy(dtype=float)
my_pred_base = indexer['my_pred'].reindex(player_names).to_numpy(dtype=float)
round_std = indexer['std'].reindex(player_names).to_numpy(dtype=float)  # for validation only

print(f"\n[sim] {n_players} players, {SIMULATIONS:,} simulations")


# ======================
# R1: Category-first draws
# ======================
cats_r1 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
sg_r1 = np.empty((n_players, SIMULATIONS), dtype=float)

for i, (mu, std_c) in enumerate(player_params_v2):
    # Category means shifted by weather delta
    cat_mu = mu + weather_delta_r1[i] * WEATHER_CAT_SPLIT
    Z = RNG.standard_normal(size=(SIMULATIONS, 4))
    corr_z = Z @ L_corr.T                          # correlated unit-variance draws
    for j in range(4):                              # apply per-player skewness
        corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[i, j])
    draws = cat_mu + corr_z * std_c                 # scale by per-player category stds
    cats_r1[i] = np.clip(draws, CLIP_CAT[0], CLIP_CAT[1])
    sg_r1[i] = cats_r1[i].sum(axis=1)

strokes_r1 = np.rint(PAR - sg_r1).astype(int)

# --- Validation: R1 ---
r1_total_std_realized = np.std(sg_r1, axis=1).mean()
r1_cat_stds_realized = np.std(cats_r1, axis=1).mean(axis=0)
print(f"\n[check] R1 total std (realized): {r1_total_std_realized:.3f}  (v1 blend would be ~{round_std.mean():.3f})")
print(f"[check] R1 category stds (realized): OTT={r1_cat_stds_realized[0]:.3f}, "
      f"APP={r1_cat_stds_realized[1]:.3f}, ARG={r1_cat_stds_realized[2]:.3f}, PUTT={r1_cat_stds_realized[3]:.3f}")
# Realized skewness check
from scipy.stats import skew as _skew_fn
r1_cat_skew_realized = np.array([_skew_fn(cats_r1[:, :, j].ravel()) for j in range(4)])
print(f"[check] R1 category skew (realized): OTT={r1_cat_skew_realized[0]:+.3f}, "
      f"APP={r1_cat_skew_realized[1]:+.3f}, ARG={r1_cat_skew_realized[2]:+.3f}, PUTT={r1_cat_skew_realized[3]:+.3f}")
print(f"[check] R1 category skew  (targets): OTT={effective_skew[:, 0].mean():+.3f}, "
      f"APP={effective_skew[:, 1].mean():+.3f}, ARG={effective_skew[:, 2].mean():+.3f}, "
      f"PUTT={effective_skew[:, 3].mean():+.3f}")
print(f"[check] R1 mean SG (field): {sg_r1.mean():.4f}  (should be near 0)")


# ======================
# R1 -> R2 skill update (same logic as v1)
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
# R2: Category-first draws with skill update shift
# ======================
cats_r2 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
sg_r2 = np.empty((n_players, SIMULATIONS), dtype=float)

for i, (mu, std_c) in enumerate(player_params_v2):
    # Category means shifted by weather delta
    cat_mu = mu + weather_delta_r2[i] * WEATHER_CAT_SPLIT
    # Skill update shift: distribute evenly across 4 categories
    base_total_mu = mu.sum() + weather_delta_r2[i]
    skill_shift = sg_r2_mean[i] - base_total_mu  # shape (SIMULATIONS,)
    cat_mu_shifted = cat_mu + skill_shift[:, None] / 4.0
    Z = RNG.standard_normal(size=(SIMULATIONS, 4))
    corr_z = Z @ L_corr.T
    for j in range(4):
        corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[i, j])
    draws = cat_mu_shifted + corr_z * std_c
    cats_r2[i] = np.clip(draws, CLIP_CAT[0], CLIP_CAT[1])
    sg_r2[i] = cats_r2[i].sum(axis=1)

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
# R2 -> R3 skill update (position buckets; uses R1+R2 stats)
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
sg_adj_r1 = ensure_array(sg_adj_r1, shape2)
total_adjustment_r2 = (tot_resid_adj_r2 + tot_sg_adj_r2) - sg_adj_r1

updated_skill_r3 = updated_skill_r2 + total_adjustment_r2
sg_r3_mean = updated_skill_r3 + (r3_mu - my_pred_base)[:, None]


# ======================
# R3: Category-first draws (no weather)
# ======================
cats_r3 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
sg_r3 = np.empty((n_players, SIMULATIONS), dtype=float)

for i, (mu, std_c) in enumerate(player_params_v2):
    # No weather delta for R3
    base_total_mu = mu.sum()
    skill_shift = sg_r3_mean[i] - base_total_mu  # shape (SIMULATIONS,)
    cat_mu_shifted = mu + skill_shift[:, None] / 4.0
    Z = RNG.standard_normal(size=(SIMULATIONS, 4))
    corr_z = Z @ L_corr.T
    for j in range(4):
        corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[i, j])
    draws = cat_mu_shifted + corr_z * std_c
    cats_r3[i] = np.clip(draws, CLIP_CAT[0], CLIP_CAT[1])
    sg_r3[i] = cats_r3[i].sum(axis=1)

strokes_r3 = np.rint(PAR - sg_r3).astype(int)
r1_r3_scores = r1_r2_scores + strokes_r3


# ======================
# R3 -> R4 (AVG-SG ONLY; position buckets)
# ======================
avg_ott_r3  = 0.66 * (0.5 * (cats_r1[:, :, 0] + cats_r2[:, :, 0])) + 0.34 * cats_r3[:, :, 0]
avg_app_r3  = 0.66 * (0.5 * (cats_r1[:, :, 1] + cats_r2[:, :, 1])) + 0.34 * cats_r3[:, :, 1]
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
# R4: Category-first draws (no weather)
# ======================
cats_r4 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
sg_r4 = np.empty((n_players, SIMULATIONS), dtype=float)

for i, (mu, std_c) in enumerate(player_params_v2):
    base_total_mu = mu.sum()
    skill_shift = sg_r4_mean[i] - base_total_mu
    cat_mu_shifted = mu + skill_shift[:, None] / 4.0
    Z = RNG.standard_normal(size=(SIMULATIONS, 4))
    corr_z = Z @ L_corr.T
    for j in range(4):
        corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[i, j])
    draws = cat_mu_shifted + corr_z * std_c
    cats_r4[i] = np.clip(draws, CLIP_CAT[0], CLIP_CAT[1])
    sg_r4[i] = cats_r4[i].sum(axis=1)

strokes_r4 = np.rint(PAR - sg_r4).astype(int)

# Missed-cut penalty for R3+R4
r3_r4 = strokes_r3 + strokes_r4
r3_r4[~made_cut_mask] = 200

# Final integer 72-hole totals
final_scores = r1_r2_scores + r3_r4


# ======================
# Validation checks
# ======================
print(f"\n{'='*50}")
print("  CATEGORY-FIRST VALIDATION CHECKS")
print(f"{'='*50}")

# 1. Total std dev per round
for rnd, sg_arr, label in [(1, sg_r1, "R1"), (2, sg_r2, "R2"), (3, sg_r3, "R3"), (4, sg_r4, "R4")]:
    std_realized = np.std(sg_arr, axis=1).mean()
    print(f"  {label} total std (realized avg): {std_realized:.3f}")

# 2. Category stds vs expected
print(f"\n  Category stds (R1 realized vs input*course_mult):")
for k, cat in enumerate(["OTT", "APP", "ARG", "PUTT"]):
    realized = np.std(cats_r1[:, :, k], axis=1).mean()
    expected_avg = np.mean([p[1][k] for p in player_params_v2])
    print(f"    {cat}: realized={realized:.3f}, expected_avg={expected_avg:.3f}, ratio={realized/expected_avg:.3f}")

# 3. Win prob sanity
simulated_winners_v2 = []
for j in range(SIMULATIONS):
    sc = final_scores[:, j]
    min_score = sc.min()
    tied = np.where(sc == min_score)[0]
    winner_idx = RNG.choice(tied)
    simulated_winners_v2.append(player_names[winner_idx])

win_counts = pd.Series(simulated_winners_v2).value_counts(normalize=True)
print(f"\n  Win prob sum: {win_counts.sum():.4f} (should be 1.0)")
print(f"  Top winner: {win_counts.index[0]} at {win_counts.iloc[0]*100:.1f}%")
if win_counts.iloc[0] > 0.25:
    print(f"  [WARN] Top win prob > 25% — may indicate variance too low")

# 4. Mean SG near 0
print(f"\n  Mean field SG: R1={sg_r1.mean():.4f}, R2={sg_r2.mean():.4f}, R3={sg_r3.mean():.4f}, R4={sg_r4.mean():.4f}")

# 5. Category correlations
r1_flat = cats_r1.reshape(-1, 4)
sample_idx = RNG.choice(r1_flat.shape[0], size=min(100000, r1_flat.shape[0]), replace=False)
r1_sample = r1_flat[sample_idx]
realized_corr = np.corrcoef(r1_sample.T)
print(f"\n  Realized category correlations (R1 sample):")
cat_labels = ["OTT", "APP", "ARG", "PUTT"]
for i_c in range(4):
    for j_c in range(i_c+1, 4):
        print(f"    {cat_labels[i_c]}-{cat_labels[j_c]}: realized={realized_corr[i_c,j_c]:.4f}, input={R[i_c,j_c]:.4f}")

print(f"{'='*50}\n")


# ======================
# Markets (WIN; Top-5/10/20 with dead-heat)
# ======================
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

print(f"[ok] Sim complete for {tourney}.")
print(f"  Players: {n_players}, Sims: {SIMULATIONS}")
print(f"  Outputs: simulated_probs.csv, top_finish_probs_{tourney}.csv, finish_equity_{tourney}.csv")

# Per-player expected SG summaries
COLS = ["ott", "app", "arg", "putt"]
r1m = cats_r1.mean(axis=1); r2m = cats_r2.mean(axis=1); r3m = cats_r3.mean(axis=1); r4m = cats_r4.mean(axis=1)
r1_total_mean = sg_r1.mean(axis=1); r2_total_mean = sg_r2.mean(axis=1)
r3_total_mean = sg_r3.mean(axis=1); r4_total_mean = sg_r4.mean(axis=1)
per_round_avg_cat = ((cats_r1 + cats_r2 + cats_r3 + cats_r4) / 4.0).mean(axis=1)
tourn_total_per_round_mean = ((sg_r1 + sg_r2 + sg_r3 + sg_r4) / 4.0).mean(axis=1)

rows_avg = []
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
    rows_avg.append(row)

df_avg = pd.DataFrame(rows_avg)
df_avg.to_csv(f"avg_expected_cat_sg_{tourney}.csv", index=False)

# Dead-heat adjusted rank probabilities (ties split fractional credit across positions)
long_df['tie_count'] = long_df.groupby(['simulation_id', 'rank'])['player_name'].transform('count')

no_tie = long_df[long_df['tie_count'] == 1][['player_name', 'rank']].copy()
no_tie['weight'] = 1.0

ties = long_df[long_df['tie_count'] > 1].copy()
if not ties.empty:
    expanded_parts = []
    for tc_val in ties['tie_count'].unique():
        tc_int = int(tc_val)
        sub = ties[ties['tie_count'] == tc_val][['player_name', 'rank']].copy()
        for offset in range(tc_int):
            part = sub.copy()
            part['rank'] = sub['rank'].astype(int) + offset
            part['weight'] = 1.0 / tc_int
            expanded_parts.append(part)
    all_rank_rows = pd.concat([no_tie] + expanded_parts, ignore_index=True)
else:
    all_rank_rows = no_tie

rank_probs = (
    all_rank_rows.groupby(['player_name', 'rank'])['weight']
    .sum()
    .div(SIMULATIONS)
    .rename('prob')
    .reset_index()
)
rank_probs_updated = rank_probs.rename(columns={'prob': 'prob_u'}).copy()
rank_probs_updated['rank'] = rank_probs_updated['rank'].astype(int)
rank_probs_updated.to_parquet(f"rank_probs_updated_{tourney}.parquet", index=False)
print(f"[ok] wrote rank_probs_updated_{tourney}.parquet")


# ============================================================
# OUTRIGHTS & TOP-N PRICING VS MARKET
# ============================================================

EDGE_THRESHOLD_WIN   = 2.0
EDGE_THRESHOLD_TOPN  = 2.0
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
    win_merged['edge'] = (p - win_merged['implied_prob']) * 100.0
    win_filtered = win_merged[win_merged['edge'] > 0].copy()
    f_star = (b * p - q) / b
    win_filtered['stake'] = (BANKROLL * KELLY_FRACTION * f_star.clip(lower=0)).astype(float)
    win_filtered['eg'] = f_star * win_filtered['edge'] / 2.0
    win_filtered['market_type'] = 'win'
    n_email = (win_filtered['edge'] > EDGE_THRESHOLD_WIN).sum()
    print(f"[finish-pos] WIN: checked {len(win_merged)} player-book lines, "
          f"best edge {win_merged['edge'].max():.1f}pp -> {len(win_filtered)} saved, {n_email} email-worthy")
else:
    win_filtered = pd.DataFrame()
    print("[finish-pos] WIN: no market data returned from API")

# --- Top-N helper ---
def process_topn_market(market, prob_col):
    data = fetch_market_data(market)
    if not data:
        print(f"[finish-pos] {market.upper()}: no market data returned from API")
        return pd.DataFrame()
    df = extract_market_rows(data, odds_key='odds')
    if df.empty:
        print(f"[finish-pos] {market.upper()}: API returned data but no parseable odds rows")
        return pd.DataFrame()
    if prob_col not in topn_df.columns:
        print(f"[finish-pos] {market.upper()}: '{prob_col}' column not found in topn_df")
        return pd.DataFrame()
    df = df.merge(topn_df[['player_name', prob_col]], on='player_name', how='inner')
    df['implied_prob'] = 1.0 / df['decimal_odds']
    df['american_odds'] = df['decimal_odds'].apply(decimal_to_american)
    p = df[prob_col].astype(float)
    if p.max() > 1.0:
        p = p / 100.0
    b = df['decimal_odds'] - 1.0
    q = 1.0 - p
    df['edge'] = (p - df['implied_prob']) * 100.0
    n_checked = len(df)
    best_edge = df['edge'].max() if not df.empty else 0.0
    df = df[df['edge'] > 0].copy()
    n_email = (df['edge'] > EDGE_THRESHOLD_TOPN).sum()
    print(f"[finish-pos] {market.upper()}: checked {n_checked} player-book lines, "
          f"best edge {best_edge:.1f}pp -> {len(df)} saved, {n_email} email-worthy")
    if df.empty:
        return df
    f_star = (b * p - q) / b
    df['stake'] = (BANKROLL * KELLY_FRACTION * f_star.clip(lower=0)).astype(float)
    df['eg'] = f_star * df['edge'] / 2.0
    df['market_type'] = market
    df['my_fair'] = p.apply(prob_to_american)
    return df

top5_bets  = process_topn_market('top_5',  'top_5')
top10_bets = process_topn_market('top_10', 'top_10')
top20_bets = process_topn_market('top_20', 'top_20')

# Combine all candidate bets
frames_to_concat = [df for df in [win_filtered, top5_bets, top10_bets, top20_bets] if not df.empty]
if frames_to_concat:
    combined_finish_df = pd.concat(frames_to_concat, ignore_index=True)
    n_email_worthy = (combined_finish_df['edge'] > EDGE_THRESHOLD_WIN).sum()
    print(f"\n[finish-pos] TOTAL: {len(combined_finish_df)} +EV bets, "
          f"{n_email_worthy} email-worthy")
else:
    combined_finish_df = pd.DataFrame()
    print("\n[finish-pos] TOTAL: 0 +EV bets found")

combined_finish_df.to_csv('finish_test.csv')

# Best price per player/market
if not combined_finish_df.empty:
    combined_finish_df = (
        combined_finish_df.sort_values(['player_name','market_type','decimal_odds'], ascending=[True, True, False])
        .drop_duplicates(subset=['player_name','market_type'], keep='first')
    )
    combined_finish_df = (
        combined_finish_df
        .merge(sample_df, on='player_name', how='left')
        .merge(pred_join, on='player_name', how='left')
    )
    combined_finish_df = combined_finish_df[
        (combined_finish_df['sample'].fillna(0) >= 0) &
        (combined_finish_df['my_pred'].fillna(0) >= -1.0)
    ].copy()

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
    eg_total = output_df['eg'].sum()
    output_df['eg_share'] = np.where(eg_total > 0, output_df['eg'] / eg_total, 0.0)
    output_df['size_grouped'] = output_df['stake'] * (output_df['bookroll'] / BANKROLL)
    output_df['size_grouped'] = np.minimum(output_df['size_grouped'], 0.15 * output_df['bookroll'])
    output_df['size'] = output_df['size_grouped'].round(2)
    output_df['rank'] = output_df.groupby(['player_name','market'])['size'].rank(method='max', ascending=False)

    fb = pd.read_csv(PRED_PATH)
    fb['player_name'] = (
        fb['player_name'].astype(str).str.lower().str.strip().replace(name_replacements)
    )
    if 'sample' in fb.columns:
        sample_map = fb.set_index('player_name')['sample']
        _key = output_df['player_name'].astype(str).str.lower().str.strip().replace(name_replacements)
        output_df['sample'] = _key.map(sample_map).combine_first(output_df['sample'])
        del _key

    output_df_all = output_df.copy()
    sharp_books_list = ['betcris','pinnacle','betonline']
    pattern = '|'.join(sharp_books_list)
    sharp_df_finish = output_df[output_df['book'].str.contains(pattern, case=False, na=False)].copy()

    # Save outputs to v2 directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    finish_dir = os.path.join(".", tourney, "finish_pos")
    os.makedirs(finish_dir, exist_ok=True)

    if not output_df.empty:
        output_df.to_csv(os.path.join(finish_dir, f"positions_{tourney}_{timestamp}.csv"), index=False)
    if not sharp_df_finish.empty:
        sharp_df_finish.sort_values("eg", ascending=False).assign(size=lambda d: d['size'].round(2)) \
                .to_csv(os.path.join(finish_dir, f"sharp_pos_{tourney}.csv"), index=False)

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


# ============================================================
# MATCHUPS PRICING (tournament matchups) + weather impact CSV
# ============================================================

# --- Weather impact report ---
wx = model_preds[['player_name', 'r1_teetime', 'r2_teetime', 'my_pred']].copy()
wx['r1_teetime'] = pd.to_datetime(wx['r1_teetime'], format='mixed', errors='coerce')
wx['r2_teetime'] = pd.to_datetime(wx['r2_teetime'], format='mixed', errors='coerce')

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
    wx[c] = wx[c] - wx[c].mean()

wx['wind_adv_r1_2'] = wx['wind_adj_r1'] + wx['wind_adj_r2']
wx['dew_adv_r1_2']  = wx['dew_adj_r1']  + wx['dew_adj_r2']

wx_out = wx[['player_name','dew_adj_r1','wind_adj_r1','dew_adj_r2','wind_adj_r2','dew_adv_r1_2','wind_adv_r1_2']].copy()
wx_out = wx_out.round(2)
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

def norm_player(s: str) -> str:
    if s is None: return None
    x = str(s).lower().strip()
    return name_replacements.get(x, x)

rows_mu = []
match_list = data_tournament.get('match_list', [])
if not isinstance(match_list, list):
    print(f"[info] No tournament matchups available: {match_list}")
    match_list = []
for m in match_list:
    p1 = norm_player(m.get('p1_player_name'))
    p2 = norm_player(m.get('p2_player_name'))
    ties_handling = m.get('ties', 'unknown')
    for book, odds in m.get('odds', {}).items():
        if book == 'datagolf':
            continue
        rows_mu.append({
            'Player 1': p1,
            'Player 2': p2,
            'Bookmaker': book,
            'P1 Odds': odds.get('p1'),
            'P2 Odds': odds.get('p2'),
            'Datagolf Odds (P1)': m.get('odds', {}).get('datagolf', {}).get('p1'),
            'Datagolf Odds (P2)': m.get('odds', {}).get('datagolf', {}).get('p2'),
            'Ties': ties_handling
        })


# ============================================================
# EMAIL: Tournament Sim Summary
# ============================================================

def build_tournament_email_html(sharp_mu_df, finish_df, sample_lookup, my_pred_lookup):
    timestamp_str = datetime.now().strftime('%B %d, %Y %I:%M %p')

    # Section 1: Tournament Matchups
    mu_html = ""
    if sharp_mu_df is not None and not sharp_mu_df.empty:
        filtered = sharp_mu_df.copy()
        if 'sample_on' not in filtered.columns:
            filtered['sample_on'] = filtered['bet_on'].str.lower().map(sample_lookup).fillna(0)
        if 'pred_on' not in filtered.columns:
            filtered['pred_on'] = filtered['bet_on'].str.lower().map(my_pred_lookup).fillna(0)

        filtered = filtered[
            (filtered['pred_on'] > EMAIL_MU_MIN_PRED) &
            (filtered['sample_on'] >= EMAIL_MU_MIN_SAMPLE)
        ]

        if not filtered.empty:
            filtered = filtered.sort_values('edge_on', ascending=False)
            rows_html = ""
            for _, row in filtered.iterrows():
                bet_player = str(row.get('bet_on', '')).title()
                opponent = (
                    str(row['Player 2']).title()
                    if str(row.get('bet_on', '')).lower() == str(row['Player 1']).lower()
                    else str(row['Player 1']).title()
                )
                book = str(row.get('Bookmaker', ''))
                ties = str(row.get('Ties', ''))
                book_odds = (
                    row['P1 Odds'] if str(row.get('bet_on', '')).lower() == str(row['Player 1']).lower()
                    else row['P2 Odds']
                )
                fair_odds = (
                    row.get('Fair_p1') if str(row.get('bet_on', '')).lower() == str(row['Player 1']).lower()
                    else row.get('Fair_p2')
                )
                edge = row.get('edge_on', 0)
                pred = row.get('pred_on', 0)
                sample = int(row.get('sample_on', 0))
                half_shot = (
                    row.get('half_shot_p1', '')
                    if str(row.get('bet_on', '')).lower() == str(row['Player 1']).lower()
                    else row.get('half_shot_p2', '')
                )

                edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"
                pred_color = "#d4edda" if pred > 1.5 else "#ffffff"
                book_str = f"{int(book_odds):+d}" if pd.notna(book_odds) else ""
                fair_str = f"{int(fair_odds):+d}" if pd.notna(fair_odds) else ""
                hs_str = f"{half_shot:.1f}" if pd.notna(half_shot) and half_shot != "" else ""

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{bet_player}</td>
                    <td style="padding:6px 10px; color:#666;">vs {opponent}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{ties}</td>
                    <td style="padding:6px 10px; text-align:center;">{book_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred:.2f}</td>
                    <td style="padding:6px 10px; text-align:center;">{sample}</td>
                    <td style="padding:6px 10px; text-align:center;">{hs_str}</td>
                </tr>"""

            mu_html = f"""
            <h3 style="color:#2c5282; margin:20px 0 8px 0;">
                Tournament Matchups (pred &gt; {EMAIL_MU_MIN_PRED}, sample &gt; {EMAIL_MU_MIN_SAMPLE})
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Bet On</th>
                    <th style="padding:6px 10px; text-align:left;">Opponent</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Ties</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Sample</th>
                    <th style="padding:6px 10px; text-align:center;">1/2 Shot</th>
                </tr>
                {rows_html}
            </table>"""
        else:
            mu_html = "<p>No tournament matchups passed filters.</p>"
    else:
        mu_html = "<p>No tournament matchup data available.</p>"

    # Section 2: Finish Positions
    fp_html = ""
    if finish_df is not None and not finish_df.empty:
        fp_filtered = finish_df[
            (finish_df['my_pred'].fillna(0) > EMAIL_FP_MIN_PRED) &
            (finish_df['edge'] > EDGE_THRESHOLD_WIN)
        ].copy()
        if not fp_filtered.empty:
            fp_filtered = fp_filtered.sort_values('edge', ascending=False)
            fp_rows = ""
            for _, row in fp_filtered.iterrows():
                player = str(row.get('player_name', '')).title()
                market = str(row.get('market_type', ''))
                book_odds_str = f"{int(row['american_odds']):+d}" if pd.notna(row.get('american_odds')) else ""
                fair_str = f"{int(row['my_fair']):+d}" if pd.notna(row.get('my_fair')) else ""
                edge = row.get('edge', 0)
                pred = row.get('my_pred', 0)
                sample = int(row.get('sample', 0)) if pd.notna(row.get('sample')) else 0
                book = str(row.get('bookmaker', ''))

                edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"

                fp_rows += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{market}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{book_odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center;">{pred:.2f}</td>
                    <td style="padding:6px 10px; text-align:center;">{sample}</td>
                </tr>"""

            fp_html = f"""
            <h3 style="color:#2c5282; margin:20px 0 8px 0;">
                Finish Position Edges (pred &gt; {EMAIL_FP_MIN_PRED})
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Market</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Sample</th>
                </tr>
                {fp_rows}
            </table>"""
        else:
            fp_html = f"<p>No finish position edges passed email filters (edge &gt; {EDGE_THRESHOLD_WIN}pp, pred &gt; {EMAIL_FP_MIN_PRED}).</p>"
    else:
        fp_html = "<p>No finish position data available.</p>"

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:960px; margin:0 auto; padding:20px;">
        <h2 style="margin-bottom:4px;">Tournament Sim &mdash; {tourney.replace('_', ' ').title()}</h2>
        <p style="color:#666; margin-top:0;">{timestamp_str} | {SIMULATIONS:,} simulations | Course mults: OTT={course_mults[0]:.2f} APP={course_mults[1]:.2f} ARG={course_mults[2]:.2f} PUTT={course_mults[3]:.2f}</p>

        {mu_html}
        {fp_html}

        <p style="color:#999; font-size:11px; margin-top:30px;">
            Fair = our no-vig price (ties push) | Edge = expected return % |
            Pred = model SG prediction | 1/2 Shot = value of half-shot spread (in edge pts)
        </p>
    </body>
    </html>"""
    return html


def send_tournament_email(sharp_mu_df, finish_df, sample_lookup, my_pred_lookup,
                          attachment_paths=None):
    if not EMAIL_PASSWORD:
        print("  [warn] EMAIL_PASSWORD not set. Skipping email.")
        return
    if not EMAIL_FROM or not EMAIL_TO or EMAIL_TO == ['']:
        print("  [warn] EMAIL_FROM or EMAIL_TO not configured. Skipping email.")
        return

    try:
        html = build_tournament_email_html(sharp_mu_df, finish_df, sample_lookup, my_pred_lookup)

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Tournament Sim -- {tourney.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)

        msg.attach(MIMEText(html, "html"))

        if attachment_paths:
            for fpath in attachment_paths:
                if fpath and os.path.exists(fpath):
                    with open(fpath, "rb") as f:
                        ext = os.path.splitext(fpath)[1].lstrip('.')
                        att = MIMEApplication(f.read(), _subtype=ext)
                        att.add_header(
                            "Content-Disposition", "attachment",
                            filename=os.path.basename(fpath),
                        )
                        msg.attach(att)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print("  [ok] Tournament sim email sent")

    except Exception as e:
        print(f"  [warn] Email failed: {e}")
        print("    (Sim outputs still saved -- email is non-blocking)")


# --- Process matchups ---
df_match = pd.DataFrame(rows_mu).drop_duplicates(subset=['Player 1','Player 2','Bookmaker'], keep='first')
if not df_match.empty:
    df_match = df_match.dropna(subset=['Player 1','Player 2'])

    # Compute matchup probabilities from sim final scores
    name_to_idx = {p: i for i, p in enumerate(player_names)}
    final_scores_np = final_scores

    def get_score_vec(pname):
        idx = name_to_idx.get(pname)
        if idx is None:
            return None
        return final_scores_np[idx]

    p1_probs, p2_probs = [], []
    p1_probs_tl, p2_probs_tl = [], []
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
        denom = max(total - ties, 1.0)
        p1_probs.append(wins_p1 / denom)
        p2_probs.append(wins_p2 / denom)
        p1_probs_tl.append(wins_p1 / total)
        p2_probs_tl.append(wins_p2 / total)

    df_match['my_odds_p1'] = p1_probs
    df_match['my_odds_p2'] = p2_probs
    df_match['my_odds_p1_ties_loss'] = p1_probs_tl
    df_match['my_odds_p2_ties_loss'] = p2_probs_tl

    df_match['P1 Odds'] = pd.to_numeric(df_match['P1 Odds'], errors='coerce')
    df_match['P2 Odds'] = pd.to_numeric(df_match['P2 Odds'], errors='coerce')

    df_match['p1_dec'] = np.where(
        df_match['P1 Odds'] > 0,
        df_match['P1 Odds'] / 100 + 1,
        100 / df_match['P1 Odds'].abs() + 1,
    )
    df_match['p2_dec'] = np.where(
        df_match['P2 Odds'] > 0,
        df_match['P2 Odds'] / 100 + 1,
        100 / df_match['P2 Odds'].abs() + 1,
    )

    use_tl = df_match['Ties'] == 'separate bet offered'
    prob_p1 = np.where(use_tl, df_match['my_odds_p1_ties_loss'], df_match['my_odds_p1'])
    prob_p2 = np.where(use_tl, df_match['my_odds_p2_ties_loss'], df_match['my_odds_p2'])

    df_match['edge_p1'] = (prob_p1 * (df_match['p1_dec'] - 1) - (1 - prob_p1)) * 100
    df_match['edge_p2'] = (prob_p2 * (df_match['p2_dec'] - 1) - (1 - prob_p2)) * 100

    df_match['Fair_p1'] = df_match['my_odds_p1'].apply(
        lambda p: implied_prob_to_american_odds(p) if pd.notna(p) and 0 < p < 1 else None)
    df_match['Fair_p2'] = df_match['my_odds_p2'].apply(
        lambda p: implied_prob_to_american_odds(p) if pd.notna(p) and 0 < p < 1 else None)

    df_match['half_shot_p1'] = (df_match['my_odds_p1'] - df_match['my_odds_p1_ties_loss']) * 400
    df_match['half_shot_p2'] = (df_match['my_odds_p2'] - df_match['my_odds_p2_ties_loss']) * 400

    df_match['edge_on'] = df_match[['edge_p1', 'edge_p2']].max(axis=1).round(1)
    df_match['bet_on'] = df_match.apply(
        lambda r: r['Player 1'] if r['edge_p1'] > r['edge_p2'] else r['Player 2'], axis=1)
    df_match['pred_on'] = df_match['bet_on'].str.lower().map(my_pred_lookup)
    df_match['sample_on'] = df_match['bet_on'].str.lower().map(sample_lookup)

    dfs_by_book = {bk: df for bk, df in df_match.groupby('Bookmaker', dropna=True)}

    round_var = 'tourn'

    for bookmaker, dfb in dfs_by_book.items():
        dfb = dfb.copy()
        dfb['p1_implied'] = dfb['P1 Odds'].apply(american_to_implied_probability).round(1)
        dfb['p2_implied'] = dfb['P2 Odds'].apply(american_to_implied_probability).round(1)
        dfb['use_ties_loss'] = (dfb['Ties'] == "separate bet offered")
        dfb['p1_decimal_odds'] = np.where(dfb['P1 Odds'] > 0, dfb['P1 Odds'] / 100 + 1, 100 / dfb['P1 Odds'].abs() + 1)
        dfb['p2_decimal_odds'] = np.where(dfb['P2 Odds'] > 0, dfb['P2 Odds'] / 100 + 1, 100 / dfb['P2 Odds'].abs() + 1)
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
        dfb['Fair_p1'] = dfb['my_odds_p1'].apply(lambda p: implied_prob_to_american_odds(p) if pd.notna(p) else None)
        dfb['Fair_p2'] = dfb['my_odds_p2'].apply(lambda p: implied_prob_to_american_odds(p) if pd.notna(p) else None)
        dfb['Round'] = round_var

        final_cols = ['Player 1','Player 2','Bookmaker','Ties','P1 Odds','P2 Odds','Fair_p1','Fair_p2','edge_p1','edge_p2','Round']
        use_cols = [c for c in final_cols if c in dfb.columns]
        dfb = dfb[use_cols].dropna(subset=['Fair_p1','Fair_p2'])

        out_name = f"{bookmaker}_odds_with_my_odds_tu.csv"
        dfb.drop_duplicates(subset=['Player 1','Player 2','Bookmaker'], keep='first').to_csv(out_name, index=False)
        print(f"[ok] wrote {out_name}")

    # Combine + filter into {tourney}/matchups
    timestamp = datetime.now().strftime('%H%M')
    matchup_dir = os.path.join(".", tourney, "matchups")
    os.makedirs(matchup_dir, exist_ok=True)

    csv_files = [f"{bk}_odds_with_my_odds_tu.csv" for bk in dfs_by_book.keys()]
    dfs_list = []
    for fpath in csv_files:
        if os.path.exists(fpath):
            dfs_list.append(pd.read_csv(fpath))

    if dfs_list:
        combined_df = pd.concat(dfs_list, ignore_index=True)
        combined_df['Sample_P1'] = combined_df['Player 1'].str.lower().map(sample_lookup)
        combined_df['Sample_P2'] = combined_df['Player 2'].str.lower().map(sample_lookup)
        combined_df['sample_on'] = combined_df.apply(
            lambda r: r['Sample_P1'] if r.get('edge_p1', 0) > r.get('edge_p2', 0) else r['Sample_P2'], axis=1
        )
        combined_df['my_pred_p1'] = combined_df['Player 1'].str.lower().map(my_pred_lookup)
        combined_df['my_pred_p2'] = combined_df['Player 2'].str.lower().map(my_pred_lookup)
        combined_df['edge_on'] = combined_df[['edge_p1','edge_p2']].max(axis=1)
        combined_df = combined_df[combined_df['edge_on'] > 3]

        combined_df['pred_on'] = combined_df.apply(
            lambda r: r['my_pred_p1'] if r['edge_p1'] > r['edge_p2'] else r['my_pred_p2'], axis=1
        )
        combined_df['pred_against'] = combined_df.apply(
            lambda r: r['my_pred_p2'] if r['edge_p1'] > r['edge_p2'] else r['my_pred_p1'],
            axis=1
        )
        combined_df['bet_on'] = combined_df.apply(
            lambda r: r['Player 1'] if r['edge_p1'] > r['edge_p2'] else r['Player 2'], axis=1
        )

        combined_csv_name = os.path.join(matchup_dir, f"matchups_ftsimp_{tourney}_{timestamp}.csv")
        combined_df.to_csv(combined_csv_name, index=False)
        print(f"[ok] combined matchups -> {combined_csv_name}")

        # Sharp filter
        sharp_books_mu = ['betonline', 'betcris', 'pinnacle']
        sharp_df = combined_df[combined_df['Bookmaker'].str.lower().isin(sharp_books_mu)].copy()

        if sharp_df.empty:
            print("[warn] No sharp-book matchups found")
            sharp_filename = os.path.join(matchup_dir, f"sharp_filtered_{tourney}_{timestamp}.csv")
            sharp_df.to_csv(sharp_filename, index=False)
        else:
            wind_lookup = dict(zip(wx['player_name'].str.lower(), wx['wind_adv_r1_2']))
            sharp_df['wind_on'] = sharp_df['bet_on'].str.lower().map(wind_lookup)
            sharp_df['bet_against'] = sharp_df.apply(
                lambda r: r['Player 2'] if r['bet_on'] == r['Player 1'] else r['Player 1'], axis=1
            )
            sharp_df['wind_against'] = sharp_df['bet_against'].str.lower().map(wind_lookup)
            sharp_df['wind_diff'] = sharp_df['wind_on'] - sharp_df['wind_against']

            sharp_df['matchup_key'] = sharp_df.apply(
                lambda r: '-'.join(sorted([r['Player 1'].lower(), r['Player 2'].lower()])),
                axis=1
            )
            sharp_df = sharp_df.sort_values('edge_on', ascending=False).drop_duplicates('matchup_key', keep='first')
            sharp_df = sharp_df.drop(columns=['matchup_key', 'Sample_P1', 'Sample_P2', 'my_pred_p1', 'my_pred_p2'], errors='ignore')
            if 'sample_on' not in sharp_df.columns and 'sample_on' in combined_df.columns:
                sharp_df = sharp_df.merge(
                    combined_df[['Player 1', 'Player 2', 'Bookmaker', 'sample_on']],
                    on=['Player 1', 'Player 2', 'Bookmaker'], how='left'
                )

            sharp_filename = os.path.join(matchup_dir, f"sharp_filtered_{tourney}_{timestamp}.csv")
            sharp_df.to_csv(sharp_filename, index=False)
            print(f"[ok] sharp filtered -> {sharp_filename}")

        # --- Send tournament email ---
        print("\n[email] Building tournament sim email...")
        _finish_for_email = combined_finish_df if ('combined_finish_df' in dir() and not combined_finish_df.empty) else None
        _attachments = [f for f in [
            combined_csv_name,
            sharp_filename,
            f"weather_impact_{tourney}.csv",
            f"finish_equity_{tourney}.csv",
        ] if os.path.exists(f)]
        send_tournament_email(
            sharp_mu_df=sharp_df,
            finish_df=_finish_for_email,
            sample_lookup=sample_lookup,
            my_pred_lookup=my_pred_lookup,
            attachment_paths=_attachments,
        )
    else:
        print("[note] no bookmaker CSVs found to combine.")

else:
    print("[warn] No valid tournament matchups found.")

# --- Storage: always attempt (finish positions don't depend on matchups) ---
from sheets_storage import (
    is_valid_run_time,
    get_spreadsheet,
    store_tournament_matchups,
    store_finish_positions,
    load_dg_id_lookup,
)

if is_valid_run_time():
    print("\n[storage] Saving to Google Sheets...")
    try:
        # Single auth for all store calls
        spreadsheet = get_spreadsheet()

        # Build dg_id lookup from the predictions file
        dg_id_lookup = load_dg_id_lookup(tourney, name_replacements)

        # 1. Tournament matchups (only if matchup data exists)
        if 'combined_df' in dir() and not combined_df.empty:
            store_tournament_matchups(
                combined_df, tourney, _event_id,
                dg_id_lookup=dg_id_lookup,
                spreadsheet=spreadsheet,
            )

        # 2. Finish position bets
        if 'combined_finish_df' in dir() and not combined_finish_df.empty:
            store_finish_positions(
                combined_finish_df, tourney, _event_id,
                dg_id_lookup=dg_id_lookup,
                spreadsheet=spreadsheet,
            )

        print("[storage] Done.")
    except Exception as e:
        print(f"[storage] FAILED: {e}")
        import traceback; traceback.print_exc()
else:
    print("[storage] Skipped — before Monday 3 PM EST cutoff.")

# Push dashboard data to Render
try:
    from push_dashboard_data import copy_files, git_push
    print(f"\n{'='*60}")
    print("  Pushing dashboard data to Render...")
    copied, skipped = copy_files()
    if copied:
        print(f"  Copied {len(copied)} files to dashboard_data/")
        git_push()
        print("  Render deploy triggered.")
    else:
        print("  No files to push.")
    print(f"{'='*60}")
except Exception as e:
    print(f"  [dashboard push] Warning: {e}")

print("\n[done] Sim complete.")

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
# Player adjustments from Sheet (survives sim_inputs overwrites)

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

# --- Bayesian wind blending: blend forecast with climatological prior ---
# Climo weight scales with actual lead time (days to round), not hard-coded.
from api_utils import (
    fetch_historical_hourly_wind, blend_wind_with_climo,
    get_round_dates, climo_weight_for_lead,
    fetch_player_decompositions, fetch_field_updates,
)
try:
    import pandas as _pd_coords
    _coords_csv = os.path.join(os.path.dirname(__file__), "permanent_data", "course_coordinates.csv")
    _coords_df = _pd_coords.read_csv(_coords_csv)
    _coords_row = _coords_df[_coords_df["course_id"] == _event_id]
    if not _coords_row.empty:
        _lat = float(_coords_row["lat"].iloc[0])
        _lon = float(_coords_row["lon"].iloc[0])
        _month = datetime.now().month
        _climo_wind = fetch_historical_hourly_wind(_lat, _lon, _month)
        _round_dates = get_round_dates()
        if _climo_wind:
            print(f"[weather] Blending forecast with {_month}-month climo prior ({_lat}, {_lon})")
            print(f"[weather] Climo avg: {np.mean(_climo_wind):.1f} mph | "
                  f"R1 fcst avg: {np.mean(wind_1):.1f} | R2 fcst avg: {np.mean(wind_2):.1f}")
            wind_1, _w1 = blend_wind_with_climo(wind_1, _climo_wind,
                                                 round_date=_round_dates[0] if _round_dates else None)
            wind_2, _w2 = blend_wind_with_climo(wind_2, _climo_wind,
                                                 round_date=_round_dates[1] if _round_dates else None)
            print(f"[weather] Blended R1 avg: {np.mean(wind_1):.1f} (w_climo={_w1:.0%}) | "
                  f"R2 avg: {np.mean(wind_2):.1f} (w_climo={_w2:.0%})")
            if _round_dates:
                _now = datetime.now()
                for _r in range(4):
                    _ld = (_round_dates[_r] - _now).total_seconds() / 86400
                    print(f"  R{_r+1}: {_ld:.1f} days out -> climo weight {climo_weight_for_lead(_ld):.0%}")
        else:
            print("[weather] Could not fetch climo wind — using raw forecast")
    else:
        print(f"[weather] No coordinates for event_id={_event_id} — skipping climo blend")
except Exception as _e:
    print(f"[weather] Climo blend failed: {_e} — using raw forecast")

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
EMAIL_MU_MIN_SAMPLE = 8
EMAIL_FP_MIN_PRED = 0.0
MIN_DIST_ROUNDS = 20  # minimum rounds in player distribution (from cat_dists_player.py)

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

# --- DG overrides + manual boosts are now applied upstream in pre_sim_skill.py ---
# pre_course_fit_{tourney}.csv arrives with overrides already baked in.
# On second pass, final_predictions has regression on top — no overrides needed.
if PRED_PATH != _final_pred_path:
    print(f"[info] Using predictions from: {PRED_PATH} (overrides applied by pre_sim_skill.py)")
else:
    print(f"[info] Using regressed predictions from: {PRED_PATH}")

# --- Archetype map (for email/diagnostics, not for boosting) ---
_precomputed_arch_map = None

# Pull sample sizes from pre_course_fit if missing
if 'sample' not in model_preds.columns and os.path.exists(_pre_course_path):
    pcf = pd.read_csv(_pre_course_path, usecols=['player_name', 'sample'])
    pcf['player_name'] = pcf['player_name'].astype(str).str.lower().str.strip().replace(name_replacements)
    model_preds = model_preds.merge(pcf, on='player_name', how='left')

# --- Fetch tee times from DataGolf API ---
for rnd_col in ['r1_teetime', 'r2_teetime']:
    fu = fetch_field_updates(API_KEY, teetime_col=rnd_col)
    if fu is not None and not fu.empty:
        model_preds = model_preds.drop(columns=[rnd_col], errors='ignore')
        model_preds = model_preds.merge(fu[['player_name', rnd_col]], on='player_name', how='left')
        n_with_tt = model_preds[rnd_col].notna().sum()
        print(f"[teetimes] Fetched {rnd_col}: {n_with_tt}/{len(model_preds)} players have tee times")
    else:
        print(f"[teetimes] WARNING: Could not fetch {rnd_col} from API")

# --- Field mismatch check: stop if pred file has players not in DG field ---
if fu is not None and not fu.empty:
    dg_field = set(fu['player_name'].str.lower().str.strip())
    our_field = set(model_preds['player_name'].str.lower().str.strip())
    missing_from_dg = our_field - dg_field
    extra_in_dg = dg_field - our_field
    if missing_from_dg:
        print(f"\n[FIELD MISMATCH] {len(missing_from_dg)} players in predictions but NOT in DG field:")
        for p in sorted(missing_from_dg):
            pred_val = model_preds.loc[model_preds['player_name'] == p, 'my_pred'].values
            print(f"  {p}  (pred={pred_val[0]:.3f})" if len(pred_val) else f"  {p}")
        if extra_in_dg:
            print(f"\n[FIELD MISMATCH] {len(extra_in_dg)} players in DG field but NOT in predictions:")
            for p in sorted(extra_in_dg):
                print(f"  {p}")
        print(f"\n[STOPPED] Re-run pre_course_fit generation with updated field, then re-run this sim.")
        raise SystemExit(1)
    if extra_in_dg:
        print(f"[field] {len(extra_in_dg)} DG field players not in predictions (OK — likely late adds without skill data)")

# Write tee times back to pre_course_fit file
if os.path.exists(_pre_course_path):
    pcf_full = pd.read_csv(_pre_course_path)
    pcf_full['player_name'] = pcf_full['player_name'].astype(str).str.lower().str.strip().replace(name_replacements)
    for col in ['r1_teetime', 'r2_teetime']:
        if col in model_preds.columns:
            pcf_full = pcf_full.drop(columns=[col], errors='ignore')
            pcf_full = pcf_full.merge(
                model_preds[['player_name', col]].drop_duplicates('player_name'),
                on='player_name', how='left'
            )
    pcf_full.to_csv(_pre_course_path, index=False)
    print(f"[teetimes] Updated {_pre_course_path} with fresh tee times")

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
model_preds['r1_pred'] = model_preds['my_pred'] - model_preds['wind_adj_r1_sim'] - model_preds['dew_adj_r1_sim']
model_preds['r2_pred'] = model_preds['my_pred'] - model_preds['wind_adj_r2_sim'] - model_preds['dew_adj_r2_sim']
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
# Build dist_rounds lookup: min rounds across all 4 categories per player
dist_rounds_lookup = dists.groupby('player_name')['n'].min().to_dict()

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
    cat_mu = mu - weather_delta_r1[i] * WEATHER_CAT_SPLIT
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
from scipy.stats import skew as _skew_fn, norm as _norm_dist
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
    cat_mu = mu - weather_delta_r2[i] * WEATHER_CAT_SPLIT
    # Skill update shift: distribute evenly across 4 categories
    base_total_mu = mu.sum() - weather_delta_r2[i]
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
np.save(f"final_scores_{tourney}.npy", final_scores)

# ── Precompute H2H matchup matrix for dashboard ──────────────────────
_h2h_rows = []
_n_players = len(player_names)
for i in range(_n_players):
    for j in range(i + 1, _n_players):
        s_i = final_scores[i]
        s_j = final_scores[j]
        wins_i = int(np.sum(s_i < s_j))
        wins_j = int(np.sum(s_i > s_j))
        denom = wins_i + wins_j
        if denom == 0:
            continue
        prob_i = wins_i / denom
        _h2h_rows.append({
            "player_a": player_names[i],
            "player_b": player_names[j],
            "prob_a": round(prob_i, 6),
        })
_h2h_df = pd.DataFrame(_h2h_rows)
_h2h_df.to_parquet(f"h2h_matrix_{tourney}.parquet", index=False)
print(f"[ok] wrote h2h_matrix_{tourney}.parquet ({len(_h2h_rows)} pairs)")


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

# 3. Win prob sanity (with skill-weighted playoff)
from sim_inputs import playoff_format, playoff_holes, aggregate_holes

# Load hole distributions for playoff holes
_hole_dist_path = os.path.join(
    os.path.expanduser("~"), "OneDrive",
    f"adj_hole_dist_{tourney}_{_event_id}.csv"
)
_playoff_hole_dists = {}  # hole_num -> (scores_array, probs_array)
if os.path.exists(_hole_dist_path):
    _hd = pd.read_csv(_hole_dist_path)
    _hd_r4 = _hd[_hd["Round"] == 4]
    for hole_num in set(playoff_holes):
        row = _hd_r4[_hd_r4["Hole"] == hole_num]
        if not row.empty:
            r = row.iloc[0]
            scores = np.array([r[str(i)] for i in range(1, 11)])
            total = scores.sum()
            if total > 0:
                _playoff_hole_dists[hole_num] = (np.arange(1, 11), scores / total)
    print(f"[playoff] Loaded hole distributions for holes {list(_playoff_hole_dists.keys())} from {_hole_dist_path}")
else:
    print(f"[playoff] Hole dist file not found: {_hole_dist_path} — using random tiebreak")


def _playoff_draw(player_idxs, hole_num, preds):
    """Draw one hole score for each player, skill-adjusted.
    Shifts the hole distribution mean by player_pred / 18 (negative = better).
    """
    if hole_num not in _playoff_hole_dists:
        return RNG.standard_normal(len(player_idxs))  # fallback

    scores, base_probs = _playoff_hole_dists[hole_num]
    base_mean = (scores * base_probs).sum()
    results = np.empty(len(player_idxs))

    for i, pidx in enumerate(player_idxs):
        # Skill shift: positive pred = better player = lower score
        shift = -preds[pidx] / 18.0
        shifted_mean = base_mean + shift
        # Scale probabilities to shift mean while keeping shape
        # Use exponential tilting: multiply by exp(-lambda * score), solve for lambda
        # Simpler: just shift and redraw from continuous normal with hole std
        hole_std = np.sqrt(((scores - base_mean) ** 2 * base_probs).sum())
        results[i] = RNG.normal(shifted_mean, hole_std)

    return results


def _run_playoff(tied_idxs, preds):
    """Run a playoff between tied player indices. Returns winner index."""
    if len(tied_idxs) == 1:
        return tied_idxs[0]

    remaining = list(tied_idxs)

    if playoff_format == "aggregate" and aggregate_holes > 0:
        # Aggregate phase: sum scores over N holes
        totals = np.zeros(len(remaining))
        hole_seq = playoff_holes if playoff_holes else [18]
        for h_idx in range(aggregate_holes):
            hole_num = hole_seq[h_idx % len(hole_seq)]
            scores = _playoff_draw(remaining, hole_num, preds)
            totals += scores
        min_total = totals.min()
        survivors = [remaining[i] for i in range(len(remaining)) if totals[i] == min_total]
        if len(survivors) == 1:
            return survivors[0]
        remaining = survivors

    # Sudden death phase
    hole_seq = playoff_holes if playoff_holes else [18]
    for attempt in range(20):  # cap at 20 holes to avoid infinite loop
        hole_num = hole_seq[attempt % len(hole_seq)]
        scores = _playoff_draw(remaining, hole_num, preds)
        min_score = scores.min()
        winners = [remaining[i] for i in range(len(remaining)) if scores[i] == min_score]
        if len(winners) == 1:
            return winners[0]
        remaining = winners

    # Fallback: random if still tied after 20 holes
    return RNG.choice(remaining)


simulated_winners_v2 = []
_playoff_count = 0
_playoff_appearances = {}  # player_idx -> count of playoffs entered
_playoff_wins = {}         # player_idx -> count of playoffs won
for j in range(SIMULATIONS):
    sc = final_scores[:, j]
    min_score = sc.min()
    tied = np.where(sc == min_score)[0]
    if len(tied) == 1:
        winner_idx = tied[0]
    else:
        for pidx in tied:
            _playoff_appearances[pidx] = _playoff_appearances.get(pidx, 0) + 1
        winner_idx = _run_playoff(tied, my_pred_base)
        _playoff_wins[winner_idx] = _playoff_wins.get(winner_idx, 0) + 1
        _playoff_count += 1
    simulated_winners_v2.append(player_names[winner_idx])

print(f"[playoff] {_playoff_count}/{SIMULATIONS} sims needed playoff ({_playoff_count/SIMULATIONS*100:.1f}%)")

# Playoff summary: players with most appearances
if _playoff_appearances:
    _po_rows = []
    for pidx in sorted(_playoff_appearances, key=lambda x: _playoff_appearances[x], reverse=True):
        apps = _playoff_appearances[pidx]
        wins = _playoff_wins.get(pidx, 0)
        _po_rows.append({
            "player": player_names[pidx],
            "pred": f"{my_pred_base[pidx]:+.2f}",
            "playoffs": apps,
            "wins": wins,
            "win_pct": f"{wins/apps*100:.1f}%",
        })
    _po_df = pd.DataFrame(_po_rows)
    print(f"\n  PLAYOFF SUMMARY (top 15):")
    print(f"  {'Player':>25s}  {'Pred':>6s}  {'In':>5s}  {'Won':>5s}  {'Win%':>6s}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*6}")
    for _, r in _po_df.head(15).iterrows():
        print(f"  {r['player']:>25s}  {r['pred']:>6s}  {r['playoffs']:>5d}  {r['wins']:>5d}  {r['win_pct']:>6s}")

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

# 6. Per-player realized vs predicted (mean & std)
print(f"\n  Per-player validation:")

# R1 mean (cleanest — no skill updates)
r1_mean_realized = sg_r1.mean(axis=1)
mean_delta = r1_mean_realized - r1_mu

# Per-round realized std
r1_std_real = sg_r1.std(axis=1)
r2_std_real = sg_r2.std(axis=1)
r3_std_real = sg_r3.std(axis=1)
r4_std_real = sg_r4.std(axis=1)
avg_std_real = (r1_std_real + r2_std_real + r3_std_real + r4_std_real) / 4.0
std_delta = avg_std_real - round_std

# Tournament-wide mean (all 4 rounds averaged)
tourn_sg = (sg_r1 + sg_r2 + sg_r3 + sg_r4) / 4.0
tourn_mean_realized = tourn_sg.mean(axis=1)
tourn_mean_delta = tourn_mean_realized - my_pred_base

print(f"  {'':30s}  {'R1 Mean':>10s}  {'Avg Std':>10s}  {'Tourn Mean':>10s}")
print(f"  {'MAE (all players)':30s}  {np.abs(mean_delta).mean():10.4f}  {np.abs(std_delta).mean():10.4f}  {np.abs(tourn_mean_delta).mean():10.4f}")
print(f"  {'Max abs delta':30s}  {np.abs(mean_delta).max():10.4f}  {np.abs(std_delta).max():10.4f}  {np.abs(tourn_mean_delta).max():10.4f}")

# Per-round std summary
print(f"\n  Std dev by round (field avg):")
print(f"    R1: pred={round_std.mean():.3f}  real={r1_std_real.mean():.3f}  d={r1_std_real.mean() - round_std.mean():+.3f}")
print(f"    R2: pred={round_std.mean():.3f}  real={r2_std_real.mean():.3f}  d={r2_std_real.mean() - round_std.mean():+.3f}")
print(f"    R3: pred={round_std.mean():.3f}  real={r3_std_real.mean():.3f}  d={r3_std_real.mean() - round_std.mean():+.3f}")
print(f"    R4: pred={round_std.mean():.3f}  real={r4_std_real.mean():.3f}  d={r4_std_real.mean() - round_std.mean():+.3f}")

# Flag players with largest discrepancies
_val_df = pd.DataFrame({
    'player_name': player_names,
    'pred': my_pred_base,
    'r1_pred': r1_mu,
    'r1_mean_real': r1_mean_realized,
    'mean_delta': mean_delta,
    'pred_std': round_std,
    'r1_std_real': r1_std_real,
    'r2_std_real': r2_std_real,
    'r3_std_real': r3_std_real,
    'r4_std_real': r4_std_real,
    'avg_std_real': avg_std_real,
    'std_delta': std_delta,
    'tourn_mean_real': tourn_mean_realized,
    'tourn_mean_delta': tourn_mean_delta,
})

# Top 5 mean misses
print(f"\n  Biggest R1 mean misses:")
for _, r in _val_df.nlargest(5, 'mean_delta').iterrows():
    print(f"    {r['player_name']:28s}  pred={r['r1_pred']:+.3f}  real={r['r1_mean_real']:+.3f}  d={r['mean_delta']:+.4f}")
for _, r in _val_df.nsmallest(5, 'mean_delta').iterrows():
    print(f"    {r['player_name']:28s}  pred={r['r1_pred']:+.3f}  real={r['r1_mean_real']:+.3f}  d={r['mean_delta']:+.4f}")

# Top 5 std misses (avg across all rounds)
print(f"\n  Biggest avg std misses (pred vs realized avg of R1-R4):")
for _, r in _val_df.nlargest(5, 'std_delta').iterrows():
    print(f"    {r['player_name']:28s}  pred={r['pred_std']:.3f}  real={r['avg_std_real']:.3f}  d={r['std_delta']:+.4f}  "
          f"[R1={r['r1_std_real']:.2f} R2={r['r2_std_real']:.2f} R3={r['r3_std_real']:.2f} R4={r['r4_std_real']:.2f}]")
for _, r in _val_df.nsmallest(5, 'std_delta').iterrows():
    print(f"    {r['player_name']:28s}  pred={r['pred_std']:.3f}  real={r['avg_std_real']:.3f}  d={r['std_delta']:+.4f}  "
          f"[R1={r['r1_std_real']:.2f} R2={r['r2_std_real']:.2f} R3={r['r3_std_real']:.2f} R4={r['r4_std_real']:.2f}]")

# Save full validation file
_val_df.to_csv(f'sim_validation_{tourney}.csv', index=False)
print(f"\n  [ok] Saved sim_validation_{tourney}.csv")

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
# Persist a copy in permanent_data/ so Monday grading pipeline can access it
os.makedirs("permanent_data", exist_ok=True)
df_avg.to_csv(f"permanent_data/avg_expected_cat_sg_{tourney}.csv", index=False)

# Non dead-heat rank probabilities (raw min-rank — ties share same position)
rank_probs_ndh = (
    long_df.groupby(['player_name', 'rank']).size()
    .div(SIMULATIONS)
    .rename('prob_ndh')
    .reset_index()
)
rank_probs_ndh['rank'] = rank_probs_ndh['rank'].astype(int)

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
    .rename('prob_u')
    .reset_index()
)
rank_probs['rank'] = rank_probs['rank'].astype(int)

# Merge dead-heat and non-dead-heat into single parquet
rank_probs_updated = rank_probs.merge(rank_probs_ndh, on=['player_name', 'rank'], how='outer').fillna(0)
rank_probs_updated.to_parquet(f"rank_probs_updated_{tourney}.parquet", index=False)
print(f"[ok] wrote rank_probs_updated_{tourney}.parquet (cols: prob_u, prob_ndh)")


# ============================================================
# OUTRIGHTS & TOP-N PRICING VS MARKET
# ============================================================

EDGE_THRESHOLD_WIN   = 2.0
EDGE_THRESHOLD_TOPN  = 2.0
BANKROLL             = 10000.0
KELLY_FRACTION       = 0.25
RETAIL_BOOKS         = ['draftkings','fanduel','betmgm','caesars','barstool','espn','pointsbet','wynnbet','unibet','betway','betfred','betrivers']

_all_sb_lines_raw = []  # accumulate ALL sportsbook lines (pre-edge-filter) for exchange comparison
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
    win_merged['market_type'] = 'win'
    win_merged['american_odds'] = win_merged['decimal_odds'].apply(decimal_to_american)
    _all_sb_lines_raw.append(win_merged[['player_name', 'market_type', 'american_odds', 'bookmaker', 'decimal_odds']].copy())
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
    df['market_type'] = market
    # Stash ALL lines (pre-filter) for SB line lookup in exchange email
    _all_sb_lines_raw.append(df[['player_name', 'market_type', 'american_odds', 'bookmaker', 'decimal_odds']].copy())
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

# Build full SB line lookup (best price per player+market across ALL sportsbooks, pre-edge-filter)
_all_sb_lines_lookup = {}  # (player, market_type) → best american odds
if _all_sb_lines_raw:
    _all_sb = pd.concat(_all_sb_lines_raw, ignore_index=True)
    # Filter to sharp books only
    _sharp = _all_sb[_all_sb['bookmaker'].str.lower().isin(['pinnacle', 'betcris', 'betonline'])]
    if not _sharp.empty:
        # Best price = highest decimal odds (lowest implied prob) per player+market
        _sharp_best = _sharp.sort_values('decimal_odds', ascending=False).drop_duplicates(
            subset=['player_name', 'market_type'], keep='first')
        for _, r in _sharp_best.iterrows():
            key = (str(r['player_name']).lower().strip(), str(r['market_type']))
            odds = r.get('american_odds')
            if pd.notna(odds):
                _all_sb_lines_lookup[key] = int(odds)
    print(f"[finish-pos] SB line lookup: {len(_all_sb_lines_lookup)} player+market combos from {len(_all_sb)} raw lines")

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

wx['wind_adv_r1_2'] = -(wx['wind_adj_r1'] + wx['wind_adj_r2'])
wx['dew_adv_r1_2']  = -(wx['dew_adj_r1']  + wx['dew_adj_r2'])

wx_out = wx[['player_name','dew_adj_r1','wind_adj_r1','dew_adj_r2','wind_adj_r2','dew_adv_r1_2','wind_adv_r1_2']].copy()
wx_out = wx_out.round(2)
wx_out.to_csv(f'weather_impact_{tourney}.csv', index=False)
print(f"[ok] wrote weather_impact_{tourney}.csv")

# --- Pull tournament matchups (scraped → DataGolf fallback) ---
from odds_loader import load_matchup_odds, scrape_betonline_live

print("[info] Fetching tournament matchup odds (scraped -> DataGolf fallback)...")
_mu_df = load_matchup_odds("tournament_matchups", api_key=API_KEY)
if _mu_df.empty or len(_mu_df) < 5:
    if not _mu_df.empty:
        print(f"[info] Only {len(_mu_df)} lines from pipeline — too few, trying live scrape...")
    else:
        print("[info] No matchups from normal pipeline — scraping BetOnline live...")
    _live = scrape_betonline_live("tournament_matchup")
    if not _live.empty:
        _mu_df = _live

if not _mu_df.empty:
    # Rename DG columns to match legacy format
    _mu_df = _mu_df.rename(columns={"DG_p1": "Datagolf Odds (P1)", "DG_p2": "Datagolf Odds (P2)"})
    print(f"  Loaded {len(_mu_df)} matchup lines ({_mu_df['source'].iloc[0]})")
    rows_mu = _mu_df.drop(columns=["source"], errors="ignore").to_dict("records")
else:
    print("[info] No tournament matchups available")
    rows_mu = []


# ============================================================
# KALSHI OUTRIGHT PRICING
# ============================================================

def _kalshi_taker_fee(price):
    """Kalshi taker fee: 7% of price * (1 - price)."""
    if price <= 0 or price >= 1:
        return 0.0
    return 0.07 * price * (1 - price)


def price_kalshi_outrights_tourney(finish_probs, pred_lookup, sample_lookup):
    """Price Kalshi outright markets using live API with orderbook-aware liquidity filtering.

    Two-stage filter:
      Stage 1: Pre-filter on bid/ask presence, spread ≤ 10c, and edge > 0.5%
      Stage 2: Fetch orderbook for survivors, compute fillable depth at ≥ 0.5% edge

    Returns DataFrame with yes-mid edges plus depth/best-fill/volume/OI columns.
    """
    import re as _re
    import time as _time
    from collections import Counter

    # ── Kalshi client + market fetch ──────────────────────────────────
    try:
        from kalshi_client import KalshiClient
        client = KalshiClient()
    except Exception as e:
        print(f"  [warn] KalshiClient init failed: {e} — skipping Kalshi pricing")
        return pd.DataFrame()

    OUTRIGHT_SERIES = {
        "KXPGATOP5": "top_5",
        "KXPGATOP10": "top_10",
        "KXPGATOP20": "top_20",
        "KXPGATOUR": "winner",
    }

    all_markets = []
    for series_ticker, mtype in OUTRIGHT_SERIES.items():
        try:
            mkts = client.get_markets(series_ticker)
            for m in mkts:
                m["_market_type"] = mtype
            all_markets.append(mkts)
        except Exception as e:
            print(f"  [warn] Failed to fetch {series_ticker}: {e}")
    all_markets = [m for batch in all_markets for m in batch]
    print(f"  Fetched {len(all_markets)} Kalshi markets from live API")

    if not all_markets:
        return pd.DataFrame()

    # Detect current tournament (most common title prefix)
    def _detect_tourney(title):
        m = _re.search(r"(?:at|in|win) the (.+?)\?", title)
        if m:
            return m.group(1).strip()
        m = _re.match(r"(.+?):\s*Will", title)
        if m:
            return m.group(1).strip()
        return ""

    tourn_counts = Counter()
    for m in all_markets:
        t = _detect_tourney(m.get("title", ""))
        if t:
            tourn_counts[t] += 1
    if tourn_counts:
        current_tourney = tourn_counts.most_common(1)[0][0]
        # Include markets whose detected tournament name overlaps with the
        # most-common name (e.g. "The Masters" and "Masters Tournament" are
        # the same event).  Overlap = any word longer than 3 chars in common.
        _ct_words = {w.lower() for w in current_tourney.split() if len(w) > 3}
        def _same_event(detected):
            if not detected:
                return True
            return bool({w.lower() for w in detected.split() if len(w) > 3} & _ct_words)
        all_markets = [m for m in all_markets
                       if _same_event(_detect_tourney(m.get("title", "")))]
        print(f"  Tournament: {current_tourney} ({len(all_markets)} markets)")

    # ── Extract player name from title ────────────────────────────────
    def _extract_player(title):
        m = _re.match(r".*:\s*Will (.+?) (?:finish|make|miss|lead|win)", title)
        if m:
            return m.group(1).strip()
        m = _re.match(r"Will (.+?) win the ", title)
        if m:
            return m.group(1).strip()
        return ""

    # ── Build nodh probabilities ──────────────────────────────────────
    _rp_path = f"rank_probs_updated_{tourney}.parquet"
    if os.path.exists(_rp_path):
        _rp = pd.read_parquet(_rp_path)
        _nodh = _rp.groupby("player_name").apply(
            lambda g: pd.Series({
                "top_5_nodh": g.loc[g["rank"] <= 5, "prob_ndh"].sum(),
                "top_10_nodh": g.loc[g["rank"] <= 10, "prob_ndh"].sum(),
                "top_20_nodh": g.loc[g["rank"] <= 20, "prob_ndh"].sum(),
                "win_nodh": g.loc[g["rank"] == 1, "prob_ndh"].sum(),
            })
        ).reset_index()
        finish_probs = finish_probs.merge(_nodh, on="player_name", how="left")
        print(f"  Built nodh probs from {_rp_path} ({len(_nodh)} players)")

    type_to_col = {
        "top_5": "top_5_nodh",
        "top_10": "top_10_nodh",
        "top_20": "top_20_nodh",
        "winner": "win_nodh",
    }
    for mtype, col in list(type_to_col.items()):
        if col not in finish_probs.columns:
            fallback = {"top_5_nodh": "top_5", "top_10_nodh": "top_10",
                        "top_20_nodh": "top_20", "win_nodh": "simulated_win_prob"}
            type_to_col[mtype] = fallback.get(col, col)

    def norm(s):
        x = s.strip().lower()
        if "," not in x:
            parts = x.rsplit(" ", 1)
            if len(parts) == 2:
                x = f"{parts[1]}, {parts[0]}"
        return name_replacements.get(x, x)

    # ── Flat-size orderbook walk ─────────────────────────────────────
    FLAT_CONTRACTS = 1000

    def _walk_book(ob_levels, sim_prob):
        """Walk orderbook for a flat 1000-contract fill.

        Returns dict with effective_price, filled, or None if empty book
        or no edge at the fill price.
        ob_levels: list of (price_dollars, qty) from the OPPOSITE side.
        """
        fill_levels = sorted([(1.0 - p, qty) for p, qty in ob_levels])
        if not fill_levels:
            return None

        target = FLAT_CONTRACTS
        filled = 0
        cost_sum = 0.0
        fee_sum = 0.0
        for price, qty in fill_levels:
            fee = _kalshi_taker_fee(price)
            take = min(qty, target - filled)
            filled += take
            cost_sum += take * price
            fee_sum += take * fee
            if filled >= target:
                break

        if filled == 0:
            return None

        vwap = cost_sum / filled
        eff_price = vwap + fee_sum / filled

        # Only return if there's edge at the fill price
        if eff_price >= sim_prob:
            return None

        return {
            "effective_price": eff_price,
            "target": target,
            "filled": filled,
        }

    # ── Stage 1: Pre-filter (no orderbook calls) ─────────────────────
    MIN_EDGE_STAGE1 = 0.5  # percentage points (on mid, loose gate)
    stage1_rows = []

    for mkt in all_markets:
        ticker = mkt.get("ticker", "")
        title = mkt.get("title", "")
        mtype = mkt.get("_market_type", "")

        # Extract bid/ask in dollars (try dollar fields first, then cents)
        bid = float(mkt.get("yes_bid_dollars") or 0)
        ask = float(mkt.get("yes_ask_dollars") or 0)
        if bid == 0 and ask == 0:
            bid = float(mkt.get("yes_bid", 0) or 0) / 100.0
            ask = float(mkt.get("yes_ask", 0) or 0) / 100.0

        # Must be two-sided
        if bid <= 0 or ask <= 0:
            continue
        # Spread filter
        if (ask - bid) > 0.10:
            continue

        # Extract volume and open interest (free from get_markets response)
        volume = int(float(mkt.get("volume_fp", 0) or 0))
        open_interest = int(float(mkt.get("open_interest_fp", 0) or 0))

        # Match player to sim
        player_raw = _extract_player(title)
        if not player_raw:
            continue
        player = norm(player_raw)

        prob_col = type_to_col.get(mtype)
        if not prob_col or prob_col not in finish_probs.columns:
            continue

        match = finish_probs[finish_probs["player_name"] == player]
        if match.empty:
            continue

        sim_yes = float(match.iloc[0][prob_col])
        if sim_yes <= 0:
            continue

        sim_no = 1.0 - sim_yes
        mid = (bid + ask) / 2.0

        # Loose edge check on mid (either side)
        yes_edge_mid = (sim_yes - mid) * 100
        no_edge_mid = (sim_no - (1 - mid)) * 100
        if max(yes_edge_mid, no_edge_mid) < MIN_EDGE_STAGE1:
            continue

        stage1_rows.append({
            "ticker": ticker,
            "player": player,
            "mtype": mtype,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "sim_yes": sim_yes,
            "sim_no": sim_no,
            "volume": volume,
            "open_interest": open_interest,
            "pred": pred_lookup.get(player),
            "sample": sample_lookup.get(player),
        })

    print(f"  Stage 1 pre-filter: {len(stage1_rows)} markets pass (edge > {MIN_EDGE_STAGE1}%)")

    if not stage1_rows:
        return pd.DataFrame()

    # ── Stage 2: Orderbook walk with Kelly sizing ────────────────────
    for row in stage1_rows:
        try:
            ob = client.get_orderbook(row["ticker"])
            _time.sleep(0.05)  # 50ms rate limit
        except Exception:
            row["yes_fill"] = None
            row["no_fill"] = None
            continue

        yes_levels = ob.get("yes", [])  # YES bids
        no_levels = ob.get("no", [])    # NO bids

        # Buy YES: walk NO levels (sellers of YES)
        row["yes_fill"] = _walk_book(no_levels, row["sim_yes"])
        # Buy NO: walk YES levels (sellers of NO)
        row["no_fill"] = _walk_book(yes_levels, row["sim_no"])

    print(f"  Stage 2 orderbook: fetched {len(stage1_rows)} orderbooks")

    # ── Build output rows ────────────────────────────────────────────
    rows = []
    for row in stage1_rows:
        oi = row["open_interest"]
        base_row = {
            "player_name": row["player"],
            "market_type": row["mtype"],
            "bookmaker": "kalshi",
            "volume": row["volume"],
            "open_interest": oi,
            "ticker": row["ticker"],
        }

        # --- YES side ---
        yf = row.get("yes_fill")
        if yf is not None:
            eff = yf["effective_price"]
            edge = (row["sim_yes"] - eff) * 100
            rows.append({
                **base_row,
                "side": "yes",
                "eff_price": eff,
                "american_odds": implied_prob_to_american_odds(eff),
                "sim_prob": row["sim_yes"],
                "edge": round(edge, 1),
                "my_fair": prob_to_american(row["sim_yes"]),
                "target": yf["target"],
                "filled": yf["filled"],
            })

        # --- NO side ---
        nf = row.get("no_fill")
        if nf is not None:
            eff = nf["effective_price"]
            edge = (row["sim_no"] - eff) * 100
            rows.append({
                **base_row,
                "side": "no",
                "eff_price": eff,
                "american_odds": implied_prob_to_american_odds(eff),
                "sim_prob": row["sim_no"],
                "edge": round(edge, 1),
                "my_fair": prob_to_american(row["sim_no"]),
                "target": nf["target"],
                "filled": nf["filled"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge", ascending=False)
        yes_pos = ((df["side"] == "yes") & (df["edge"] > 0)).sum()
        no_pos = ((df["side"] == "no") & (df["edge"] > 0)).sum()
        print(f"  Kalshi outrights: {len(df)} lines ({FLAT_CONTRACTS}-contract fill), "
              f"{yes_pos} YES +edge, {no_pos} NO +edge")
    return df


# ============================================================
# NOVIG OUTRIGHT PRICING
# ============================================================

def price_novig_outrights_tourney(finish_probs, pred_lookup, sample_lookup):
    """Price NoVig outright markets using live GraphQL API (no auth needed).

    Uses the `available` field (best offer / ask price) on each outcome.
    Probabilities are no-dead-heat (NoVig pays on exact finish, not DH-adjusted).

    Returns DataFrame with same schema as Kalshi pricing for easy merging.
    """
    import httpx as _httpx

    GRAPHQL_URL = "https://api.novig.us/v1/graphql"
    GQL_HEADERS = {
        "Content-Type": "application/json",
        "Origin": "https://novig.com",
        "Referer": "https://novig.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    NOVIG_TYPE_MAP = {
        "TOP_FIVE_FINISH": "top_5",
        "TOP_TEN_FINISH": "top_10",
        "TOP_TWENTY_FINISH": "top_20",
        "WINNER": "winner",
    }

    # ── Build nodh probabilities ──────────────────────────────────────
    _rp_path = f"rank_probs_updated_{tourney}.parquet"
    if os.path.exists(_rp_path):
        _rp = pd.read_parquet(_rp_path)
        _nodh = _rp.groupby("player_name").apply(
            lambda g: pd.Series({
                "top_5_nodh": g.loc[g["rank"] <= 5, "prob_ndh"].sum(),
                "top_10_nodh": g.loc[g["rank"] <= 10, "prob_ndh"].sum(),
                "top_20_nodh": g.loc[g["rank"] <= 20, "prob_ndh"].sum(),
                "win_nodh": g.loc[g["rank"] == 1, "prob_ndh"].sum(),
            })
        ).reset_index()
        finish_probs = finish_probs.merge(_nodh, on="player_name", how="left")

    type_to_col = {
        "top_5": "top_5_nodh", "top_10": "top_10_nodh",
        "top_20": "top_20_nodh", "winner": "win_nodh",
    }
    for mtype, col in list(type_to_col.items()):
        if col not in finish_probs.columns:
            fallback = {"top_5_nodh": "top_5", "top_10_nodh": "top_10",
                        "top_20_nodh": "top_20", "win_nodh": "simulated_win_prob"}
            type_to_col[mtype] = fallback.get(col, col)

    def norm(s):
        x = s.strip().lower()
        if "," not in x:
            parts = x.rsplit(" ", 1)
            if len(parts) == 2:
                x = f"{parts[1]}, {parts[0]}"
        return name_replacements.get(x, x)

    # ── Find current PGA tournament on NoVig ──────────────────────────
    try:
        client = _httpx.Client(timeout=15.0)

        def gql(query, variables=None):
            payload = {"query": query}
            if variables:
                payload["variables"] = variables
            resp = client.post(GRAPHQL_URL, headers=GQL_HEADERS, json=payload)
            resp.raise_for_status()
            return resp.json()

        # Get tournaments, pick one with most child events
        data = gql("""query {
  event(where: {league: {_eq: "PGA"}, type: {_eq: "Tournament"}, status: {_eq: "OPEN_PREGAME"}}
        order_by: {scheduled_start: asc}) {
    id description
    child_events_aggregate: events_aggregate(where: {status: {_eq: "OPEN_PREGAME"}}) {
      aggregate { count }
    }
  }
}""")
        events = data.get("data", {}).get("event", [])
        if not events:
            print("  No open PGA tournament on NoVig")
            return pd.DataFrame()
        best = max(events, key=lambda e: e.get("child_events_aggregate", {})
                   .get("aggregate", {}).get("count", 0))
        tourn_id = best["id"]
        tourn_name = best["description"]
        print(f"  NoVig tournament: {tourn_name}")
    except Exception as e:
        print(f"  [warn] NoVig tournament lookup failed: {e}")
        return pd.DataFrame()

    # ── Fetch outright markets ────────────────────────────────────────
    rows = []
    for novig_type, our_type in NOVIG_TYPE_MAP.items():
        try:
            data = gql("""query($tournId: uuid!, $mtype: String!) {
  event(where: {parent_event: {id: {_eq: $tournId}}, type: {_eq: "Future"},
                status: {_eq: "OPEN_PREGAME"},
                markets: {type: {_eq: $mtype}, status: {_eq: "OPEN"}}}) {
    id description
    markets(where: {type: {_eq: $mtype}, status: {_eq: "OPEN"}}) {
      id type volume description
      outcomes { id index description available }
    }
  }
}""", {"tournId": tourn_id, "mtype": novig_type})
            events = data.get("data", {}).get("event", [])
            # Pick the right event (skip "With Ties" and "End Of Round")
            target = None
            for e in events:
                desc = e.get("description", "")
                if "With Ties" in desc or "End Of Round" in desc:
                    continue
                if e.get("markets"):
                    target = e
                    break
            if not target:
                continue

            for m in target.get("markets", []):
                outcomes = m.get("outcomes", [])
                yes_price = None
                no_price = None
                for o in outcomes:
                    if o["index"] == 0:
                        yes_price = o.get("available")
                    elif o["index"] == 1:
                        no_price = o.get("available")
                if not yes_price or yes_price <= 0:
                    continue

                # Extract player name
                desc = m.get("description", "")
                player_raw = desc.replace(f" {novig_type}", "").strip()
                if not player_raw:
                    continue
                player = norm(player_raw)

                prob_col = type_to_col.get(our_type)
                if not prob_col or prob_col not in finish_probs.columns:
                    continue
                match = finish_probs[finish_probs["player_name"] == player]
                if match.empty:
                    continue
                sim_prob = float(match.iloc[0][prob_col])
                if sim_prob <= 0:
                    continue

                volume = float(m.get("volume", 0) or 0)

                # YES side: edge = sim_prob - ask_price
                yes_edge = (sim_prob - yes_price) * 100
                american_odds = implied_prob_to_american_odds(yes_price)
                if american_odds is not None:
                    rows.append({
                        "player_name": player,
                        "market_type": our_type,
                        "bookmaker": "novig",
                        "side": "yes",
                        "eff_price": yes_price,
                        "american_odds": american_odds,
                        "sim_prob": sim_prob,
                        "edge": round(yes_edge, 1),
                        "my_fair": prob_to_american(sim_prob),
                        "volume": volume,
                        "open_interest": 0,
                        "target": 0,
                        "filled": 0,
                        "ticker": m.get("id", ""),
                    })

                # NO side: edge = (1-sim_prob) - (1-yes_price) ... but use no_price if available
                if no_price and no_price > 0:
                    sim_no = 1.0 - sim_prob
                    no_edge = (sim_no - no_price) * 100
                    no_american = implied_prob_to_american_odds(no_price)
                    if no_american is not None:
                        rows.append({
                            "player_name": player,
                            "market_type": our_type,
                            "bookmaker": "novig",
                            "side": "no",
                            "eff_price": no_price,
                            "american_odds": no_american,
                            "sim_prob": sim_no,
                            "edge": round(no_edge, 1),
                            "my_fair": prob_to_american(sim_no),
                            "volume": volume,
                            "open_interest": 0,
                            "target": 0,
                            "filled": 0,
                            "ticker": m.get("id", ""),
                        })

        except Exception as e:
            print(f"  [warn] NoVig {novig_type} failed: {e}")

    client.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge", ascending=False)
        yes_pos = ((df["side"] == "yes") & (df["edge"] > 0)).sum()
        no_pos = ((df["side"] == "no") & (df["edge"] > 0)).sum()
        print(f"  NoVig outrights: {len(df)} lines, {yes_pos} YES +edge, {no_pos} NO +edge")
    return df


# ============================================================
# KALSHI MATCHUP PRICING
# ============================================================

def price_kalshi_matchups_tourney(model_preds_df, final_scores_arr=None, player_names_list=None):
    """Price Kalshi H2H matchups using live API with 1000-contract orderbook walk.

    Returns DataFrame with matchup edges computed from effective fill prices.
    """
    import re as _re
    import time as _time
    from collections import Counter

    try:
        from kalshi_client import KalshiClient
        client = KalshiClient()
    except Exception as e:
        print(f"  [warn] KalshiClient init failed: {e} — skipping Kalshi matchups")
        return pd.DataFrame()

    FLAT_STAKE = 1000  # contracts

    try:
        all_mkts = []
        cursor = None
        while True:
            params = {"limit": 200, "status": "open", "series_ticker": "KXPGAH2H"}
            if cursor:
                params["cursor"] = cursor
            data = client._get("/markets", params=params)
            mkts = data.get("markets", [])
            all_mkts.extend(mkts)
            cursor = data.get("cursor")
            if not cursor or len(mkts) < 200:
                break
        print(f"  Kalshi H2H: fetched {len(all_mkts)} markets")
    except Exception as e:
        print(f"  [warn] Kalshi H2H fetch failed: {e}")
        return pd.DataFrame()

    if not all_mkts:
        return pd.DataFrame()

    # Detect current tournament
    def _detect(title):
        m = _re.search(r"(?:at|in|win) the (.+?)\?", title)
        if m:
            return m.group(1).strip()
        m = _re.match(r"(.+?):\s*Will", title)
        if m:
            return m.group(1).strip()
        return ""

    tourn_counts = Counter()
    for m in all_mkts:
        t = _detect(m.get("title", ""))
        if t:
            tourn_counts[t] += 1
    if tourn_counts:
        current_tourney = tourn_counts.most_common(1)[0][0]
        all_mkts = [m for m in all_mkts if _detect(m.get("title", "")) in (current_tourney, "")]

    # Group by event_ticker to pair the two sides of each H2H
    by_event = {}
    for m in all_mkts:
        et = m.get("event_ticker", "")
        by_event.setdefault(et, []).append(m)

    def _extract_player(title):
        m = _re.match(r"Will (.+?) beat (.+?) in the", title)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return "", ""

    def norm(s):
        x = s.strip().lower()
        if "," not in x:
            parts = x.rsplit(" ", 1)
            if len(parts) == 2:
                x = f"{parts[1]}, {parts[0]}"
        return name_replacements.get(x, x)

    # Build player skill lookup from model_preds
    pred_lookup = dict(zip(model_preds_df["player_name"], model_preds_df["my_pred"]))

    # Build sim win prob lookup if final_scores available
    _name_to_idx = {p: i for i, p in enumerate(player_names_list)} if player_names_list is not None else {}

    def _sim_win_prob(p1, p2):
        """Compute sim P(p1 beats p2) from final_scores, ties pushed."""
        if final_scores_arr is None or not _name_to_idx:
            return None
        i1 = _name_to_idx.get(p1)
        i2 = _name_to_idx.get(p2)
        if i1 is None or i2 is None:
            return None
        s1 = final_scores_arr[i1]
        s2 = final_scores_arr[i2]
        wins = int(np.sum(s1 < s2))
        losses = int(np.sum(s1 > s2))
        denom = wins + losses
        return wins / denom if denom > 0 else None

    rows = []
    ob_count = 0
    for event_ticker, mkts in by_event.items():
        if len(mkts) < 2:
            continue

        # Parse both sides
        sides = []
        for m in mkts:
            player_raw, opponent_raw = _extract_player(m.get("title", ""))
            if not player_raw:
                continue
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)
            if bid <= 0 or ask <= 0:
                continue
            sides.append({
                "ticker": m.get("ticker", ""),
                "player_raw": player_raw,
                "player": norm(player_raw),
                "opponent_raw": opponent_raw,
                "opponent": norm(opponent_raw),
                "bid": bid,
                "ask": ask,
            })

        if len(sides) < 2:
            continue

        a, b = sides[0], sides[1]

        # Walk orderbook for each side with flat 1000-contract stake
        for side_info in [a, b]:
            player = side_info["player"]
            if player not in pred_lookup:
                continue
            try:
                ob = client.get_orderbook(side_info["ticker"])
                _time.sleep(0.05)
                ob_count += 1
            except Exception:
                continue

            no_levels = ob.get("no", [])
            # Buy YES: walk NO levels (sellers of YES)
            fill_levels = sorted([(1.0 - p, qty) for p, qty in no_levels])
            if not fill_levels:
                continue

            # Walk book for FLAT_STAKE contracts
            filled = 0
            cost_sum = 0.0
            fee_sum = 0.0
            for price, qty in fill_levels:
                fee = _kalshi_taker_fee(price)
                take = min(qty, FLAT_STAKE - filled)
                filled += take
                cost_sum += take * price
                fee_sum += take * fee
                if filled >= FLAT_STAKE:
                    break

            if filled == 0:
                continue

            vwap = cost_sum / filled
            eff_price = vwap + (fee_sum / filled)

            # Compute edge: sim win prob vs effective price
            # For matchups, sim_prob is derived from the skill difference
            # A beats B: P(A wins) ≈ from model predictions
            mid = (side_info["bid"] + side_info["ask"]) / 2.0
            american = implied_prob_to_american_odds(eff_price)
            if american is None:
                continue

            sim_p = _sim_win_prob(player, side_info["opponent"])
            edge_val = (sim_p - eff_price) * 100 if sim_p is not None else None

            rows.append({
                "ticker": side_info["ticker"],
                "player_name": player,
                "opponent": side_info["opponent"],
                "market_type": "tournament_matchup",
                "bookmaker": "kalshi",
                "eff_price": eff_price,
                "mid": mid,
                "american_odds": american,
                "sim_prob": sim_p,
                "edge": round(edge_val, 1) if edge_val is not None else None,
                "my_pred": pred_lookup.get(player, 0),
                "opp_pred": pred_lookup.get(side_info["opponent"], 0),
                "target": FLAT_STAKE,
                "filled": filled,
            })

    client._client.close()
    print(f"  Kalshi matchups: {len(rows)} sides priced ({ob_count} orderbooks fetched)")

    df = pd.DataFrame(rows)
    return df


# ============================================================
# NOVIG MATCHUP PRICING
# ============================================================

def price_novig_matchups_tourney(model_preds_df, final_scores_arr=None, player_names_list=None):
    """Price NoVig H2H matchups using live GraphQL API.

    Uses `available` (ask) price on each outcome. No orderbook depth.
    Returns DataFrame with matchup edges.
    """
    import httpx as _httpx

    GRAPHQL_URL = "https://api.novig.us/v1/graphql"
    GQL_HEADERS = {
        "Content-Type": "application/json",
        "Origin": "https://novig.com",
        "Referer": "https://novig.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    def norm(s):
        x = s.strip().lower()
        if "," not in x:
            parts = x.rsplit(" ", 1)
            if len(parts) == 2:
                x = f"{parts[1]}, {parts[0]}"
        return name_replacements.get(x, x)

    pred_lookup = dict(zip(model_preds_df["player_name"], model_preds_df["my_pred"]))

    # Build sim win prob lookup if final_scores available
    _name_to_idx = {p: i for i, p in enumerate(player_names_list)} if player_names_list is not None else {}

    def _sim_win_prob(p1, p2):
        if final_scores_arr is None or not _name_to_idx:
            return None
        i1 = _name_to_idx.get(p1)
        i2 = _name_to_idx.get(p2)
        if i1 is None or i2 is None:
            return None
        s1 = final_scores_arr[i1]
        s2 = final_scores_arr[i2]
        wins = int(np.sum(s1 < s2))
        losses = int(np.sum(s1 > s2))
        denom = wins + losses
        return wins / denom if denom > 0 else None

    try:
        client = _httpx.Client(timeout=15.0)

        def gql(query, variables=None):
            payload = {"query": query}
            if variables:
                payload["variables"] = variables
            resp = client.post(GRAPHQL_URL, headers=GQL_HEADERS, json=payload)
            resp.raise_for_status()
            return resp.json()

        # Find tournament
        data = gql("""query {
  event(where: {league: {_eq: "PGA"}, type: {_eq: "Tournament"}, status: {_eq: "OPEN_PREGAME"}}
        order_by: {scheduled_start: asc}) {
    id description
    child_events_aggregate: events_aggregate(where: {status: {_eq: "OPEN_PREGAME"}}) {
      aggregate { count }
    }
  }
}""")
        events = data.get("data", {}).get("event", [])
        if not events:
            print("  No open PGA tournament on NoVig for matchups")
            return pd.DataFrame()
        best = max(events, key=lambda e: e.get("child_events_aggregate", {})
                   .get("aggregate", {}).get("count", 0))
        tourn_id = best["id"]

        # Fetch H2H matchups (tournament MONEY only — round matchups use different sim)
        data = gql("""query($tournId: uuid!) {
  event(where: {parent_event: {id: {_eq: $tournId}}, type: {_eq: "Game"},
                status: {_eq: "OPEN_PREGAME"}}) {
    id description
    game {
      homeTeam { name }
      awayTeam { name }
    }
    markets(where: {status: {_eq: "OPEN"}, type: {_eq: "MONEY"}}) {
      id type
      outcomes {
        index available
        competitor { name }
      }
    }
  }
}""", {"tournId": tourn_id})

        rows = []
        for e in data.get("data", {}).get("event", []):
            game = e.get("game")
            mkts = e.get("markets", [])
            if not game or not mkts:
                continue
            home = game.get("homeTeam", {}).get("name", "")
            away = game.get("awayTeam", {}).get("name", "")
            if not home or not away:
                continue

            home_norm = norm(home)
            away_norm = norm(away)

            for m in mkts:
                outcomes = m.get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                prices = {}
                for o in outcomes:
                    avail = o.get("available")
                    comp = o.get("competitor", {}).get("name", "")
                    if avail and avail > 0 and comp:
                        prices[comp] = avail

                home_price = prices.get(home, 0)
                away_price = prices.get(away, 0)

                mtype = "tournament_matchup"

                # Add row for each side that has a price and a model prediction
                for player_name, player_norm, opp_norm, price in [
                    (home, home_norm, away_norm, home_price),
                    (away, away_norm, home_norm, away_price),
                ]:
                    if price <= 0 or player_norm not in pred_lookup:
                        continue
                    american = implied_prob_to_american_odds(price)
                    if american is None:
                        continue
                    sim_p = _sim_win_prob(player_norm, opp_norm)
                    edge_val = (sim_p - price) * 100 if sim_p is not None else None
                    rows.append({
                        "ticker": m.get("id", ""),
                        "player_name": player_norm,
                        "opponent": opp_norm,
                        "market_type": mtype,
                        "bookmaker": "novig",
                        "eff_price": price,
                        "mid": price,  # no bid/ask on NoVig public data
                        "american_odds": american,
                        "sim_prob": sim_p,
                        "edge": round(edge_val, 1) if edge_val is not None else None,
                        "my_pred": pred_lookup.get(player_norm, 0),
                        "opp_pred": pred_lookup.get(opp_norm, 0),
                        "target": 0,
                        "filled": 0,
                    })

        client.close()
        print(f"  NoVig matchups: {len(rows)} sides priced")
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  [warn] NoVig matchup pricing failed: {e}")
        return pd.DataFrame()


# ============================================================
# EMAIL: Tournament Sim Summary
# ============================================================


def build_tournament_email_html(sharp_mu_df, finish_df, sample_lookup, my_pred_lookup,
                                wx_lookup=None, kalshi_df=None, novig_df=None,
                                kalshi_mu_df=None, novig_mu_df=None, arch_map=None,
                                sb_lines_lookup=None):
    timestamp_str = datetime.now().strftime('%B %d, %Y %I:%M %p')


    # Section 1: Tournament Matchups
    _exch_mu_replacements = []  # exchange-replaced matchups for Sheets storage

    # Build exchange matchup lookup: (player, opponent) → best exchange price
    _exch_mu_lookup = {}  # key: (player_norm, opp_norm) → {book, american_odds, eff_price}
    for mu_df, book_name in [(kalshi_mu_df, "kalshi"), (novig_mu_df, "novig")]:
        if mu_df is None or mu_df.empty:
            continue
        for _, r in mu_df.iterrows():
            if r.get('market_type') != 'tournament_matchup':
                continue
            player = str(r.get('player_name', '')).lower().strip()
            opp = str(r.get('opponent', '')).lower().strip()
            eff = r.get('eff_price', 0) or 0
            if eff <= 0:
                continue
            key = (player, opp)
            # Lower eff_price = better odds for the bettor
            if key not in _exch_mu_lookup or eff < _exch_mu_lookup[key]['eff_price']:
                _exch_mu_lookup[key] = {
                    'book': book_name,
                    'eff_price': eff,
                    'american_odds': r.get('american_odds'),
                }

    mu_html = ""
    if sharp_mu_df is not None and not sharp_mu_df.empty:
        filtered = sharp_mu_df.copy()
        if 'sample_on' not in filtered.columns:
            filtered['sample_on'] = filtered['bet_on'].str.lower().map(sample_lookup).fillna(0)
        if 'pred_on' not in filtered.columns:
            filtered['pred_on'] = filtered['bet_on'].str.lower().map(my_pred_lookup).fillna(0)

        filtered['dist_rounds_on'] = filtered['bet_on'].str.lower().map(dist_rounds_lookup).fillna(0)
        filtered = filtered[
            (filtered['pred_on'] > EMAIL_MU_MIN_PRED) &
            (filtered['sample_on'] >= EMAIL_MU_MIN_SAMPLE) &
            (filtered['dist_rounds_on'] >= MIN_DIST_ROUNDS)
        ]

        if not filtered.empty:
            filtered = filtered.sort_values('edge_on', ascending=False)
            rows_html = ""
            _exch_mu_replacements = []  # track exchange-replaced matchups for Sheets
            for _, row in filtered.iterrows():
                bet_on_lower = str(row.get('bet_on', '')).lower().strip()
                bet_player = str(row.get('bet_on', '')).title()
                opponent = (
                    str(row['Player 2']).title()
                    if bet_on_lower == str(row['Player 1']).lower()
                    else str(row['Player 1']).title()
                )
                _opp_lower = opponent.lower().strip()
                book = str(row.get('Bookmaker', ''))
                ties = str(row.get('Ties', ''))
                book_odds = (
                    row['P1 Odds'] if bet_on_lower == str(row['Player 1']).lower()
                    else row['P2 Odds']
                )
                fair_odds = (
                    row.get('Fair_p1') if bet_on_lower == str(row['Player 1']).lower()
                    else row.get('Fair_p2')
                )
                edge = row.get('edge_on', 0)
                pred = row.get('pred_on', 0)
                sample = int(row.get('sample_on', 0))
                _was_replaced = False

                # Check if an exchange has a better price for this matchup
                _opp_lower = opponent.lower().strip()
                _bet_norm = name_replacements.get(bet_on_lower.strip(), bet_on_lower.strip())
                _opp_norm = name_replacements.get(_opp_lower, _opp_lower)
                _exch_key = (_bet_norm, _opp_norm)
                _exch = _exch_mu_lookup.get(_exch_key)
                if _exch:
                    # Compare: exchange eff_price vs sportsbook implied prob
                    sb_implied = abs(book_odds) / (abs(book_odds) + 100) if pd.notna(book_odds) and book_odds != 0 else 1.0
                    if pd.notna(book_odds) and book_odds > 0:
                        sb_implied = 100 / (book_odds + 100)
                    exch_implied = _exch['eff_price']
                    if exch_implied < sb_implied:
                        # Exchange is cheaper — full row replacement
                        book = _exch['book']
                        book_odds = _exch['american_odds']
                        # Recompute edge: same sim prob, better price
                        sim_prob_on = row.get('my_odds_p1') if bet_on_lower == str(row['Player 1']).lower() else row.get('my_odds_p2')
                        if sim_prob_on and sim_prob_on > 0:
                            edge = (sim_prob_on - exch_implied) * 100
                        _was_replaced = True
                        # Build row matching store_tournament_matchups format
                        _repl = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
                        _repl['Bookmaker'] = book
                        _repl['edge_on'] = edge
                        if bet_on_lower == str(row['Player 1']).lower():
                            _repl['P1 Odds'] = book_odds
                        else:
                            _repl['P2 Odds'] = book_odds
                        _exch_mu_replacements.append(_repl)

                # Weather SG differential
                wx_sg = row.get('wx_diff', 0) if pd.notna(row.get('wx_diff', None)) else 0

                # Archetypes (bet_on + opponent)
                archetype = arch_map.get(str(row.get('bet_on', '')).lower().strip(), "") if arch_map else ""
                _opp_name = (
                    str(row['Player 2']).lower().strip()
                    if bet_on_lower == str(row['Player 1']).lower()
                    else str(row['Player 1']).lower().strip()
                )
                arch_against = arch_map.get(_opp_name, "") if arch_map else ""

                edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"
                pred_color = "#d4edda" if pred > 1.5 else "#ffffff"
                book_str = f"{int(book_odds):+d}" if pd.notna(book_odds) else ""
                fair_str = f"{int(fair_odds):+d}" if pd.notna(fair_odds) else ""
                book_color = "color:#6b46c1; font-weight:500;" if book in ("kalshi", "novig") else ""

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{bet_player}</td>
                    <td style="padding:6px 10px; color:#666;">vs {opponent}</td>
                    <td style="padding:6px 10px; text-align:center; {book_color}">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{ties}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{arch_against}</td>
                    <td style="padding:6px 10px; text-align:center;">{book_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px;">{wx_sg:+.2f}</td>
                    <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred:.2f}</td>
                    <td style="padding:6px 10px; text-align:center;">{sample}</td>
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
                    <th style="padding:6px 10px; text-align:center;">Type</th>
                    <th style="padding:6px 10px; text-align:center;">Vs Type</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Wx</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Sample</th>
                </tr>
                {rows_html}
            </table>"""
        else:
            mu_html = "<p>No matchup edges above threshold in this run.</p>"
    else:
        mu_html = "<p>No tournament matchup data available.</p>"

    # Section 2: Finish Positions (sharp books only, best price, no edge filter)
    SHARP_FP_BOOKS = {"pinnacle", "betonline", "betcris"}
    SHARP_FP_MIN_EDGE = 0.4
    fp_html = ""
    if finish_df is not None and not finish_df.empty:
        fp_filtered = finish_df[
            finish_df['bookmaker'].str.lower().isin(SHARP_FP_BOOKS)
        ].copy()
        # Best price per player/market (highest edge), minimum 0.4% edge
        if not fp_filtered.empty:
            fp_filtered = fp_filtered.sort_values('edge', ascending=False)
            fp_filtered = fp_filtered.drop_duplicates(subset=['player_name', 'market_type'], keep='first')
            fp_filtered = fp_filtered[fp_filtered['edge'] >= SHARP_FP_MIN_EDGE]
            fp_filtered = fp_filtered.sort_values('edge', ascending=False)

            fp_rows = ""
            for _, row in fp_filtered.iterrows():
                player = str(row.get('player_name', '')).title()
                market = str(row.get('market_type', ''))
                book = str(row.get('bookmaker', ''))
                book_odds_str = f"{int(row['american_odds']):+d}" if pd.notna(row.get('american_odds')) else ""
                fair_str = f"{int(row['my_fair']):+d}" if pd.notna(row.get('my_fair')) else ""
                edge = row.get('edge', 0)
                pred = row.get('my_pred', 0)
                sample = int(row.get('sample', 0)) if pd.notna(row.get('sample')) else 0

                _fp_wx_sg = 0.0
                if wx_lookup:
                    _fp_wx_sg = wx_lookup.get(str(row.get('player_name', '')).lower().strip(), 0)
                _fp_wx_color = "#d4edda" if abs(_fp_wx_sg) > 0.3 else "#ffffff"
                _fp_wx_str = f"{_fp_wx_sg:+.2f}" if _fp_wx_sg != 0 else "0.00"

                archetype = arch_map.get(str(row.get('player_name', '')).lower().strip(), "") if arch_map else ""

                edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"

                fp_rows += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{market}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                    <td style="padding:6px 10px; text-align:center;">{book_odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center;">{pred:.2f}</td>
                    <td style="padding:6px 10px; text-align:center;">{sample}</td>
                    <td style="padding:6px 10px; text-align:center; background:{_fp_wx_color};">{_fp_wx_str}</td>
                </tr>"""

            fp_html = f"""
            <h3 style="color:#2c5282; margin:20px 0 8px 0;">
                Finish Positions — Sharp Books (best price)
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Market</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Type</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Sample</th>
                    <th style="padding:6px 10px; text-align:center;">Wx SG</th>
                </tr>
                {fp_rows}
            </table>"""
        else:
            fp_html = "<p>No finish position data from sharp books.</p>"
    else:
        fp_html = "<p>No finish position data available.</p>"

    # Section 3: Inject best exchange rows into finish position table
    # For each player+market, if Kalshi or NoVig has a better edge than any
    # sportsbook, add that row to the finish position table.
    exchange_inject_html = ""
    _exchange_bets_for_sheets = []  # collect qualifying exchange bets for Sheets storage

    # Build lookup of best sportsbook edge per (player, market_type) from +EV bets
    _sb_best_edge = {}
    if finish_df is not None and not finish_df.empty:
        for _, r in finish_df.iterrows():
            key = (str(r.get('player_name', '')).lower().strip(), str(r.get('market_type', '')))
            edge = r.get('edge', 0) or 0
            if key not in _sb_best_edge or edge > _sb_best_edge[key]:
                _sb_best_edge[key] = edge

    # Use the full SB lines lookup (all lines, not just +EV) passed from caller
    _sb_best_line = sb_lines_lookup or {}

    # Collect exchange YES-side outrights that beat sportsbooks
    _exchange_winners = []
    for exch_df, exch_name in [(kalshi_df, "kalshi"), (novig_df, "novig")]:
        if exch_df is None or exch_df.empty:
            continue
        yes_df = exch_df[exch_df['side'] == 'yes'] if 'side' in exch_df.columns else exch_df
        for _, r in yes_df.iterrows():
            edge = r.get('edge', 0)
            if edge < 0.5:
                continue
            key = (str(r.get('player_name', '')).lower().strip(), str(r.get('market_type', '')))
            sb_edge = _sb_best_edge.get(key, -999)
            if edge > sb_edge:
                _exchange_winners.append(r)
                # Build row for Sheets storage (match sportsbook format)
                eff = r.get('eff_price', 0) or 0
                sim_p = r.get('sim_prob', 0) or 0
                dec_odds = round(1.0 / eff, 4) if eff > 0 else None
                # Kelly stake: same formula as sportsbook finish positions
                _b = (dec_odds - 1.0) if dec_odds and dec_odds > 1 else 0
                _q = 1.0 - sim_p
                _f = max((_b * sim_p - _q) / _b, 0) if _b > 0 else 0
                _kelly_stake = round(BANKROLL * KELLY_FRACTION * _f, 2)
                player_key = str(r.get('player_name', '')).lower().strip()
                _exchange_bets_for_sheets.append({
                    'player_name': r.get('player_name', ''),
                    'market_type': r.get('market_type', ''),
                    'bookmaker': exch_name,
                    'american_odds': r.get('american_odds'),
                    'decimal_odds': dec_odds,
                    'my_fair': r.get('my_fair'),
                    'sim_prob': sim_p,
                    'edge': edge,
                    'stake': _kelly_stake,
                    'my_pred': my_pred_lookup.get(player_key, 0),
                    'sample': sample_lookup.get(player_key, 0),
                })

    if _exchange_winners:
        _exch_df = pd.DataFrame([r.to_dict() if hasattr(r, 'to_dict') else dict(r) for r in _exchange_winners])
        _exch_df = _exch_df.sort_values('edge', ascending=False)

        exch_rows_html = ""
        for _, row in _exch_df.iterrows():
            player = str(row.get('player_name', '')).title()
            market = str(row.get('market_type', ''))
            book = str(row.get('bookmaker', ''))

            eff = row.get('eff_price', 0) or 0
            eff_american = f"{int(row['american_odds']):+d}" if pd.notna(row.get('american_odds')) else ""
            fair_str = f"{int(row['my_fair']):+d}" if pd.notna(row.get('my_fair')) else ""
            edge = row.get('edge', 0)
            sim_prob = row.get('sim_prob', 0)
            edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"

            # Best sportsbook line for this player+market
            key = (str(row.get('player_name', '')).lower().strip(), market)
            sb_line = _sb_best_line.get(key)
            sb_line_str = f"{sb_line:+d}" if sb_line is not None else "—"

            exch_rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{market}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500; color:#6b46c1;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{eff * 100:.1f}¢</td>
                    <td style="padding:6px 10px; text-align:center;">{eff_american}</td>
                    <td style="padding:6px 10px; text-align:center; color:#888;">{sb_line_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center;">{sim_prob:.1%}</td>
                </tr>"""

        exchange_inject_html = f"""
            <h3 style="color:#6b46c1; margin:20px 0 8px 0;">
                Exchange Edges — Better Than Sportsbooks (nodh)
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Market</th>
                    <th style="padding:6px 10px; text-align:center;">Exchange</th>
                    <th style="padding:6px 10px; text-align:center;">Price ¢</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">SB Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Sim Prob</th>
                </tr>
                {exch_rows_html}
            </table>"""

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:960px; margin:0 auto; padding:20px;">
        <h2 style="margin-bottom:4px;">Tournament Sim &mdash; {tourney.replace('_', ' ').title()}</h2>
        <p style="color:#666; margin-top:0;">{timestamp_str} | {SIMULATIONS:,} simulations | Course mults: OTT={course_mults[0]:.2f} APP={course_mults[1]:.2f} ARG={course_mults[2]:.2f} PUTT={course_mults[3]:.2f}</p>

        {mu_html}
        {fp_html}
        {exchange_inject_html}

        <p style="color:#999; font-size:11px; margin-top:30px;">
            Fair = our no-vig price (ties push, except exchanges = nodh) | Edge = expected return % |
            Wx = R1+R2 weather SG advantage vs opponent (positive = favorable) |
            Pred = model SG prediction
        </p>
    </body>
    </html>"""

    # Stash for caller access
    build_tournament_email_html._exchange_bets = _exchange_bets_for_sheets
    build_tournament_email_html._exchange_mu_replacements = _exch_mu_replacements
    return html


def send_tournament_email(sharp_mu_df, finish_df, sample_lookup, my_pred_lookup,
                          attachment_paths=None, wx_lookup=None, kalshi_df=None,
                          novig_df=None, kalshi_mu_df=None, novig_mu_df=None,
                          arch_map=None, sb_lines_lookup=None):
    if not EMAIL_PASSWORD:
        print("  [warn] EMAIL_PASSWORD not set. Skipping email.")
        return
    if not EMAIL_FROM or not EMAIL_TO or EMAIL_TO == ['']:
        print("  [warn] EMAIL_FROM or EMAIL_TO not configured. Skipping email.")
        return

    try:
        html = build_tournament_email_html(sharp_mu_df, finish_df, sample_lookup, my_pred_lookup,
                                           wx_lookup=wx_lookup, kalshi_df=kalshi_df,
                                           novig_df=novig_df, kalshi_mu_df=kalshi_mu_df,
                                           novig_mu_df=novig_mu_df,
                                           arch_map=arch_map, sb_lines_lookup=sb_lines_lookup)

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Tournament Sim -- {tourney.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)

        msg.attach(MIMEText(html, "html", "utf-8"))

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

    # Store qualifying exchange bets to Google Sheets
    _exch_bets = getattr(build_tournament_email_html, '_exchange_bets', [])
    _exch_mu = getattr(build_tournament_email_html, '_exchange_mu_replacements', [])
    if _exch_bets or _exch_mu:
        try:
            from sheets_storage import get_spreadsheet, store_finish_positions, store_tournament_matchups, load_dg_id_lookup
            _exch_ss = get_spreadsheet()
            _exch_dg = load_dg_id_lookup(tourney, name_replacements)

            # Store exchange outright bets
            if _exch_bets:
                _exch_store_df = pd.DataFrame(_exch_bets)
                store_finish_positions(
                    _exch_store_df, tourney, _event_id,
                    dg_id_lookup=_exch_dg,
                    spreadsheet=_exch_ss,
                )
                print(f"  [ok] Stored {len(_exch_bets)} exchange outright bets to Sheets")

            # Store exchange matchup replacements
            if _exch_mu:
                _exch_mu_df = pd.DataFrame(_exch_mu)
                store_tournament_matchups(
                    _exch_mu_df, tourney, _event_id,
                    dg_id_lookup=_exch_dg,
                    spreadsheet=_exch_ss,
                )
                print(f"  [ok] Stored {len(_exch_mu)} exchange matchup bets to Sheets")

        except Exception as _se:
            print(f"  [warn] Exchange Sheets storage failed: {_se}")


def send_exchange_email(kalshi_df=None, novig_df=None, kalshi_mu_df=None, novig_mu_df=None):
    """Email 2: Exchange-only opportunities (Kalshi + NoVig, independent tables + fades)."""
    if not EMAIL_PASSWORD or not EMAIL_FROM or not EMAIL_TO or EMAIL_TO == ['']:
        return

    def _exch_row(row, book):
        player = str(row.get('player_name', '')).title()
        market = str(row.get('market_type', ''))
        eff = row.get('eff_price', 0) or 0
        sim_prob = row.get('sim_prob', 0)
        edge = row.get('edge', 0)
        american = f"{int(row['american_odds']):+d}" if pd.notna(row.get('american_odds')) else ""
        fair = f"{int(row['my_fair']):+d}" if pd.notna(row.get('my_fair')) else ""
        target = int(row.get('target', 0) or 0)
        filled = int(row.get('filled', 0) or 0)
        fill_str = f"{filled:,}/{target:,}" if target else "—"
        edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"
        return f"""<tr>
            <td style="padding:5px 8px; font-weight:600;">{player}</td>
            <td style="padding:5px 8px; text-align:center;">{market}</td>
            <td style="padding:5px 8px; text-align:center;">{eff*100:.1f}¢</td>
            <td style="padding:5px 8px; text-align:center;">{sim_prob*100:.1f}¢</td>
            <td style="padding:5px 8px; text-align:center;">{american}</td>
            <td style="padding:5px 8px; text-align:center;">{fair}</td>
            <td style="padding:5px 8px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
            <td style="padding:5px 8px; text-align:center;">{fill_str}</td>
        </tr>"""

    def _mu_row(row):
        player = str(row.get('player_name', '')).title()
        opp = str(row.get('opponent', '')).title()
        eff = row.get('eff_price', 0) or 0
        american = f"{int(row['american_odds']):+d}" if pd.notna(row.get('american_odds')) else ""
        sim_p = row.get('sim_prob')
        fair_str = f"{int(implied_prob_to_american_odds(sim_p)):+d}" if sim_p and sim_p > 0 else "—"
        edge = row.get('edge', 0) or 0
        edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"
        mtype = str(row.get('market_type', ''))
        target = int(row.get('target', 0) or 0)
        filled = int(row.get('filled', 0) or 0)
        fill_str = f"{filled:,}/{target:,}" if target else "—"
        return f"""<tr>
            <td style="padding:5px 8px; font-weight:600;">{player}</td>
            <td style="padding:5px 8px;">{opp}</td>
            <td style="padding:5px 8px; text-align:center;">{mtype}</td>
            <td style="padding:5px 8px; text-align:center;">{eff*100:.1f}¢</td>
            <td style="padding:5px 8px; text-align:center;">{american}</td>
            <td style="padding:5px 8px; text-align:center;">{fair_str}</td>
            <td style="padding:5px 8px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
            <td style="padding:5px 8px; text-align:center;">{fill_str}</td>
        </tr>"""

    _hdr = """<table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:12px; width:100%;">
        <tr style="background:#343a40; color:white;">
            <th style="padding:5px 8px; text-align:left;">Player</th>
            <th style="padding:5px 8px; text-align:center;">Market</th>
            <th style="padding:5px 8px; text-align:center;">Price ¢</th>
            <th style="padding:5px 8px; text-align:center;">Fair ¢</th>
            <th style="padding:5px 8px; text-align:center;">Line</th>
            <th style="padding:5px 8px; text-align:center;">Fair</th>
            <th style="padding:5px 8px; text-align:center;">Edge</th>
            <th style="padding:5px 8px; text-align:center;">Fill</th>
        </tr>"""

    _mu_hdr = """<table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:12px; width:100%;">
        <tr style="background:#343a40; color:white;">
            <th style="padding:5px 8px; text-align:left;">Bet On</th>
            <th style="padding:5px 8px; text-align:left;">Opponent</th>
            <th style="padding:5px 8px; text-align:center;">Type</th>
            <th style="padding:5px 8px; text-align:center;">Price ¢</th>
            <th style="padding:5px 8px; text-align:center;">Line</th>
            <th style="padding:5px 8px; text-align:center;">Fair</th>
            <th style="padding:5px 8px; text-align:center;">Edge</th>
            <th style="padding:5px 8px; text-align:center;">Fill</th>
        </tr>"""

    sections = []
    timestamp_str = datetime.now().strftime('%B %d, %Y %I:%M %p')

    # Kalshi YES outrights
    if kalshi_df is not None and not kalshi_df.empty:
        yes = kalshi_df[(kalshi_df['side'] == 'yes') & (kalshi_df['edge'] > 0)].sort_values('edge', ascending=False)
        if not yes.empty:
            rows_html = "".join(_exch_row(r, "kalshi") for _, r in yes.head(20).iterrows())
            sections.append(f'<h3 style="color:#2c5282;">Kalshi YES — Top 20</h3>{_hdr}{rows_html}</table>')

        # Kalshi fades
        no = kalshi_df[(kalshi_df['side'] == 'no') & (kalshi_df['edge'] > 0.5)].sort_values('edge', ascending=False)
        if not no.empty:
            rows_html = "".join(_exch_row(r, "kalshi") for _, r in no.head(20).iterrows())
            sections.append(f'<h3 style="color:#9b2c2c;">Kalshi NO (Fade) — edge &gt; 0.5%</h3>{_hdr}{rows_html}</table>')

    # NoVig YES outrights
    if novig_df is not None and not novig_df.empty:
        yes = novig_df[(novig_df['side'] == 'yes') & (novig_df['edge'] > 0)].sort_values('edge', ascending=False)
        if not yes.empty:
            rows_html = "".join(_exch_row(r, "novig") for _, r in yes.head(20).iterrows())
            sections.append(f'<h3 style="color:#2c5282;">NoVig YES — Top 20</h3>{_hdr}{rows_html}</table>')

        # NoVig fades
        no = novig_df[(novig_df['side'] == 'no') & (novig_df['edge'] > 0.5)].sort_values('edge', ascending=False)
        if not no.empty:
            rows_html = "".join(_exch_row(r, "novig") for _, r in no.head(20).iterrows())
            sections.append(f'<h3 style="color:#9b2c2c;">NoVig NO (Fade) — edge &gt; 0.5%</h3>{_hdr}{rows_html}</table>')

    # Kalshi matchups (edge > 5%)
    if kalshi_mu_df is not None and not kalshi_mu_df.empty:
        _kmu = kalshi_mu_df[kalshi_mu_df['edge'].notna() & (kalshi_mu_df['edge'] > 5)].sort_values('edge', ascending=False)
        if not _kmu.empty:
            rows_html = "".join(_mu_row(r) for _, r in _kmu.iterrows())
            sections.append(f'<h3 style="color:#2c5282;">Kalshi Matchups — edge &gt; 5%</h3>{_mu_hdr}{rows_html}</table>')

    # NoVig matchups (edge > 5%)
    if novig_mu_df is not None and not novig_mu_df.empty:
        _nmu = novig_mu_df[novig_mu_df['edge'].notna() & (novig_mu_df['edge'] > 5)].sort_values('edge', ascending=False)
        if not _nmu.empty:
            rows_html = "".join(_mu_row(r) for _, r in _nmu.iterrows())
            sections.append(f'<h3 style="color:#2c5282;">NoVig Matchups — edge &gt; 5%</h3>{_mu_hdr}{rows_html}</table>')

    if not sections:
        print("  [exchange email] No exchange data — skipping")
        return

    body = f"""<html><body style="font-family:Arial,sans-serif; max-width:960px; margin:0 auto; padding:20px;">
        <h2>Exchange Opportunities &mdash; {tourney.replace('_', ' ').title()}</h2>
        <p style="color:#666;">{timestamp_str} | All prices are nodh (no dead heat)</p>
        {''.join(sections)}
        <p style="color:#999; font-size:11px; margin-top:30px;">
            Kalshi: Kelly-walked effective fill ($30K @ 0.6x, incl. 7% taker fee) |
            NoVig: ask price (no fees for manual orders)
        </p>
    </body></html>"""

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Exchange Opportunities -- {tourney.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)
        msg.attach(MIMEText(body, "html", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print("  [ok] Exchange opportunities email sent")
    except Exception as e:
        print(f"  [warn] Exchange email failed: {e}")


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
    # Drop rows with missing odds or probabilities
    _before = len(df_match)
    df_match = df_match.dropna(subset=['P1 Odds', 'P2 Odds', 'my_odds_p1', 'my_odds_p2']).copy()
    if len(df_match) < _before:
        print(f"  [matchups] Dropped {_before - len(df_match)} rows with missing odds/probs")

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

    df_match['edge_p1'] = (prob_p1 - 1.0 / df_match['p1_dec']) * 100
    df_match['edge_p2'] = (prob_p2 - 1.0 / df_match['p2_dec']) * 100

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
            (dfb['my_odds_p1_ties_loss'] - 1.0 / dfb['p1_decimal_odds']) * 100,
            (dfb['my_odds_p1'] - 1.0 / dfb['p1_decimal_odds']) * 100
        )
        dfb['edge_p2'] = np.where(
            dfb['use_ties_loss'],
            (dfb['my_odds_p2_ties_loss'] - 1.0 / dfb['p2_decimal_odds']) * 100,
            (dfb['my_odds_p2'] - 1.0 / dfb['p2_decimal_odds']) * 100
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

        # Add wind advantage columns to combined_df (for Sheets storage)
        wind_lookup = dict(zip(wx['player_name'].str.lower(), wx['wind_adv_r1_2']))
        combined_df['wind_on'] = combined_df['bet_on'].str.lower().map(wind_lookup)
        combined_df['bet_against'] = combined_df.apply(
            lambda r: r['Player 2'] if r['bet_on'] == r['Player 1'] else r['Player 1'], axis=1
        )
        combined_df['wind_against'] = combined_df['bet_against'].str.lower().map(wind_lookup)
        combined_df['wind_diff'] = combined_df['wind_on'] - combined_df['wind_against']

        # --- Weather SG differential (R1+R2) ---
        # Total weather benefit per player = wind_adv + dew_adv (both centered, positive = helps player)
        dew_lookup = dict(zip(wx['player_name'].str.lower(), wx['dew_adv_r1_2']))
        combined_df['dew_on'] = combined_df['bet_on'].str.lower().map(dew_lookup)
        combined_df['dew_against'] = combined_df['bet_against'].str.lower().map(dew_lookup)
        combined_df['wx_on'] = combined_df['wind_on'] + combined_df['dew_on']
        combined_df['wx_against'] = combined_df['wind_against'] + combined_df['dew_against']
        combined_df['wx_diff'] = combined_df['wx_on'] - combined_df['wx_against']

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

        # --- Compute archetypes (before email so type_on appears in tables) ---
        if _precomputed_arch_map:
            _arch_map = _precomputed_arch_map
            print(f"[archetypes] Reusing precomputed map ({len(_arch_map)} players)")
        else:
            _arch_map = {}
            try:
                from sg_diagnostic import compute_rolling_archetypes
                field_players = model_preds['player_name'].unique().tolist()
                _arch_df = compute_rolling_archetypes(_event_id, field_players)
                _arch_map = dict(zip(_arch_df['player_name'], _arch_df['archetype']))
                print(f"[archetypes] Computed for {len(_arch_map)} players")
            except Exception as _arch_err:
                print(f"[archetypes] Computation skipped: {_arch_err}")

        # --- Send tournament email ---
        print("\n[email] Building tournament sim email...")
        _finish_for_email = combined_finish_df if ('combined_finish_df' in dir() and not combined_finish_df.empty) else None
        _attachments = []
        # Weather lookup for finish position context: total SG shift R1+R2
        _wx_fp_lookup = dict(zip(
            wx['player_name'].str.lower(),
            wx['wind_adv_r1_2'] + wx['dew_adv_r1_2'],
        ))
        # Price exchange outrights + matchups for email
        _kalshi_df = pd.DataFrame()
        _novig_df = pd.DataFrame()
        _kalshi_mu_df = pd.DataFrame()
        _novig_mu_df = pd.DataFrame()
        try:
            _kalshi_df = price_kalshi_outrights_tourney(finish_equity_df, my_pred_lookup, sample_lookup)
        except Exception as _ke:
            print(f"  [warn] Kalshi outright pricing failed: {_ke}")
        try:
            _novig_df = price_novig_outrights_tourney(finish_equity_df, my_pred_lookup, sample_lookup)
        except Exception as _ke:
            print(f"  [warn] NoVig outright pricing failed: {_ke}")
        try:
            _kalshi_mu_df = price_kalshi_matchups_tourney(model_preds, final_scores, player_names)
        except Exception as _ke:
            print(f"  [warn] Kalshi matchup pricing failed: {_ke}")
        try:
            _novig_mu_df = price_novig_matchups_tourney(model_preds, final_scores, player_names)
        except Exception as _ke:
            print(f"  [warn] NoVig matchup pricing failed: {_ke}")

        _skip = bool(os.getenv("SKIP_STORAGE"))
        if _skip:
            print("\n[skip] SKIP_STORAGE set — skipping emails and storage")

        # Email 1: bettable edges (sportsbooks + best exchange rows)
        if not _skip:
            send_tournament_email(
            sharp_mu_df=sharp_df,
            finish_df=_finish_for_email,
            sample_lookup=sample_lookup,
            my_pred_lookup=my_pred_lookup,
            attachment_paths=_attachments,
            wx_lookup=_wx_fp_lookup,
            kalshi_df=_kalshi_df if not _kalshi_df.empty else None,
            novig_df=_novig_df if not _novig_df.empty else None,
            kalshi_mu_df=_kalshi_mu_df if not _kalshi_mu_df.empty else None,
            novig_mu_df=_novig_mu_df if not _novig_mu_df.empty else None,
            arch_map=_arch_map,
            sb_lines_lookup=_all_sb_lines_lookup,
        )

        # --- Correlated Kelly staking (console diagnostic only) ---
        try:
            from correlated_kelly import optimize_correlated_kelly, print_correlated_kelly_report, build_correlated_kelly_email_html

            # Re-filter finish bets (same criteria as email builder)
            _ck_finish = []
            if _finish_for_email is not None and not _finish_for_email.empty:
                _ck_fp = _finish_for_email[
                    _finish_for_email['bookmaker'].str.lower().isin({"pinnacle", "betonline", "betcris"})
                ].copy()
                if not _ck_fp.empty:
                    _ck_fp = _ck_fp.sort_values('edge', ascending=False)
                    _ck_fp = _ck_fp.drop_duplicates(subset=['player_name', 'market_type'], keep='first')
                    _ck_fp = _ck_fp[_ck_fp['edge'] >= 0.4]
                    _fp_prob_col = {'win': 'simulated_win_prob', 'top_5': 'top_5',
                                    'top_10': 'top_10', 'top_20': 'top_20'}
                    for _, _r in _ck_fp.iterrows():
                        _mt = str(_r['market_type'])
                        _pcol = _fp_prob_col.get(_mt, _mt)
                        _sp = float(_r[_pcol]) if _pcol in _r.index and pd.notna(_r[_pcol]) else 0.0
                        _ck_finish.append({
                            'player_name': str(_r['player_name']).lower().strip(),
                            'market_type': _mt,
                            'decimal_odds': float(_r['decimal_odds']),
                            'sim_prob': _sp,
                            'bookmaker': str(_r.get('bookmaker', '')),
                            'american_odds': _r.get('american_odds', ''),
                        })

            # Add exchange finish bets (beat sportsbook edge, edge > 0.5%)
            _ck_exch_fp = getattr(build_tournament_email_html, '_exchange_bets', [])
            for _eb in _ck_exch_fp:
                _pn = str(_eb.get('player_name', '')).lower().strip()
                _mt = str(_eb.get('market_type', ''))
                _sp = float(_eb.get('sim_prob', 0))
                _dec = float(_eb.get('decimal_odds', 0))
                if _sp > 0 and _dec > 1:
                    _ck_finish.append({
                        'player_name': _pn,
                        'market_type': _mt,
                        'decimal_odds': _dec,
                        'sim_prob': _sp,
                        'bookmaker': str(_eb.get('bookmaker', '')),
                        'american_odds': _eb.get('american_odds', ''),
                    })

            # Re-filter matchup bets (same criteria as email builder)
            _ck_matchups = []
            if not sharp_df.empty:
                _ck_mu = sharp_df.copy()
                if 'sample_on' not in _ck_mu.columns:
                    _ck_mu['sample_on'] = _ck_mu['bet_on'].str.lower().map(sample_lookup).fillna(0)
                if 'pred_on' not in _ck_mu.columns:
                    _ck_mu['pred_on'] = _ck_mu['bet_on'].str.lower().map(my_pred_lookup).fillna(0)
                _ck_mu['dist_rounds_on'] = _ck_mu['bet_on'].str.lower().map(dist_rounds_lookup).fillna(0)
                _ck_mu = _ck_mu[
                    (_ck_mu['pred_on'] > EMAIL_MU_MIN_PRED) &
                    (_ck_mu['sample_on'] >= EMAIL_MU_MIN_SAMPLE) &
                    (_ck_mu['dist_rounds_on'] >= MIN_DIST_ROUNDS)
                ]
                for _, _r in _ck_mu.iterrows():
                    _bet_on = str(_r['bet_on']).lower().strip()
                    _bet_against = str(_r.get('bet_against', '')).lower().strip()
                    if not _bet_against:
                        _bet_against = (str(_r['Player 2']).lower().strip()
                                        if _bet_on == str(_r['Player 1']).lower().strip()
                                        else str(_r['Player 1']).lower().strip())
                    # Decimal odds and sim probability for the bet_on side
                    def _fair_to_prob(fair_am):
                        """Convert Fair american odds to implied probability."""
                        if pd.isna(fair_am) or fair_am == 0:
                            return 0.0
                        fair_am = float(fair_am)
                        if fair_am < 0:
                            return abs(fair_am) / (abs(fair_am) + 100)
                        return 100 / (fair_am + 100)

                    if _bet_on == str(_r['Player 1']).lower().strip():
                        _p1_odds = float(_r['P1 Odds'])
                        _dec = (_p1_odds / 100 + 1) if _p1_odds > 0 else (100 / abs(_p1_odds) + 1)
                        _sim_p = _fair_to_prob(_r.get('Fair_p1'))
                        _am = _r['P1 Odds']
                    else:
                        _p2_odds = float(_r['P2 Odds'])
                        _dec = (_p2_odds / 100 + 1) if _p2_odds > 0 else (100 / abs(_p2_odds) + 1)
                        _sim_p = _fair_to_prob(_r.get('Fair_p2'))
                        _am = _r['P2 Odds']
                    _ck_matchups.append({
                        'bet_on': _bet_on,
                        'opponent': _bet_against,
                        'decimal_odds': _dec,
                        'sim_win_prob': _sim_p,
                        'bookmaker': str(_r.get('Bookmaker', '')),
                        'american_odds': _am,
                    })

            # Add exchange-replaced matchups (better exchange price, sized alongside sportsbook)
            _ck_exch_mu = getattr(build_tournament_email_html, '_exchange_mu_replacements', [])
            for _em in _ck_exch_mu:
                _bet_on = str(_em.get('bet_on', '')).lower().strip()
                if not _bet_on:
                    _bet_on = (str(_em['Player 1']).lower().strip()
                               if _em.get('edge_p1', 0) > _em.get('edge_p2', 0)
                               else str(_em['Player 2']).lower().strip())
                _opp = (str(_em['Player 2']).lower().strip()
                        if _bet_on == str(_em['Player 1']).lower().strip()
                        else str(_em['Player 1']).lower().strip())
                if _bet_on == str(_em['Player 1']).lower().strip():
                    _am = _em.get('P1 Odds', 0)
                    _sim_p = float(_em.get('my_odds_p1', 0) or 0)
                else:
                    _am = _em.get('P2 Odds', 0)
                    _sim_p = float(_em.get('my_odds_p2', 0) or 0)
                _am_f = float(_am) if pd.notna(_am) else 0
                _dec = (_am_f / 100 + 1) if _am_f > 0 else (100 / abs(_am_f) + 1) if _am_f < 0 else 0
                if _sim_p > 0 and _dec > 1:
                    _ck_matchups.append({
                        'bet_on': _bet_on,
                        'opponent': _opp,
                        'decimal_odds': _dec,
                        'sim_win_prob': _sim_p,
                        'bookmaker': str(_em.get('Bookmaker', '')),
                        'american_odds': _am,
                    })

            if _ck_finish or _ck_matchups:
                # Build finish prob lookup from sim's authoritative calculations
                _ck_fprobs = {}
                if 'finish_equity_df' in dir() and not finish_equity_df.empty:
                    for _, _frow in finish_equity_df.iterrows():
                        _ck_fprobs[str(_frow['player_name']).lower().strip()] = {
                            'win': float(_frow.get('simulated_win_prob', 0)),
                            'top_5': float(_frow.get('top_5', 0)),
                            'top_10': float(_frow.get('top_10', 0)),
                            'top_20': float(_frow.get('top_20', 0)),
                        }
                _ck_result = optimize_correlated_kelly(
                    _ck_finish, _ck_matchups,
                    final_scores, player_names,
                    finish_probs=_ck_fprobs,
                    bankroll=30_000.0,
                    kelly_fraction=0.60,
                    book_caps={'pinnacle': 500},
                )
                print_correlated_kelly_report(_ck_result)

                # Email 3: correlated Kelly staking
                if _ck_result.get('bets') and EMAIL_PASSWORD and EMAIL_FROM and EMAIL_TO and EMAIL_TO != [''] and not _skip:
                    try:
                        _ck_html = build_correlated_kelly_email_html(_ck_result, tourney=tourney)
                        _ck_msg = MIMEMultipart("mixed")
                        _ck_msg["Subject"] = f"Correlated Kelly -- {tourney.replace('_', ' ').title()}"
                        _ck_msg["From"] = EMAIL_FROM
                        _ck_msg["To"] = ", ".join(EMAIL_TO)
                        _ck_msg.attach(MIMEText(_ck_html, "html", "utf-8"))
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as _ck_srv:
                            _ck_srv.login(EMAIL_FROM, EMAIL_PASSWORD)
                            _ck_srv.sendmail(EMAIL_FROM, EMAIL_TO, _ck_msg.as_string())
                        print("  [ok] Correlated Kelly email sent")
                    except Exception as _ck_email_err:
                        print(f"  [warn] Correlated Kelly email failed: {_ck_email_err}")
            else:
                print("[correlated-kelly] No qualifying bets for correlated sizing.")
        except Exception as _ck_err:
            print(f"[correlated-kelly] Error: {_ck_err}")

        # Email 2: exchange-only opportunities
        if not _skip:
            send_exchange_email(
                kalshi_df=_kalshi_df if not _kalshi_df.empty else None,
                novig_df=_novig_df if not _novig_df.empty else None,
                kalshi_mu_df=_kalshi_mu_df if not _kalshi_mu_df.empty else None,
                novig_mu_df=_novig_mu_df if not _novig_mu_df.empty else None,
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

if is_valid_run_time() and not os.getenv("SKIP_STORAGE"):
    print("\n[storage] Saving to Google Sheets...")
    try:
        # Single auth for all store calls
        spreadsheet = get_spreadsheet()

        # Build dg_id lookup from the predictions file
        dg_id_lookup = load_dg_id_lookup(tourney, name_replacements)

        # Reuse archetypes computed before email (or compute if not yet done)
        if not _arch_map:
            try:
                from sg_diagnostic import compute_rolling_archetypes
                field_players = model_preds['player_name'].unique().tolist()
                _arch_df = compute_rolling_archetypes(_event_id, field_players)
                _arch_map = dict(zip(_arch_df['player_name'], _arch_df['archetype']))
                print(f"[storage] Computed archetypes for {len(_arch_map)} players")
            except Exception as _arch_err:
                print(f"[storage] Archetype computation skipped: {_arch_err}")

        # Add type_on to matchup and finish position DataFrames
        if _arch_map:
            if 'combined_df' in dir() and not combined_df.empty:
                combined_df['type_on'] = (
                    combined_df['bet_on'].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
                )
            if 'combined_finish_df' in dir() and not combined_finish_df.empty:
                combined_finish_df['type_on'] = (
                    combined_finish_df['player_name'].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
                )

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

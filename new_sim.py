# ============================
# PGA SIM: Category-First Draws with Course Variance Multipliers
# Draws SG categories from course-adjusted multivariate normal, sums to total.
# ============================

import argparse
import os
import sys
import numpy as np
import pandas as pd
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from numpy.linalg import cholesky

# --- CLI flags (parsed early so the rest of the script can branch on them) ---
# --sim-only:    run sim, save cache, exit before pricing/email/storage.
# --price-only:  load cached sim outputs, skip Monte Carlo, run pricing + email
#                + storage against fresh API/scraped odds.
# --reprice:     like --price-only but route Sheets storage through dedup so
#                only newly-priced rows get appended.
_parser = argparse.ArgumentParser(description="Pre-tournament category-first sim")
_parser.add_argument("--sim-only", action="store_true",
                     help="Run sim, save cache, exit before pricing/email")
_parser.add_argument("--price-only", action="store_true",
                     help="Skip sim (load cache), run pricing + email + storage")
_parser.add_argument("--reprice", action="store_true",
                     help="Like --price-only but dedup against existing Sheets rows")
_parser.add_argument("--use-python", action="store_true",
                     help="Use the legacy Python draw cascade instead of the Rust "
                          "sims_kernel, which is now the PRODUCTION DEFAULT. The "
                          "Python kernel still runs as a shadow; this flag just "
                          "keeps its output instead of the Rust one.")
_parser.add_argument("--no-week-latent", action="store_true",
                     help="Disable the shared week-level form draw (sim_inputs."
                          "WEEK_LATENT_SD). Off-state is bit-identical to the "
                          "pre-latent cascade; use for parity fixture capture.")
_parser.add_argument("--no-skew-cal", action="store_true",
                     help="Disable the 72-hole totals skew top-up "
                          "(skew_calibration.calibrate_total_skew).")
args = _parser.parse_args()
if args.reprice:
    args.price_only = True
if args.sim_only and args.price_only:
    _parser.error("Cannot use --sim-only and --price-only/--reprice together")

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
_course_id       = _cfg["course_id"]
# Player adjustments from Sheet (survives sim_inputs overwrites)
_sheet_manual_boosts = _cfg.get("manual_boosts", {})
_sheet_archetype_boosts = _cfg.get("archetype_boosts", {})

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
import sim_inputs as _sim_inputs_mod
# Week-level form latent sigma (SG/round); getattr so a stale synced
# sim_inputs copy degrades to latent-off instead of crashing the sim.
WEEK_LATENT_SD = float(getattr(_sim_inputs_mod, "WEEK_LATENT_SD", 0.0))

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
    fetch_field_updates,
)
try:
    import pandas as _pd_coords
    _coords_csv = os.path.join(os.path.dirname(__file__), "permanent_data", "course_coordinates.csv")
    _coords_df = _pd_coords.read_csv(_coords_csv)
    _coords_row = _coords_df[_coords_df["course_id"] == _course_id]
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
        print(f"[weather] No coordinates for course_id={_course_id} — skipping climo blend")
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

# Input predictions. Preference order:
#   1. final_predictions_{tourney}.csv  (post-mkt_regress, second-pass input)
#   2. pre_sim_summary_{tourney}.csv    (pre_sim_skill output: pred with manual
#                                         boosts + DG blend + sheet boosts baked in)
#   3. pre_course_fit_{tourney}.csv     (raw external skill, last-resort fallback
#                                         if pre_sim_skill hasn't been run)
_final_pred_path = f"final_predictions_{tourney}.csv"
_pre_sim_path = f"pre_sim_summary_{tourney}.csv"
_pre_course_path = f"pre_course_fit_{tourney}.csv"
if os.path.exists(_final_pred_path):
    PRED_PATH = _final_pred_path
elif os.path.exists(_pre_sim_path):
    PRED_PATH = _pre_sim_path
else:
    PRED_PATH = _pre_course_path
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

def format_units(stake_dollars):
    """Format a $ stake as units (1u = $200). Returns '—' for non-positive.
    Sub-0.3u stakes show 2 decimals so 0.05u doesn't render as '0.1u'."""
    if not stake_dollars or stake_dollars <= 0:
        return "—"
    u = stake_dollars / 200.0
    return f"{u:.2f}u" if u < 0.3 else f"{u:.1f}u"

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

# --- Archetype boosts from Google Sheet (first pass only) ---
if _sheet_archetype_boosts and PRED_PATH != _final_pred_path:
    try:
        from sg_diagnostic import compute_rolling_archetypes
        field_players = model_preds['player_name'].tolist()
        _arch_boost_df = compute_rolling_archetypes(_event_id, field_players)
        _arch_boost_map = dict(zip(_arch_boost_df['player_name'], _arch_boost_df['archetype']))
        _arch_boost_applied = 0
        for idx, row in model_preds.iterrows():
            arch = _arch_boost_map.get(row['player_name'], "")
            if arch in _sheet_archetype_boosts:
                boost = _sheet_archetype_boosts[arch]
                old_val = row['my_pred']
                model_preds.at[idx, 'my_pred'] = old_val + boost
                _arch_boost_applied += 1
        if _arch_boost_applied:
            print(f"[archetype boost] Applied to {_arch_boost_applied} players: " +
                  ", ".join(f"{k}: {v:+.2f}" for k, v in _sheet_archetype_boosts.items()))
        # Cache archetype map so we don't recompute later
        _precomputed_arch_map = _arch_boost_map
    except Exception as e:
        print(f"[archetype boost] Skipped: {e}")
        _precomputed_arch_map = None
elif _sheet_archetype_boosts:
    print("[archetype boost] Second pass - skipping (already baked into final_predictions)")
    _precomputed_arch_map = None
else:
    _precomputed_arch_map = None

# --- Manual boosts from Google Sheet (pre_course_fit fallback only) ---
# pre_sim_skill.py already bakes sheet boosts into pre_sim_summary's pred
# (tracked in its 'sheet_boost' column), and final_predictions inherits them
# through init_sim_skill -> mkt_regress. Only the raw pre_course_fit fallback
# has never seen them — re-applying on the pre_sim_summary path double-boosts.
if _sheet_manual_boosts and PRED_PATH == _pre_course_path:
    _boost_applied = 0
    for name, boost in _sheet_manual_boosts.items():
        mask = model_preds['player_name'] == name
        if mask.any():
            old_val = model_preds.loc[mask, 'my_pred'].iloc[0]
            model_preds.loc[mask, 'my_pred'] += boost
            print(f"[boost] {name}: {old_val:.3f} -> {old_val + boost:.3f} ({boost:+.3f})")
            _boost_applied += 1
        else:
            print(f"[boost] Warning: {name} not found in field")
    print(f"[boost] Applied {_boost_applied} manual boosts from Sheet")
elif _sheet_manual_boosts and PRED_PATH == _pre_sim_path:
    print("[boost] Skipping manual boosts (already baked into pre_sim_summary by pre_sim_skill)")
elif _sheet_manual_boosts:
    print("[boost] Second pass - skipping manual boosts (already baked into final_predictions)")

# --- Save init_sim_skill for mkt_regress (first pass only) ---
# DG blending is handled upstream by pre_sim_skill.py — no override here.
# Saved AFTER the boost blocks so archetype/manual boosts reach mkt_regress's
# base pred; saving first made mkt_regress regress boosted sim probs against an
# un-boosted base, partially inverting the boost in final_predictions.
if PRED_PATH != _final_pred_path:
    _pss_path = f"pre_sim_summary_{tourney}.csv"
    if os.path.exists(_pss_path):
        _pss_df = pd.read_csv(_pss_path)
        _pss_df['player_name'] = _pss_df['player_name'].str.lower().str.strip().replace(name_replacements)
        _init_skill = model_preds[['player_name', 'my_pred']].rename(columns={'my_pred': 'pred'})
        _init_skill = _init_skill.merge(_pss_df[['player_name', 'c_adj', 'sample']], on='player_name', how='left')
        _init_skill.to_csv(f"init_sim_skill_{tourney}.csv", index=False)
        print(f"[ok] Saved init_sim_skill_{tourney}.csv ({len(_init_skill)} players)")
    else:
        print(f"[warn] {_pss_path} not found — init_sim_skill_{tourney}.csv not saved (mkt_regress may fail)")
else:
    print("[init_sim_skill] Second pass — skipping (regressed predictions already in final_predictions)")

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

# Field-average skill = mean my_pred over the field (matches the Rust/ETR sim's
# ModelPredictions::field_skill = sum(my_pred)/n). Added to the R1 skill-update
# residual so the residual baseline matches the ETR sim:
#   resid_r1 = sg_r1 + field_skill - my_pred   (was: sg_r1 - my_pred)
field_skill = float(np.nanmean(my_pred_base))
print(f"[skill] field_skill (mean my_pred over field) = {field_skill:.4f}")

print(f"\n[sim] {n_players} players, {SIMULATIONS:,} simulations")


# ======================
# R1: Category-first draws
# ======================
# ─── Sim block (skip when --price-only loads cached arrays) ───────
if not args.price_only:
    # ─── Week-level form latent (2026-08 joint distributional fix, step 1) ───
    # One shared draw per (player, sim) added as +w/4 to every round's category
    # means; idiosyncratic category stds shrunk so per-round total variance is
    # unchanged (round products don't reprice — only the cross-round linkage).
    # Drawn from a SEPARATE RNG stream so latent-off stays bit-identical to the
    # pre-latent cascade (the seed-456 stream is never touched by this block).
    WEEK_LATENT = 0.0 if args.no_week_latent else WEEK_LATENT_SD
    if WEEK_LATENT > 0.0:
        _round_var = np.array([std_c @ R @ std_c for (_mu_i, std_c) in player_params_v2])
        _latent_shrink = np.sqrt(np.clip(1.0 - WEEK_LATENT**2 / _round_var, 0.05, 1.0))
        W_LATENT = (np.random.default_rng(789)
                    .standard_normal((n_players, SIMULATIONS)) * WEEK_LATENT)
        _exp_ratio = float(np.mean(1.0 + 3.0 * WEEK_LATENT**2 / _round_var))
        print(f"[week-latent] sigma={WEEK_LATENT:.2f} SG/round, shrink "
              f"{_latent_shrink.min():.3f}-{_latent_shrink.max():.3f}, "
              f"expected 72h variance ratio ~{_exp_ratio:.3f} pre-cascade "
              f"(target 1.15-1.20)")
    else:
        _latent_shrink = np.ones(n_players)
        W_LATENT = None
        print("[week-latent] disabled")

    # ─── Fixture dump hook (parity re-capture; see rust/fixtures/verify_cascade_against_prod.py) ───
    # SIMS_DUMP_FIXTURE=<path.npz> (or =1 for the default fixtures path) freezes the
    # fully-assembled kernel inputs so the Rust kernel and the Python cascade can be
    # compared on IDENTICAL real arrays. Pair with --use-python so the saved
    # final_scores support the bit-for-bit transcription check.
    _dump_path = os.getenv("SIMS_DUMP_FIXTURE")
    if _dump_path:
        if _dump_path == "1":
            _dump_path = os.path.join("rust", "fixtures", f"prod_input_{tourney}.npz")
        _c1 = lambda c: np.asarray([c['ott'], c['putt'], c['residual'], c['residual2']], dtype=float)
        _c2 = lambda c: np.asarray([c['residual'], c['residual2'], c['residual3'], c['avg_ott'],
                                    c['avg_putt'], c['avg_app'], c['avg_arg'], c['delta_app']], dtype=float)
        _c3 = lambda c: np.asarray([c['sg_ott_avg'], c['sg_putt_avg'], c['sg_app_avg'],
                                    c['sg_arg_avg'], c.get('pos_6_10', 0.0)], dtype=float)
        np.savez_compressed(
            _dump_path,
            mu=np.stack([m for (m, s) in player_params_v2]),
            std_course=np.stack([s for (m, s) in player_params_v2]),
            eff_skew=np.asarray(effective_skew, dtype=float),
            l_corr=np.asarray(L_corr, dtype=float),
            my_pred_base=np.asarray(my_pred_base, dtype=float),
            r2_mu=np.asarray(r2_mu, dtype=float), r3_mu=np.asarray(r3_mu, dtype=float),
            r4_mu=np.asarray(r4_mu, dtype=float),
            weather_delta_r1=np.asarray(weather_delta_r1, dtype=float),
            weather_delta_r2=np.asarray(weather_delta_r2, dtype=float),
            r1_high=_c1(coefficients_r1_high), r1_midh=_c1(coefficients_r1_midh),
            r1_midl=_c1(coefficients_r1_midl), r1_low=_c1(coefficients_r1_low),
            r2_lt6=_c2(coefficients_r2), r2_6_30=_c2(coefficients_r2_6_30),
            r2_30up=_c2(coefficients_r2_30_up),
            r3_lt6=_c3(coefficients_r3), r3_6_20=_c3(coefficients_r3_mid),
            r3_30up=_c3(coefficients_r3_high),
            cut_line=int(CUT_LINE), use_10_shot_rule=bool(USE_10_SHOT_RULE),
            sims=int(SIMULATIONS), seed=456, week_latent_sd=float(WEEK_LATENT),
            player_names=np.asarray(player_names, dtype=object),
        )
        print(f"  [fixture] dumped kernel inputs -> {_dump_path}"
              + ("" if WEEK_LATENT == 0.0 else
                 "  [NOTE: latent ON — bit-for-bit A-check needs --no-week-latent]"))

    # ─── Rust kernel (PRODUCTION DEFAULT; --use-python forces legacy Python draw) ───
    # Compute final_scores + per-category SG means via the Rust sims_kernel up
    # front (inputs map 1:1 to the seed-456 dump-hook contract). On --use-python
    # or any Rust error, leave final_scores=None so the legacy Python draw below
    # runs as the fallback (and still emits validation diagnostics). Everything
    # downstream consumes only final_scores + win prob + the per-category means.
    final_scores = None
    _rust_cat_means = None
    if not args.use_python:
        try:
            import sims_kernel as _sk
            _A = np.ascontiguousarray
            _r1 = lambda c: [c['ott'], c['putt'], c['residual'], c['residual2']]
            _r2 = lambda c: [c['residual'], c['residual2'], c['residual3'], c['avg_ott'],
                             c['avg_putt'], c['avg_app'], c['avg_arg'], c['delta_app']]
            _r3 = lambda c: [c['sg_ott_avg'], c['sg_putt_avg'], c['sg_app_avg'], c['sg_arg_avg'],
                             c.get('pos_6_10', 0.0)]
            _mu = np.stack([m for (m, s) in player_params_v2])
            _std = np.stack([s for (m, s) in player_params_v2])
            _ret = _sk.run_pretournament(
                _A(_mu), _A(_std), _A(effective_skew), _A(L_corr),
                _A(my_pred_base), _A(r2_mu), _A(r3_mu), _A(r4_mu),
                _A(weather_delta_r1), _A(weather_delta_r2),
                _r1(coefficients_r1_high), _r1(coefficients_r1_midh),
                _r1(coefficients_r1_midl), _r1(coefficients_r1_low),
                _r2(coefficients_r2), _r2(coefficients_r2_6_30), _r2(coefficients_r2_30_up),
                _r3(coefficients_r3), _r3(coefficients_r3_mid), _r3(coefficients_r3_high),
                int(CUT_LINE), bool(USE_10_SHOT_RULE), int(SIMULATIONS), 456,
                float(WEEK_LATENT),
            )
            if len(_ret) >= 7:
                _fs, _win, _cm1, _cm2, _cm3, _cm4, _mc = _ret[:7]
                made_cut_mask = np.ascontiguousarray(_mc).astype(bool)
            else:  # pre-mask wheel: no cut mask (board's make_cut stays on the copula)
                _fs, _win, _cm1, _cm2, _cm3, _cm4 = _ret
                made_cut_mask = None
            final_scores = np.ascontiguousarray(_fs).astype(np.int32)
            _rust_cat_means = (_cm1, _cm2, _cm3, _cm4)
            np.save(f"final_scores_{tourney}.npy", final_scores)
            print(f"[rust] final_scores + cat-means from sims_kernel.run_pretournament "
                  f"v{_sk.version()} (shape={final_scores.shape}, dtype={final_scores.dtype})")
        except Exception as _rust_err:
            print(f"[rust] WARNING: Rust kernel unavailable/failed ({_rust_err!r}); "
                  f"falling back to Python draw output. Pass --use-python to silence.")
            final_scores = None
            _rust_cat_means = None
    _python_drew = final_scores is None

    if _python_drew:
        cats_r1 = np.empty((n_players, SIMULATIONS, 4), dtype=float)
        sg_r1 = np.empty((n_players, SIMULATIONS), dtype=float)

        for i, (mu, std_c) in enumerate(player_params_v2):
            # Category means shifted by weather delta
            cat_mu = mu - weather_delta_r1[i] * WEATHER_CAT_SPLIT
            if W_LATENT is not None:                        # week latent: +w/4 per category
                cat_mu = cat_mu + W_LATENT[i][:, None] / 4.0
            Z = RNG.standard_normal(size=(SIMULATIONS, 4))
            corr_z = Z @ L_corr.T                          # correlated unit-variance draws
            for j in range(4):                              # apply per-player skewness
                corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[i, j])
            draws = cat_mu + corr_z * (std_c * _latent_shrink[i])
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
        resid_r1  = sg_r1 + field_skill - my_pred_base[:, None]
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
        # User risk rule 2026-08: residual adjustments capped at +/-0.5
        # everywhere (former -0.75 band retired); parity with engine / Rust.
        tot_resid_adj_r1 = np.maximum(
            np.minimum(np.where(mask_bad, 0.2, tot_resid_adj_r1), 0.5), -0.5)

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
            if W_LATENT is not None:                     # week latent rides the shift (/4)
                skill_shift = skill_shift + W_LATENT[i]
            cat_mu_shifted = cat_mu + skill_shift[:, None] / 4.0
            Z = RNG.standard_normal(size=(SIMULATIONS, 4))
            corr_z = Z @ L_corr.T
            for j in range(4):
                corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[i, j])
            draws = cat_mu_shifted + corr_z * (std_c * _latent_shrink[i])
            cats_r2[i] = np.clip(draws, CLIP_CAT[0], CLIP_CAT[1])
            sg_r2[i] = cats_r2[i].sum(axis=1)

        strokes_r2 = np.rint(PAR - sg_r2).astype(int)
        r1_r2_scores = strokes_r1 + strokes_r2


        # ======================
        # CUT LOGIC after 36 (Top-N and ties)
        # ======================
        made_cut_mask = np.ones_like(r1_r2_scores, dtype=bool)
        if CUT_LINE >= n_players:
            print(f"[cut] CUT_LINE={CUT_LINE} >= field size ({n_players}); treating as no-cut event")
        else:
            for j in range(SIMULATIONS):
                sc = r1_r2_scores[:, j]
                cut_score = np.sort(sc)[CUT_LINE - 1]
                top_cut = sc <= cut_score
                if USE_10_SHOT_RULE:
                    within_10 = sc <= (sc.min() + 10)
                    made_cut_mask[:, j] = top_cut | within_10
                else:
                    made_cut_mask[:, j] = top_cut

        # Persist exact P(make cut) for the odds board. The downstream rank-prob
        # re-derivation is biased (dead-heat spread + ignores the 10-shot rule),
        # so ship the simulated cut mask mean directly (true cut: top-N + ties,
        # plus the 10-shot rule when enabled). Rows are in player_names order.
        try:
            pd.DataFrame({"player_name": player_names,
                          "make_cut": made_cut_mask.mean(axis=1)}).to_csv(
                f"make_cut_probs_{tourney}.csv", index=False)
            print(f"  [make_cut] wrote make_cut_probs_{tourney}.csv")
        except Exception as _mc_e:
            print(f"  [make_cut] persist failed: {_mc_e}")


        # ======================
        # R2 -> R3 skill update (position buckets; uses R1+R2 stats)
        # ======================
        # Residual capped at +6 before the cubic: beyond training support it
        # explodes positive (parity: live_stats_engine.py RESID_FIX_CAP and
        # the Rust kernel).
        resid_r2  = np.minimum(sg_r2 - sg_r2_mean, 6.0)
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
        # +/-0.5 clip (user risk rule 2026-08): parity with engine / Rust
        tot_resid_adj_r2 = np.minimum(np.maximum(
            ensure_array(adj_sum.get('residual_adj', 0.0),  shape2) +
            ensure_array(adj_sum.get('residual2_adj', 0.0), shape2) +
            ensure_array(adj_sum.get('residual3_adj', 0.0), shape2),
            -0.5,
        ), 0.5)
        tot_sg_adj_r2 = (
            ensure_array(adj_sum.get('avg_ott_adj', 0.0),   shape2) +
            ensure_array(adj_sum.get('avg_putt_adj', 0.0),  shape2) +
            ensure_array(adj_sum.get('avg_app_adj', 0.0),   shape2) +
            ensure_array(adj_sum.get('avg_arg_adj', 0.0),   shape2) +
            ensure_array(adj_sum.get('delta_app_adj', 0.0), shape2)
        )
        # R2's fresh adjustments REPLACE R1's entirely — sg AND residual (R1's
        # residual predicts nothing beyond R2; horizon regression 2026-08).
        # Parity with live_stats_engine reset-to-base.
        total_adjustment_r1 = ensure_array(total_adjustment_r1, shape2)
        total_adjustment_r2 = (tot_resid_adj_r2 + tot_sg_adj_r2) - total_adjustment_r1

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
            if W_LATENT is not None:                     # week latent rides the shift (/4)
                skill_shift = skill_shift + W_LATENT[i]
            cat_mu_shifted = mu + skill_shift[:, None] / 4.0
            Z = RNG.standard_normal(size=(SIMULATIONS, 4))
            corr_z = Z @ L_corr.T
            for j in range(4):
                corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[i, j])
            draws = cat_mu_shifted + corr_z * (std_c * _latent_shrink[i])
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
        pos_6_10_mask_r3  = np.zeros((n_players, SIMULATIONS), dtype=bool)
        for j in range(SIMULATIONS):
            pos = rank_positions_from_strokes(r1_r3_scores[:, j])
            pos_lt_6_mask_r3[:, j]  = (pos < 6)
            pos_6_20_mask_r3[:, j]  = (pos >= 6) & (pos <= 20)
            pos_gt_20_mask_r3[:, j] = (pos > 20)
            pos_6_10_mask_r3[:, j]  = (pos >= 6) & (pos <= 10)

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
        # pos_6_10 LEVEL term (parity with the Rust kernel): positions 6-10 into R4 only
        tot_sg_adj_r3 = tot_sg_adj_r3 + np.where(
            pos_6_10_mask_r3, coefficients_r3_mid.get('pos_6_10', 0.0), 0.0
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
            if W_LATENT is not None:                     # week latent rides the shift (/4)
                skill_shift = skill_shift + W_LATENT[i]
            cat_mu_shifted = mu + skill_shift[:, None] / 4.0
            Z = RNG.standard_normal(size=(SIMULATIONS, 4))
            corr_z = Z @ L_corr.T
            for j in range(4):
                corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[i, j])
            draws = cat_mu_shifted + corr_z * (std_c * _latent_shrink[i])
            cats_r4[i] = np.clip(draws, CLIP_CAT[0], CLIP_CAT[1])
            sg_r4[i] = cats_r4[i].sum(axis=1)

        strokes_r4 = np.rint(PAR - sg_r4).astype(int)

        # Missed-cut penalty for R3+R4
        r3_r4 = strokes_r3 + strokes_r4
        r3_r4[~made_cut_mask] = 200

        # Final integer 72-hole totals
        final_scores = r1_r2_scores + r3_r4
        np.save(f"final_scores_{tourney}.npy", final_scores)


    if _python_drew:
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

        # 1b. 72-hole dependence: total variance / sum of round variances.
        # Independent rounds -> 1.0; empirical player-weeks 1.09-1.38; the week
        # latent targets 1.15-1.20 (see sim_inputs.WEEK_LATENT_SD).
        _tot_sg = sg_r1 + sg_r2 + sg_r3 + sg_r4
        _vr = float(np.mean(_tot_sg.var(axis=1) /
                            (sg_r1.var(axis=1) + sg_r2.var(axis=1)
                             + sg_r3.var(axis=1) + sg_r4.var(axis=1))))
        print(f"  72h variance ratio: {_vr:.3f}  (target 1.15-1.20; independent rounds = 1.0)")

        # 2. Category stds vs expected
        print(f"\n  Category stds (R1 realized vs input*course_mult):")
        for k, cat in enumerate(["OTT", "APP", "ARG", "PUTT"]):
            realized = np.std(cats_r1[:, :, k], axis=1).mean()
            expected_avg = np.mean([p[1][k] for p in player_params_v2])
            print(f"    {cat}: realized={realized:.3f}, expected_avg={expected_avg:.3f}, ratio={realized/expected_avg:.3f}")

    # ─── 72-hole totals skew top-up (joint fix step 2 — ALWAYS after the cascade;
    # self-calibrating to the absolute empirical target, so running it last makes
    # the joint fit with the week latent automatic. Cut-conditional at cut events;
    # monotone per player, preserves per-player mean/std and the copula. Applied
    # BEFORE win/top-N aggregation and the cache save so --price-only/--reprice
    # inherit it. The raw pre-top-up final_scores_{tourney}.npy saved above stays
    # the parity-oracle artifact.) ───
    if not args.price_only:
        if not args.no_skew_cal:
            from skew_calibration import calibrate_total_skew
            final_scores = np.asarray(final_scores)
            calibrate_total_skew(final_scores, made_cut_mask)
        else:
            print("  [skew-cal] totals top-up disabled (--no-skew-cal)")

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

    if _python_drew:
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

    # Top-5/10/20 dead-heat finish probabilities via the Rust kernel
    # (sims_kernel.aggregate_round) — the same path round_sim uses. Replaces the
    # 100k-group x 156-row iterrows loop that was the ~7-minute bottleneck.
    # aggregate_round returns top_dh ALREADY divided by sims (probabilities),
    # validated integer-exact vs the pandas dead-heat (~1e-15). Falls back to the
    # pandas loop on --use-python or any Rust error.
    topn_df = None
    if not args.use_python:
        try:
            import sims_kernel as _sk
            _fs = np.ascontiguousarray(np.asarray(final_scores, dtype=np.int64))
            _, _top_dh, _ = _sk.aggregate_round(_fs)
            topn_df = pd.DataFrame({
                "player_name": list(player_names),
                "top_5":  _top_dh[:, 0],
                "top_10": _top_dh[:, 1],
                "top_20": _top_dh[:, 2],
            })
            print(f"[rust] top-5/10/20 via sims_kernel.aggregate_round "
                  f"({n_players} players x {SIMULATIONS:,} sims)")
        except Exception as _agg_err:
            print(f"[rust] WARNING: aggregate_round top-N failed ({_agg_err!r}); "
                  f"falling back to pandas loop. Pass --use-python to silence.")
            topn_df = None

    if topn_df is None:
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
    if _python_drew:
        r1m = cats_r1.mean(axis=1); r2m = cats_r2.mean(axis=1); r3m = cats_r3.mean(axis=1); r4m = cats_r4.mean(axis=1)
    else:
        r1m, r2m, r3m, r4m = _rust_cat_means   # each (n,4) from sims_kernel
    # totals via sum-of-category-means == mean-of-sg (identical to the old code)
    r1_total_mean = r1m.sum(axis=1); r2_total_mean = r2m.sum(axis=1)
    r3_total_mean = r3m.sum(axis=1); r4_total_mean = r4m.sum(axis=1)
    per_round_avg_cat = (r1m + r2m + r3m + r4m) / 4.0
    tourn_total_per_round_mean = per_round_avg_cat.sum(axis=1)

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
    # Persist sim outputs so a future --price-only / --reprice run can skip
    # the Monte Carlo entirely. final_scores is the big one (~players * sims
    # * 4 bytes). Everything else (topn_df, finish_equity_df, rank_probs_updated,
    # top_finish_probs CSV) is already written to disk above.
    import json as _sim_json
    _cache_dir = f"./{tourney}"
    os.makedirs(_cache_dir, exist_ok=True)
    np.save(os.path.join(_cache_dir, "final_scores.npy"), final_scores)
    with open(os.path.join(_cache_dir, "player_names.json"), "w") as _f:
        _sim_json.dump(list(player_names), _f)
    # Made-cut mask on the SAME draw axis as final_scores — published so the board
    # prices make_cut off the tournament joint instead of an independent copula
    # draw. None only on a pre-mask Rust wheel.
    if made_cut_mask is not None:
        np.save(os.path.join(_cache_dir, "made_cut.npy"), made_cut_mask)
    print(f"[sim-cache] Saved final_scores + player_names"
          f"{' + made_cut' if made_cut_mask is not None else ''} to {_cache_dir}")
else:
    # ─── Load cached sim outputs (--price-only / --reprice) ─────
    import json as _sim_json
    _cache_dir = f"./{tourney}"
    _fs_path = os.path.join(_cache_dir, "final_scores.npy")
    _pn_path = os.path.join(_cache_dir, "player_names.json")
    _rp_path = f"rank_probs_updated_{tourney}.parquet"
    _topn_path = f"top_finish_probs_{tourney}.csv"
    _fe_path = f"finish_equity_{tourney}.csv"
    if not (os.path.exists(_fs_path) and os.path.exists(_pn_path)
            and os.path.exists(_rp_path) and os.path.exists(_topn_path)
            and os.path.exists(_fe_path)):
        raise FileNotFoundError(
            "Missing one or more sim cache files. Run new_sim.py without "
            "--price-only / --reprice first to populate the cache."
        )
    final_scores = np.load(_fs_path)
    with open(_pn_path) as _f:
        player_names = _sim_json.load(_f)
    n_players = len(player_names)
    topn_df = pd.read_csv(_topn_path)
    finish_equity_df = pd.read_csv(_fe_path)
    rank_probs_updated = pd.read_parquet(_rp_path)
    sim_win_probs = finish_equity_df[["player_name", "simulated_win_prob"]].copy()
    print(f"[sim-cache] Loaded cached sim outputs ({len(player_names)} players, "
          f"{final_scores.shape[1]:,} sims)")

# Precomputed pairwise H2H matrix for the dashboard. Ships to Render via
# push_dashboard_data.py; the raw final_scores.npy is too big to deploy.
# Ties pushed — same convention as the Kalshi/NoVig matchup pricers below.
import time as _h2h_time
_h2h_t0 = _h2h_time.time()
_n_players_h2h = len(player_names)
_h2h_records = []
for _i in range(_n_players_h2h):
    _s_i = final_scores[_i]
    for _j in range(_i + 1, _n_players_h2h):
        _s_j = final_scores[_j]
        _wins = int(np.sum(_s_i < _s_j))
        _losses = int(np.sum(_s_i > _s_j))
        _ties = int(np.sum(_s_i == _s_j))
        _denom = _wins + _losses
        if _denom == 0:
            continue
        _total = _wins + _losses + _ties
        _h2h_records.append((
            player_names[_i],
            player_names[_j],
            _wins / _denom,
            _ties / _total if _total else 0.0,
        ))
_h2h_df = pd.DataFrame(_h2h_records, columns=["player_a", "player_b", "prob_a", "tie_pct"])
_h2h_path = f"h2h_matrix_{tourney}.parquet"
_h2h_df.to_parquet(_h2h_path, index=False)
print(f"[h2h-matrix] Wrote {len(_h2h_df):,} pairs to {_h2h_path} "
      f"({_h2h_time.time() - _h2h_t0:.1f}s)")

if args.sim_only:
    print("\n[sim-only] Cache populated, skipping pricing/email/storage. Done.")
    sys.exit(0)



# ============================================================
# OUTRIGHTS & TOP-N PRICING VS MARKET
# ============================================================

EDGE_THRESHOLD_WIN   = 2.0
EDGE_THRESHOLD_TOPN  = 2.0
BANKROLL             = 10000.0
KELLY_FRACTION       = 0.25
RETAIL_BOOKS         = ['draftkings','fanduel','betmgm','caesars','barstool','espn','wynnbet','unibet','betway','betfred','betrivers']

_all_sb_lines_raw = []  # accumulate ALL sportsbook lines (pre-edge-filter) for exchange comparison
BANKROLLS            = {'pinnacle': 10000, 'betcris': 10000, 'betonline': 8000, 'retail': 4000, 'bovada': 3000}

books_to_use = ['betcris', 'betmgm', 'betonline', 'bovada', 'caesars', 'draftkings', 'fanduel', 'pinnacle', 'unibet']

def decimal_to_american(decimal_odds):
    if pd.isna(decimal_odds):
        return np.nan
    if decimal_odds <= 1.0:
        # Not a real bettable line (1.0 = stake back, no profit; <1.0 impossible).
        # Shows up as a missing-book placeholder (e.g. when scraped Betcris odds
        # are unavailable). Treat as no-odds, same as NaN, to avoid /0 at -100/(d-1).
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


# Fetch scraped Betcris outrights once. Used as a fallback when DataGolf
# returns no betcris row for a (player, market_type) — DG drops books that
# don't post a price; we backfill from our own scrape so betcris-only lines
# are still priced.
try:
    from odds_loader import load_betcris_outrights as _load_betcris_outrights
    _BETCRIS_FALLBACK_OUTRIGHTS = _load_betcris_outrights()
    if not _BETCRIS_FALLBACK_OUTRIGHTS.empty:
        # Match the new_sim convention: lowercase + name_replacements
        _BETCRIS_FALLBACK_OUTRIGHTS["player_name"] = (
            _BETCRIS_FALLBACK_OUTRIGHTS["player_name"]
            .str.lower().str.strip()
            .replace(name_replacements)
        )
        _by_mkt = _BETCRIS_FALLBACK_OUTRIGHTS.groupby("market_type").size()
        print(f"[betcris-fallback] Loaded scraped outright odds: "
              f"{', '.join(f'{k}={v}' for k, v in _by_mkt.items())}")
    else:
        print("[betcris-fallback] No scraped Betcris outright odds available")
except Exception as _e:
    print(f"[betcris-fallback] Loader failed: {_e}")
    _BETCRIS_FALLBACK_OUTRIGHTS = pd.DataFrame()


def _apply_betcris_fallback(dg_df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Append scraped Betcris rows for any (player) where DG has no betcris row.

    Only operates on the given market (`win` / `top_5` / `top_10` / `top_20`).
    DG sometimes returns betcris odds, sometimes doesn't — when it doesn't,
    fall back to our scrape so betcris pricing still flows downstream.
    """
    if _BETCRIS_FALLBACK_OUTRIGHTS.empty:
        return dg_df

    # Map DG's "win" market to our scrape's "winner" market_type
    scraped_mt = "winner" if market == "win" else market
    scrape = _BETCRIS_FALLBACK_OUTRIGHTS[
        _BETCRIS_FALLBACK_OUTRIGHTS["market_type"] == scraped_mt
    ]
    if scrape.empty:
        return dg_df

    # Players that already have a betcris row from DG for this market
    if not dg_df.empty:
        dg_betcris_players = set(
            dg_df.loc[dg_df["bookmaker"].astype(str).str.lower() == "betcris", "player_name"]
        )
    else:
        dg_betcris_players = set()

    backfill = scrape[~scrape["player_name"].isin(dg_betcris_players)][
        ["player_name", "bookmaker", "decimal_odds"]
    ].copy()
    if backfill.empty:
        return dg_df

    print(f"[betcris-fallback] {market.upper()}: DG had betcris for {len(dg_betcris_players)} players, "
          f"backfilled {len(backfill)} more from scrape")
    return pd.concat([dg_df, backfill], ignore_index=True) if not dg_df.empty else backfill

model_preds['player_name'] = model_preds['player_name'].str.lower().str.strip()

sample_df = model_preds[['player_name','sample']].copy() if 'sample' in model_preds.columns else \
            pd.DataFrame({'player_name': model_preds['player_name'], 'sample': np.nan})

pred_join = model_preds[['player_name','r3_pred']].rename(columns={'r3_pred':'my_pred'})
if pred_join['my_pred'].isna().all() and 'my_pred' in model_preds.columns:
    pred_join = model_preds[['player_name','my_pred']].copy()

# --- WIN market ---
data_win = fetch_market_data('win')
win_df = extract_market_rows(data_win)
win_df = _apply_betcris_fallback(win_df, 'win')
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
    df = _apply_betcris_fallback(df, market)
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

# Snapshot per-book outright edges (pre-groupby; combined_finish_df gets its bookmaker
# field comma-joined below). Feeds the standalone Outrights email.
_outright_sb_perbook = combined_finish_df.copy() if not combined_finish_df.empty else pd.DataFrame()

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

    # ── Kalshi public API (no auth needed for market data) ──────────
    import httpx as _kalshi_httpx
    import time as _time
    _KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
    _KALSHI_HEADERS = {"Accept": "application/json"}
    _kalshi_client = _kalshi_httpx.Client(timeout=15.0, headers=_KALSHI_HEADERS)

    def _kalshi_get_markets(series_ticker):
        """Fetch all open markets for a series, handling pagination + 429 retry."""
        all_mkts = []
        cursor = None
        while True:
            params = {"limit": 200, "status": "open", "series_ticker": series_ticker}
            if cursor:
                params["cursor"] = cursor
            # 429-retry: Kalshi throttles burst paginated reads; back off and retry
            for attempt in range(5):
                resp = _kalshi_client.get(f"{_KALSHI_API}/markets", params=params)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("retry-after", 1.0)) * (attempt + 1)
                    _time.sleep(min(wait, 5.0))
                    continue
                resp.raise_for_status()
                break
            else:
                resp.raise_for_status()  # re-raise final 429 after retries
            data = resp.json()
            mkts = data.get("markets", [])
            all_mkts.extend(mkts)
            cursor = data.get("cursor")
            if not cursor or len(mkts) < 200:
                break
        return all_mkts

    def _kalshi_get_orderbook(ticker):
        """Fetch orderbook for a market (public, no auth)."""
        resp = _kalshi_client.get(f"{_KALSHI_API}/markets/{ticker}/orderbook")
        resp.raise_for_status()
        data = resp.json()
        ob = data.get("orderbook", data.get("orderbook_fp", data))
        def parse_levels(raw):
            levels = []
            for item in (raw or []):
                if isinstance(item, list) and len(item) == 2:
                    price = float(item[0])
                    qty = int(float(item[1]))
                    levels.append((price, qty))
            return levels
        return {
            "yes": parse_levels(ob.get("yes_dollars") or ob.get("yes", [])),
            "no": parse_levels(ob.get("no_dollars") or ob.get("no", [])),
        }

    OUTRIGHT_SERIES = {
        "KXPGATOP5": "top_5",
        "KXPGATOP10": "top_10",
        "KXPGATOP20": "top_20",
        "KXPGATOUR": "winner",
    }

    all_markets = []
    for series_ticker, mtype in OUTRIGHT_SERIES.items():
        try:
            mkts = _kalshi_get_markets(series_ticker)
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

    # Match Kalshi tournament names against the configured `tourney` from sim_inputs.
    # Generic stopwords removed so "cadillac" doesn't match "PGA Championship" via
    # the shared "championship" token. If every word is generic (e.g. tourney=
    # "pga_championship"), require ALL words to appear in the title.
    _GENERIC = {"the", "a", "an", "of", "at", "in", "tournament", "championship",
                "open", "classic", "invitational", "cup", "pga", "tour"}
    _all_target_words = {w for w in _re.split(r"[\s_]+", tourney.lower()) if w}
    _distinct_target_words = _all_target_words - _GENERIC

    def _matches_target(detected_title):
        t = detected_title.lower()
        if _distinct_target_words:
            return any(w in t for w in _distinct_target_words)
        return all(w in t for w in _all_target_words)

    tourn_counts = Counter()
    for m in all_markets:
        t = _detect_tourney(m.get("title", ""))
        if t:
            tourn_counts[t] += 1
    if tourn_counts:
        matched = [t for t in tourn_counts.keys() if _matches_target(t)]
        if matched:
            matched_lower = {t.lower() for t in matched}
            all_markets = [m for m in all_markets
                           if _detect_tourney(m.get("title", "")).lower() in matched_lower
                           or _detect_tourney(m.get("title", "")) == ""]
            label = ", ".join(matched)
            print(f"  Tournament: {label} ({len(all_markets)} markets, matched on tourney='{tourney}')")
        else:
            current_tourney = tourn_counts.most_common(1)[0][0]
            all_markets = [m for m in all_markets
                           if _detect_tourney(m.get("title", "")) in (current_tourney, "")]
            print(f"  WARNING: no Kalshi tournament matches '{tourney}'; falling back to most-common: {current_tourney}")
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

    # Winner uses dead-heat-resolved sim prob (one actual tournament winner
    # via uniform random tie-break in the sim, equivalent to DH 1/N split for
    # tied-at-top sims). Top-5/10/20 stay nodh because exchanges pay full on
    # any rank-in-region (no fractional DH payout).
    type_to_col = {
        "top_5": "top_5_nodh",
        "top_10": "top_10_nodh",
        "top_20": "top_20_nodh",
        "winner": "simulated_win_prob",
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

    # ── Flat 1000-contract orderbook walk (matches matchup pricer) ──
    FLAT_STAKE = 1000
    MIN_DEPTH = 300  # require at least 300 contracts of fill to call the VWAP "real"

    def _walk_book(ob_levels, sim_prob, target=FLAT_STAKE):
        """Walk orderbook to compute effective fill for a flat target-contract order.

        Args:
            ob_levels: raw levels from the OPPOSITE side of the book.
                       For buying YES: pass NO levels (no_price, qty).
                       For buying NO:  pass YES levels (yes_price, qty).
            sim_prob:  our probability for the side we're betting (used only for
                       the no-edge short-circuit at best price).
            target:    contracts to fill (default FLAT_STAKE=1000).

        Returns dict with effective_price, target, filled, vwap, avg_fee,
        or None if no edge at best price or empty book.
        """
        # Convert to (cost_dollars, qty) sorted cheapest-first
        # ob_levels are from the OPPOSITE side in dollar format (0.0-1.0)
        # Cost to buy our side = 1.0 - opposite_price
        fill_levels = sorted(
            [(1.0 - p, qty) for p, qty in ob_levels],
        )
        if not fill_levels:
            return None

        # No-edge short-circuit at best price (post-fee)
        best = fill_levels[0][0]
        fee_at_best = _kalshi_taker_fee(best)
        eff_best = best + fee_at_best
        if eff_best <= 0 or eff_best >= 1 or eff_best >= sim_prob:
            return None

        # Depth available at the headline (best) price — i.e. how many contracts
        # we could take without paying up to the next level. Kalshi consolidates
        # same-price quantity into one level but sum just in case.
        depth_at_best = sum(qty for price, qty in fill_levels if price == best)

        # Walk the book for `target` contracts
        filled = 0
        cost_sum = 0.0   # dollars
        fee_sum = 0.0     # dollars
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
        avg_fee = fee_sum / filled
        eff_price = vwap + avg_fee

        return {
            "effective_price": eff_price,
            "target": target,
            "filled": filled,
            "depth_at_best": int(depth_at_best),
            "vwap": vwap,
            "avg_fee": avg_fee,
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
            ob = _kalshi_get_orderbook(row["ticker"])
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
            passes_liq = yf["filled"] >= MIN_DEPTH
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
                "depth_at_best": yf.get("depth_at_best", 0),
                "passes_liq": passes_liq,
            })

        # --- NO side ---
        nf = row.get("no_fill")
        if nf is not None:
            eff = nf["effective_price"]
            edge = (row["sim_no"] - eff) * 100
            passes_liq = nf["filled"] >= MIN_DEPTH
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
                "depth_at_best": nf.get("depth_at_best", 0),
                "passes_liq": passes_liq,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        pre_filter = len(df)
        df = df[df["passes_liq"]].drop(columns=["passes_liq"])
        df = df.sort_values("edge", ascending=False)
        yes_pos = ((df["side"] == "yes") & (df["edge"] > 0)).sum()
        no_pos = ((df["side"] == "no") & (df["edge"] > 0)).sum()
        print(f"  Kalshi outrights: {len(df)} lines pass liquidity (dropped {pre_filter - len(df)}), "
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

    # Winner uses dead-heat-resolved sim prob (one actual tournament winner
    # via uniform random tie-break in the sim, equivalent to DH 1/N split for
    # tied-at-top sims). Top-5/10/20 stay nodh because exchanges pay full on
    # any rank-in-region (no fractional DH payout).
    type_to_col = {
        "top_5": "top_5_nodh", "top_10": "top_10_nodh",
        "top_20": "top_20_nodh", "winner": "simulated_win_prob",
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
  event(where: {league: {_eq: "PGA"}, type: {_eq: "Tournament"}, status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]}}
        order_by: {scheduled_start: asc}) {
    id description
    child_events_aggregate: events_aggregate(where: {status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]}}) {
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
                status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]},
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

    import httpx as _kalshi_httpx_mu
    _KALSHI_API_MU = "https://api.elections.kalshi.com/trade-api/v2"
    _KALSHI_HEADERS_MU = {"Accept": "application/json"}
    _kalshi_mu_client = _kalshi_httpx_mu.Client(timeout=15.0, headers=_KALSHI_HEADERS_MU)

    def _kalshi_mu_get_orderbook(ticker):
        resp = _kalshi_mu_client.get(f"{_KALSHI_API_MU}/markets/{ticker}/orderbook")
        resp.raise_for_status()
        data = resp.json()
        ob = data.get("orderbook", data.get("orderbook_fp", data))
        def parse_levels(raw):
            levels = []
            for item in (raw or []):
                if isinstance(item, list) and len(item) == 2:
                    levels.append((float(item[0]), int(float(item[1]))))
            return levels
        return {
            "yes": parse_levels(ob.get("yes_dollars") or ob.get("yes", [])),
            "no": parse_levels(ob.get("no_dollars") or ob.get("no", [])),
        }

    FLAT_STAKE = 1000  # contracts

    try:
        all_mkts = []
        cursor = None
        while True:
            params = {"limit": 200, "status": "open", "series_ticker": "KXPGAH2H"}
            if cursor:
                params["cursor"] = cursor
            resp = _kalshi_mu_client.get(f"{_KALSHI_API_MU}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
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

    # Match against the configured `tourney` so simultaneously-open events
    # (e.g. PGA Championship + Cadillac) don't get confused.
    _GENERIC_MU = {"the", "a", "an", "of", "at", "in", "tournament", "championship",
                   "open", "classic", "invitational", "cup", "pga", "tour"}
    _all_target_words_mu = {w for w in _re.split(r"[\s_]+", tourney.lower()) if w}
    _distinct_target_words_mu = _all_target_words_mu - _GENERIC_MU

    def _matches_target_mu(detected_title):
        t = detected_title.lower()
        if _distinct_target_words_mu:
            return any(w in t for w in _distinct_target_words_mu)
        return all(w in t for w in _all_target_words_mu)

    tourn_counts = Counter()
    for m in all_mkts:
        t = _detect(m.get("title", ""))
        if t:
            tourn_counts[t] += 1
    if tourn_counts:
        matched = [t for t in tourn_counts.keys() if _matches_target_mu(t)]
        if matched:
            matched_lower = {t.lower() for t in matched}
            all_mkts = [m for m in all_mkts
                        if _detect(m.get("title", "")).lower() in matched_lower
                        or _detect(m.get("title", "")) == ""]
            print(f"  Kalshi H2H: matched on tourney='{tourney}' -> {', '.join(matched)} ({len(all_mkts)} markets)")
        else:
            current_tourney = tourn_counts.most_common(1)[0][0]
            all_mkts = [m for m in all_mkts if _detect(m.get("title", "")) in (current_tourney, "")]
            print(f"  Kalshi H2H: WARNING no match for tourney='{tourney}', falling back to most-common: {current_tourney}")
            print(f"  Kalshi H2H: detected tournaments in raw feed: {dict(tourn_counts)}")
    else:
        print(f"  Kalshi H2H: WARNING no titles parsed for tournament name. Sample titles: "
              f"{[m.get('title', '')[:80] for m in all_mkts[:3]]}")

    if not all_mkts:
        print(f"  Kalshi H2H: 0 markets after tournament filter — nothing to price")
        return pd.DataFrame()

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

    # Diagnostic drop counters — printed in summary at the end so it's easy
    # to see exactly where markets fall out of the pipeline.
    diag = {
        "events_total": len(by_event),
        "events_single_market": 0,        # event had < 2 markets in group
        "sides_total": 0,                  # markets considered across all events
        "sides_title_unparsed": 0,         # _extract_player returned ""
        "sides_no_quote": 0,               # bid<=0 or ask<=0
        "events_fewer_than_2_sides": 0,    # after parsing, < 2 valid sides
        "sides_player_not_in_preds": 0,
        "sides_orderbook_fail": 0,
        "sides_empty_book": 0,             # no fill_levels on the buy side
        "sides_zero_filled": 0,            # walk produced 0 contracts
        "sides_american_none": 0,
        "sides_sim_prob_none": 0,
        "name_mismatches": set(),          # players in Kalshi that weren't in pred_lookup
    }

    rows = []
    ob_count = 0
    for event_ticker, mkts in by_event.items():
        if len(mkts) < 2:
            diag["events_single_market"] += 1
            continue

        # Parse both sides
        sides = []
        for m in mkts:
            diag["sides_total"] += 1
            player_raw, opponent_raw = _extract_player(m.get("title", ""))
            if not player_raw:
                diag["sides_title_unparsed"] += 1
                continue
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)
            if bid <= 0 or ask <= 0:
                diag["sides_no_quote"] += 1
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
            diag["events_fewer_than_2_sides"] += 1
            continue

        a, b = sides[0], sides[1]

        # Walk orderbook for each side with flat 1000-contract stake
        for side_info in [a, b]:
            player = side_info["player"]
            if player not in pred_lookup:
                diag["sides_player_not_in_preds"] += 1
                diag["name_mismatches"].add(player)
                continue
            try:
                ob = _kalshi_mu_get_orderbook(side_info["ticker"])
                _time.sleep(0.05)
                ob_count += 1
            except Exception:
                diag["sides_orderbook_fail"] += 1
                continue

            no_levels = ob.get("no", [])
            # Buy YES: walk NO levels (sellers of YES)
            # Prices are in dollar format (0.0-1.0); cost = 1.0 - opposite_price
            fill_levels = sorted([(1.0 - p, qty) for p, qty in no_levels])
            if not fill_levels:
                diag["sides_empty_book"] += 1
                continue

            # Depth at the headline (best) price — max take with no slippage
            best_price_mu = fill_levels[0][0]
            depth_at_best_mu = sum(q for p, q in fill_levels if p == best_price_mu)

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
                diag["sides_zero_filled"] += 1
                continue

            vwap = cost_sum / filled
            eff_price = vwap + (fee_sum / filled)

            # Compute edge: sim win prob vs effective price
            # For matchups, sim_prob is derived from the skill difference
            # A beats B: P(A wins) ≈ from model predictions
            mid = (side_info["bid"] + side_info["ask"]) / 2.0
            american = implied_prob_to_american_odds(eff_price)
            if american is None:
                diag["sides_american_none"] += 1
                continue

            sim_p = _sim_win_prob(player, side_info["opponent"])
            if sim_p is None:
                diag["sides_sim_prob_none"] += 1
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
                "depth_at_best": int(depth_at_best_mu),
            })

    _kalshi_mu_client.close()
    print(f"  Kalshi matchups: {len(rows)} sides priced ({ob_count} orderbooks fetched)")

    # Diagnostic drop-reason summary — surfaces exactly where markets fall out
    # of the pipeline so we don't have to guess on "missing markets" reports.
    print(f"  [kalshi-mu diag] events={diag['events_total']} "
          f"(single_market_drop={diag['events_single_market']}, "
          f"under_2_sides_drop={diag['events_fewer_than_2_sides']})")
    print(f"  [kalshi-mu diag] sides_seen={diag['sides_total']} "
          f"(title_unparsed={diag['sides_title_unparsed']}, "
          f"no_quote={diag['sides_no_quote']}, "
          f"player_not_in_preds={diag['sides_player_not_in_preds']}, "
          f"orderbook_fail={diag['sides_orderbook_fail']}, "
          f"empty_book={diag['sides_empty_book']}, "
          f"zero_filled={diag['sides_zero_filled']}, "
          f"sim_prob_none={diag['sides_sim_prob_none']})")
    if diag["name_mismatches"]:
        _shown = sorted(diag["name_mismatches"])[:10]
        more = "" if len(diag["name_mismatches"]) <= 10 else f" (+{len(diag['name_mismatches']) - 10} more)"
        print(f"  [kalshi-mu diag] name mismatches (Kalshi normalized -> not in pred_lookup): "
              f"{', '.join(_shown)}{more}")

    df = pd.DataFrame(rows)
    if not df.empty and diag["name_mismatches"]:
        df.attrs["name_mismatches"] = diag["name_mismatches"]
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
  event(where: {league: {_eq: "PGA"}, type: {_eq: "Tournament"}, status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]}}
        order_by: {scheduled_start: asc}) {
    id description
    child_events_aggregate: events_aggregate(where: {status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]}}) {
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
                status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]}}) {
    id description
    game {
      homeTeam { name }
      awayTeam { name }
    }
    markets(where: {status: {_eq: "OPEN"}, type: {_eq: "MONEY"}}) {
      id type volume
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
                volume = float(m.get("volume", 0) or 0)

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
                        "volume": volume,
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

                # Raw probability edge (sim_prob - book_implied)
                # Derive sim_prob from fair_odds (American) since my_odds_p1/p2 may not be in filtered df
                edge_pct = 0
                if pd.notna(book_odds) and book_odds != 0 and pd.notna(fair_odds) and fair_odds != 0:
                    if book_odds < 0:
                        book_implied = abs(book_odds) / (abs(book_odds) + 100)
                    else:
                        book_implied = 100 / (book_odds + 100)
                    if fair_odds < 0:
                        sim_implied = abs(fair_odds) / (abs(fair_odds) + 100)
                    else:
                        sim_implied = 100 / (fair_odds + 100)
                    edge_pct = (sim_implied - book_implied) * 100

                edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"
                pred_color = "#d4edda" if pred > 1.5 else "#ffffff"
                book_str = f"{int(book_odds):+d}" if pd.notna(book_odds) else ""
                fair_str = f"{int(fair_odds):+d}" if pd.notna(fair_odds) else ""
                book_color = "color:#dd6b20; font-weight:500;" if book == "kalshi" else "color:#6b46c1; font-weight:500;" if book == "novig" else ""

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{bet_player}</td>
                    <td style="padding:6px 10px; color:#666;">vs {opponent}</td>
                    <td style="padding:6px 10px; text-align:center; {book_color}">{book}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{arch_against}</td>
                    <td style="padding:6px 10px; text-align:center;">{book_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center; color:#555;">{edge_pct:.1f}%</td>
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
                    <th style="padding:6px 10px; text-align:center;">Type</th>
                    <th style="padding:6px 10px; text-align:center;">Vs Type</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Edge %</th>
                    <th style="padding:6px 10px; text-align:center;">Wx</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Sample</th>
                </tr>
                {rows_html}
            </table>"""
        else:
            mu_html = "<p>No tournament matchups passed filters.</p>"
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

                # 1 unit = $200 (matches dashboard/performance convention)
                stake_dollars = float(row.get('stake', 0) or 0)
                units_str = format_units(stake_dollars)

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
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{units_str}</td>
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
                    <th style="padding:6px 10px; text-align:center;">Units</th>
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

            book_color_hex = "#dd6b20" if book == "kalshi" else "#6b46c1" if book == "novig" else "#000000"
            if book == "kalshi":
                _filled = int(row.get('filled', 0) or 0)
                fill_str = f"{_filled:,}c" if _filled else "—"
                _depth_best = int(row.get('depth_at_best', 0) or 0)
                top_str = f"{_depth_best:,}c" if _depth_best else "—"
            else:
                fill_str = "—"  # NoVig: depth not exposed publicly
                top_str = "—"

            # Kelly stake (same formula as Sheet storage above) → units (1u = $200)
            _dec = round(1.0 / eff, 4) if eff > 0 else None
            _b = (_dec - 1.0) if _dec and _dec > 1 else 0
            _q = 1.0 - sim_prob
            _f = max((_b * sim_prob - _q) / _b, 0) if _b > 0 else 0
            _stake_dollars = round(BANKROLL * KELLY_FRACTION * _f, 2)
            units_str = format_units(_stake_dollars)

            exch_rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{market}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500; color:{book_color_hex};">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{eff * 100:.1f}¢</td>
                    <td style="padding:6px 10px; text-align:center;">{eff_american}</td>
                    <td style="padding:6px 10px; text-align:center; color:#888;">{sb_line_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center;">{sim_prob:.1%}</td>
                    <td style="padding:6px 10px; text-align:center;">{fill_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{top_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{units_str}</td>
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
                    <th style="padding:6px 10px; text-align:center;">Fill</th>
                    <th style="padding:6px 10px; text-align:center;">Top</th>
                    <th style="padding:6px 10px; text-align:center;">Units</th>
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


def _compute_sb_priority_keys(combined_finish_df, kalshi_df=None, novig_df=None):
    """Return the set of (player_lower, market_type) pairs that would appear in the
    sportsbook-priority email — i.e. where the best sharp-book edge is >=0.5pp AND
    either no YES-side exchange edge exists OR SB beats Exch by >0.5pp.

    Used by send_exchange_email to suppress players whose YES bet is "better"
    at sportsbooks.
    """
    keys = set()
    if combined_finish_df is None or combined_finish_df.empty:
        return keys

    def _sharp_books_in(bm_str):
        tokens = [t.strip().lower() for t in str(bm_str or '').split(',')]
        return [t for t in tokens if t in SHARP_BOOKS]

    sharp_df = combined_finish_df[
        combined_finish_df['bookmaker'].apply(lambda v: bool(_sharp_books_in(v)))
    ]
    if sharp_df.empty:
        return keys

    sb_best = {}
    for _, r in sharp_df.iterrows():
        try:
            edge = float(r.get('edge', 0) or 0)
        except (TypeError, ValueError):
            continue
        if edge <= 0:
            continue
        key = (str(r.get('player_name', '')).lower().strip(),
               str(r.get('market_type', '')))
        if not key[0] or not key[1]:
            continue
        if key not in sb_best or edge > sb_best[key]:
            sb_best[key] = edge

    exch_best = {}
    for exch_df in [kalshi_df, novig_df]:
        if exch_df is None or exch_df.empty:
            continue
        sub = exch_df
        if 'side' in sub.columns:
            sub = sub[sub['side'] == 'yes']
        if 'pricing' in sub.columns:
            sub = sub[sub['pricing'] != 'mid']
        for _, r in sub.iterrows():
            try:
                edge = float(r.get('edge', 0) or 0)
            except (TypeError, ValueError):
                continue
            key = (str(r.get('player_name', '')).lower().strip(),
                   str(r.get('market_type', '')))
            if not key[0] or not key[1]:
                continue
            if key not in exch_best or edge > exch_best[key]:
                exch_best[key] = edge

    for key, sb_edge in sb_best.items():
        if sb_edge < 0.5:
            continue
        ex_edge = exch_best.get(key)
        delta = sb_edge - (ex_edge if ex_edge is not None else 0.0)
        if ex_edge is None or delta > 0.5:
            keys.add(key)
    return keys


def send_sportsbook_priority_email(combined_finish_df, kalshi_df=None, novig_df=None,
                                    arch_map=None, wx_lookup=None):
    """Email: capital-deployment guide.

    For each (player, market_type), surface rows where sportsbook edge is
    materially better than the best exchange edge (or where exchange has no
    edge at all). Goal: tell us where to prioritise sportsbook deployment.

    Inclusion rule:
        sb_best_edge >= 0.5 AND
        (exch_best_edge is None OR sb_best_edge - exch_best_edge > 0.5)
    """
    if not EMAIL_PASSWORD or not EMAIL_FROM or not EMAIL_TO or EMAIL_TO == ['']:
        return
    if combined_finish_df is None or combined_finish_df.empty:
        print("  [sb-priority] no sportsbook outright rows — skipping email")
        return

    # Sharp-book filter — only consider edges at pinnacle / betonline / betcris.
    # Note: combined_finish_df's `bookmaker` field is post-groupby comma-joined
    # when multiple books share the same decimal_odds (e.g. "betcris, betonline").
    # A naive isin() check drops every concat row, which silently buries
    # tied-best lines (often where Betcris is among the books). Filter on
    # whether ANY token in the concat is sharp.
    def _sharp_books_in(bm_str: str) -> list[str]:
        tokens = [t.strip().lower() for t in str(bm_str or '').split(',')]
        return [t for t in tokens if t in SHARP_BOOKS]

    sharp_df = combined_finish_df[
        combined_finish_df['bookmaker'].apply(lambda v: bool(_sharp_books_in(v)))
    ]
    if sharp_df.empty:
        print("  [sb-priority] no sharp-book outright rows — skipping email")
        return

    # ── Build SB-best per (player, market_type) ────────────────────────────
    # When the row is a concat ("betcris, betonline"), display only the sharp
    # tokens — retail-book co-occurrences can leak into the concat too.
    sb_best = {}
    for _, r in sharp_df.iterrows():
        try:
            edge = float(r.get('edge', 0) or 0)
        except (TypeError, ValueError):
            continue
        if edge <= 0:
            continue
        key = (str(r.get('player_name', '')).lower().strip(),
               str(r.get('market_type', '')))
        if not key[0] or not key[1]:
            continue
        if key not in sb_best or edge > sb_best[key]['edge']:
            sharp_tokens = _sharp_books_in(r.get('bookmaker', ''))
            sb_best[key] = {
                'book': ', '.join(sharp_tokens),
                'edge': edge,
                'american_odds': r.get('american_odds'),
                'my_pred': r.get('my_pred'),
                'stake': r.get('stake'),
                'sample': r.get('sample'),
            }

    # ── Build exchange-best per (player, market_type) ──────────────────────
    exch_best = {}
    for exch_df, exch_name in [(kalshi_df, 'kalshi'), (novig_df, 'novig')]:
        if exch_df is None or exch_df.empty:
            continue
        sub = exch_df
        if 'side' in sub.columns:
            sub = sub[sub['side'] == 'yes']
        if 'pricing' in sub.columns:
            sub = sub[sub['pricing'] != 'mid']  # taker only — actual fills
        for _, r in sub.iterrows():
            try:
                edge = float(r.get('edge', 0) or 0)
            except (TypeError, ValueError):
                continue
            key = (str(r.get('player_name', '')).lower().strip(),
                   str(r.get('market_type', '')))
            if not key[0] or not key[1]:
                continue
            if key not in exch_best or edge > exch_best[key]['edge']:
                exch_best[key] = {
                    'book': exch_name,
                    'edge': edge,
                    'american_odds': r.get('american_odds'),
                }

    # ── Filter + sort ──────────────────────────────────────────────────────
    rows = []
    for key, sb in sb_best.items():
        if sb['edge'] < 0.5:
            continue
        ex = exch_best.get(key)
        ex_edge = ex['edge'] if ex else None
        delta = sb['edge'] - (ex_edge if ex_edge is not None else 0.0)
        if ex_edge is None or delta > 0.5:
            rows.append({
                'player': key[0],
                'market': key[1],
                'sb_book': sb['book'],
                'sb_edge': sb['edge'],
                'sb_odds': sb['american_odds'],
                'sb_stake': sb.get('stake') or 0,
                'sb_pred': sb.get('my_pred'),
                'sb_sample': sb.get('sample'),
                'exch_book': ex['book'] if ex else None,
                'exch_edge': ex_edge,
                'exch_odds': ex['american_odds'] if ex else None,
                'delta': delta,
            })

    if not rows:
        print("  [sb-priority] no edges where sportsbook beats exchange by >0.5pp")
        return

    rows.sort(key=lambda x: x['sb_edge'], reverse=True)
    print(f"  [sb-priority] {len(rows)} priority rows "
          f"({sum(1 for r in rows if r['exch_edge'] is None)} sb-only, "
          f"{sum(1 for r in rows if r['exch_edge'] is not None)} both-but-sb-bigger)")

    # ── HTML ───────────────────────────────────────────────────────────────
    rows_html = ""
    for r in rows:
        player = r['player'].title()
        market = r['market']
        sb_book = r['sb_book']
        sb_odds_str = f"{int(r['sb_odds']):+d}" if pd.notna(r.get('sb_odds')) else ""
        sb_edge = r['sb_edge']
        sb_stake = r['sb_stake'] or 0
        units_str = format_units(sb_stake)

        if r['exch_edge'] is None:
            exch_book_str = "—"
            exch_odds_str = "—"
            exch_edge_str = "—"
            delta_str = "SB only"
            row_bg = "#fff8e1"  # light amber for SB-only — biggest priority
        else:
            exch_book_str = r['exch_book']
            exch_odds_str = f"{int(r['exch_odds']):+d}" if pd.notna(r.get('exch_odds')) else "—"
            exch_edge_str = f"{r['exch_edge']:.1f}%"
            delta_str = f"+{r['delta']:.1f}pp"
            row_bg = "#f0f4f8" if r['delta'] > 2 else "#ffffff"

        sb_edge_color = "#d4edda" if sb_edge > 5 else "#fff3cd" if sb_edge > 2 else "#ffffff"

        archetype = arch_map.get(r['player'], "") if arch_map else ""
        wx_sg = (wx_lookup or {}).get(r['player'], 0) if wx_lookup else 0
        wx_str = f"{wx_sg:+.2f}" if wx_sg else ""

        pred = r.get('sb_pred')
        pred_str = f"{pred:.2f}" if pd.notna(pred) else ""
        sample = r.get('sb_sample')
        sample_str = f"{int(sample)}" if pd.notna(sample) else ""

        rows_html += f"""
            <tr style="background:{row_bg};">
                <td style="padding:6px 10px; font-weight:600;">{player}</td>
                <td style="padding:6px 10px; text-align:center;">{market}</td>
                <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                <td style="padding:6px 10px; text-align:center; font-weight:500;">{sb_book}</td>
                <td style="padding:6px 10px; text-align:center;">{sb_odds_str}</td>
                <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{sb_edge_color};">{sb_edge:.1f}%</td>
                <td style="padding:6px 10px; text-align:center; font-weight:500;">{units_str}</td>
                <td style="padding:6px 10px; text-align:center; color:#666;">{exch_book_str}</td>
                <td style="padding:6px 10px; text-align:center; color:#666;">{exch_odds_str}</td>
                <td style="padding:6px 10px; text-align:center; color:#666;">{exch_edge_str}</td>
                <td style="padding:6px 10px; text-align:center; font-weight:600; color:#2c5282;">{delta_str}</td>
                <td style="padding:6px 10px; text-align:center;">{pred_str}</td>
                <td style="padding:6px 10px; text-align:center;">{sample_str}</td>
                <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{wx_str}</td>
            </tr>"""

    body = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:1200px; margin:0 auto; padding:20px;">
        <h2 style="margin-bottom:4px;">Sportsbook Priority &mdash; {tourney.replace('_', ' ').title()}</h2>
        <p style="color:#666; margin-top:0;">
            Players with sportsbook edge &ge; 0.5pp where SB edge beats exchange edge by &gt;0.5pp, or where no exchange edge exists.
            Amber rows = SB-only (no exchange offer). Sorted by SB edge desc.
        </p>
        <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
            <tr style="background:#343a40; color:white;">
                <th style="padding:6px 10px; text-align:left;">Player</th>
                <th style="padding:6px 10px; text-align:center;">Market</th>
                <th style="padding:6px 10px; text-align:center;">Type</th>
                <th style="padding:6px 10px; text-align:center;">SB Book</th>
                <th style="padding:6px 10px; text-align:center;">SB Line</th>
                <th style="padding:6px 10px; text-align:center;">SB Edge</th>
                <th style="padding:6px 10px; text-align:center;">Units</th>
                <th style="padding:6px 10px; text-align:center;">Exch Book</th>
                <th style="padding:6px 10px; text-align:center;">Exch Line</th>
                <th style="padding:6px 10px; text-align:center;">Exch Edge</th>
                <th style="padding:6px 10px; text-align:center;">Delta</th>
                <th style="padding:6px 10px; text-align:center;">Pred</th>
                <th style="padding:6px 10px; text-align:center;">Sample</th>
                <th style="padding:6px 10px; text-align:center;">Wx</th>
            </tr>
            {rows_html}
        </table>
        <p style="color:#999; font-size:11px; margin-top:20px;">
            Edge = (sim_prob - implied_prob) * 100. SB edge is the best edge across all sportsbook books for this
            (player, market). Exchange edge uses YES-side taker (walked-fillable) rows only. Units = SB kelly stake / $200.
        </p>
    </body>
    </html>"""

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Sportsbook Priority -- {tourney.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"  [ok] Sportsbook priority email sent ({len(rows)} rows)")
    except Exception as e:
        print(f"  [warn] Sportsbook priority email failed: {e}")


def send_exchange_email(kalshi_df=None, novig_df=None, kalshi_mu_df=None, novig_mu_df=None,
                        sb_priority_keys=None):
    """Email 2: Exchange-only opportunities (Kalshi + NoVig, independent tables + fades).

    Per-player dedup: each player appears at most once across the entire email, on
    their single highest-edge bet (compared across Kalshi/NoVig YES/NO outrights and
    matchups). If that bet (player, market_type) is also flagged in the sportsbook
    priority email (SB beats Exch by >0.5pp on the same YES bet, or SB-only),
    the player is dropped entirely — surface SB bets in the SB priority email instead.
    Matchup edges are NOT filtered by sb_priority_keys (no SB matchup comparison).
    """
    if not EMAIL_PASSWORD or not EMAIL_FROM or not EMAIL_TO or EMAIL_TO == ['']:
        return
    sb_priority_keys = sb_priority_keys or set()

    def _fill_str(row):
        book = str(row.get('bookmaker', ''))
        if book == 'kalshi':
            filled = int(row.get('filled', 0) or 0)
            return f"{filled:,}c" if filled else "—"
        # NoVig doesn't publish depth without auth; cumulative volume is not fillable size
        return "—"

    def _top_str(row):
        book = str(row.get('bookmaker', ''))
        if book == 'kalshi':
            depth = int(row.get('depth_at_best', 0) or 0)
            return f"{depth:,}c" if depth else "—"
        return "—"

    def _units_str(row):
        eff = row.get('eff_price', 0) or 0
        sim_p = row.get('sim_prob', 0) or 0
        if eff <= 0 or eff >= 1 or sim_p <= 0:
            return "—"
        dec = 1.0 / eff
        b = dec - 1.0
        if b <= 0:
            return "—"
        f_star = max((b * sim_p - (1 - sim_p)) / b, 0)
        stake = BANKROLL * KELLY_FRACTION * f_star
        return format_units(stake)

    def _exch_row(row, book):
        player = str(row.get('player_name', '')).title()
        market = str(row.get('market_type', ''))
        eff = row.get('eff_price', 0) or 0
        sim_prob = row.get('sim_prob', 0)
        edge = row.get('edge', 0)
        american = f"{int(row['american_odds']):+d}" if pd.notna(row.get('american_odds')) else ""
        fair = f"{int(row['my_fair']):+d}" if pd.notna(row.get('my_fair')) else ""
        fill_str = _fill_str(row)
        top_str = _top_str(row)
        units_str = _units_str(row)
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
            <td style="padding:5px 8px; text-align:center;">{top_str}</td>
            <td style="padding:5px 8px; text-align:center; font-weight:500;">{units_str}</td>
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
        fill_str = _fill_str(row)
        top_str = _top_str(row)
        units_str = _units_str(row)
        return f"""<tr>
            <td style="padding:5px 8px; font-weight:600;">{player}</td>
            <td style="padding:5px 8px;">{opp}</td>
            <td style="padding:5px 8px; text-align:center;">{mtype}</td>
            <td style="padding:5px 8px; text-align:center;">{eff*100:.1f}¢</td>
            <td style="padding:5px 8px; text-align:center;">{american}</td>
            <td style="padding:5px 8px; text-align:center;">{fair_str}</td>
            <td style="padding:5px 8px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
            <td style="padding:5px 8px; text-align:center;">{fill_str}</td>
            <td style="padding:5px 8px; text-align:center;">{top_str}</td>
            <td style="padding:5px 8px; text-align:center; font-weight:500;">{units_str}</td>
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
            <th style="padding:5px 8px; text-align:center;">Top</th>
            <th style="padding:5px 8px; text-align:center;">Units</th>
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
            <th style="padding:5px 8px; text-align:center;">Top</th>
            <th style="padding:5px 8px; text-align:center;">Units</th>
        </tr>"""

    sections = []
    timestamp_str = datetime.now().strftime('%B %d, %Y %I:%M %p')

    # ── Build unified pool: every candidate row across all six display buckets,
    # already past its per-section threshold. Then keep only each player's single
    # biggest-edge row, dropping YES outrights whose (player, market_type) is in
    # sb_priority_keys (= the SB bet is materially better than the exchange offer).
    pool = []  # list of dicts: {player, edge, bucket, row}

    def _push(df, bucket, kind, side=None):
        if df is None or df.empty:
            return
        sub = df
        if side is not None and 'side' in sub.columns:
            sub = sub[sub['side'] == side]
        for _, r in sub.iterrows():
            e = r.get('edge')
            try:
                e = float(e) if e is not None else None
            except (TypeError, ValueError):
                e = None
            if e is None:
                continue
            # Per-bucket threshold (mirrors the original section filters)
            if kind == 'yes' and e <= 0:
                continue
            if kind == 'no' and e <= 0.5:
                continue
            if kind == 'mu' and e <= 0.5:
                continue
            player = str(r.get('player_name', '')).lower().strip()
            if not player:
                continue
            market = str(r.get('market_type', ''))
            # Skip YES outrights covered better by sportsbooks
            if kind == 'yes' and (player, market) in sb_priority_keys:
                continue
            pool.append({'player': player, 'edge': e, 'bucket': bucket, 'row': r})

    _push(kalshi_df, 'kalshi_yes', 'yes', side='yes')
    _push(kalshi_df, 'kalshi_no', 'no', side='no')
    _push(novig_df, 'novig_yes', 'yes', side='yes')
    _push(novig_df, 'novig_no', 'no', side='no')
    _push(kalshi_mu_df, 'kalshi_mu', 'mu')
    _push(novig_mu_df, 'novig_mu', 'mu')

    # Keep only each player's biggest-edge row
    best = {}
    for item in pool:
        cur = best.get(item['player'])
        if cur is None or item['edge'] > cur['edge']:
            best[item['player']] = item

    by_bucket = {b: [] for b in
                 ('kalshi_yes', 'kalshi_no', 'novig_yes', 'novig_no', 'kalshi_mu', 'novig_mu')}
    for item in best.values():
        by_bucket[item['bucket']].append(item['row'])

    def _bucket_df(bucket):
        rows = by_bucket.get(bucket, [])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values('edge', ascending=False)

    _ky = _bucket_df('kalshi_yes')
    _kn = _bucket_df('kalshi_no')
    _ny = _bucket_df('novig_yes')
    _nn = _bucket_df('novig_no')
    _km = _bucket_df('kalshi_mu')
    _nm = _bucket_df('novig_mu')

    if not _ky.empty:
        rows_html = "".join(_exch_row(r, "kalshi") for _, r in _ky.head(20).iterrows())
        sections.append(f'<h3 style="color:#2c5282;">Kalshi YES — Top 20</h3>{_hdr}{rows_html}</table>')
    if not _kn.empty:
        rows_html = "".join(_exch_row(r, "kalshi") for _, r in _kn.head(20).iterrows())
        sections.append(f'<h3 style="color:#9b2c2c;">Kalshi NO (Fade) — edge &gt; 0.5%</h3>{_hdr}{rows_html}</table>')
    if not _ny.empty:
        rows_html = "".join(_exch_row(r, "novig") for _, r in _ny.head(20).iterrows())
        sections.append(f'<h3 style="color:#2c5282;">NoVig YES — Top 20</h3>{_hdr}{rows_html}</table>')
    if not _nn.empty:
        rows_html = "".join(_exch_row(r, "novig") for _, r in _nn.head(20).iterrows())
        sections.append(f'<h3 style="color:#9b2c2c;">NoVig NO (Fade) — edge &gt; 0.5%</h3>{_hdr}{rows_html}</table>')
    if not _km.empty:
        rows_html = "".join(_mu_row(r) for _, r in _km.iterrows())
        sections.append(f'<h3 style="color:#2c5282;">Kalshi Matchups — edge &gt; 0.5%</h3>{_mu_hdr}{rows_html}</table>')
    if not _nm.empty:
        rows_html = "".join(_mu_row(r) for _, r in _nm.iterrows())
        sections.append(f'<h3 style="color:#2c5282;">NoVig Matchups — edge &gt; 0.5%</h3>{_mu_hdr}{rows_html}</table>')

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
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print("  [ok] Exchange opportunities email sent")
    except Exception as e:
        print(f"  [warn] Exchange email failed: {e}")


def send_outrights_email(sb_outrights_df=None, kalshi_df=None, novig_df=None):
    """Outrights email — pure edge tables, no email-worthiness threshold.

    Table 1: ALL outright edges (edge > 0) at the sharp sportsbooks
    (betcris / betonline / pinnacle) PLUS Kalshi/NoVig YES outrights.
    Table 2: Kalshi/NoVig NO-side edges (edge > 0) — the fade side.
    """
    if not EMAIL_PASSWORD or not EMAIL_FROM or not EMAIL_TO or EMAIL_TO == ['']:
        return
    SHARP = {'betcris', 'betonline', 'pinnacle'}
    WIN = {'win', 'winner'}  # winner market ONLY (sportsbooks use 'win'; Kalshi/NoVig use 'winner')

    def _fair(p):
        try:
            return f"{int(prob_to_american(float(p))):+d}"
        except Exception:
            return "—"

    def _line(v):
        return f"{int(v):+d}" if pd.notna(v) else "—"

    rows_yes, rows_no = [], []

    # Sportsbook YES outrights (sharp books only, per-book, edge > 0)
    if sb_outrights_df is not None and not sb_outrights_df.empty and 'bookmaker' in sb_outrights_df.columns:
        sb = sb_outrights_df[
            sb_outrights_df['bookmaker'].astype(str).str.lower().isin(SHARP)
            & sb_outrights_df['market_type'].astype(str).str.lower().isin(WIN)
            & (pd.to_numeric(sb_outrights_df['edge'], errors='coerce') > 0)
        ]
        for _, r in sb.iterrows():
            imp = pd.to_numeric(r.get('implied_prob'), errors='coerce')
            edge = float(r['edge'])
            our_p = (imp + edge / 100.0) if pd.notna(imp) else float('nan')
            rows_yes.append({
                'player': str(r['player_name']).title(),
                'market': str(r['market_type']),
                'book': str(r['bookmaker']).lower(),
                'our_p': our_p,
                'line': r.get('american_odds'),
                'edge': edge,
                'implied': imp,  # market implied prob (1/decimal); lower = better price
            })

    # Exchange YES + NO (kalshi, novig)
    for exch in (kalshi_df, novig_df):
        if exch is None or exch.empty or 'side' not in exch.columns:
            continue
        for _, r in exch.iterrows():
            edge = pd.to_numeric(r.get('edge'), errors='coerce')
            if pd.isna(edge) or edge <= 0:
                continue
            if str(r.get('market_type', '')).lower() not in WIN:
                continue
            rec = {
                'player': str(r.get('player_name', '')).title(),
                'market': str(r.get('market_type', '')),
                'book': str(r.get('bookmaker', '')).lower(),
                'our_p': pd.to_numeric(r.get('sim_prob'), errors='coerce'),
                'line': r.get('american_odds'),
                'edge': float(edge),
                'implied': pd.to_numeric(r.get('eff_price'), errors='coerce'),  # exch buy price
            }
            (rows_yes if r['side'] == 'yes' else rows_no).append(rec)

    rows_yes.sort(key=lambda x: x['edge'], reverse=True)
    rows_no.sort(key=lambda x: x['edge'], reverse=True)

    # Flag, per player, the row with the BEST price across our books (lowest implied
    # = highest payout). Ties flag all. Used to green-highlight the player name.
    def _mark_best(rows):
        best = {}
        for x in rows:
            imp = pd.to_numeric(x.get('implied'), errors='coerce')
            if pd.isna(imp) or imp <= 0:
                continue
            k = x['player'].lower()
            if k not in best or imp < best[k]:
                best[k] = imp
        for x in rows:
            imp = pd.to_numeric(x.get('implied'), errors='coerce')
            k = x['player'].lower()
            x['is_best'] = bool(pd.notna(imp) and imp > 0 and k in best and abs(imp - best[k]) < 1e-9)

    _mark_best(rows_yes)
    _mark_best(rows_no)

    if not rows_yes and not rows_no:
        print("  [outrights email] No outright edges — skipping")
        return

    def _tbl(rows):
        hdr = ('<table style="border-collapse:collapse; font-family:Arial,sans-serif; '
               'font-size:12px; width:100%;">'
               '<tr style="background:#343a40; color:white;">'
               '<th style="padding:5px 8px; text-align:left;">Player</th>'
               '<th style="padding:5px 8px; text-align:center;">Market</th>'
               '<th style="padding:5px 8px; text-align:center;">Book</th>'
               '<th style="padding:5px 8px; text-align:center;">Our %</th>'
               '<th style="padding:5px 8px; text-align:center;">Fair</th>'
               '<th style="padding:5px 8px; text-align:center;">Line</th>'
               '<th style="padding:5px 8px; text-align:center;">Edge</th></tr>')
        out = [hdr]
        for x in rows:
            ec = "#d4edda" if x['edge'] > 5 else "#fff3cd" if x['edge'] > 2 else "#ffffff"
            op = f"{x['our_p'] * 100:.1f}%" if pd.notna(x['our_p']) else "—"
            # Green-highlight the player name where this row is the best price for them.
            pstyle = "padding:5px 8px; font-weight:600;"
            if x.get('is_best'):
                pstyle += " background:#c6f6d5; color:#1a7f37;"
            out.append(
                "<tr>"
                f'<td style="{pstyle}">{x["player"]}</td>'
                f'<td style="padding:5px 8px; text-align:center;">{x["market"]}</td>'
                f'<td style="padding:5px 8px; text-align:center;">{x["book"]}</td>'
                f'<td style="padding:5px 8px; text-align:center;">{op}</td>'
                f'<td style="padding:5px 8px; text-align:center;">{_fair(x["our_p"])}</td>'
                f'<td style="padding:5px 8px; text-align:center;">{_line(x["line"])}</td>'
                f'<td style="padding:5px 8px; text-align:center; font-weight:bold; background:{ec};">{x["edge"]:.1f}%</td>'
                "</tr>")
        out.append("</table>")
        return "".join(out)

    timestamp_str = datetime.now().strftime('%B %d, %Y %I:%M %p')
    t1 = _tbl(rows_yes) if rows_yes else "<p style='color:#999;'>No YES outright edges.</p>"
    t2 = _tbl(rows_no) if rows_no else "<p style='color:#999;'>No NO-side edges.</p>"
    body = f"""<html><body style="font-family:Arial,sans-serif; max-width:1000px; margin:0 auto; padding:20px;">
        <h2>Outrights &mdash; {tourney.replace('_', ' ').title()}</h2>
        <p style="color:#666;">{timestamp_str} | every outright (winner) edge down to 0% | <span style="background:#c6f6d5; color:#1a7f37; font-weight:600; padding:1px 4px;">green</span> = best price across our books</p>
        <h3 style="color:#2c5282;">Outright edges &mdash; betcris / betonline / pinnacle / kalshi / novig ({len(rows_yes)})</h3>
        {t1}
        <h3 style="color:#9b2c2c; margin-top:26px;">NO edges &mdash; kalshi / novig fade side ({len(rows_no)})</h3>
        {t2}
        <p style="color:#999; font-size:11px; margin-top:30px;">
            Sportsbooks: per-book line, edge = our prob &minus; implied. Exchanges: ask/effective
            price (Kalshi incl. fees), edge = our prob &minus; price. Our % is the model probability.
        </p>
    </body></html>"""

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Outrights -- {tourney.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"  [ok] Outrights email sent ({len(rows_yes)} YES, {len(rows_no)} NO)")
    except Exception as e:
        print(f"  [warn] Outrights email failed: {e}")


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

    else:
        print("[note] no bookmaker CSVs found to combine.")

else:
    print("[warn] No valid tournament matchups found.")

# --- Emails: always send. Finish positions, outrights, and exchange edges do
# not depend on matchup markets, and Monday runs typically have no matchup
# boards posted yet — the sharp-matchup table is simply omitted when empty.
_sharp_for_email = (
    sharp_df
    if ('sharp_df' in dir() and isinstance(sharp_df, pd.DataFrame) and not sharp_df.empty)
    else None
)

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

# Email 1: bettable edges (sportsbooks + best exchange rows)
send_tournament_email(
    sharp_mu_df=_sharp_for_email,
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

# Email 2: exchange-only opportunities — pre-compute the sportsbook
# priority key set so we can suppress players whose YES bet is "better"
# at sportsbooks (= already covered by Email 3).
try:
    _sb_priority_keys = _compute_sb_priority_keys(
        combined_finish_df if 'combined_finish_df' in dir() and not combined_finish_df.empty else None,
        _kalshi_df if not _kalshi_df.empty else None,
        _novig_df if not _novig_df.empty else None,
    )
except Exception as _ke:
    print(f"  [warn] sb_priority_keys compute failed: {_ke}")
    _sb_priority_keys = set()

send_exchange_email(
    kalshi_df=_kalshi_df if not _kalshi_df.empty else None,
    novig_df=_novig_df if not _novig_df.empty else None,
    kalshi_mu_df=_kalshi_mu_df if not _kalshi_mu_df.empty else None,
    novig_mu_df=_novig_mu_df if not _novig_mu_df.empty else None,
    sb_priority_keys=_sb_priority_keys,
)

# Email 4: Outrights — every outright edge (down to 0%) at the sharp
# sportsbooks + Kalshi/NoVig YES, plus a Kalshi/NoVig NO-edge table.
try:
    _sb_outright_perbook = (_outright_sb_perbook
                            if '_outright_sb_perbook' in dir() and not _outright_sb_perbook.empty
                            else None)
    send_outrights_email(
        sb_outrights_df=_sb_outright_perbook,
        kalshi_df=_kalshi_df if not _kalshi_df.empty else None,
        novig_df=_novig_df if not _novig_df.empty else None,
    )
except Exception as _oe:
    print(f"  [warn] Outrights email failed: {_oe}")

# Email 3: sportsbook priority (capital-deployment guide)
try:
    send_sportsbook_priority_email(
        combined_finish_df=combined_finish_df if 'combined_finish_df' in dir() and not combined_finish_df.empty else None,
        kalshi_df=_kalshi_df if not _kalshi_df.empty else None,
        novig_df=_novig_df if not _novig_df.empty else None,
        arch_map=_arch_map,
        wx_lookup=_wx_fp_lookup,
    )
except Exception as _spe:
    print(f"  [warn] Sportsbook priority email failed: {_spe}")

# --- Storage: always attempt (finish positions don't depend on matchups) ---
# In --reprice mode, skip Sheets storage entirely. Pricing + emails still run
# (so the user sees fresh edges) but we don't append duplicate rows to the
# Tournament Matchups / Finish Positions tabs. The bet ledger has its own
# dedup but the Sheet tabs do not, so a naive re-store would double-count.
if args.reprice:
    print("\n[reprice] Skipping Sheets storage (re-priced rows not re-appended).")
    sys.exit(0)

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

        # Reuse archetypes computed before email (or compute if not yet done).
        # A no-matchups run skips the entire matchup branch, so _arch_map may
        # not exist at all — finish positions still need it for type_on.
        if '_arch_map' not in dir():
            _arch_map = {}
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

# Publish compact sim fair probabilities to R2 for the golf odds board
try:
    import publish_sim_fairs
    print(f"\n{'='*60}")
    print("  Publishing sim fairs to R2 (odds board)...")
    publish_sim_fairs.publish()
    print(f"{'='*60}")
except Exception as e:
    print(f"  [publish_sim_fairs] Warning: {e}")

print("\n[done] Sim complete.")

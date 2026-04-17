"""
round_sim.py — Unified Round Simulation for Matchups + Score Line Pricing + Tournament Outrights

Replaces: round_mu_sim.py + round_scores.py (without HTML scraping)

Reads model_predictions_rN.csv (created by live_stats_engine.py).
Simulates round scores, prices matchups vs DataGolf API odds,
generates fair score-line pricing cards.

NEW (v2): Also simulates remaining rounds through R4 for outright/finish position pricing.

Usage:
    python round_sim.py                        (reads config from Google Sheet)
    python round_sim.py --cli --sim-round 2 --expected-avg 72.2

Outputs (saved to {tourney}/ folder):
    round_{N}_sim_{timestamp}.xlsx    — Matchup tabs + Score Card tab + Outrights tabs
    round_{N}_sim_scores.csv          — Raw simulated score distributions
    simulated_probs_live.csv          — Live win probabilities
    top_finish_probs_live_{tourney}.csv — Top 5/10/20 probabilities
    finish_equity_live_{tourney}.csv  — Combined finish position edges
"""

import os
import argparse
import numpy as np
import pandas as pd
import requests
import smtplib
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from numpy.linalg import cholesky

# --- Weekly-changing config from Google Sheet ---
from sheet_config import load_config as _load_sheet_config
_cfg = _load_sheet_config()
tourney          = _cfg["tourney"]
STD_DEV          = _cfg["std_dev"]
PAR              = 72  # constant
CUT_LINE         = _cfg["cut_line"]
USE_10_SHOT_RULE = _cfg["use_10_shot_rule"]
SIMULATIONS      = _cfg["simulations"]
_event_id        = _cfg["event_id"]

# --- Stable model params from sim_inputs ---
from sim_inputs import (
    name_replacements,
    # R1 update coefficient sets
    coefficients_r1_high, coefficients_r1_midh, coefficients_r1_midl, coefficients_r1_low,
    # R2 update sets (position buckets)
    coefficients_r2, coefficients_r2_6_30, coefficients_r2_30_up,
    # R3 update sets (avg SG only)
    coefficients_r3, coefficients_r3_mid, coefficients_r3_high,
)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("DATAGOLF_API_KEY")
MATCHUPS_URL = "https://feeds.datagolf.com/betting-tools/matchups"
OUTRIGHTS_URL = "https://feeds.datagolf.com/betting-tools/outrights"

NUM_SIMULATIONS = 100_000        # For single-round matchup sim
TOURNAMENT_SIMULATIONS = 100_000  # For tournament sim (R through R4)

SHARP_BOOKS = ["pinnacle", "betonline", "betcris"]
HALF_SHOT_ADJ = {"betonline": 25, "betcris": 30}

# Score card: generate fair UNDER prices at these offsets from expected avg
SCORE_CARD_RANGE = 3.0        # +-3 strokes from expected
SCORE_CARD_STEP = 0.5         # half-stroke intervals
MIN_PRED_FOR_CARD = -0.5      # exclude players with pred below this

# Email
EMAIL_FROM = os.getenv("EMAIL_USER")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")

# Matchup email filter thresholds
EMAIL_MIN_PRED = 0.75
EMAIL_MIN_SAMPLE = 20

# Outright market filter thresholds
EDGE_THRESHOLD_WIN = 2.0     # minimum edge (pp) sim_prob - implied_prob
EDGE_THRESHOLD_TOPN = 2.0   # minimum edge (pp) sim_prob - implied_prob
BANKROLL = 10000.0
KELLY_FRACTION = 0.25
RETAIL_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'barstool', 'espn', 'pointsbet']
BOOKS_TO_USE = ['betcris', 'betmgm', 'betonline', 'bovada', 'caesars', 'draftkings', 'fanduel', 'pinnacle', 'unibet']

# Category order for correlation matrix
CAT_ORDER = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
CLIP_CAT = (-8.0, 8.0)

# Correlation matrix file preferences
CORR_PREFS = [
    "permanent_data/sg_cat_corr_tour_within_player_pearson.csv",
    "permanent_data/sg_cat_corr_tour_spearman.csv",
    "permanent_data/sg_cat_corr_tour_pearson.csv",
]

# Random number generator for reproducibility
RNG = np.random.default_rng(42)

# ── Category-first sim constants ─────────────────────────────────────────────
DISTS_FILE_V2 = "this_week_dists_v2.csv"

COURSE_CAT_MULTS = _cfg.get("course_cat_mults", {})
COURSE_CAT_SKEW  = _cfg.get("course_cat_skew", {})
BASELINE_CAT_SKEW = {
    'sg_ott': -0.93, 'sg_app': -0.21, 'sg_arg': -0.18, 'sg_putt': -0.05,
}

_course_mults_cf = np.array([COURSE_CAT_MULTS.get(c, 1.0) for c in CAT_ORDER])
_course_skew_cf = np.array([
    COURSE_CAT_SKEW.get(c, BASELINE_CAT_SKEW.get(c, 0.0)) for c in CAT_ORDER
])

WEATHER_CAT_SPLIT = np.array([0.35, 0.35, 0.15, 0.15])

SKEW_BLEND_MAX_CF = 0.5
SKEW_CONFIDENCE_N_CF = 100.0

RNG_CF = np.random.default_rng(789)  # separate seed for catfirst draws


# ══════════════════════════════════════════════════════════════════════════════
# Tournament Sim Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def load_corr_matrix(cat_order):
    """Load category correlation matrix from file or return identity."""
    for fn in CORR_PREFS:
        if os.path.exists(fn):
            R = pd.read_csv(fn, index_col=0)
            R = R.loc[cat_order, cat_order]
            return R.values
    return np.eye(len(cat_order))


def _cf_calibration_multiplier(gamma):
    """Correction multiplier for first-order Cornish-Fisher saturation."""
    ag = abs(gamma)
    if ag < 0.2:
        return 1.0
    return 1.0 + 0.0234 * ag**2 + 0.0125 * ag**3


def _apply_skew(z, gamma):
    """Apply Cornish-Fisher skewness to standard-normal draws."""
    if abs(gamma) < 0.01:
        return z
    gamma_adj = gamma * _cf_calibration_multiplier(gamma)
    z_skewed = z + (gamma_adj / 6.0) * (z ** 2 - 1.0)
    z_skewed /= np.sqrt(1.0 + gamma_adj ** 2 / 18.0)
    return z_skewed


def _load_catfirst_dists(player_names):
    """Load v2 dists + correlation + per-player params for category-first draws.

    Returns (player_cf_params, effective_skew, L_corr) or None if data unavailable.
      player_cf_params: list of (mu, std_course) per player — mu NOT re-centered
      effective_skew: np.array (n_players, 4)
      L_corr: (4, 4) Cholesky of correlation matrix
    """
    if not os.path.exists(DISTS_FILE_V2):
        print(f"  [catfirst] Warning: {DISTS_FILE_V2} not found")
        return None

    dists = pd.read_csv(DISTS_FILE_V2)
    dists['player_name'] = (
        dists['player_name'].astype(str).str.lower().str.strip()
        .replace(name_replacements)
    )

    need_cols = {'player_name', 'category_clean', 'mean', 'std', 'skew', 'n_eff'}
    missing = need_cols - set(dists.columns)
    if missing:
        print(f"  [catfirst] Warning: {DISTS_FILE_V2} missing columns: {missing}")
        return None

    mu_w   = dists.pivot(index='player_name', columns='category_clean', values='mean')
    std_w  = dists.pivot(index='player_name', columns='category_clean', values='std')
    skew_w = dists.pivot(index='player_name', columns='category_clean', values='skew')
    neff_w = dists.pivot(index='player_name', columns='category_clean', values='n_eff')
    global_mu  = dists.groupby('category_clean')['mean'].mean()
    global_std = dists.groupby('category_clean')['std'].median()

    # Load correlation matrix and Cholesky
    R = load_corr_matrix(CAT_ORDER)
    try:
        L_corr = cholesky(R)
    except np.linalg.LinAlgError:
        R = 0.95 * R + 0.05 * np.eye(4)
        L_corr = cholesky(R)

    player_cf_params = []
    effective_skew = np.zeros((len(player_names), 4), dtype=float)

    for idx, player in enumerate(player_names):
        # Build category means/stds
        if player in mu_w.index:
            mu_row = mu_w.loc[player].reindex(CAT_ORDER)
            std_row = std_w.loc[player].reindex(CAT_ORDER)
        else:
            mu_row = pd.Series(index=CAT_ORDER, dtype=float)
            std_row = pd.Series(index=CAT_ORDER, dtype=float)

        mu = mu_row.fillna(global_mu.reindex(CAT_ORDER)).to_numpy(dtype=float)
        std = std_row.fillna(global_std.reindex(CAT_ORDER)).to_numpy(dtype=float).clip(1e-6)

        # Apply course variance multipliers
        std_course = std * _course_mults_cf

        player_cf_params.append((mu, std_course))

        # Per-player effective skewness (confidence-weighted blend)
        for j, cat in enumerate(CAT_ORDER):
            p_skew = (skew_w.at[player, cat]
                      if player in skew_w.index and cat in skew_w.columns
                         and pd.notna(skew_w.at[player, cat])
                      else 0.0)
            p_neff = (neff_w.at[player, cat]
                      if player in neff_w.index and cat in neff_w.columns
                         and pd.notna(neff_w.at[player, cat])
                      else 0.0)
            confidence = min(p_neff / SKEW_CONFIDENCE_N_CF, 1.0)
            blend_w = SKEW_BLEND_MAX_CF * confidence
            effective_skew[idx, j] = (1 - blend_w) * _course_skew_cf[j] + blend_w * p_skew

    return player_cf_params, effective_skew, L_corr


def _catfirst_draw(mu, std_c, eff_skew, skill_mean, L_corr, rng, num_sims):
    """Category-first draw for one player (no weather splitting).

    Args:
        mu: (4,) base category means (un-recentered)
        std_c: (4,) course-adjusted category stds
        eff_skew: (4,) effective skewness per category
        skill_mean: scalar OR (num_sims,) per-sim-path skill target
        L_corr: (4,4) Cholesky of correlation matrix
        rng: numpy RNG
        num_sims: number of simulations

    Returns:
        cats: (num_sims, 4) category draws
        sg_total: (num_sims,) total SG = sum of categories
    """
    skill_shift = skill_mean - mu.sum()
    if np.ndim(skill_shift) == 0:
        cat_mu = mu + skill_shift / 4.0  # (4,)
    else:
        cat_mu = mu + skill_shift[:, None] / 4.0  # (num_sims, 4)

    Z = rng.standard_normal(size=(num_sims, 4))
    corr_z = Z @ L_corr.T
    for j in range(4):
        corr_z[:, j] = _apply_skew(corr_z[:, j], eff_skew[j])
    draws = cat_mu + corr_z * std_c
    cats = np.clip(draws, CLIP_CAT[0], CLIP_CAT[1])
    return cats, cats.sum(axis=1)


def rank_positions_from_strokes(strokes_asc_int):
    """Get rank positions from stroke array."""
    s = pd.Series(strokes_asc_int)
    return s.rank(method='min').astype(int).to_numpy()


def coeff_vec_r1(cdict):
    """Build coefficient vector for R1 update: [ott, app, arg, putt, residual, residual2]."""
    return np.array([
        cdict.get('ott', 0.0), 0.0, 0.0, cdict.get('putt', 0.0),
        cdict.get('residual', 0.0), cdict.get('residual2', 0.0)
    ], dtype=float)


def ensure_array(x, shape):
    """Ensure x is an array of the given shape, defaulting to zeros."""
    return x if isinstance(x, np.ndarray) else np.zeros(shape, dtype=float)


def dead_heat_factor(position, tie_count, threshold):
    """Calculate dead heat factor for top-N finish."""
    start = position
    end = position + tie_count - 1
    overlap_start = max(start, 1)
    overlap_end = min(end, threshold)
    overlap_count = max(0, overlap_end - overlap_start + 1)
    return overlap_count / tie_count


def parse_time(teetime):
    """Parse tee time string to datetime."""
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
    """Calculate 5-hour average wind starting from tee time."""
    parsed = parse_time(teetime)
    if parsed is None or not wind_data:
        return 0.0
    dec_hour = parsed.hour + parsed.minute / 60.0
    start_idx = dec_hour - 6  # wind array starts at 6 AM
    end_idx = start_idx + 5
    minutes = np.arange(start_idx, end_idx, 1/60.0)
    return float(np.mean(np.interp(minutes, np.arange(len(wind_data)), wind_data)))


# ══════════════════════════════════════════════════════════════════════════════
# Tournament Sim Config Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_tournament_config(sheet_config):
    """
    Build tournament sim config from sheet config.

    Returns dict with:
        course_map: dict {course_code: {"par": int, "expected_r2": float, ...}}
        default_par: int
        wind_arrays: dict {2: [...], 3: [...], 4: [...]}
        dew_arrays: dict {2: [...], 3: [...], 4: [...]}
    """
    course_codes = sheet_config.get("course_codes", [])
    course_pars = sheet_config.get("course_pars", [])

    # Build course map
    course_map = {}
    if course_codes and course_pars:
        for i, code in enumerate(course_codes):
            par = course_pars[i] if i < len(course_pars) else PAR
            course_map[code] = {"par": int(par)}

            # Add per-round expected scores if available
            for rnd, key in [(1, "expected_score_r1"), (2, "expected_score_r2"), (3, "expected_score_r3"), (4, "expected_score_r4")]:
                exp_list = sheet_config.get(key, [])
                if exp_list:
                    exp_val = exp_list[i] if i < len(exp_list) else exp_list[0]
                    course_map[code][f"expected_r{rnd}"] = exp_val

    # Default expected score fallback (use expected_score_1 from sheet or PAR)
    es1 = sheet_config.get("expected_score_1", 0)
    default_expected = es1 if abs(es1) > 50 else es1 + PAR

    # Wind arrays per round (fallback to generic 'wind')
    default_wind = sheet_config.get("wind", [])
    wind_arrays = {
        1: sheet_config.get("wind_r1", []) or default_wind,
        2: sheet_config.get("wind_r2", []) or default_wind,
        3: sheet_config.get("wind_r3", []) or default_wind,
        4: sheet_config.get("wind_r4", []) or default_wind,
    }

    # Dew arrays per round (fallback to generic 'dew')
    default_dew = sheet_config.get("dew", [])
    dew_arrays = {
        1: sheet_config.get("dew_r1", []) or default_dew,
        2: sheet_config.get("dew_r2", []) or default_dew,
        3: sheet_config.get("dew_r3", []) or default_dew,
        4: sheet_config.get("dew_r4", []) or default_dew,
    }

    return {
        "course_map": course_map,
        "default_par": PAR,
        "default_expected": default_expected,
        "wind_arrays": wind_arrays,
        "dew_arrays": dew_arrays,
    }


def load_known_rounds(completed_round, course_map, default_par):
    """
    Load actual scores and SG categories from completed rounds.

    Reads: r1_live_model.csv, r2_live_model.csv, etc.

    Returns dict with:
        player_names: list[str]
        strokes: dict {round_num: np.array}
        categories: dict {round_num: np.array (n_players, 4)}
        cumulative: np.array (total strokes through completed rounds)
        made_cut: np.array[bool] (True if player made cut)
        course_x: dict {player: course_code}
    """
    result = {
        "player_names": [],
        "strokes": {},
        "categories": {},
        "cumulative": None,
        "made_cut": None,
        "course_x": {},
        "player_preds": {},  # base predictions for skill updates
    }

    # Load R1 live model (always needed if completed_round >= 1)
    all_players = None

    for rnd in range(1, completed_round + 1):
        live_file = f"r{rnd}_live_model.csv"
        if not os.path.exists(live_file):
            print(f"  Warning: {live_file} not found. Skipping round {rnd}.")
            continue

        df = pd.read_csv(live_file)
        df['player_name'] = df['player_name'].str.lower().str.strip().replace(name_replacements)

        if all_players is None:
            # Exclude players with NaN pred — they can't be simulated and
            # NaN propagates through skill updates, producing garbage scores
            pred_col = 'pred' if 'pred' in df.columns else ('my_pred' if 'my_pred' in df.columns else None)
            if pred_col:
                nan_pred = df[df[pred_col].isna()]['player_name'].tolist()
                if nan_pred:
                    print(f"    Excluding {len(nan_pred)} player(s) with NaN pred: {nan_pred}")
                    df = df[df[pred_col].notna()].copy()

            all_players = df['player_name'].tolist()
            result["player_names"] = all_players

            # Get course assignments
            if 'course_x' in df.columns:
                result["course_x"] = dict(zip(df['player_name'], df['course_x']))

            # Get base predictions
            if pred_col:
                result["player_preds"] = dict(zip(df['player_name'], df[pred_col]))

        # Calculate strokes for this round
        # strokes = par - sg_total (or use 'total' column if available)
        n_players = len(all_players)
        strokes_arr = np.zeros(n_players)
        cats_arr = np.zeros((n_players, 4))

        for i, player in enumerate(all_players):
            row = df[df['player_name'] == player]
            if row.empty:
                strokes_arr[i] = PAR  # default to par if missing
                continue

            row = row.iloc[0]

            # Get player's par
            player_course = result["course_x"].get(player)
            player_par = course_map.get(player_course, {}).get("par", default_par) if player_course else default_par

            # Get strokes
            sg_col = f"sg_total_r{rnd}" if f"sg_total_r{rnd}" in df.columns else "sg_total"
            if sg_col in df.columns and pd.notna(row.get(sg_col)):
                strokes_arr[i] = player_par - row[sg_col]
            elif 'total' in df.columns:
                strokes_arr[i] = row['total']
            else:
                strokes_arr[i] = player_par

            # Get categories
            for j, cat in enumerate(["sg_ott", "sg_app", "sg_arg", "sg_putt"]):
                cat_col = f"{cat}_r{rnd}" if f"{cat}_r{rnd}" in df.columns else cat
                if cat_col in df.columns and pd.notna(row.get(cat_col)):
                    cats_arr[i, j] = row[cat_col]

        result["strokes"][rnd] = strokes_arr.astype(int)
        result["categories"][rnd] = cats_arr

    # Calculate cumulative strokes
    if result["strokes"]:
        result["cumulative"] = sum(result["strokes"].values())

        # Determine made cut status (if completed_round >= 2)
        if completed_round >= 2:
            r1_r2 = result["strokes"].get(1, 0) + result["strokes"].get(2, 0)
            if isinstance(r1_r2, np.ndarray):
                cut_score = np.sort(r1_r2)[min(CUT_LINE - 1, len(r1_r2) - 1)]
                result["made_cut"] = r1_r2 <= cut_score
                if USE_10_SHOT_RULE:
                    within_10 = r1_r2 <= (r1_r2.min() + 10)
                    result["made_cut"] = result["made_cut"] | within_10
            else:
                result["made_cut"] = np.ones(len(result["player_names"]), dtype=bool)
        else:
            result["made_cut"] = np.ones(len(result["player_names"]), dtype=bool)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Tournament Simulation Engine
# ══════════════════════════════════════════════════════════════════════════════

def simulate_remaining_rounds(
    completed_round,
    player_names,
    known_strokes,
    known_categories,
    model_preds,
    player_cf_params,
    effective_skew,
    L_corr,
    tournament_config,
    player_preds_base,
    num_sims=TOURNAMENT_SIMULATIONS,
):
    """
    Simulate from round (completed_round + 1) through R4 using category-first draws.

    Returns:
        final_scores: np.array (n_players, num_sims) - 72-hole totals
        made_cut_mask: np.array[bool] (n_players, num_sims)
    """
    n_players = len(player_names)
    default_par = tournament_config["default_par"]

    # Get base predictions for each player
    my_pred_base = np.array([player_preds_base.get(p, 0.0) for p in player_names])

    # Per-player expected score for R2 (multi-course aware)
    player_expected_r2 = np.full(n_players, default_par, dtype=float)
    if model_preds is not None and 'course_score_adj' in model_preds.columns:
        for i, p in enumerate(player_names):
            row = model_preds[model_preds['player_name'] == p]
            if not row.empty and pd.notna(row['course_score_adj'].iloc[0]):
                player_expected_r2[i] = row['course_score_adj'].iloc[0]
        if np.unique(player_expected_r2).size > 1:
            print(f"    Multi-course R2: expected scores = {dict(zip(*np.unique(player_expected_r2, return_counts=True)))}")

    # Initialize accumulators
    if completed_round >= 1 and 1 in known_strokes:
        strokes_r1 = np.tile(known_strokes[1][:, np.newaxis], (1, num_sims))
        cats_r1 = np.tile(known_categories.get(1, np.zeros((n_players, 4)))[:, np.newaxis, :], (1, num_sims, 1))
    else:
        # Simulate R1 — category-first draws
        cats_r1 = np.empty((n_players, num_sims, 4), dtype=float)
        sg_r1 = np.empty((n_players, num_sims), dtype=float)
        for i, (mu, std_c) in enumerate(player_cf_params):
            cats_r1[i], sg_r1[i] = _catfirst_draw(
                mu, std_c, effective_skew[i], my_pred_base[i],
                L_corr, RNG, num_sims
            )
        strokes_r1 = np.clip(np.rint(default_par - sg_r1), default_par - 12, default_par + 12).astype(int)

    # R1 -> R2 skill update
    sg_r1_actual = default_par - strokes_r1.astype(float)

    resid_r1 = sg_r1_actual - my_pred_base[:, None]
    resid2_r1 = resid_r1 ** 2
    ott_r1 = cats_r1[:, :, 0]
    putt_r1 = cats_r1[:, :, 3]

    # Skill buckets for R1
    high_m = (my_pred_base > 1.0)
    midh_m = (my_pred_base > 0.5) & (my_pred_base <= 1.0)
    midl_m = (my_pred_base > -0.5) & (my_pred_base <= 0.5)
    low_m = (my_pred_base <= -0.5)

    C_high = coeff_vec_r1(coefficients_r1_high)
    C_midh = coeff_vec_r1(coefficients_r1_midh)
    C_midl = coeff_vec_r1(coefficients_r1_midl)
    C_low = coeff_vec_r1(coefficients_r1_low)

    C = np.zeros((n_players, 6), dtype=float)
    C[high_m] = C_high
    C[midh_m] = C_midh
    C[midl_m] = C_midl
    C[low_m] = C_low

    tot_resid_adj_r1 = resid_r1 * C[:, [4]] + resid2_r1 * C[:, [5]]
    mask_bad = (resid_r1 < 0) & (tot_resid_adj_r1 > 0.2)
    tot_resid_adj_r1 = np.minimum(np.where(mask_bad, 0.2, tot_resid_adj_r1), 0.5)

    ott_adj_r1 = ott_r1 * C[:, [0]]
    putt_adj_r1 = putt_r1 * C[:, [3]]
    sg_adj_r1 = ott_adj_r1 + putt_adj_r1
    total_adjustment_r1 = tot_resid_adj_r1 + sg_adj_r1

    updated_skill_r2 = my_pred_base[:, None] + total_adjustment_r1

    # R2 simulation or use known
    if completed_round >= 2 and 2 in known_strokes:
        strokes_r2 = np.tile(known_strokes[2][:, np.newaxis], (1, num_sims))
        cats_r2 = np.tile(known_categories.get(2, np.zeros((n_players, 4)))[:, np.newaxis, :], (1, num_sims, 1))
        sg_r2 = (default_par - strokes_r2.astype(float))
    else:
        # Category-first draws — per-sim-path skill mean
        cats_r2 = np.empty((n_players, num_sims, 4), dtype=float)
        sg_r2 = np.empty((n_players, num_sims), dtype=float)
        for i, (mu, std_c) in enumerate(player_cf_params):
            cats_r2[i], sg_r2[i] = _catfirst_draw(
                mu, std_c, effective_skew[i], updated_skill_r2[i],
                L_corr, RNG, num_sims
            )
        strokes_r2 = np.clip(np.rint(player_expected_r2[:, None] - sg_r2), (player_expected_r2 - 12)[:, None], (player_expected_r2 + 12)[:, None]).astype(int)

    r1_r2_scores = strokes_r1 + strokes_r2

    # Cut logic after 36 holes
    made_cut_mask = np.ones_like(r1_r2_scores, dtype=bool)
    if completed_round < 2:
        # Simulate cut
        for j in range(num_sims):
            sc = r1_r2_scores[:, j]
            cut_score = np.sort(sc)[min(CUT_LINE - 1, len(sc) - 1)]
            top_cut = sc <= cut_score
            if USE_10_SHOT_RULE:
                within_10 = sc <= (sc.min() + 10)
                made_cut_mask[:, j] = top_cut | within_10
            else:
                made_cut_mask[:, j] = top_cut
    else:
        # Use known cut status - broadcast to all sims
        # Players who missed cut in reality stay out
        pass  # made_cut_mask already all True, we'll handle in final scores

    # R2 -> R3 skill update
    resid_r2 = sg_r2 - updated_skill_r2
    resid2_r2 = resid_r2 ** 2
    resid3_r2 = resid_r2 ** 3

    avg_ott_r2 = 0.5 * (cats_r1[:, :, 0] + cats_r2[:, :, 0])
    avg_app_r2 = 0.5 * (cats_r1[:, :, 1] + cats_r2[:, :, 1])
    avg_arg_r2 = 0.5 * (cats_r1[:, :, 2] + cats_r2[:, :, 2])
    avg_putt_r2 = 0.5 * (cats_r1[:, :, 3] + cats_r2[:, :, 3])
    delta_app_r2 = cats_r2[:, :, 1] - cats_r1[:, :, 1]

    # Position buckets for R2->R3
    pos_lt_6_mask = np.zeros((n_players, num_sims), dtype=bool)
    pos_6_30_mask = np.zeros((n_players, num_sims), dtype=bool)
    pos_gt_30_mask = np.zeros((n_players, num_sims), dtype=bool)

    for j in range(num_sims):
        pos = rank_positions_from_strokes(r1_r2_scores[:, j])
        pos_lt_6_mask[:, j] = (pos < 6)
        pos_6_30_mask[:, j] = (pos >= 6) & (pos <= 30)
        pos_gt_30_mask[:, j] = (pos > 30)

    def apply_block_r2(adj_dict, mask, resid_r2_arr, resid2_r2_arr, resid3_r2_arr,
                       avg_ott_arr, avg_putt_arr, avg_app_arr, avg_arg_arr, delta_app_arr):
        out = {}
        for key, coeff in adj_dict.items():
            if key == 'residual':
                base = resid_r2_arr
            elif key == 'residual2':
                base = resid2_r2_arr
            elif key == 'residual3':
                base = resid3_r2_arr
            elif key == 'avg_ott':
                base = avg_ott_arr
            elif key == 'avg_putt':
                base = avg_putt_arr
            elif key == 'avg_app':
                base = avg_app_arr
            elif key == 'avg_arg':
                base = avg_arg_arr
            elif key == 'delta_app':
                base = delta_app_arr
            else:
                continue
            out[f"{key}_adj"] = np.where(mask, base * coeff, 0.0)
        return out

    adj_lt6 = apply_block_r2(coefficients_r2, pos_lt_6_mask, resid_r2, resid2_r2, resid3_r2,
                             avg_ott_r2, avg_putt_r2, avg_app_r2, avg_arg_r2, delta_app_r2)
    adj_6_30 = apply_block_r2(coefficients_r2_6_30, pos_6_30_mask, resid_r2, resid2_r2, resid3_r2,
                              avg_ott_r2, avg_putt_r2, avg_app_r2, avg_arg_r2, delta_app_r2)
    adj_30up = apply_block_r2(coefficients_r2_30_up, pos_gt_30_mask, resid_r2, resid2_r2, resid3_r2,
                              avg_ott_r2, avg_putt_r2, avg_app_r2, avg_arg_r2, delta_app_r2)

    all_keys = set(adj_lt6) | set(adj_6_30) | set(adj_30up)
    adj_sum = {}
    for k in all_keys:
        adj_sum[k] = adj_lt6.get(k, 0.0) + adj_6_30.get(k, 0.0) + adj_30up.get(k, 0.0)

    shape2 = (n_players, num_sims)
    tot_resid_adj_r2 = (
        ensure_array(adj_sum.get('residual_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('residual2_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('residual3_adj', 0.0), shape2)
    )
    tot_sg_adj_r2 = (
        ensure_array(adj_sum.get('avg_ott_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('avg_putt_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('avg_app_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('avg_arg_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('delta_app_adj', 0.0), shape2)
    )

    total_adjustment_r2 = (tot_resid_adj_r2 + tot_sg_adj_r2) - ensure_array(sg_adj_r1, shape2)
    updated_skill_r3 = updated_skill_r2 + total_adjustment_r2

    # R3 simulation or use known
    if completed_round >= 3 and 3 in known_strokes:
        strokes_r3 = np.tile(known_strokes[3][:, np.newaxis], (1, num_sims))
        cats_r3 = np.tile(known_categories.get(3, np.zeros((n_players, 4)))[:, np.newaxis, :], (1, num_sims, 1))
        sg_r3 = (default_par - strokes_r3.astype(float))
    else:
        # Category-first draws — per-sim-path skill mean
        cats_r3 = np.empty((n_players, num_sims, 4), dtype=float)
        sg_r3 = np.empty((n_players, num_sims), dtype=float)
        for i, (mu, std_c) in enumerate(player_cf_params):
            cats_r3[i], sg_r3[i] = _catfirst_draw(
                mu, std_c, effective_skew[i], updated_skill_r3[i],
                L_corr, RNG, num_sims
            )
        strokes_r3 = np.clip(np.rint(default_par - sg_r3), default_par - 12, default_par + 12).astype(int)

    r1_r3_scores = r1_r2_scores + strokes_r3

    # R3 -> R4 skill update (SG-only, no residual)
    avg_ott_r3 = 0.66 * avg_ott_r2 + 0.34 * cats_r3[:, :, 0]
    avg_app_r3 = 0.66 * avg_app_r2 + 0.34 * cats_r3[:, :, 1]
    avg_arg_r3 = 0.66 * avg_arg_r2 + 0.34 * cats_r3[:, :, 2]
    avg_putt_r3 = 0.66 * avg_putt_r2 + 0.34 * cats_r3[:, :, 3]

    pos_lt_6_mask_r3 = np.zeros((n_players, num_sims), dtype=bool)
    pos_6_20_mask_r3 = np.zeros((n_players, num_sims), dtype=bool)
    pos_gt_20_mask_r3 = np.zeros((n_players, num_sims), dtype=bool)

    for j in range(num_sims):
        pos = rank_positions_from_strokes(r1_r3_scores[:, j])
        pos_lt_6_mask_r3[:, j] = (pos < 6)
        pos_6_20_mask_r3[:, j] = (pos >= 6) & (pos <= 20)
        pos_gt_20_mask_r3[:, j] = (pos > 20)

    def apply_block_r3_avg(adj_dict, mask, avg_ott, avg_putt, avg_app, avg_arg):
        out = {}
        for key, coeff in adj_dict.items():
            if key == 'sg_ott_avg':
                base = avg_ott
            elif key == 'sg_putt_avg':
                base = avg_putt
            elif key == 'sg_app_avg':
                base = avg_app
            elif key == 'sg_arg_avg':
                base = avg_arg
            else:
                continue
            out[f"{key}_adj_r3"] = np.where(mask, base * coeff, 0.0)
        return out

    adj_lt6_r3 = apply_block_r3_avg(coefficients_r3, pos_lt_6_mask_r3, avg_ott_r3, avg_putt_r3, avg_app_r3, avg_arg_r3)
    adj_6_20_r3 = apply_block_r3_avg(coefficients_r3_mid, pos_6_20_mask_r3, avg_ott_r3, avg_putt_r3, avg_app_r3, avg_arg_r3)
    adj_20up_r3 = apply_block_r3_avg(coefficients_r3_high, pos_gt_20_mask_r3, avg_ott_r3, avg_putt_r3, avg_app_r3, avg_arg_r3)

    all_keys_r3 = set(adj_lt6_r3) | set(adj_6_20_r3) | set(adj_20up_r3)
    adj_sum_r3 = {}
    for k in all_keys_r3:
        adj_sum_r3[k] = adj_lt6_r3.get(k, 0.0) + adj_6_20_r3.get(k, 0.0) + adj_20up_r3.get(k, 0.0)

    tot_sg_adj_r3 = (
        ensure_array(adj_sum_r3.get('sg_ott_avg_adj_r3', 0.0), shape2) +
        ensure_array(adj_sum_r3.get('sg_putt_avg_adj_r3', 0.0), shape2) +
        ensure_array(adj_sum_r3.get('sg_app_avg_adj_r3', 0.0), shape2) +
        ensure_array(adj_sum_r3.get('sg_arg_avg_adj_r3', 0.0), shape2)
    )

    # Undo R2 adjustments, apply R3 adjustments
    updated_skill_r4 = updated_skill_r3 - (tot_sg_adj_r2 + tot_resid_adj_r2) + tot_sg_adj_r3

    # R4 simulation — category-first draws, per-sim-path skill mean
    cats_r4 = np.empty((n_players, num_sims, 4), dtype=float)
    sg_r4 = np.empty((n_players, num_sims), dtype=float)
    for i, (mu, std_c) in enumerate(player_cf_params):
        cats_r4[i], sg_r4[i] = _catfirst_draw(
            mu, std_c, effective_skew[i], updated_skill_r4[i],
            L_corr, RNG, num_sims
        )
    strokes_r4 = np.clip(np.rint(default_par - sg_r4), default_par - 12, default_par + 12).astype(int)

    # Missed-cut penalty
    r3_r4 = strokes_r3 + strokes_r4
    r3_r4[~made_cut_mask] = 200

    # Final 72-hole totals
    final_scores = r1_r2_scores + r3_r4

    return final_scores, made_cut_mask


def compute_finish_probabilities(final_scores, player_names, made_cut_mask, num_sims):
    """
    Compute win and top-N probabilities from simulated 72-hole totals.

    Returns DataFrame with columns: player_name, simulated_win_prob, top_5, top_10, top_20
    """
    n_players = len(player_names)

    # Win probabilities (playoff tiebreaker: random winner)
    simulated_winners = []
    for j in range(num_sims):
        sc = final_scores[:, j]
        min_score = sc.min()
        tied = np.where(sc == min_score)[0]
        winner_idx = RNG.choice(tied)
        simulated_winners.append(player_names[winner_idx])

    win_counts = pd.Series(simulated_winners).value_counts(normalize=True)
    sim_win_probs = win_counts.rename_axis('player_name').reset_index(name='simulated_win_prob')

    # Top-N with dead-heat adjustment
    df_long = pd.DataFrame(final_scores, index=player_names).T
    df_long['simulation_id'] = np.arange(num_sims)
    long_df = df_long.melt(id_vars='simulation_id', var_name='player_name', value_name='score')
    long_df['rank'] = long_df.groupby('simulation_id')['score'].rank(method='min')

    player_stats = {p: {
        "top_5": 0.0, "top_10": 0.0, "top_20": 0.0,
        "top_5_nodh": 0.0, "top_10_nodh": 0.0, "top_20_nodh": 0.0,
    } for p in player_names}
    for sim_id, group in long_df.groupby("simulation_id", sort=False):
        pos_counts = group['rank'].value_counts().to_dict()
        for _, row in group.iterrows():
            p = row['player_name']
            pos = int(row['rank'])
            tie_ct = pos_counts[pos]
            # Dead-heat adjusted (traditional sportsbooks)
            player_stats[p]["top_5"] += dead_heat_factor(pos, tie_ct, 5)
            player_stats[p]["top_10"] += dead_heat_factor(pos, tie_ct, 10)
            player_stats[p]["top_20"] += dead_heat_factor(pos, tie_ct, 20)
            # No dead-heat (Kalshi — ties all count as finishing inside)
            player_stats[p]["top_5_nodh"] += 1.0 if pos <= 5 else 0.0
            player_stats[p]["top_10_nodh"] += 1.0 if pos <= 10 else 0.0
            player_stats[p]["top_20_nodh"] += 1.0 if pos <= 20 else 0.0

    topn_df = pd.DataFrame.from_dict(player_stats, orient='index')
    topn_df = topn_df.div(num_sims).reset_index().rename(columns={'index': 'player_name'})

    # Merge win probs and top-N
    finish_probs = pd.merge(sim_win_probs, topn_df, on="player_name", how="outer").fillna(0)

    # Per-player, per-position rank probabilities (for distribution dashboard)
    rank_probs = (long_df.groupby(['player_name', 'rank']).size()
                  .div(num_sims).rename('prob_u').reset_index())
    rank_probs['rank'] = rank_probs['rank'].astype(int)

    return finish_probs, rank_probs


# ══════════════════════════════════════════════════════════════════════════════
# Outright Market Pricing
# ══════════════════════════════════════════════════════════════════════════════

def decimal_to_american(decimal_odds):
    """Convert decimal odds to American odds."""
    if pd.isna(decimal_odds):
        return np.nan
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(round(-100 / (decimal_odds - 1)))


def fetch_outright_odds(market_name):
    """Fetch outright odds from DataGolf API."""
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
        print(f"  Warning: Failed to fetch {market_name}: {e}")
        return {}


def extract_market_rows(json_obj, odds_key='odds'):
    """Extract rows from outright market JSON."""
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

        for book in BOOKS_TO_USE:
            odds = entry.get(book)
            if odds is not None:
                rows.append({
                    'player_name': player,
                    'bookmaker': book,
                    'decimal_odds': float(odds)
                })

    return pd.DataFrame(rows)


def price_outrights(finish_probs, pred_lookup, sample_lookup):
    """
    Price outright and top-N markets against DataGolf odds.

    Returns dict of DataFrames: {'win': df, 'top_5': df, 'top_10': df, 'top_20': df}
    """
    results = {}
    markets = [
        ('win', 'simulated_win_prob', EDGE_THRESHOLD_WIN),
        ('top_5', 'top_5', EDGE_THRESHOLD_TOPN),
        ('top_10', 'top_10', EDGE_THRESHOLD_TOPN),
        ('top_20', 'top_20', EDGE_THRESHOLD_TOPN),
    ]

    for market_name, prob_col, edge_threshold in markets:
        data = fetch_outright_odds(market_name)
        if not data:
            results[market_name] = pd.DataFrame()
            continue

        df = extract_market_rows(data, odds_key='odds')
        if df.empty:
            results[market_name] = pd.DataFrame()
            continue

        # Normalize player names
        df['player_name'] = df['player_name'].str.lower().str.strip().replace(name_replacements)

        # Merge with sim probabilities
        if prob_col not in finish_probs.columns:
            results[market_name] = pd.DataFrame()
            continue

        df = df.merge(
            finish_probs[['player_name', prob_col]],
            on='player_name',
            how='inner'
        )

        if df.empty:
            results[market_name] = pd.DataFrame()
            continue

        # Calculate edge (sim prob minus implied prob, in percentage points)
        df['implied_prob'] = 1.0 / df['decimal_odds']
        df['american_odds'] = df['decimal_odds'].apply(decimal_to_american)

        p = df[prob_col].astype(float)
        b = df['decimal_odds'] - 1.0
        q = 1.0 - p
        df['edge'] = (p - df['implied_prob']) * 100.0

        # Filter by edge threshold
        df = df[df['edge'] > edge_threshold].copy()

        if df.empty:
            results[market_name] = pd.DataFrame()
            continue

        # Kelly sizing
        f_star = (b * p - q) / b
        df['stake'] = (BANKROLL * KELLY_FRACTION * f_star.clip(lower=0)).astype(float)
        df['eg'] = f_star * df['edge'] / 2.0
        df['market_type'] = market_name

        # Add pred and sample
        df['my_pred'] = df['player_name'].map(pred_lookup)
        df['sample'] = df['player_name'].map(sample_lookup)

        # Fair odds (clip to avoid None from 0/1 boundary)
        df['my_fair'] = df[prob_col].clip(1e-4, 1 - 1e-4).apply(implied_to_american)

        results[market_name] = df

    return results


def _kalshi_taker_fee(price):
    """Kalshi taker fee: 7% of price * (1 - price), matching scraper formula."""
    if price <= 0 or price >= 1:
        return 0.0
    return 0.07 * price * (1 - price)


def price_kalshi_outrights(finish_probs, pred_lookup, sample_lookup):
    """Price Kalshi outright markets using live API with orderbook-aware liquidity filtering.

    Two-stage approach (matches new_sim.py):
      Stage 1: Pre-filter on bid/ask presence, spread ≤ 10c, mid edge > 0.5%
      Stage 2: Fetch orderbook, Kelly-walk for effective fill, gate taker rows on MIN_DEPTH

    Taker rows are the walked VWAP (incl. fee) — only emitted when fillable depth ≥ MIN_DEPTH.
    Mid rows are the bid/ask midpoint (maker, no fee) — emitted for any stage-1 survivor.
    Downstream splits on `pricing` (taker → outrights pipeline; mid → maker-opportunity email).
    """
    import re as _re
    import time as _time
    from collections import Counter
    import httpx as _httpx

    try:
        from sim_inputs import name_replacements
    except ImportError:
        name_replacements = {}

    _KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
    _client = _httpx.Client(timeout=15.0, headers={"Accept": "application/json"})

    def _get_markets(series_ticker):
        all_mkts, cursor = [], None
        while True:
            params = {"limit": 200, "status": "open", "series_ticker": series_ticker}
            if cursor:
                params["cursor"] = cursor
            resp = _client.get(f"{_KALSHI_API}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            mkts = data.get("markets", [])
            all_mkts.extend(mkts)
            cursor = data.get("cursor")
            if not cursor or len(mkts) < 200:
                break
        return all_mkts

    def _get_orderbook(ticker):
        resp = _client.get(f"{_KALSHI_API}/markets/{ticker}/orderbook")
        resp.raise_for_status()
        data = resp.json()
        ob = data.get("orderbook", data.get("orderbook_fp", data))
        def parse_levels(raw):
            out = []
            for item in (raw or []):
                if isinstance(item, list) and len(item) == 2:
                    out.append((float(item[0]), int(float(item[1]))))
            return out
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
            mkts = _get_markets(series_ticker)
            for m in mkts:
                m["_market_type"] = mtype
            all_markets.extend(mkts)
        except Exception as e:
            print(f"  [warn] Failed to fetch {series_ticker}: {e}")
    print(f"  Fetched {len(all_markets)} Kalshi markets from live API")

    if not all_markets:
        return pd.DataFrame()

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
        all_markets = [m for m in all_markets
                       if _detect_tourney(m.get("title", "")) in (current_tourney, "")]
        print(f"  Tournament: {current_tourney} ({len(all_markets)} markets)")

    def _extract_player(title):
        m = _re.match(r".*:\s*Will (.+?) (?:finish|make|miss|lead|win)", title)
        if m:
            return m.group(1).strip()
        m = _re.match(r"Will (.+?) win the ", title)
        if m:
            return m.group(1).strip()
        return ""

    type_to_col = {
        "top_5": "top_5_nodh",
        "top_10": "top_10_nodh",
        "top_20": "top_20_nodh",
        "winner": "simulated_win_prob",
    }

    def norm(s):
        x = s.strip().lower()
        if "," not in x:
            parts = x.rsplit(" ", 1)
            if len(parts) == 2:
                x = f"{parts[1]}, {parts[0]}"
        return name_replacements.get(x, x)

    KALSHI_BANKROLL = 30_000.0
    KALSHI_KELLY = 0.60
    MIN_DEPTH = 400
    MIN_EDGE_STAGE1 = 0.5  # pct points on mid — loose gate

    def _walk_book(ob_levels, sim_prob, bankroll, kelly_frac):
        """Walk opposite-side levels to compute Kelly-sized VWAP fill."""
        fill_levels = sorted([(1.0 - p, qty) for p, qty in ob_levels])
        if not fill_levels:
            return None
        best = fill_levels[0][0]
        eff_best = best + _kalshi_taker_fee(best)
        if eff_best <= 0 or eff_best >= 1 or eff_best >= sim_prob:
            return None
        b = (1.0 / eff_best) - 1.0
        if b <= 0:
            return None
        f_star = max((b * sim_prob - (1 - sim_prob)) / b, 0) * kelly_frac
        target = int(bankroll * f_star / eff_best)
        if target <= 0:
            return None
        filled, cost_sum, fee_sum = 0, 0.0, 0.0
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
        return {"effective_price": vwap + avg_fee, "target": target, "filled": filled}

    # ── Stage 1: pre-filter on bid/ask + mid edge ────────────────────
    stage1 = []
    kalshi_mismatches = set()
    for mkt in all_markets:
        ticker = mkt.get("ticker", "")
        title = mkt.get("title", "")
        mtype = mkt.get("_market_type", "")

        bid = float(mkt.get("yes_bid_dollars") or 0)
        ask = float(mkt.get("yes_ask_dollars") or 0)
        if bid == 0 and ask == 0:
            bid = float(mkt.get("yes_bid", 0) or 0) / 100.0
            ask = float(mkt.get("yes_ask", 0) or 0) / 100.0
        if bid <= 0 or ask <= 0:
            continue
        if (ask - bid) > 0.10:
            continue

        player_raw = _extract_player(title)
        if not player_raw:
            continue
        player = norm(player_raw)

        prob_col = type_to_col.get(mtype)
        if not prob_col or prob_col not in finish_probs.columns:
            continue
        match = finish_probs[finish_probs["player_name"] == player]
        if match.empty:
            kalshi_mismatches.add(player)
            continue

        sim_yes = float(match.iloc[0][prob_col])
        if sim_yes <= 0:
            continue
        sim_no = 1.0 - sim_yes
        mid = (bid + ask) / 2.0
        yes_mid_edge = (sim_yes - mid) * 100
        no_mid_edge = (sim_no - (1 - mid)) * 100
        if max(yes_mid_edge, no_mid_edge) < MIN_EDGE_STAGE1:
            continue

        stage1.append({
            "ticker": ticker, "player": player, "mtype": mtype,
            "bid": bid, "ask": ask, "mid": mid,
            "sim_yes": sim_yes, "sim_no": sim_no,
            "yes_mid_edge": yes_mid_edge, "no_mid_edge": no_mid_edge,
        })

    print(f"  Stage 1 pre-filter: {len(stage1)} markets pass (edge > {MIN_EDGE_STAGE1}%)")
    if not stage1:
        df = pd.DataFrame()
        if kalshi_mismatches:
            df.attrs["name_mismatches"] = kalshi_mismatches
        return df

    # ── Stage 2: orderbook walk with Kelly sizing ────────────────────
    rows = []
    ob_fail = 0
    for s in stage1:
        try:
            ob = _get_orderbook(s["ticker"])
            _time.sleep(0.05)
        except Exception:
            ob_fail += 1
            ob = {"yes": [], "no": []}

        base_row = {
            "player_name": s["player"],
            "market_type": s["mtype"],
            "bookmaker": "kalshi",
            "my_pred": pred_lookup.get(s["player"]),
            "sample": sample_lookup.get(s["player"]),
        }

        # YES taker: walk NO levels (sellers of YES)
        yf = _walk_book(ob.get("no", []), s["sim_yes"], KALSHI_BANKROLL, KALSHI_KELLY)
        if yf and yf["filled"] >= MIN_DEPTH:
            eff = yf["effective_price"]
            rows.append({
                **base_row, "side": "yes", "pricing": "taker",
                "american_odds": implied_to_american(min(max(eff, 1e-4), 1 - 1e-4)),
                "sim_prob": s["sim_yes"], "implied_prob": eff,
                "edge": round((s["sim_yes"] - eff) * 100, 1),
                "my_fair": implied_to_american(min(max(s["sim_yes"], 1e-4), 1 - 1e-4)),
                "target": yf["target"], "filled": yf["filled"],
            })

        # NO taker: walk YES levels (sellers of NO)
        nf = _walk_book(ob.get("yes", []), s["sim_no"], KALSHI_BANKROLL, KALSHI_KELLY)
        if nf and nf["filled"] >= MIN_DEPTH:
            eff = nf["effective_price"]
            rows.append({
                **base_row, "side": "no", "pricing": "taker",
                "american_odds": implied_to_american(min(max(eff, 1e-4), 1 - 1e-4)),
                "sim_prob": s["sim_no"], "implied_prob": eff,
                "edge": round((s["sim_no"] - eff) * 100, 1),
                "my_fair": implied_to_american(min(max(s["sim_no"], 1e-4), 1 - 1e-4)),
                "target": nf["target"], "filled": nf["filled"],
            })

        # YES mid (maker, no fees)
        yes_mid_am = implied_to_american(min(max(s["mid"], 1e-4), 1 - 1e-4))
        if yes_mid_am is not None:
            rows.append({
                **base_row, "side": "yes", "pricing": "mid",
                "american_odds": yes_mid_am,
                "sim_prob": s["sim_yes"], "implied_prob": s["mid"],
                "edge": round(s["yes_mid_edge"], 1),
                "my_fair": implied_to_american(min(max(s["sim_yes"], 1e-4), 1 - 1e-4)),
            })

        # NO mid (maker, no fees)
        no_mid_price = 1 - s["mid"]
        no_mid_am = implied_to_american(min(max(no_mid_price, 1e-4), 1 - 1e-4))
        if no_mid_am is not None:
            rows.append({
                **base_row, "side": "no", "pricing": "mid",
                "american_odds": no_mid_am,
                "sim_prob": s["sim_no"], "implied_prob": no_mid_price,
                "edge": round(s["no_mid_edge"], 1),
                "my_fair": implied_to_american(min(max(s["sim_no"], 1e-4), 1 - 1e-4)),
            })

    print(f"  Stage 2 orderbook: fetched {len(stage1) - ob_fail}/{len(stage1)} orderbooks")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge", ascending=False)
        taker_pos = ((df["pricing"] == "taker") & (df["edge"] > 0)).sum()
        mid_pos = ((df["pricing"] == "mid") & (df["edge"] > 0)).sum()
        print(f"  Kalshi outrights: {len(df)} lines priced, "
              f"{taker_pos} taker edge (walked >= {MIN_DEPTH}), {mid_pos} maker edge")

    if kalshi_mismatches:
        print(f"  Warning: {len(kalshi_mismatches)} Kalshi outright players not found in sim")
        df.attrs["name_mismatches"] = kalshi_mismatches

    return df


def price_novig_outrights(finish_probs, pred_lookup, sample_lookup, tourney_name=None):
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

    type_to_col = {
        "top_5": "top_5_nodh",
        "top_10": "top_10_nodh",
        "top_20": "top_20_nodh",
        "winner": "simulated_win_prob",
    }
    for mtype, col in list(type_to_col.items()):
        if col not in finish_probs.columns:
            fallback = {"top_5_nodh": "top_5", "top_10_nodh": "top_10",
                        "top_20_nodh": "top_20"}
            type_to_col[mtype] = fallback.get(col, col)

    try:
        from sim_inputs import name_replacements as _nr
    except ImportError:
        _nr = {}

    def norm(s):
        x = s.strip().lower()
        if "," not in x:
            parts = x.rsplit(" ", 1)
            if len(parts) == 2:
                x = f"{parts[1]}, {parts[0]}"
        return _nr.get(x, x)

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

        # Fuzzy-match to tourney name from sim_inputs if provided
        best = None
        if tourney_name:
            from difflib import SequenceMatcher
            scored = [(e, SequenceMatcher(None, tourney_name.lower(),
                       e.get("description", "").lower()).ratio() * 100) for e in events]
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] >= 50:
                best = scored[0][0]
                print(f"  NoVig tournament: {best['description']} (fuzzy={scored[0][1]:.0f})")
        if best is None:
            best = max(events, key=lambda e: e.get("child_events_aggregate", {})
                       .get("aggregate", {}).get("count", 0))
            print(f"  NoVig tournament: {best['description']}")

        tourn_id = best["id"]
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

                # YES side
                yes_edge = (sim_prob - yes_price) * 100
                yes_american = implied_to_american(yes_price)
                if yes_american is not None:
                    base_row = {
                        "player_name": player,
                        "market_type": our_type,
                        "bookmaker": "novig",
                        "my_pred": pred_lookup.get(player),
                        "sample": sample_lookup.get(player),
                        "side": "yes",
                        "pricing": "taker",
                        "american_odds": yes_american,
                        "sim_prob": sim_prob,
                        "implied_prob": yes_price,
                        "edge": round(yes_edge, 1),
                        "my_fair": implied_to_american(min(max(sim_prob, 1e-4), 1 - 1e-4)),
                        "volume": volume,
                    }
                    rows.append(base_row)

                # NO side
                if no_price and no_price > 0:
                    sim_no = 1.0 - sim_prob
                    no_edge = (sim_no - no_price) * 100
                    no_american = implied_to_american(no_price)
                    if no_american is not None:
                        rows.append({
                            "player_name": player,
                            "market_type": our_type,
                            "bookmaker": "novig",
                            "my_pred": pred_lookup.get(player),
                            "sample": sample_lookup.get(player),
                            "side": "no",
                            "pricing": "taker",
                            "american_odds": no_american,
                            "sim_prob": sim_no,
                            "implied_prob": no_price,
                            "edge": round(no_edge, 1),
                            "my_fair": implied_to_american(min(max(sim_no, 1e-4), 1 - 1e-4)),
                            "volume": volume,
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


def build_finish_outputs(priced_markets, pred_lookup, sample_lookup):
    """
    Build combined and sharp outputs for finish positions.

    Returns (combined_df, sharp_df).
    """
    # Combine all markets
    all_dfs = [df for df in priced_markets.values() if not df.empty]
    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Sort by edge
    combined = combined.sort_values('edge', ascending=False)

    # Sharp: only sharp books, deduplicate by player+market
    sharp = combined[combined['bookmaker'].str.lower().isin([b.lower() for b in SHARP_BOOKS])].copy()
    if not sharp.empty:
        sharp['key'] = sharp['player_name'] + '_' + sharp['market_type']
        sharp = sharp.sort_values('edge', ascending=False).drop_duplicates('key', keep='first')
        sharp = sharp.drop(columns='key')

    return combined, sharp


# ══════════════════════════════════════════════════════════════════════════════
# Outright Win Edge CSVs
# ══════════════════════════════════════════════════════════════════════════════

def build_win_edges_csv(finish_probs, pred_lookup, sample_lookup, out_dir):
    """
    Fetch live outright win odds, merge with model win probabilities,
    and save a CSV of the largest POSITIVE edges (players we think will win
    more often than the market implies).

    Saves: {out_dir}/outright_win_edges.csv
    """
    data = fetch_outright_odds('win')
    if not data:
        print("    No win market data available for positive edge CSV")
        return pd.DataFrame(), None, pd.DataFrame()

    df = extract_market_rows(data, odds_key='odds')
    if df.empty:
        print("    No win market rows extracted")
        return pd.DataFrame(), None, pd.DataFrame()

    df['player_name'] = df['player_name'].str.lower().str.strip().replace(name_replacements)

    if 'simulated_win_prob' not in finish_probs.columns:
        print("    No simulated_win_prob column in finish_probs")
        return pd.DataFrame(), None, pd.DataFrame()

    df = df.merge(
        finish_probs[['player_name', 'simulated_win_prob']],
        on='player_name', how='inner'
    )
    if df.empty:
        return pd.DataFrame(), None, pd.DataFrame()

    df['implied_prob'] = 1.0 / df['decimal_odds']
    df['american_odds'] = df['decimal_odds'].apply(decimal_to_american)
    p = df['simulated_win_prob'].astype(float)
    b = df['decimal_odds'] - 1.0
    q = 1.0 - p
    df['edge'] = (p - df['implied_prob']) * 100.0  # probability edge in pp
    f_star = (b * p - q) / b
    df['kelly'] = (BANKROLL * KELLY_FRACTION * f_star.clip(lower=0)).astype(float)
    df['my_fair'] = p.clip(1e-4, 1 - 1e-4).apply(implied_to_american)
    df['my_pred'] = df['player_name'].map(pred_lookup)
    df['sample'] = df['player_name'].map(sample_lookup)

    # Keep only positive edges, sort by Kelly stake
    pos = df[df['edge'] > 0].copy()
    pos = pos.sort_values('kelly', ascending=False)

    # Best price per player (highest Kelly across books)
    pos = pos.drop_duplicates('player_name', keep='first')

    cols = ['player_name', 'bookmaker', 'american_odds', 'implied_prob',
            'simulated_win_prob', 'my_fair', 'edge', 'kelly', 'my_pred', 'sample']
    pos = pos[[c for c in cols if c in pos.columns]]

    path = os.path.join(out_dir, "outright_win_edges.csv")
    pos.to_csv(path, index=False)
    print(f"    Saved {path} ({len(pos)} positive win edges)")
    return pos, path, df


def build_betonline_negative_edges_csv(finish_probs, pred_lookup, sample_lookup, out_dir):
    """
    Fetch live outright win odds, isolate BetOnline, devig using the
    multiplicative method (divide each implied prob by the total overround),
    then compare to model win probabilities.

    Players where the model gives a LOWER win probability than BetOnline's
    devigged implied probability have negative edges — the market overrates
    them, i.e., players we think WON'T win.

    Saves: {out_dir}/betonline_devig_fades.csv
    """
    data = fetch_outright_odds('win')
    if not data:
        print("    No win market data available for BetOnline devig CSV")
        return pd.DataFrame()

    # Extract ALL books first so we can get BetOnline rows
    entries = data.get('odds', [])
    if not isinstance(entries, list):
        print("    Unexpected win market format")
        return pd.DataFrame()

    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        player = entry.get('player_name', '')
        if not player:
            continue
        player = player.lower().strip()
        odds = entry.get('betonline')
        if odds is not None:
            rows.append({
                'player_name': player,
                'decimal_odds': float(odds),
            })

    if not rows:
        print("    No BetOnline win odds found")
        return pd.DataFrame()

    bol = pd.DataFrame(rows)
    bol['player_name'] = bol['player_name'].str.lower().str.strip().replace(name_replacements)
    bol['implied_prob'] = 1.0 / bol['decimal_odds']

    # Devig: multiplicative method
    total_overround = bol['implied_prob'].sum()
    bol['devigged_prob'] = bol['implied_prob'] / total_overround
    bol['devigged_decimal'] = 1.0 / bol['devigged_prob']
    bol['devigged_american'] = bol['devigged_decimal'].apply(decimal_to_american)
    bol['raw_american'] = bol['decimal_odds'].apply(decimal_to_american)

    print(f"    BetOnline overround: {total_overround:.4f} "
          f"({(total_overround - 1) * 100:.1f}% vig on {len(bol)} players)")

    if 'simulated_win_prob' not in finish_probs.columns:
        print("    No simulated_win_prob column in finish_probs")
        return pd.DataFrame()

    bol = bol.merge(
        finish_probs[['player_name', 'simulated_win_prob']],
        on='player_name', how='inner'
    )
    if bol.empty:
        print("    No player overlap between BetOnline odds and model")
        return pd.DataFrame()

    # Edge vs devigged line: negative = model thinks player is WORSE than market
    p = bol['simulated_win_prob'].astype(float)
    bol['edge_vs_devig'] = (p - bol['devigged_prob']) * 100.0  # probability edge in pp

    bol['model_fair_american'] = p.clip(1e-4, 1 - 1e-4).apply(implied_to_american)
    bol['my_pred'] = bol['player_name'].map(pred_lookup)
    bol['sample'] = bol['player_name'].map(sample_lookup)

    # Sort by most negative edge (biggest fades first)
    bol = bol.sort_values('edge_vs_devig', ascending=True)

    cols = ['player_name', 'raw_american', 'implied_prob', 'devigged_prob',
            'devigged_american', 'simulated_win_prob', 'model_fair_american',
            'edge_vs_devig', 'my_pred', 'sample']
    bol = bol[[c for c in cols if c in bol.columns]]

    path = os.path.join(out_dir, "betonline_devig_fades.csv")
    bol.to_csv(path, index=False)
    print(f"    Saved {path} ({len(bol)} players, "
          f"most negative edge: {bol['edge_vs_devig'].iloc[0]:.1f}%)")
    return bol


# ══════════════════════════════════════════════════════════════════════════════
# Odds Conversion Helpers
# ══════════════════════════════════════════════════════════════════════════════

def american_to_implied(odds):
    """American odds → implied probability (0–1)."""
    if pd.isna(odds) or odds == 0:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def implied_to_american(prob):
    """Implied probability (0–1) → American odds (int)."""
    if prob is None or pd.isna(prob) or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


def _send_telegram(text):
    """Send a Telegram message. Non-blocking — logs warning on failure."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        print("  Warning: Telegram alert failed")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Score Simulation (shared by matchups + score card)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_round_scores(model_preds, sim_round, expected_avg, num_sims=NUM_SIMULATIONS):
    """
    Simulate integer round scores for every player.

    Formula per player:
        actual_score = round( expected_avg − Normal(scores_rN, STD_DEV) )

    For multi-course events, each player's expected_avg comes from the
    'course_score_adj' column if present; otherwise uses the global expected_avg.

    Returns
    -------
    sim_dict : dict
        player_name → np.ndarray of simulated integer scores (shape: num_sims)
    """
    scores_col = f"scores_r{sim_round}"
    if scores_col not in model_preds.columns:
        raise ValueError(f"Column '{scores_col}' not found in predictions file. "
                         f"Available: {list(model_preds.columns)}")

    has_course_adj = "course_score_adj" in model_preds.columns

    sim_dict = {}
    for _, row in model_preds.iterrows():
        player = row["player_name"]
        skill = row[scores_col]

        # Skip players with missing predictions
        if pd.isna(skill):
            continue

        # Per-player expected avg (multi-course) or global
        if has_course_adj and pd.notna(row.get("course_score_adj")):
            player_avg = row["course_score_adj"]
        else:
            player_avg = expected_avg

        raw = np.random.normal(loc=skill, scale=STD_DEV, size=num_sims)
        scores = np.round(player_avg - raw).astype(int)
        sim_dict[player] = np.clip(scores, int(round(player_avg)) - 12, int(round(player_avg)) + 12)

    print(f"  Simulated {len(sim_dict)} players × {num_sims:,} iterations")
    return sim_dict


def simulate_round_scores_catfirst(model_preds, sim_round, expected_avg,
                                   wx_lookup, num_sims=NUM_SIMULATIONS):
    """
    Category-first round score simulation.

    Draws each SG category from a course-adjusted multivariate normal with
    Cornish-Fisher skewness, then sums to total. Mirrors new_sim.py's approach
    but re-centers to the per-player skill prediction (post skill-update, pre-weather).

    Returns same format as simulate_round_scores(): {player_name: np.array of int scores}
    """
    scores_col = f"scores_r{sim_round}"
    if scores_col not in model_preds.columns:
        raise ValueError(f"Column '{scores_col}' not found in predictions file.")

    # Load v2 distributions via shared helper
    dists_result = _load_catfirst_dists(list(model_preds["player_name"]))
    if dists_result is None:
        print(f"  [catfirst] Falling back to standard sim")
        return simulate_round_scores(model_preds, sim_round, expected_avg, num_sims)
    player_cf_params, effective_skew, L_corr = dists_result

    has_course_adj = "course_score_adj" in model_preds.columns

    sim_dict = {}
    for idx, (_, row) in enumerate(model_preds.iterrows()):
        player = row["player_name"]
        scores_rn = row[scores_col]
        if pd.isna(scores_rn):
            continue

        # Per-player expected avg (multi-course) or global
        if has_course_adj and pd.notna(row.get("course_score_adj")):
            player_avg = row["course_score_adj"]
        else:
            player_avg = expected_avg

        mu, std_c = player_cf_params[idx]

        # Decompose: skill = scores_rN - weather
        wx_delta = wx_lookup.get(player, 0.0)
        skill = scores_rn - wx_delta

        # Re-center + weather split (inline, not via _catfirst_draw)
        shift = (skill - mu.sum()) / 4.0
        cat_mu = mu + shift + wx_delta * WEATHER_CAT_SPLIT

        # Draw: correlated standard normals → skew → scale → clip → sum
        Z = RNG_CF.standard_normal(size=(num_sims, 4))
        corr_z = Z @ L_corr.T
        for j in range(4):
            corr_z[:, j] = _apply_skew(corr_z[:, j], effective_skew[idx, j])
        draws = np.clip(cat_mu + corr_z * std_c, CLIP_CAT[0], CLIP_CAT[1])
        sg_total = draws.sum(axis=1)

        # Convert SG to integer scores
        scores = np.rint(player_avg - sg_total).astype(int)
        sim_dict[player] = np.clip(scores, int(round(player_avg)) - 12,
                                   int(round(player_avg)) + 12)

    print(f"  [catfirst] Simulated {len(sim_dict)} players × {num_sims:,} iterations")
    print(f"  [catfirst] Course mults: OTT={_course_mults_cf[0]:.3f}, APP={_course_mults_cf[1]:.3f}, "
          f"ARG={_course_mults_cf[2]:.3f}, PUTT={_course_mults_cf[3]:.3f}")
    return sim_dict


def print_catfirst_comparison(sim_old, sim_cf, model_preds, pred_col, expected_avg):
    """Print a console comparison table between standard and category-first sims."""
    common = sorted(set(sim_old.keys()) & set(sim_cf.keys()))
    if not common:
        print("  [catfirst] No common players to compare.")
        return

    pred_lookup = dict(zip(model_preds["player_name"], model_preds[pred_col]))

    rows = []
    for p in common:
        old_scores = sim_old[p].astype(float)
        cf_scores = sim_cf[p].astype(float)
        pred = pred_lookup.get(p, 0.0)
        rows.append({
            "player": p,
            "pred": pred,
            "std_mean": old_scores.mean(),
            "std_sd": old_scores.std(),
            "cf_mean": cf_scores.mean(),
            "cf_sd": cf_scores.std(),
            "delta_mean": cf_scores.mean() - old_scores.mean(),
        })

    rows.sort(key=lambda r: r["pred"], reverse=True)

    print(f"\n{'-'*76}")
    print(f"{'CATEGORY-FIRST vs STANDARD SIM COMPARISON':^76}")
    print(f"{'-'*76}")
    print(f"  {'Player':<22s} {'Pred':>6s}  {'Std-Mean':>8s} {'Std-SD':>7s}  "
          f"{'CF-Mean':>8s} {'CF-SD':>7s}  {'d Mean':>7s}")
    print(f"{'-'*76}")

    for r in rows[:30]:
        print(f"  {r['player']:<22s} {r['pred']:>6.2f}  {r['std_mean']:>8.2f} {r['std_sd']:>7.2f}  "
              f"{r['cf_mean']:>8.2f} {r['cf_sd']:>7.2f}  {r['delta_mean']:>+7.2f}")

    if len(rows) > 30:
        print(f"  ... ({len(rows) - 30} more players)")

    abs_deltas = [abs(r["delta_mean"]) for r in rows]
    std_sds = [r["std_sd"] for r in rows]
    cf_sds = [r["cf_sd"] for r in rows]

    print(f"{'-'*76}")
    print(f"  SUMMARY:  Mean abs d = {np.mean(abs_deltas):.2f} strokes,  "
          f"Max d = {max(abs_deltas):.2f}")
    print(f"            Std-SD avg = {np.mean(std_sds):.2f},  CF-SD avg = {np.mean(cf_sds):.2f}")

    # Matchup win probability comparison (sample top players by pred)
    players_by_pred = sorted(rows, key=lambda r: r["pred"], reverse=True)
    top_players = [r["player"] for r in players_by_pred[:20]]
    matchup_rows = []
    for i in range(0, len(top_players) - 1, 2):
        p1, p2 = top_players[i], top_players[i + 1]
        p1_wins_old = (sim_old[p1] < sim_old[p2]).sum()
        p1_pct_old = p1_wins_old / len(sim_old[p1]) * 100
        p1_wins_cf = (sim_cf[p1] < sim_cf[p2]).sum()
        p1_pct_cf = p1_wins_cf / len(sim_cf[p1]) * 100
        matchup_rows.append({
            "matchup": f"{p1} vs {p2}",
            "std_p1": p1_pct_old,
            "cf_p1": p1_pct_cf,
            "delta": p1_pct_cf - p1_pct_old,
        })

    if matchup_rows:
        print(f"\n{'-'*70}")
        print(f"{'MATCHUP WIN PROBABILITY COMPARISON (sample)':^70}")
        print(f"{'-'*70}")
        print(f"  {'Matchup':<34s} {'Std P1%':>8s}  {'CF P1%':>8s}  {'d':>6s}")
        print(f"{'-'*70}")
        for r in matchup_rows:
            print(f"  {r['matchup']:<34s} {r['std_p1']:>7.1f}%  {r['cf_p1']:>7.1f}%  {r['delta']:>+5.1f}%")
        abs_d = [abs(r["delta"]) for r in matchup_rows]
        print(f"{'-'*70}")
        print(f"  Mean abs d = {np.mean(abs_d):.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Matchup Pricing
# ══════════════════════════════════════════════════════════════════════════════

def fetch_matchup_odds():
    """Fetch round matchup odds from DataGolf API."""
    params = {
        "tour": "pga",
        "market": "round_matchups",
        "odds_format": "american",
        "file_format": "json",
        "key": API_KEY,
    }
    resp = requests.get(MATCHUPS_URL, params=params, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Matchup API failed ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    rows = []
    for match in data.get("match_list", []):
        p1 = match["p1_player_name"].lower()
        p2 = match["p2_player_name"].lower()
        ties = match.get("ties", "unknown")

        for book, odds in match.get("odds", {}).items():
            if book == "datagolf":
                continue
            rows.append({
                "Player 1": p1,
                "Player 2": p2,
                "Bookmaker": book,
                "P1 Odds": odds.get("p1"),
                "P2 Odds": odds.get("p2"),
                "DG_p1": match["odds"].get("datagolf", {}).get("p1"),
                "DG_p2": match["odds"].get("datagolf", {}).get("p2"),
                "Ties": ties,
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Player 1", "Player 2", "Bookmaker"], keep="first")
    df["P1 Odds"] = pd.to_numeric(df["P1 Odds"], errors="coerce")
    df["P2 Odds"] = pd.to_numeric(df["P2 Odds"], errors="coerce")
    print(f"  Fetched {len(df)} matchup lines across {df['Bookmaker'].nunique()} books")
    return df


def price_matchups(matchup_df, sim_dict):
    """
    Attach fair win probabilities to each matchup row.

    Two probability modes per side:
        my_odds_pN       — ties are a push (excluded from total)
        my_odds_pN_tl    — ties count as losses

    Tracks name mismatches in matchup_df._name_mismatches (set of (player, bookmaker)).
    """
    cols = {"fair_p1": [], "fair_p2": [], "tl_p1": [], "tl_p2": []}
    name_mismatches = defaultdict(set)  # player -> set of bookmakers

    for _, row in matchup_df.iterrows():
        p1, p2 = row["Player 1"], row["Player 2"]
        book = row.get("Bookmaker", "unknown")

        if p1 not in sim_dict or p2 not in sim_dict:
            if p1 not in sim_dict:
                name_mismatches[p1].add(book)
            if p2 not in sim_dict:
                name_mismatches[p2].add(book)
            for k in cols:
                cols[k].append(None)
            continue

        s1, s2 = sim_dict[p1], sim_dict[p2]
        w1 = (s1 < s2).sum()
        w2 = (s1 > s2).sum()
        ties = (s1 == s2).sum()
        total = len(s1)
        non_tie = w1 + w2

        cols["fair_p1"].append(w1 / non_tie if non_tie else 0.5)
        cols["fair_p2"].append(w2 / non_tie if non_tie else 0.5)
        cols["tl_p1"].append(w1 / total)
        cols["tl_p2"].append(w2 / total)

    matchup_df["my_odds_p1"] = cols["fair_p1"]
    matchup_df["my_odds_p2"] = cols["fair_p2"]
    matchup_df["my_odds_p1_tl"] = cols["tl_p1"]
    matchup_df["my_odds_p2_tl"] = cols["tl_p2"]

    if name_mismatches:
        print(f"  Warning: {len(name_mismatches)} scraped players not found in sim")
        matchup_df.attrs["name_mismatches"] = dict(name_mismatches)

    return matchup_df


def calculate_edges(df):
    """
    Calculate edges, fair odds, half-shot spreads for all matchup rows.
    Operates on the combined DataFrame (all bookmakers).
    """
    df = df.dropna(subset=["my_odds_p1", "my_odds_p2"]).copy()

    # Decimal odds from American
    df["p1_dec"] = np.where(
        df["P1 Odds"] > 0,
        df["P1 Odds"] / 100 + 1,
        100 / df["P1 Odds"].abs() + 1,
    )
    df["p2_dec"] = np.where(
        df["P2 Odds"] > 0,
        df["P2 Odds"] / 100 + 1,
        100 / df["P2 Odds"].abs() + 1,
    )

    # Which probability to use for edge: ties-loss when "separate bet offered"
    use_tl = df["Ties"] == "separate bet offered"

    prob_p1 = np.where(use_tl, df["my_odds_p1_tl"], df["my_odds_p1"])
    prob_p2 = np.where(use_tl, df["my_odds_p2_tl"], df["my_odds_p2"])

    # Edge = (prob × (decimal − 1) − (1 − prob)) × 100
    df["edge_p1"] = (prob_p1 * (df["p1_dec"] - 1) - (1 - prob_p1)) * 100
    df["edge_p2"] = (prob_p2 * (df["p2_dec"] - 1) - (1 - prob_p2)) * 100

    # Fair American odds (ties push)
    df["Fair_p1"] = df["my_odds_p1"].apply(
        lambda p: implied_to_american(p) if pd.notna(p) else None
    )
    df["Fair_p2"] = df["my_odds_p2"].apply(
        lambda p: implied_to_american(p) if pd.notna(p) else None
    )

    # Book implied probabilities (%)
    df["p1_implied"] = df["P1 Odds"].apply(
        lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
    )
    df["p2_implied"] = df["P2 Odds"].apply(
        lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
    )

    # Half-shot values: the value of a half-shot of spread
    df["half_shot_p1"] = (df["my_odds_p1"] - df["my_odds_p1_tl"]) * 400
    df["half_shot_p2"] = (df["my_odds_p2"] - df["my_odds_p2_tl"]) * 400

    # Push-wins: P(win or tie)
    df["p1_pushwins"] = (1 - df["my_odds_p2_tl"]) * 100
    df["p2_pushwins"] = (1 - df["my_odds_p1_tl"]) * 100

    # No-push: P(win, no tie) = ties-loss prob
    df["p1_nopush"] = df["my_odds_p1_tl"] * 100
    df["p2_nopush"] = df["my_odds_p2_tl"] * 100

    # ±0.5 spread edges for betonline / betcris
    for book, adj in HALF_SHOT_ADJ.items():
        mask = df["Bookmaker"].str.lower() == book
        if not mask.any():
            continue
        for side, odds_col in [("p1", "P1 Odds"), ("p2", "P2 Odds")]:
            pw_imp = (df.loc[mask, odds_col] - adj).apply(
                lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
            )
            np_imp = (df.loc[mask, odds_col] + adj).apply(
                lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
            )
            df.loc[mask, f"{side}_pushwins_imp"] = pw_imp
            df.loc[mask, f"{side}_nopush_imp"] = np_imp
            df.loc[mask, f"{side}_+0.5"] = df.loc[mask, f"{side}_pushwins"] - pw_imp
            df.loc[mask, f"{side}_-0.5"] = df.loc[mask, f"{side}_nopush"] - np_imp

    return df


def build_matchup_outputs(df, sim_round, pred_lookup, sample_lookup, wx_lookup=None):
    """
    Filter, annotate, and split matchup DataFrame into combined + sharp outputs.

    Returns (combined_df, sharp_df).
    """
    # Merge predictions and sample sizes
    df["p1_pred"] = df["Player 1"].map(pred_lookup)
    df["p2_pred"] = df["Player 2"].map(pred_lookup)
    df["Sample_P1"] = df["Player 1"].map(sample_lookup)
    df["Sample_P2"] = df["Player 2"].map(sample_lookup)
    df["Round"] = f"r{sim_round}"

    # Derived columns
    df["edge_on"] = df[["edge_p1", "edge_p2"]].max(axis=1).round(1)
    df["bet_on"] = df.apply(
        lambda r: r["Player 1"] if r["edge_p1"] > r["edge_p2"] else r["Player 2"],
        axis=1,
    )
    df["pred_on"] = df.apply(
        lambda r: r["p1_pred"] if r["edge_p1"] > r["edge_p2"] else r["p2_pred"],
        axis=1,
    )
    df["pred_against"] = df.apply(
        lambda r: r["p2_pred"] if r["edge_p1"] > r["edge_p2"] else r["p1_pred"],
        axis=1,
    )
    df["sample_on"] = df.apply(
        lambda r: r["Sample_P1"] if r["edge_p1"] > r["edge_p2"] else r["Sample_P2"],
        axis=1,
    )

    # --- Weather SG differential ---
    if wx_lookup:
        df["wx_on"] = df["bet_on"].map(wx_lookup).fillna(0)
        df["wx_against"] = df.apply(
            lambda r: wx_lookup.get(
                r["Player 2"] if r["bet_on"] == r["Player 1"] else r["Player 1"], 0
            ),
            axis=1,
        )
        df["wx_diff"] = df["wx_on"] - df["wx_against"]

    # --- Combined: basic filters ---
    combined = df[df["edge_on"] > 3].copy()
    combined = combined[combined["sample_on"].fillna(0) >= 20]
    combined = combined[
        ((combined["pred_on"] > 0) & (combined["edge_on"] > 7))
        | (combined["pred_on"] > 1)
    ]
    combined = combined[
        ~((combined["edge_on"] < 5) & (combined["pred_on"] < 1))
    ]

    # --- Sharp: pinnacle / betonline / betcris, deduplicate by highest edge ---
    sharp = combined[combined["Bookmaker"].str.lower().isin(SHARP_BOOKS)].copy()
    sharp["matchup_key"] = [
        "-".join(sorted([p1, p2]))
        for p1, p2 in zip(sharp["Player 1"], sharp["Player 2"])
    ]
    sharp = sharp.sort_values("edge_on", ascending=False).drop_duplicates(
        "matchup_key", keep="first"
    )
    sharp = sharp.drop(columns="matchup_key")

    # --- Clean up display columns ---
    for out in [combined, sharp]:
        out["p1_pred"] = out["p1_pred"].round(2)
        out["p2_pred"] = out["p2_pred"].round(2)
        out["edge_p1"] = out["edge_p1"].round(1)
        out["edge_p2"] = out["edge_p2"].round(1)

    # Column ordering for output
    display_cols = [
        "Player 1", "Player 2", "Round", "Bookmaker", "Ties",
        "P1 Odds", "P2 Odds", "Fair_p1", "Fair_p2",
        "edge_p1", "edge_p2", "edge_on", "bet_on",
        "p1_pred", "p2_pred", "pred_on", "pred_against",
        "Sample_P1", "Sample_P2", "sample_on",
        "half_shot_p1", "half_shot_p2",
    ]
    # Add weather decomposition columns if available
    for col in ["wx_diff"]:
        if col in combined.columns:
            display_cols.append(col)
    # Add spread columns if they exist
    for col in ["p1_+0.5", "p2_+0.5", "p1_-0.5", "p2_-0.5"]:
        if col in combined.columns:
            display_cols.append(col)

    combined = combined[[c for c in display_cols if c in combined.columns]]
    sharp = sharp[[c for c in display_cols if c in sharp.columns]]

    print(f"  Combined matchups: {len(combined)} rows")
    print(f"  Sharp filtered:    {len(sharp)} rows")

    return combined, sharp


def build_betonline_all_matchups_csv(matchup_df, sim_round, out_dir):
    """
    Extract ALL BetOnline matchup rows (no edge/sample/pred filters) and save as CSV.

    Includes book odds, fair odds, edges, pred, and sample for every matchup
    BetOnline prices — even negative edges and low-sample players.
    Sorted by edge_on descending (highest edge first).
    """
    bol = matchup_df[matchup_df["Bookmaker"].str.lower() == "betonline"].copy()
    if bol.empty:
        print("  No BetOnline matchups found")
        return None

    # Round numeric columns for readability
    for col in ["edge_p1", "edge_p2", "edge_on", "p1_pred", "p2_pred",
                "pred_on", "half_shot_p1", "half_shot_p2"]:
        if col in bol.columns:
            bol[col] = bol[col].round(2)

    display_cols = [
        "Player 1", "Player 2", "Ties",
        "P1 Odds", "P2 Odds", "Fair_p1", "Fair_p2",
        "edge_p1", "edge_p2", "edge_on", "bet_on",
        "p1_pred", "p2_pred", "pred_on",
        "Sample_P1", "Sample_P2", "sample_on",
        "half_shot_p1", "half_shot_p2",
        "p1_+0.5", "p2_+0.5", "p1_-0.5", "p2_-0.5",
    ]
    bol = bol[[c for c in display_cols if c in bol.columns]]
    bol = bol.sort_values("edge_on", ascending=False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"betonline_all_matchups_r{sim_round}.csv")
    bol.to_csv(path, index=False)
    print(f"  BetOnline all matchups: {len(bol)} rows -> {path}")
    return path


def build_all_books_fair_csv(matchup_df, sim_round, out_dir):
    """
    Build a CSV with fair price vs book odds for every matchup line across
    betonline, pinnacle, and betcris. No filters — all edges, all samples.

    Sorted by Bookmaker then edge_on descending.
    """
    target_books = {"betonline", "pinnacle", "betcris"}
    mask = matchup_df["Bookmaker"].str.lower().isin(target_books)
    df = matchup_df[mask].copy()
    if df.empty:
        print("  No betonline/pinnacle/betcris matchups found")
        return None

    for col in ["edge_p1", "edge_p2", "edge_on", "p1_pred", "p2_pred",
                "pred_on", "half_shot_p1", "half_shot_p2"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    display_cols = [
        "Bookmaker", "Player 1", "Player 2", "Ties",
        "P1 Odds", "P2 Odds", "Fair_p1", "Fair_p2",
        "edge_p1", "edge_p2", "edge_on", "bet_on",
        "p1_pred", "p2_pred", "pred_on",
        "Sample_P1", "Sample_P2", "sample_on",
        "half_shot_p1", "half_shot_p2",
        "p1_+0.5", "p2_+0.5", "p1_-0.5", "p2_-0.5",
    ]
    df = df[[c for c in display_cols if c in df.columns]]
    df = df.sort_values(["Bookmaker", "edge_on"], ascending=[True, False])

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"all_books_fair_matchups_r{sim_round}.csv")
    df.to_csv(path, index=False)
    print(f"  All-books fair matchups: {len(df)} rows ({df['Bookmaker'].nunique()} books) -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Score Line Fair Card
# ══════════════════════════════════════════════════════════════════════════════

def build_score_card(sim_dict, expected_avg, pred_lookup):
    """
    Generate fair UNDER prices at half-stroke intervals around expected_avg.

    For each player and each line (e.g. 69.5, 70.5, ...):
        P(under) = P(score <= floor(line))  [no push at .5 lines]
        Fair UNDER = implied_to_american(P(under))

    Players with pred < MIN_PRED_FOR_CARD are excluded.

    Returns DataFrame with columns: Player, Pred, line_1, line_2, ...
    """
    # Generate standard .5 lines sportsbooks use (e.g. 68.5, 69.5, 70.5...)
    low = int(expected_avg - SCORE_CARD_RANGE)      # e.g. 69
    high = int(expected_avg + SCORE_CARD_RANGE) + 1  # e.g. 76
    lines = [x + 0.5 for x in range(low, high)]     # [69.5, 70.5, ..., 75.5]

    rows = []
    for player, scores in sim_dict.items():
        pred = pred_lookup.get(player)
        if pred is None or pred < MIN_PRED_FOR_CARD:
            continue

        row = {"Player": player, "Pred": round(pred, 2)}
        for line in lines:
            threshold = int(line)  # e.g. 70.5 → count scores ≤ 70
            under_pct = (scores <= threshold).mean()
            fair_under = implied_to_american(under_pct)
            row[str(line)] = fair_under

        rows.append(row)

    card = pd.DataFrame(rows)
    card = card.sort_values("Pred", ascending=False)
    print(f"  Score card: {len(card)} players × {len(lines)} lines ({lines[0]}–{lines[-1]})")
    return card


# ══════════════════════════════════════════════════════════════════════════════
# Step 3b: Price Score Lines vs FanDuel Market Odds
# ══════════════════════════════════════════════════════════════════════════════

def load_score_lines():
    """Load scraped round score O/U lines (FanDuel etc.).

    Follows the same GitHub-first pattern as odds_loader._fetch_scraped_json():
    1. Try GitHub API (mslade50/golf_scraping) for fresh data
    2. Fall back to local paths

    Staleness rule: data must be from today (UTC). Old tournament lines
    should never be priced against the current field.
    """
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    import requests

    filename = "round_scores_latest.json"
    today_utc = datetime.now(timezone.utc).date()

    # 1. Try GitHub API first
    gh_token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    api_url = f"https://api.github.com/repos/mslade50/golf_scraping/contents/data/{filename}?ref=master"
    try:
        headers = {"Accept": "application/vnd.github.raw+json"}
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"
        resp = requests.get(api_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        last_updated = data.get("last_updated", "")
        if last_updated:
            try:
                ts = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
                if ts.date() < today_utc:
                    print(f"  GitHub score lines from {ts.date()} (not today), skipping")
                else:
                    lines = data.get("lines", [])
                    if lines:
                        print(f"  Loaded {len(lines)} score lines from GitHub (updated {last_updated})")
                        return lines
            except ValueError:
                pass  # can't verify date, skip
        else:
            pass  # no timestamp, can't verify, skip
    except Exception as e:
        print(f"  GitHub fetch failed for {filename}: {e}")

    # 2. Fall back to local paths
    search_paths = [
        Path(__file__).parent / "permanent_data" / "scraped_odds",
        Path.home() / "Documents" / "golf_scraping" / "data",
    ]

    for base in search_paths:
        path = base / filename
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if mtime.date() < today_utc:
                print(f"  Local score lines from {mtime.date()} (not today): {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            lines = data.get("lines", [])
            if lines:
                print(f"  Loaded {len(lines)} score lines from {path.name}")
                return lines

    print("  No score lines from today found (GitHub + local)")
    return []


def price_score_lines(score_card, market_lines):
    """
    Join sim fair card with scraped market O/U lines. Compute edges.

    Args:
        score_card: DataFrame from build_score_card() with columns:
                    Player, Pred, "69.5", "70.5", ... (fair under odds)
        market_lines: list of dicts from round_scores_latest.json, each:
                    {player_name, round, line, odds: {fanduel: {over, under}}}

    Returns:
        DataFrame with columns:
            Player, Line, Book, Mkt_Under, Mkt_Over, Fair_Under, Fair_Over,
            Edge_Under, Edge_Over, Best_Edge, Best_Side
    """
    if score_card.empty or not market_lines:
        return pd.DataFrame()

    # Build lookup: normalized player name → row from score card
    try:
        from sim_inputs import name_replacements
    except ImportError:
        name_replacements = {}

    def norm(s):
        x = s.strip().lower()
        # Handle "Last, First" → "last, first"
        if "," not in x:
            # "First Last" → "last, first"
            parts = x.rsplit(" ", 1)
            if len(parts) == 2:
                x = f"{parts[1]}, {parts[0]}"
        return name_replacements.get(x, x)

    card_lookup = {}
    for _, row in score_card.iterrows():
        card_lookup[row["Player"]] = row

    rows = []
    for ml in market_lines:
        player_norm = norm(ml["player_name"])
        line = ml["line"]
        line_str = str(line)

        card_row = card_lookup.get(player_norm)
        if card_row is None or line_str not in card_row.index:
            continue

        fair_under = card_row[line_str]
        if fair_under is None or pd.isna(fair_under):
            continue

        # Convert fair under odds → implied prob → fair over
        fair_under_prob = american_to_implied(fair_under)
        if fair_under_prob is None:
            continue
        fair_over_prob = 1 - fair_under_prob
        fair_over = implied_to_american(fair_over_prob)

        for book, odds in ml.get("odds", {}).items():
            mkt_under = int(odds.get("under", 0))
            mkt_over = int(odds.get("over", 0))

            if mkt_under == 0 and mkt_over == 0:
                continue

            # Edge = (fair_prob × (market_decimal - 1) - (1 - fair_prob)) × 100
            mkt_under_imp = american_to_implied(mkt_under)
            mkt_over_imp = american_to_implied(mkt_over)

            if mkt_under_imp and fair_under_prob:
                mkt_under_dec = 1 / mkt_under_imp
                edge_under = (fair_under_prob * (mkt_under_dec - 1) - (1 - fair_under_prob)) * 100
            else:
                edge_under = 0

            if mkt_over_imp and fair_over_prob:
                mkt_over_dec = 1 / mkt_over_imp
                edge_over = (fair_over_prob * (mkt_over_dec - 1) - (1 - fair_over_prob)) * 100
            else:
                edge_over = 0

            best_edge = max(edge_under, edge_over)
            best_side = "Under" if edge_under >= edge_over else "Over"

            rows.append({
                "Player": player_norm,
                "Line": line,
                "Book": book,
                "Mkt_Under": mkt_under,
                "Mkt_Over": mkt_over,
                "Fair_Under": int(fair_under) if fair_under else None,
                "Fair_Over": int(fair_over) if fair_over else None,
                "Edge_Under": round(edge_under, 1),
                "Edge_Over": round(edge_over, 1),
                "Best_Edge": round(best_edge, 1),
                "Best_Side": best_side,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Best_Edge", ascending=False)
        print(f"  Priced {len(df)} score lines, {(df['Best_Edge'] > 5).sum()} with edge > 5%")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Export
# ══════════════════════════════════════════════════════════════════════════════

def export_results(combined, sharp, score_card, sim_round,
                   outrights_combined=None, outrights_sharp=None, finish_probs=None,
                   score_cards_by_course=None, score_edges=None):
    """Save all outputs to an Excel workbook + CSV backup."""
    timestamp = datetime.now().strftime("%H%M")
    out_dir = f"./{tourney}"
    os.makedirs(out_dir, exist_ok=True)

    excel_path = os.path.join(out_dir, f"round_{sim_round}_sim_{timestamp}.xlsx")

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # --- Matchups: Combined ---
        if not combined.empty:
            combined.to_excel(writer, sheet_name="matchups_all", index=False)
            _format_matchup_sheet(writer, workbook, "matchups_all", combined)

        # --- Matchups: Sharp ---
        if not sharp.empty:
            sharp.to_excel(writer, sheet_name="matchups_sharp", index=False)
            _format_matchup_sheet(writer, workbook, "matchups_sharp", sharp)

        # --- Score Card(s) ---
        def _write_score_card_sheet(card_df, sheet_name):
            """Write and format a single score card sheet."""
            card_df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            green = workbook.add_format({"bg_color": "#d4edda"})
            red = workbook.add_format({"bg_color": "#f8d7da"})
            # Find first numeric column (skip Player, Pred, Course)
            first_num = next(
                (i for i, c in enumerate(card_df.columns)
                 if c not in ("Player", "Pred", "Course")), 2
            )
            for col_idx in range(first_num, len(card_df.columns)):
                ws.conditional_format(
                    1, col_idx, len(card_df), col_idx,
                    {"type": "cell", "criteria": "<", "value": 0, "format": green},
                )
                ws.conditional_format(
                    1, col_idx, len(card_df), col_idx,
                    {"type": "cell", "criteria": ">", "value": 0, "format": red},
                )
            for i, col in enumerate(card_df.columns):
                ws.set_column(i, i, max(len(str(col)) + 2, 8))

        if score_cards_by_course:
            # Multi-course: separate tab per course
            for course_name, card_df in score_cards_by_course.items():
                if not card_df.empty:
                    sheet_name = f"card_{course_name}"[:31]
                    _write_score_card_sheet(card_df, sheet_name)
        elif not score_card.empty:
            _write_score_card_sheet(score_card, "score_card")

        # --- Outrights: Combined ---
        if outrights_combined is not None and not outrights_combined.empty:
            outrights_combined.to_excel(writer, sheet_name="outrights", index=False)
            _format_outright_sheet(writer, workbook, "outrights", outrights_combined)

        # --- Outrights: Sharp ---
        if outrights_sharp is not None and not outrights_sharp.empty:
            outrights_sharp.to_excel(writer, sheet_name="outrights_sharp", index=False)
            _format_outright_sheet(writer, workbook, "outrights_sharp", outrights_sharp)

        # --- Finish Probabilities ---
        if finish_probs is not None and not finish_probs.empty:
            finish_probs.to_excel(writer, sheet_name="finish_probs", index=False)
            ws = writer.sheets["finish_probs"]
            for i, col in enumerate(finish_probs.columns):
                ws.set_column(i, i, max(len(str(col)) + 2, 10))

        # --- Score Line Edges ---
        if score_edges is not None and not score_edges.empty:
            score_edges.to_excel(writer, sheet_name="score_edges", index=False)
            ws = writer.sheets["score_edges"]
            for i, col in enumerate(score_edges.columns):
                ws.set_column(i, i, max(len(str(col)) + 2, 10))

    print(f"\n  Saved {excel_path}")

    # Also save score card(s) as standalone CSV for easy reference
    if score_cards_by_course:
        for course_name, card_df in score_cards_by_course.items():
            if not card_df.empty:
                csv_path = os.path.join(out_dir, f"fair_card_r{sim_round}_{course_name}.csv")
                card_df.to_csv(csv_path, index=False)
                print(f"  Saved {csv_path}")
        # Combined CSV with Course column
        card_csv = os.path.join(out_dir, f"fair_card_r{sim_round}.csv")
        score_card.to_csv(card_csv, index=False)
        print(f"  Saved {card_csv} (combined)")
    else:
        card_csv = os.path.join(out_dir, f"fair_card_r{sim_round}.csv")
        score_card.to_csv(card_csv, index=False)
        print(f"  Saved {card_csv}")

    return excel_path, card_csv


def _format_outright_sheet(writer, workbook, sheet_name, df):
    """Apply formatting to outright sheet."""
    ws = writer.sheets[sheet_name]
    green = workbook.add_format({"bg_color": "#d4edda"})
    yellow = workbook.add_format({"bg_color": "#FFFF00"})

    # Highlight high edge rows
    if "edge" in df.columns:
        edge_col_idx = df.columns.get_loc("edge")
        ws.conditional_format(
            1, 0, len(df), len(df.columns) - 1,
            {
                "type": "formula",
                "criteria": f'=${chr(65 + edge_col_idx)}2>10',
                "format": green,
            },
        )

    # Auto-width columns
    for i, col in enumerate(df.columns):
        max_len = max(df[col].astype(str).map(len).max() if len(df) > 0 else 0, len(col)) + 2
        ws.set_column(i, i, min(max_len, 20))


def _format_matchup_sheet(writer, workbook, sheet_name, df):
    """Apply conditional formatting to a matchup sheet."""
    ws = writer.sheets[sheet_name]
    yellow = workbook.add_format({"bg_color": "#FFFF00"})

    # Highlight rows where pred_on > 1 (strong conviction bets)
    if "pred_on" in df.columns:
        pred_col_idx = df.columns.get_loc("pred_on")
        ws.conditional_format(
            1, 0, len(df), len(df.columns) - 1,
            {
                "type": "formula",
                "criteria": f'=${chr(65 + pred_col_idx)}2>1',
                "format": yellow,
            },
        )

    # Auto-width columns
    for i, col in enumerate(df.columns):
        max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
        ws.set_column(i, i, min(max_len, 20))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_pred_col(model_preds, sim_round):
    """Find the best prediction column for display/filtering."""
    candidates = [
        f"scores_r{sim_round}",              # always exists
        f"my_pred{sim_round}" if sim_round > 1 else "my_pred",
        f"updated_pred_r{sim_round}",
        "updated_pred",
        "pred",
        "my_pred",
    ]
    for col in candidates:
        if col in model_preds.columns:
            return col
    return f"scores_r{sim_round}"


def load_sample_data():
    """Load sample sizes from pre_sim_summary if it exists."""
    path = f"pre_sim_summary_{tourney}.csv"
    if os.path.exists(path):
        sample = pd.read_csv(path)
        sample["player_name"] = sample["player_name"].str.lower().str.strip()
        return dict(zip(sample["player_name"], sample["sample"]))
    print(f"  Warning: {path} not found. Sample filter disabled.")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Email
# ══════════════════════════════════════════════════════════════════════════════

def build_matchup_email_html(sharp_df, sim_round, sample_lookup, outrights_sharp=None,
                             win_positive_top10=None, win_negative_top10=None,
                             wx_lookup=None, score_edges=None, kalshi_mids=None):
    """
    Build HTML email body with a table of sharp matchup picks, finish position edges,
    and outright win edge tables (top positive + top negative).

    Filters sharp_df to rows where:
        - bet_on player's pred > EMAIL_MIN_PRED
        - bet_on player's sample > EMAIL_MIN_SAMPLE
    """
    matchups_html = ""
    if not sharp_df.empty:
        # Filter: pred and sample thresholds on the bet_on side
        filtered = sharp_df.copy()
        filtered["sample_on"] = filtered["bet_on"].map(sample_lookup).fillna(0)
        filtered = filtered[
            (
                (filtered["pred_on"] > EMAIL_MIN_PRED)
                | ((filtered["pred_on"] > 0) & (filtered["edge_on"] > 7))
            )
            & (filtered["sample_on"] >= EMAIL_MIN_SAMPLE)
        ]

        if not filtered.empty:
            # Sort by edge descending
            filtered = filtered.sort_values("edge_on", ascending=False)

            # Build table rows
            rows_html = ""
            for _, row in filtered.iterrows():
                bet_player = row["bet_on"].title()
                opponent = (
                    row["Player 2"].title()
                    if row["bet_on"] == row["Player 1"]
                    else row["Player 1"].title()
                )
                book = row.get("Bookmaker", "")
                ties = row.get("Ties", "")
                book_odds = (
                    row["P1 Odds"] if row["bet_on"] == row["Player 1"] else row["P2 Odds"]
                )
                fair_odds = (
                    row["Fair_p1"] if row["bet_on"] == row["Player 1"] else row["Fair_p2"]
                )
                edge = row["edge_on"]
                pred = row["pred_on"]
                sample = int(row["sample_on"])
                archetype = row.get("type_on", "")
                half_shot = (
                    row.get("half_shot_p1", "")
                    if row["bet_on"] == row["Player 1"]
                    else row.get("half_shot_p2", "")
                )

                # Weather SG differential
                wx_sg = row.get("wx_diff", 0) if pd.notna(row.get("wx_diff", None)) else 0

                # Color coding
                edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"
                pred_color = "#d4edda" if pred > 1.5 else "#ffffff"

                # Format odds
                book_str = f"{int(book_odds):+d}" if pd.notna(book_odds) else ""
                fair_str = f"{int(fair_odds):+d}" if pd.notna(fair_odds) else ""

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{bet_player}</td>
                    <td style="padding:6px 10px; color:#666;">vs {opponent}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{ties}</td>
                    <td style="padding:6px 10px; text-align:center;">{book_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px;">{wx_sg:+.2f}</td>
                    <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred:.2f}</td>
                    <td style="padding:6px 10px; text-align:center;">{sample}</td>
                </tr>"""

            matchups_html = f"""
            <h3 style="color:#2c5282; margin:20px 0 8px 0;">
                Sharp Matchup Picks (pred &gt; {EMAIL_MIN_PRED}, sample &gt; {EMAIL_MIN_SAMPLE})
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Bet On</th>
                    <th style="padding:6px 10px; text-align:left;">Opponent</th>
                    <th style="padding:6px 10px; text-align:center;">Type</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Ties</th>
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
            matchups_html = "<p>No matchups passed filters (pred &gt; 0.75, sample &gt; 30).</p>"
    else:
        matchups_html = "<p>No sharp matchup picks for this round.</p>"

    # Build finish position edges section
    outrights_html = ""
    if outrights_sharp is not None and not outrights_sharp.empty:
        # Filter to positive pred players
        filtered_out = outrights_sharp[outrights_sharp['my_pred'].fillna(-1) > 0].copy()
        if not filtered_out.empty:
            filtered_out = filtered_out.sort_values('edge', ascending=False).head(20)

            rows_html = ""
            for _, row in filtered_out.iterrows():
                player = row['player_name'].title()
                market = row['market_type'].replace('_', ' ').title()
                book = row.get('bookmaker', '')
                side = row.get('side', '')
                odds = row.get('american_odds', '')
                fair = row.get('my_fair', '')
                edge = row.get('edge', 0)
                pred = row.get('my_pred', 0)
                stake = row.get('stake', 0)
                archetype = row.get('type_on', '')

                # Weather context
                _fp_wx_sg = 0.0
                if wx_lookup:
                    _fp_wx_sg = wx_lookup.get(str(row.get('player_name', '')).lower().strip(), 0)
                _fp_wx_color = "#d4edda" if abs(_fp_wx_sg) > 0.3 else "#ffffff"
                _fp_wx_str = f"{_fp_wx_sg:+.2f}" if _fp_wx_sg != 0 else "0.00"

                edge_color = "#d4edda" if edge > 10 else "#fff3cd" if edge > 5 else "#ffffff"
                pred_color = "#d4edda" if pred and pred > 1.5 else "#ffffff"
                side_str = str(side).upper() if side and pd.notna(side) else ""
                side_color = "#e8f4fd" if side == "yes" else "#fde8e8" if side == "no" else "#ffffff"

                odds_str = f"{int(odds):+d}" if pd.notna(odds) else ""
                fair_str = f"{int(fair):+d}" if pd.notna(fair) else ""
                pred_str = f"{pred:.2f}" if pd.notna(pred) else ""
                stake_str = f"${stake:.0f}" if pd.notna(stake) and stake > 0 else ""

                # Cent pricing for exchange rows (Kalshi): effective fill price incl. fee
                is_exchange = str(book).lower() in ("kalshi", "novig")
                imp_prob = row.get("implied_prob")
                sim_prob_row = row.get("sim_prob")
                line_c_str = f"{imp_prob * 100:.1f}&cent;" if is_exchange and pd.notna(imp_prob) else ""
                fair_c_str = f"{sim_prob_row * 100:.1f}&cent;" if is_exchange and pd.notna(sim_prob_row) else ""

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{market}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center; background:{side_color}; font-weight:600;">{side_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                    <td style="padding:6px 10px; text-align:center;">{odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; color:#2c5282;">{line_c_str}</td>
                    <td style="padding:6px 10px; text-align:center; color:#2c5282; font-weight:500;">{fair_c_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{stake_str}</td>
                    <td style="padding:6px 10px; text-align:center; background:{_fp_wx_color};">{_fp_wx_str}</td>
                </tr>"""

            outrights_html = f"""
            <h3 style="color:#2c5282; margin:30px 0 8px 0;">
                Finish Position Edges (Live Tournament Sim)
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Market</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Side</th>
                    <th style="padding:6px 10px; text-align:center;">Type</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Line&cent;</th>
                    <th style="padding:6px 10px; text-align:center;">Fair&cent;</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Stake</th>
                    <th style="padding:6px 10px; text-align:center;">Wx SG</th>
                </tr>
                {rows_html}
            </table>"""

    # Build outright win — top positive edges table
    win_pos_html = ""
    if win_positive_top10 is not None and not win_positive_top10.empty:
        rows_html = ""
        for _, row in win_positive_top10.iterrows():
            player = row['player_name'].title()
            book = row.get('bookmaker', '')
            odds = row.get('american_odds', '')
            fair = row.get('my_fair', '')
            edge = row.get('edge', 0)
            sim_prob = row.get('simulated_win_prob', 0)
            kelly = row.get('kelly', 0)

            edge_color = "#d4edda" if edge > 10 else "#fff3cd" if edge > 5 else "#ffffff"

            odds_str = f"{int(odds):+d}" if pd.notna(odds) else ""
            fair_str = f"{int(fair):+d}" if pd.notna(fair) else ""
            sim_str = f"{sim_prob*100:.2f}%" if pd.notna(sim_prob) else ""
            kelly_str = f"${kelly:.0f}" if pd.notna(kelly) and kelly > 0 else "$0"

            rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center;">{sim_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{kelly_str}</td>
                </tr>"""

        win_pos_html = f"""
            <h3 style="color:#2c5282; margin:30px 0 8px 0;">
                Outright Win — Top Positive Edges (by Kelly)
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Sim Win%</th>
                    <th style="padding:6px 10px; text-align:center;">Kelly</th>
                </tr>
                {rows_html}
            </table>"""

    # Build outright win — top negative edges (fades) table
    win_neg_html = ""
    if win_negative_top10 is not None and not win_negative_top10.empty:
        rows_html = ""
        for _, row in win_negative_top10.iterrows():
            player = row['player_name'].title()
            book = row.get('bookmaker', '')
            odds = row.get('american_odds', '')
            fair = row.get('my_fair', '')
            edge = row.get('edge', 0)
            sim_prob = row.get('simulated_win_prob', 0)
            implied = row.get('implied_prob', 0)

            edge_color = "#f8d7da" if edge < -10 else "#fff3cd" if edge < -5 else "#ffffff"

            odds_str = f"{int(odds):+d}" if pd.notna(odds) else ""
            fair_str = f"{int(fair):+d}" if pd.notna(fair) else ""
            sim_str = f"{sim_prob*100:.2f}%" if pd.notna(sim_prob) else ""
            impl_str = f"{implied*100:.2f}%" if pd.notna(implied) else ""

            rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center;">{sim_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{impl_str}</td>
                </tr>"""

        win_neg_html = f"""
            <h3 style="color:#2c5282; margin:30px 0 8px 0;">
                Outright Win — Top Fades (&lt;30:1 odds, most overpriced)
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Sim Win%</th>
                    <th style="padding:6px 10px; text-align:center;">Mkt Implied%</th>
                </tr>
                {rows_html}
            </table>"""

    # Build score line edges table (>5% edge only)
    score_edges_html = ""
    if score_edges is not None and not score_edges.empty:
        edge_rows = score_edges[score_edges["Best_Edge"] > 5].copy()
        if not edge_rows.empty:
            edge_rows = edge_rows.sort_values("Best_Edge", ascending=False)
            rows_html = ""
            for _, row in edge_rows.iterrows():
                player = row["Player"].title() if "," not in row["Player"] else row["Player"].split(",")[1].strip().title() + " " + row["Player"].split(",")[0].strip().title()
                line = row["Line"]
                book = row["Book"]
                side = row["Best_Side"]
                mkt_odds = row["Mkt_Under"] if side == "Under" else row["Mkt_Over"]
                fair_odds = row["Fair_Under"] if side == "Under" else row["Fair_Over"]
                edge = row["Best_Edge"]

                edge_color = "#d4edda" if edge > 8 else "#fff3cd"
                mkt_str = f"{int(mkt_odds):+d}" if pd.notna(mkt_odds) else ""
                fair_str = f"{int(fair_odds):+d}" if pd.notna(fair_odds) else ""

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{side} {line}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{mkt_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                </tr>"""

            score_edges_html = f"""
            <h3 style="color:#2c5282; margin:30px 0 8px 0;">
                Round Score O/U Edges (&gt;5%)
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Bet</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                </tr>
                {rows_html}
            </table>"""

    # Build Kalshi Maker Opportunities table (mid pricing, no fees)
    kalshi_mids_html = ""
    if kalshi_mids is not None and not kalshi_mids.empty:
        filtered_mids = kalshi_mids[
            (kalshi_mids["edge"] > 0)
            & (kalshi_mids["my_pred"].fillna(-1) > 0)
        ].copy()
        if not filtered_mids.empty:
            filtered_mids = filtered_mids.sort_values("edge", ascending=False).head(20)
            rows_html = ""
            for _, row in filtered_mids.iterrows():
                player = row["player_name"].title()
                market = row["market_type"].replace("_", " ").title()
                side = row.get("side", "yes")
                odds = row.get("american_odds", "")
                fair = row.get("my_fair", "")
                edge = row.get("edge", 0)
                pred = row.get("my_pred", 0)

                _mk_wx_sg = 0.0
                if wx_lookup:
                    _mk_wx_sg = wx_lookup.get(str(row.get("player_name", "")).lower().strip(), 0)

                edge_color = "#d4edda" if edge > 10 else "#fff3cd" if edge > 5 else "#ffffff"
                pred_color = "#d4edda" if pred and pred > 1.5 else "#ffffff"
                side_color = "#e8f4fd" if side == "yes" else "#fde8e8"

                odds_str = f"{int(odds):+d}" if pd.notna(odds) else ""
                fair_str = f"{int(fair):+d}" if pd.notna(fair) else ""
                pred_str = f"{pred:.2f}" if pd.notna(pred) else ""
                wx_str = f"{_mk_wx_sg:+.2f}" if _mk_wx_sg != 0 else "0.00"

                # Cent pricing (Kalshi maker = mid price, no fees)
                imp_prob = row.get("implied_prob")
                sim_prob_row = row.get("sim_prob")
                line_c_str = f"{imp_prob * 100:.1f}&cent;" if pd.notna(imp_prob) else ""
                fair_c_str = f"{sim_prob_row * 100:.1f}&cent;" if pd.notna(sim_prob_row) else ""

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{market}</td>
                    <td style="padding:6px 10px; text-align:center; background:{side_color}; font-weight:600;">{side.upper()}</td>
                    <td style="padding:6px 10px; text-align:center;">{odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; color:#2c5282;">{line_c_str}</td>
                    <td style="padding:6px 10px; text-align:center; color:#2c5282; font-weight:500;">{fair_c_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{wx_str}</td>
                </tr>"""

            kalshi_mids_html = f"""
            <h3 style="color:#2c5282; margin:30px 0 8px 0;">
                Kalshi Maker Opportunities (Mid Price, No Fees)
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Market</th>
                    <th style="padding:6px 10px; text-align:center;">Side</th>
                    <th style="padding:6px 10px; text-align:center;">Mid Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Mid&cent;</th>
                    <th style="padding:6px 10px; text-align:center;">Fair&cent;</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Wx SG</th>
                </tr>
                {rows_html}
            </table>"""

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:960px; margin:0 auto; padding:20px;">
        <h2 style="margin-bottom:4px;">R{sim_round} Round Sim - {tourney.replace('_', ' ').title()}</h2>
        <p style="color:#666; margin-top:0;">{datetime.now().strftime('%B %d, %Y %I:%M %p')}</p>

        {matchups_html}

        {score_edges_html}

        {outrights_html}

        {kalshi_mids_html}

        {win_pos_html}

        {win_neg_html}

        <p style="color:#999; font-size:11px; margin-top:30px;">
            Fair = our no-vig price | Edge = expected return % |
            Wx = weather SG advantage vs opponent (positive = favorable) |
            Pred = model SG prediction | Stake = suggested Kelly stake
        </p>
        <p style="color:#999; font-size:11px;">
            Attachments: fair score card (CSV), full matchup workbook (XLSX)
        </p>
    </body>
    </html>"""

    return html


def send_round_sim_email(sharp_df, sim_round, sample_lookup,
                         excel_path=None, card_csv_path=None, outrights_sharp=None,
                         win_edges_csv_path=None, bol_matchups_csv_path=None,
                         all_books_csv_path=None,
                         finish_equity_csv_path=None,
                         win_positive_top10=None, win_negative_top10=None,
                         wx_lookup=None, score_edges=None, kalshi_mids=None):
    """
    Send round sim email with:
        - HTML body: filtered sharp matchup table + finish position edges
                     + outright win positive/negative edge tables
        - Attachment 1: full matchup + score card Excel workbook
        - Attachment 2: BetOnline all matchups CSV (unfiltered)

    Non-blocking: prints warning on failure but doesn't crash.
    """
    password = os.getenv("EMAIL_PASSWORD")
    if not password:
        print("  Warning: GMAIL_APP_PASSWORD not set. Skipping email.")
        return

    try:
        html = build_matchup_email_html(sharp_df, sim_round, sample_lookup, outrights_sharp,
                                        win_positive_top10=win_positive_top10,
                                        win_negative_top10=win_negative_top10,
                                        wx_lookup=wx_lookup,
                                        score_edges=score_edges,
                                        kalshi_mids=kalshi_mids)

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"R{sim_round} Round Sim — {tourney.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)

        # HTML body
        msg.attach(MIMEText(html, "html"))

        # Attach Excel workbook
        if excel_path and os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                att = MIMEApplication(
                    f.read(),
                    _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                att.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(excel_path),
                )
                msg.attach(att)

        # Attach win edges CSV
        if win_edges_csv_path and os.path.exists(win_edges_csv_path):
            with open(win_edges_csv_path, "rb") as f:
                att = MIMEApplication(f.read(), _subtype="csv")
                att.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(win_edges_csv_path),
                )
                msg.attach(att)

        # Attach BetOnline all matchups CSV (unfiltered)
        if bol_matchups_csv_path and os.path.exists(bol_matchups_csv_path):
            with open(bol_matchups_csv_path, "rb") as f:
                att = MIMEApplication(f.read(), _subtype="csv")
                att.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(bol_matchups_csv_path),
                )
                msg.attach(att)

        # Attach all-books fair matchups CSV (betonline + pinnacle + betcris)
        if all_books_csv_path and os.path.exists(all_books_csv_path):
            with open(all_books_csv_path, "rb") as f:
                att = MIMEApplication(f.read(), _subtype="csv")
                att.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(all_books_csv_path),
                )
                msg.attach(att)

        # Attach finish equity CSV (all outright edges)
        if finish_equity_csv_path and os.path.exists(finish_equity_csv_path):
            with open(finish_equity_csv_path, "rb") as f:
                att = MIMEApplication(f.read(), _subtype="csv")
                att.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(finish_equity_csv_path),
                )
                msg.attach(att)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, password)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print("  Round sim email sent")

    except Exception as e:
        print(f"  Warning: Email failed: {e}")
        print("    (Sim outputs still saved — email is non-blocking)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Entry point. Reads config from Google Sheet or CLI args.

    The Google Sheet provides:
        round_num        -> sim_round = round_num + 1
        expected_score_1 -> expected scoring average (or first course for multi-course)

    NEW (v2): Also runs tournament simulation for outright/finish position pricing.
    """
    parser = argparse.ArgumentParser(description="Round Simulation - Matchups + Score Cards + Outrights")
    parser.add_argument("--cli", action="store_true",
                        help="Use CLI args instead of Google Sheet config")
    parser.add_argument("--sim-round", type=int,
                        help="Round to simulate (e.g. 2 = simulate R2 scores)")
    parser.add_argument("--expected-avg", type=float,
                        help="Expected field scoring average for the round")
    parser.add_argument("--skip-tournament-sim", action="store_true",
                        help="Skip tournament simulation (matchups + score card only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip email sending and bet storage")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy N(mu, STD_DEV) draws instead of category-first (default: catfirst)")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    sheet_config = None
    if not args.cli:
        try:
            sheet_config = _cfg  # reuse config loaded at import time
            round_num = sheet_config["round_num"]
            sim_round = round_num + 1 if round_num < 4 else 4
            expected_avg = sheet_config.get("expected_score_1")
            if expected_avg is None:
                expected_avg = PAR
                print(f"  Warning: No expected_score_1 in sheet, using PAR={PAR}")
            elif abs(expected_avg) > 50:
                # Full expected score entered (e.g. 68.7), use as-is
                print(f"  Note: expected_score_1={expected_avg} detected as full score")
            else:
                # Small value = adjustment from par (e.g. -3.3)
                expected_avg = PAR + expected_avg
        except Exception as e:
            print(f"\nWarning: Could not read Google Sheet: {e}")
            if args.sim_round is None:
                parser.error("Sheet unavailable and no --sim-round provided.")
            sim_round = args.sim_round
            expected_avg = args.expected_avg or PAR
            round_num = sim_round - 1
    else:
        if args.sim_round is None:
            parser.error("--sim-round is required in CLI mode")
        sim_round = args.sim_round
        expected_avg = args.expected_avg or PAR
        round_num = sim_round - 1

    # ── Load predictions ─────────────────────────────────────────────────
    pred_file = f"model_predictions_r{sim_round}.csv"
    if not os.path.exists(pred_file):
        alt = os.path.join("dashboard_data", pred_file)
        if os.path.exists(alt):
            print(f"  [resolve] {pred_file} not in root, using {alt}")
            pred_file = alt
        else:
            raise FileNotFoundError(
                f"{pred_file} not found. Run live_stats_engine.py first."
            )

    model_preds = pd.read_csv(pred_file)
    model_preds["player_name"] = (
        model_preds["player_name"].str.lower().str.strip().replace(name_replacements)
    )

    pred_col = find_pred_col(model_preds, sim_round)
    pred_lookup = dict(zip(model_preds["player_name"], model_preds[pred_col]))
    sample_lookup = load_sample_data()

    print(f"\n{'='*60}")
    print(f"  ROUND {sim_round} SIMULATION - {tourney}")
    print(f"{'='*60}")
    print(f"  Predictions:  {pred_file} ({len(model_preds)} players)")
    print(f"  Expected avg: {expected_avg}")
    print(f"  Draw mode:    {'legacy (STD_DEV=' + str(STD_DEV) + ')' if args.legacy else 'category-first'}")
    print(f"  Simulations:  {NUM_SIMULATIONS:,}")
    print(f"  Pred column:  {pred_col}")

    # ── Build weather lookup from model predictions ─────────────────────
    _wind_col = f"wind_adj{sim_round}"
    _dew_col = f"dew_adj{sim_round}"
    _wx_lookup = {}
    if _wind_col in model_preds.columns and _dew_col in model_preds.columns:
        _avg_wind = model_preds[_wind_col].mean()
        _wx_lookup = dict(zip(
            model_preds["player_name"],
            _avg_wind - model_preds[_wind_col] + model_preds[_dew_col],
        ))
        print(f"  Weather lookup: {len(_wx_lookup)} players (wind col={_wind_col}, dew col={_dew_col})")
    else:
        print(f"  No weather columns ({_wind_col}, {_dew_col}) — weather decomposition disabled")

    # ── Step 1: Simulate scores ──────────────────────────────────────────
    print(f"\n  Simulating R{sim_round} scores...")

    # Category-first is the default; --legacy falls back to N(mu, STD_DEV) draws
    sim_dict_cf = simulate_round_scores_catfirst(
        model_preds, sim_round, expected_avg, _wx_lookup
    )

    if args.legacy:
        sim_dict_legacy = simulate_round_scores(model_preds, sim_round, expected_avg)
        print_catfirst_comparison(sim_dict_legacy, sim_dict_cf, model_preds, pred_col, expected_avg)
        print("  [legacy] Using standard N(mu, STD_DEV) draws for matchup pricing")
        sim_dict = sim_dict_legacy
    else:
        sim_dict = sim_dict_cf

    # ── Step 2: Matchup pricing ──────────────────────────────────────────
    print(f"\n  Fetching matchup odds (scraped -> DataGolf fallback)...")
    try:
        from odds_loader import load_matchup_odds
        matchup_df = load_matchup_odds("round_matchups", api_key=API_KEY)
        matchup_df = price_matchups(matchup_df, sim_dict)
        matchup_df = calculate_edges(matchup_df)
        combined, sharp = build_matchup_outputs(
            matchup_df, sim_round, pred_lookup, sample_lookup, wx_lookup=_wx_lookup
        )
        # Build unfiltered BetOnline matchup CSV (all edges, all samples)
        out_dir = f"./{tourney}"
        bol_matchups_csv = build_betonline_all_matchups_csv(
            matchup_df, sim_round, out_dir
        )
        # Build fair price vs all 3 scraped books CSV
        all_books_csv = build_all_books_fair_csv(
            matchup_df, sim_round, out_dir
        )
    except Exception as e:
        print(f"  Warning: Matchup pricing failed: {e}")
        combined = pd.DataFrame()
        sharp = pd.DataFrame()
        bol_matchups_csv = None
        all_books_csv = None

    # ── Step 3: Score card ───────────────────────────────────────────────
    # Multi-course: build separate score cards per course
    score_cards_by_course = {}
    if ("course_score_adj" in model_preds.columns
            and model_preds["course_score_adj"].nunique() > 1):
        for course_adj in sorted(model_preds["course_score_adj"].unique()):
            mask = model_preds["course_score_adj"] == course_adj
            course_players = set(model_preds.loc[mask, "player_name"])
            course_name = model_preds.loc[mask, "course_x"].iloc[0] if "course_x" in model_preds.columns else f"{course_adj}"
            course_sim = {p: s for p, s in sim_dict.items() if p in course_players}
            if course_sim:
                print(f"\n  Building score card for {course_name} (expected = {course_adj})...")
                card = build_score_card(course_sim, course_adj, pred_lookup)
                card.insert(0, "Course", course_name)
                score_cards_by_course[course_name] = card
        # Combine for CSV / email; keep course column for identification
        score_card = pd.concat(score_cards_by_course.values(), ignore_index=True) if score_cards_by_course else pd.DataFrame()
    else:
        print(f"\n  Building fair score card (expected avg = {expected_avg})...")
        score_card = build_score_card(sim_dict, expected_avg, pred_lookup)

    # ── Step 3b: Price score lines vs market ─────────────────────────────
    score_edges = pd.DataFrame()
    try:
        market_lines = load_score_lines()
        if market_lines:
            score_edges = price_score_lines(score_card, market_lines)
    except Exception as e:
        print(f"  Warning: Score line pricing failed: {e}")

    # ── Step 4: Tournament Simulation (NEW) ──────────────────────────────
    outrights_combined = pd.DataFrame()
    outrights_sharp = pd.DataFrame()
    kalshi_mids = pd.DataFrame()
    kalshi_edges = pd.DataFrame()
    novig_edges = pd.DataFrame()
    finish_probs = pd.DataFrame()
    win_edges_csv_path = None
    finish_equity_csv_path = None
    win_positive_top10 = pd.DataFrame()
    win_negative_top10 = pd.DataFrame()

    if not args.skip_tournament_sim and round_num >= 1:
        print(f"\n  Running tournament simulation (R{round_num} complete -> R4)...")
        try:
            # Load tournament config from sheet
            if sheet_config:
                tourn_config = load_tournament_config(sheet_config)
            else:
                tourn_config = {
                    "course_map": {},
                    "default_par": PAR,
                    "default_expected": expected_avg,
                    "wind_arrays": {2: [], 3: [], 4: []},
                    "dew_arrays": {2: [], 3: [], 4: []},
                }

            # Load known rounds
            known_data = load_known_rounds(
                round_num,
                tourn_config["course_map"],
                tourn_config["default_par"]
            )

            if known_data["player_names"]:
                player_names = known_data["player_names"]
                print(f"    Loaded {len(player_names)} players from R1-R{round_num} data")

                # Load category-first distribution params
                cf_result = _load_catfirst_dists(player_names)
                if cf_result is None:
                    print("    Warning: catfirst dists unavailable, skipping tournament sim")
                    raise RuntimeError("catfirst dists unavailable")
                player_cf_params, effective_skew, L_corr = cf_result
                print(f"    Loaded catfirst distribution parameters")

                # Simulate remaining rounds
                print(f"    Simulating remaining rounds ({TOURNAMENT_SIMULATIONS:,} sims)...")
                final_scores, made_cut_mask = simulate_remaining_rounds(
                    completed_round=round_num,
                    player_names=player_names,
                    known_strokes=known_data["strokes"],
                    known_categories=known_data["categories"],
                    model_preds=model_preds,
                    player_cf_params=player_cf_params,
                    effective_skew=effective_skew,
                    L_corr=L_corr,
                    tournament_config=tourn_config,
                    player_preds_base=known_data["player_preds"],
                    num_sims=TOURNAMENT_SIMULATIONS,
                )

                # Compute finish probabilities
                print(f"    Computing finish probabilities...")
                finish_probs, rank_probs = compute_finish_probabilities(
                    final_scores, player_names, made_cut_mask, TOURNAMENT_SIMULATIONS
                )

                # Save finish probs
                finish_probs.to_csv("simulated_probs_live.csv", index=False)
                finish_probs.to_csv(f"top_finish_probs_live_{tourney}.csv", index=False)
                print(f"    Saved simulated_probs_live.csv")

                # Save rank probabilities for distribution dashboard
                rank_probs.to_parquet(f"rank_probs_live_{tourney}.parquet", index=False)
                print(f"    Saved rank_probs_live_{tourney}.parquet")

                # Price outrights against market
                print(f"    Fetching outright odds and calculating edges...")
                priced_markets = price_outrights(finish_probs, pred_lookup, sample_lookup)

                # Build outputs
                outrights_combined, outrights_sharp = build_finish_outputs(
                    priced_markets, pred_lookup, sample_lookup
                )

                if not outrights_combined.empty:
                    finish_equity_csv_path = f"finish_equity_live_{tourney}.csv"
                    outrights_combined.to_csv(finish_equity_csv_path, index=False)
                    print(f"    Saved {finish_equity_csv_path}")
                    print(f"    Outrights: {len(outrights_combined)} edges found, {len(outrights_sharp)} sharp")
                else:
                    print(f"    No outright edges above threshold")

                # Price Kalshi outrights (no dead-heat)
                print(f"\n    Pricing Kalshi outrights (no dead-heat)...")
                kalshi_edges = price_kalshi_outrights(finish_probs, pred_lookup, sample_lookup)
                kalshi_mids = pd.DataFrame()
                if not kalshi_edges.empty:
                    # Split taker vs mid pricing
                    kalshi_taker = kalshi_edges[kalshi_edges["pricing"] == "taker"].copy()
                    kalshi_mids = kalshi_edges[kalshi_edges["pricing"] == "mid"].copy()

                    # Merge taker rows into outrights_combined/sharp (existing flow)
                    if not kalshi_taker.empty:
                        outrights_combined = pd.concat([outrights_combined, kalshi_taker], ignore_index=True)
                        kalshi_sharp = kalshi_taker[kalshi_taker["edge"] > EDGE_THRESHOLD_TOPN].copy()
                        if not kalshi_sharp.empty:
                            outrights_sharp = pd.concat([outrights_sharp, kalshi_sharp], ignore_index=True)
                            outrights_sharp = outrights_sharp.sort_values("edge", ascending=False)

                # Price NoVig outrights (no dead-heat)
                print(f"\n    Pricing NoVig outrights (no dead-heat)...")
                try:
                    novig_edges = price_novig_outrights(finish_probs, pred_lookup, sample_lookup, tourney_name=tourney)
                    if not novig_edges.empty:
                        novig_taker = novig_edges[novig_edges["edge"] > 0].copy()
                        if not novig_taker.empty:
                            outrights_combined = pd.concat([outrights_combined, novig_taker], ignore_index=True)
                            novig_sharp = novig_taker[novig_taker["edge"] > EDGE_THRESHOLD_TOPN].copy()
                            if not novig_sharp.empty:
                                outrights_sharp = pd.concat([outrights_sharp, novig_sharp], ignore_index=True)
                                outrights_sharp = outrights_sharp.sort_values("edge", ascending=False)
                except Exception as e:
                    print(f"    Warning: NoVig pricing failed: {e}")

                # --- Win market edge CSVs ---
                print(f"\n    Building outright win edge CSVs...")
                out_dir = f"./{tourney}"
                os.makedirs(out_dir, exist_ok=True)
                _, win_edges_csv_path, win_all_edges = build_win_edges_csv(finish_probs, pred_lookup, sample_lookup, out_dir)
                build_betonline_negative_edges_csv(finish_probs, pred_lookup, sample_lookup, out_dir)

                # Build top-10 positive and negative win edge tables for email
                if not win_all_edges.empty:
                    pos = win_all_edges[win_all_edges['edge'] > 0].copy()
                    pos = pos.sort_values('kelly', ascending=False).drop_duplicates('player_name', keep='first').head(10)
                    win_positive_top10 = pos

                    neg = win_all_edges[(win_all_edges['edge'] < 0) & (win_all_edges['decimal_odds'] < 31.0)].copy()
                    neg = neg.sort_values('edge', ascending=True).drop_duplicates('player_name', keep='first').head(10)
                    win_negative_top10 = neg

            else:
                print(f"    No player data found for tournament sim")

        except Exception as e:
            print(f"    Warning: Tournament simulation failed: {e}")
            import traceback
            traceback.print_exc()
    elif args.skip_tournament_sim:
        print(f"\n  Skipping tournament simulation (--skip-tournament-sim)")
    else:
        print(f"\n  Skipping tournament simulation (round_num < 1)")

    # ── Name mismatch alert ─────────────────────────────────────────────
    # Collect mismatches from matchup pricing + Kalshi outrights and alert
    _all_mismatches = {}
    try:
        if matchup_df is not None and hasattr(matchup_df, 'attrs') and matchup_df.attrs.get("name_mismatches"):
            _all_mismatches.update(matchup_df.attrs["name_mismatches"])
    except NameError:
        pass
    try:
        if kalshi_edges is not None and not kalshi_edges.empty and kalshi_edges.attrs.get("name_mismatches"):
            for name in kalshi_edges.attrs["name_mismatches"]:
                _all_mismatches.setdefault(name, set()).add("kalshi_outrights")
    except NameError:
        pass

    if _all_mismatches and not args.dry_run:
        _mm_lines = [f"<b>R{sim_round} Name Mismatches — {tourney.replace('_', ' ').title()}</b>", ""]
        _mm_lines.append(f"{len(_all_mismatches)} scraped players not found in sim:")
        for name, books in sorted(_all_mismatches.items()):
            book_str = ", ".join(sorted(books)) if isinstance(books, set) else str(books)
            _mm_lines.append(f"  • {name}  ({book_str})")
        _mm_lines.append("")
        _mm_lines.append("Fix: add to name_replacements in sim_inputs.py")
        _send_telegram("\n".join(_mm_lines))
        print(f"  Sent Telegram alert for {len(_all_mismatches)} name mismatches")

    # ── Compute archetypes (before export + email so type_on appears everywhere) ──
    try:
        from sg_diagnostic import compute_rolling_archetypes
        _field = model_preds['player_name'].unique().tolist() if model_preds is not None else []
        _arch_df = compute_rolling_archetypes(_event_id, _field)
        _arch_map = dict(zip(_arch_df['player_name'], _arch_df['archetype']))
        print(f"  Computed archetypes for {len(_arch_map)} players")
        if not combined.empty:
            combined['type_on'] = (
                combined['bet_on'].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
            )
        if not sharp.empty:
            sharp['type_on'] = (
                sharp['bet_on'].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
            )
        if not outrights_combined.empty:
            _fp_name_col = 'player_name' if 'player_name' in outrights_combined.columns else 'Player'
            outrights_combined['type_on'] = (
                outrights_combined[_fp_name_col].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
            )
        if not outrights_sharp.empty:
            _fp_name_col = 'player_name' if 'player_name' in outrights_sharp.columns else 'Player'
            outrights_sharp['type_on'] = (
                outrights_sharp[_fp_name_col].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
            )
        if not kalshi_mids.empty:
            kalshi_mids['type_on'] = (
                kalshi_mids['player_name'].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
            )
    except Exception as _arch_err:
        print(f"  Archetype computation skipped: {_arch_err}")

    # ── Step 5: Export ───────────────────────────────────────────────────
    excel_path, card_csv = export_results(
        combined, sharp, score_card, sim_round,
        outrights_combined=outrights_combined,
        outrights_sharp=outrights_sharp,
        finish_probs=finish_probs,
        score_cards_by_course=score_cards_by_course if score_cards_by_course else None,
        score_edges=score_edges,
    )

    # ── Step 6: Email ────────────────────────────────────────────────────
    if not args.dry_run:
        print(f"\n  Sending email...")
        send_round_sim_email(
            sharp_df=sharp,
            sim_round=sim_round,
            sample_lookup=sample_lookup,
            excel_path=excel_path,
            card_csv_path=card_csv,
            outrights_sharp=outrights_sharp,
            win_edges_csv_path=win_edges_csv_path,
            bol_matchups_csv_path=bol_matchups_csv,
            all_books_csv_path=all_books_csv,
            finish_equity_csv_path=finish_equity_csv_path,
            win_positive_top10=win_positive_top10,
            win_negative_top10=win_negative_top10,
            wx_lookup=_wx_lookup,
            score_edges=score_edges,
            kalshi_mids=kalshi_mids,
        )
    else:
        print(f"\n  [dry-run] Skipping email")

    # ── Storage ──────────────────────────────────────────────────────────────
    if not args.dry_run:
        from sheets_storage import (
            is_valid_run_time,
            get_spreadsheet,
            store_round_matchups,
            store_finish_positions,
            store_score_edges,
            load_dg_id_lookup,
        )

        if is_valid_run_time():
            print("\n[storage] Saving to Google Sheets...")
            try:
                # Single auth for all store calls
                spreadsheet = get_spreadsheet()

                # Build dg_id lookup (may not have all round-sim players, but best effort)
                dg_id_lookup = load_dg_id_lookup(tourney, name_replacements)

                # 1. All filtered round matchups
                store_round_matchups(
                    combined, sim_round, tourney, _event_id,
                    dg_id_lookup=dg_id_lookup,
                    spreadsheet=spreadsheet,
                )

                # 2. Finish position bets (live tab — not graded)
                if not outrights_combined.empty:
                    store_finish_positions(
                        outrights_combined, tourney, _event_id,
                        dg_id_lookup=dg_id_lookup,
                        spreadsheet=spreadsheet,
                        tab_name="Live",
                    )

                # 3. Score edges (round O/U)
                if score_edges is not None and not score_edges.empty:
                    store_score_edges(
                        score_edges, sim_round, tourney, _event_id,
                        spreadsheet=spreadsheet,
                    )

                print("[storage] Done.")
            except Exception as e:
                print(f"[storage] Warning: Failed: {e}")
                import traceback; traceback.print_exc()
        else:
            print("[storage] Skipped - before Monday 3 PM EST cutoff.")
    else:
        print(f"  [dry-run] Skipping bet storage")
    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")

    # Push dashboard data to Render (skip on dry-run)
    if not args.dry_run:
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


if __name__ == "__main__":
    main()
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
from category_distribution_guard import require_complete_category_distributions
from shot_dispersion_overlay import apply_shot_dispersion_overlay
from sim_health_gate import (
    SimulationHealthError,
    build_simulation_manifest,
    collect_overlay_provenance,
    configured_round_scoring_baselines,
    derive_authoritative_scoring_targets,
    names_sha256,
    require_bound_artifact,
    require_exact_simulation_source,
    require_live_tournament_alignment,
    require_market_outputs_healthy,
    require_round_score_probability_table,
    require_simulation_healthy,
    sealed_cache_expected_avg,
    write_bound_artifact_manifest,
)

from score_centering import (
    CENTERING_VERSION,
    validate_field_relative_predictions,
)
from score_reprice import (
    FRACTIONAL_SCORE_REPRICE_METHOD,
    fractional_settlement_pmf,
    score_est_requires_live_refresh,
    uniformly_shift_score_tape,
)

# --- Weekly-changing config from Google Sheet ---
from sheet_config import load_config as _load_sheet_config
_cfg = _load_sheet_config()
tourney          = _cfg["tourney"]
STD_DEV          = _cfg["std_dev"]
_sheet_pars      = _cfg.get("course_pars") or []
PAR              = int(_sheet_pars[0]) if _sheet_pars else 72  # sheet course_pars wins; 72 fallback
CUT_LINE         = _cfg["cut_line"]   # from golf_sims sheet (per event); 0 or >= field = NO CUT
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
from round_matchup_coverage import (
    DEFAULT_REQUIRED_BOOKS,
    resolve_required_matchup_books,
)

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("DATAGOLF_API_KEY")
MATCHUPS_URL = "https://feeds.datagolf.com/betting-tools/matchups"
OUTRIGHTS_URL = "https://feeds.datagolf.com/betting-tools/outrights"

NUM_SIMULATIONS = 100_000        # For single-round matchup sim
TOURNAMENT_SIMULATIONS = 100_000  # For tournament sim (R through R4)

SHARP_BOOKS = ["pinnacle", "betonline", "betcris"]
HALF_SHOT_ADJ = {"betonline": 25, "betcris": 30}
# Email banner warns (orange) when a sharp book prices fewer than this many
# matchup lines, and flags red at 0 ("NO LINES") — catches a book offline / barely posting.
MATCHUP_LINE_WARN_THRESHOLD = 10
REQUIRED_MATCHUP_BOOKS = DEFAULT_REQUIRED_BOOKS

# Score card: generate fair UNDER prices at these offsets from expected avg
SCORE_CARD_RANGE = 3.0        # +-3 strokes from expected
SCORE_CARD_STEP = 0.5         # half-stroke intervals
MIN_PRED_FOR_CARD = -0.5      # exclude players with pred below this

# Email
EMAIL_FROM = os.getenv("EMAIL_USER")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")


class EmailDeliveryError(RuntimeError):
    """A required round report was not accepted by the mail transport."""


def _round_email_required():
    return os.getenv("REQUIRE_ROUND_SIM_EMAIL", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _required_matchup_line_floor():
    try:
        value = int(os.getenv("ROUND_SIM_MIN_BOOK_MATCHUPS", "5"))
    except (TypeError, ValueError):
        value = 5
    return max(1, value)


def require_pricing_pipeline_healthy(
    *,
    matchup_error=None,
    threeball_error=None,
    score_line_error=None,
    matchup_book_counts=None,
    matchup_name_mismatches=None,
    require_complete_email=False,
    require_live_tournament=False,
    tournament_error=None,
    finish_probs=None,
    required_matchup_books=None,
):
    """Reject technically incomplete reports before any external side effect.

    Zero qualifying edges is a valid model result. A loader/pricer exception,
    insufficient required-book coverage in production, or an absent live
    tournament family means we cannot know that the report is complete.
    """
    if matchup_error is not None:
        raise SimulationHealthError(
            f"BLOCKED — round matchup pricing did not complete: {matchup_error}"
        ) from matchup_error

    if matchup_name_mismatches:
        names = ", ".join(
            sorted(str(name) for name in matchup_name_mismatches)[:8]
        )
        raise SimulationHealthError(
            "BLOCKED — fresh round-matchup lines contain player names that "
            f"did not join the active simulation field: {names}"
        )

    if require_complete_email:
        counts = matchup_book_counts or {}
        floor = _required_matchup_line_floor()
        required_matchup_books = (
            tuple(required_matchup_books)
            if required_matchup_books is not None
            else resolve_required_matchup_books()
        )
        if not required_matchup_books:
            raise SimulationHealthError(
                "BLOCKED — no required matchup books are configured"
            )
        missing = {
            book: int(counts.get(book, 0) or 0)
            for book in required_matchup_books
            if int(counts.get(book, 0) or 0) < floor
        }
        if missing:
            detail = ", ".join(
                f"{book}={count}/{floor}" for book, count in missing.items()
            )
            raise SimulationHealthError(
                f"BLOCKED — required matchup-book coverage is incomplete ({detail})"
            )
        if threeball_error is not None:
            raise SimulationHealthError(
                "BLOCKED — 3-ball pricing failed, so the email would be "
                f"incomplete: {threeball_error}"
            ) from threeball_error
        if score_line_error is not None:
            raise SimulationHealthError(
                "BLOCKED — score-line pricing failed, so the email would be "
                f"incomplete: {score_line_error}"
            ) from score_line_error

    if require_live_tournament:
        if tournament_error is not None:
            raise SimulationHealthError(
                "BLOCKED — live tournament/finish pricing did not complete: "
                f"{tournament_error}"
            ) from tournament_error
        if finish_probs is None or getattr(finish_probs, "empty", True):
            raise SimulationHealthError(
                "BLOCKED — live tournament/finish outputs are missing; rerun the "
                "full tournament simulation or explicitly use --skip-tournament-sim"
            )

# Matchup email filter thresholds
EMAIL_MIN_PRED = 0.75
EMAIL_MIN_SAMPLE = 20

# Outright market filter thresholds
EDGE_THRESHOLD_WIN = 2.0     # minimum edge (pp) sim_prob - implied_prob
EDGE_THRESHOLD_TOPN = 2.0   # minimum edge (pp) sim_prob - implied_prob
BANKROLL = 10000.0
KELLY_FRACTION = 0.25
RETAIL_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'barstool', 'espn']
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

# Set in main() from --use-python. When False (default), the heavy sim draws
# (tournament cascade seed 42 + single-round score card seed 789) run via the Rust
# sims_kernel; the Python draws are available only through the explicit flag.
_USE_PYTHON = False


def _handle_rust_kernel_failure(component, error):
    """Never let an incidental dependency failure select different sim math."""
    raise RuntimeError(
        f"{component} failed; refusing to silently switch simulation engines. "
        "Fix sims_kernel or rerun with --use-python as an explicit operator choice."
    ) from error


# ══════════════════════════════════════════════════════════════════════════════
# Tournament Sim Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def load_corr_matrix(cat_order):
    """Load the tracked production correlation matrix without model fallback."""
    production_path = CORR_PREFS[0]
    if not os.path.isfile(production_path):
        raise FileNotFoundError(
            "Required production category correlation matrix is missing: "
            f"{production_path}"
        )
    R = pd.read_csv(production_path, index_col=0)
    try:
        values = R.loc[cat_order, cat_order].to_numpy(dtype=float)
    except KeyError as exc:
        raise ValueError(
            f"Production correlation matrix lacks required categories: {cat_order}"
        ) from exc
    if values.shape != (len(cat_order), len(cat_order)) or not np.isfinite(values).all():
        raise ValueError("Production correlation matrix is incomplete or non-finite")
    return values


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


def _load_catfirst_dists(player_names, *, allow_player_subset=False):
    """Load v2 dists + correlation + per-player params for category-first draws.

    Returns (player_cf_params, effective_skew, L_corr). Production never
    substitutes the legacy normal model when these inputs are unavailable.
      player_cf_params: list of (mu, std_course) per player — mu NOT re-centered
      effective_skew: np.array (n_players, 4)
      L_corr: (4, 4) Cholesky of correlation matrix
    """
    if not os.path.exists(DISTS_FILE_V2):
        raise FileNotFoundError(
            f"Required category-first distributions not found: {DISTS_FILE_V2}"
        )

    dists, active_players = require_complete_category_distributions(
        pd.read_csv(DISTS_FILE_V2),
        player_names,
        CAT_ORDER,
        name_replacements=name_replacements,
        source_label=DISTS_FILE_V2,
    )

    mu_w   = dists.pivot(index='player_name', columns='category_clean', values='mean')
    std_w  = dists.pivot(index='player_name', columns='category_clean', values='std')
    skew_w = dists.pivot(index='player_name', columns='category_clean', values='skew')
    neff_w = dists.pivot(index='player_name', columns='category_clean', values='n_eff')

    # Use the same frozen pre-event shot dispersion in every round. This is
    # applied before course multipliers and does not change category means.
    std_w = apply_shot_dispersion_overlay(
        std_w,
        active_players,
        CAT_ORDER,
        tourney=tourney,
        event_id=_event_id,
        dists_path=DISTS_FILE_V2,
        allow_active_subset=allow_player_subset,
    )
    active_stds = std_w.loc[active_players, CAT_ORDER].to_numpy(dtype=float)
    if not np.isfinite(active_stds).all() or np.any(active_stds <= 0.0):
        raise ValueError(
            "Shot-dispersion overlay produced invalid active-field category "
            "standard deviations"
        )

    # Load correlation matrix and Cholesky
    R = load_corr_matrix(CAT_ORDER)
    try:
        L_corr = cholesky(R)
    except np.linalg.LinAlgError:
        R = 0.95 * R + 0.05 * np.eye(4)
        L_corr = cholesky(R)

    player_cf_params = []
    effective_skew = np.zeros((len(active_players), 4), dtype=float)

    for idx, player in enumerate(active_players):
        # Coverage was validated above; direct lookup prevents field-wide
        # fallback values from changing an active player's production model.
        mu = mu_w.loc[player, CAT_ORDER].to_numpy(dtype=float)
        std = std_w.loc[player, CAT_ORDER].to_numpy(dtype=float)

        # Apply course variance multipliers
        std_course = std * _course_mults_cf

        player_cf_params.append((mu, std_course))

        # Per-player effective skewness (confidence-weighted blend)
        for j, cat in enumerate(CAT_ORDER):
            p_skew = float(skew_w.at[player, cat])
            p_neff = float(neff_w.at[player, cat])
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


def _filter_to_active_players(result, completed_round):
    """
    Drop players who are out of the tournament (missed cut / WD / DQ) from the
    known-rounds result before simulating R{completed_round+1}..R4.

    Primary rule: a player with no DataGolf tee time for the NEXT round is out.
    This also covers the rare secondary cut after R3 (filters on r4_teetime) and
    mid-event WDs. Fallback when tee times aren't posted yet: the score-derived
    cut line already computed in result["made_cut"].

    Without this filter, eliminated players get simulated weekend rounds (and a
    phantom par round for any round file they're missing from), polluting
    finish probabilities and final_scores for every alive player.
    """
    players = result["player_names"]
    if not players:
        return result
    next_round = completed_round + 1
    teetime_col = f"r{next_round}_teetime"

    active_mask = None
    source = None

    field = None
    try:
        from api_utils import fetch_field_updates
        field = fetch_field_updates(API_KEY, teetime_col=teetime_col,
                                    fill_missing_teetimes=False)
    except Exception as e:
        print(f"    Warning: field updates fetch failed ({e})")

    if field is not None and teetime_col in field.columns and field[teetime_col].notna().any():
        names = (field["player_name"].astype(str).str.lower().str.strip()
                 .replace(name_replacements))
        active = set(names[field[teetime_col].notna()])
        active_mask = np.array([p in active for p in players])
        source = f"R{next_round} tee times"

        # Guard: the next-round tee-time field should be (almost) a subset of the
        # players we loaded known-rounds data for. If fewer than 90% of the tee-time
        # players are present in the known-rounds set, the r*_live_model.csv are
        # almost certainly from a DIFFERENT event than the current field (the
        # Memorial-vs-RBC bug). Abort loudly instead of simulating a stale field and
        # storing garbage bets.
        n_field_active = len(active)
        match_rate = int(active_mask.sum()) / n_field_active if n_field_active else 0.0
        if match_rate < 0.90:
            raise RuntimeError(
                f"Field/known-rounds event mismatch: only {int(active_mask.sum())}/"
                f"{n_field_active} ({match_rate:.0%}) of R{next_round} tee-time players "
                f"are present in the loaded known-rounds data (r*_live_model.csv). This "
                f"means the live-model files are from a different event than the current "
                f"field. Regenerate the live models for the current event before simulating."
            )
    elif result.get("made_cut") is not None:
        active_mask = np.asarray(result["made_cut"], dtype=bool)
        source = "computed cut line (tee times unavailable)"
        print(f"    Warning: R{next_round} tee times unavailable — "
              f"falling back to score-derived cut line for field filtering")

    if active_mask is None:
        print("    Warning: cannot determine active players — field NOT filtered for the cut")
        return result

    n_active = int(active_mask.sum())
    if n_active < 5:
        print(f"    Warning: only {n_active} active players via {source} — "
              f"looks wrong, field NOT filtered")
        return result

    dropped = [p for p, a in zip(players, active_mask) if not a]
    if not dropped:
        print(f"    Field filter ({source}): all {len(players)} players active")
        return result

    print(f"    Field filter ({source}): keeping {n_active} of {len(players)} players; "
          f"dropped {len(dropped)} (cut/WD): {dropped}")

    result["player_names"] = [p for p, a in zip(players, active_mask) if a]
    result["strokes"] = {r: arr[active_mask] for r, arr in result["strokes"].items()}
    result["categories"] = {r: arr[active_mask] for r, arr in result["categories"].items()}
    if result.get("cumulative") is not None:
        result["cumulative"] = result["cumulative"][active_mask]
    if result.get("made_cut") is not None:
        result["made_cut"] = np.asarray(result["made_cut"], dtype=bool)[active_mask]
    keep = set(result["player_names"])
    result["player_preds"] = {p: v for p, v in result["player_preds"].items() if p in keep}
    result["course_x"] = {p: v for p, v in result["course_x"].items() if p in keep}
    return result


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
            # Root copy is gitignored and only synced via the post-merge hook;
            # fall back to the tracked dashboard_data/ copy (same pattern as
            # model_predictions below) so a fresh pull works without the sync —
            # but ONLY when the sync manifest attributes the file to the
            # current event; a previous week's tracked copy would silently
            # feed last week's rounds in (2026-08 audit).
            alt = os.path.join("dashboard_data", live_file)
            from sync_event_files import _manifest_allowed
            if os.path.exists(alt) and \
                    f"r{rnd}_live_model.csv" in _manifest_allowed(str(tourney).lower()):
                print(f"  [resolve] {live_file} not in root, using {alt} "
                      f"(manifest-verified for {tourney})")
                live_file = alt
            else:
                if os.path.exists(alt):
                    print(f"  Warning: {live_file} in dashboard_data/ is not "
                          f"manifest-attributed to '{tourney}' — refusing stale "
                          f"fallback. Skipping round {rnd}.")
                else:
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

        # Determine made cut status (if completed_round >= 2).
        # CUT_LINE comes from the golf_sims sheet (per event). Convention:
        # cut_line <= 0  OR  cut_line >= field size  ==  NO CUT (signature events,
        # Tour Championship, etc.) -> everyone is "made". Otherwise top-N + 10-shot.
        if completed_round >= 2 and CUT_LINE >= 1:
            r1_r2 = result["strokes"].get(1, 0) + result["strokes"].get(2, 0)
            if isinstance(r1_r2, np.ndarray):
                _nf = len(r1_r2)
                if CUT_LINE >= _nf:
                    result["made_cut"] = np.ones(_nf, dtype=bool)
                    print(f"    Cut rule: NO CUT (cut_line={CUT_LINE} >= field {_nf}) — all made")
                else:
                    cut_score = np.sort(r1_r2)[CUT_LINE - 1]
                    result["made_cut"] = r1_r2 <= cut_score
                    if USE_10_SHOT_RULE:
                        within_10 = r1_r2 <= (r1_r2.min() + 10)
                        result["made_cut"] = result["made_cut"] | within_10
                    print(f"    Cut rule: top-{CUT_LINE}"
                          f"{' + 10-shot' if USE_10_SHOT_RULE else ''} (field {_nf})")
            else:
                result["made_cut"] = np.ones(len(result["player_names"]), dtype=bool)
        else:
            result["made_cut"] = np.ones(len(result["player_names"]), dtype=bool)
            if completed_round >= 2:
                print(f"    Cut rule: NO CUT (cut_line={CUT_LINE}) — all made")

    # Once a real cut exists, remove eliminated players from the field entirely
    # so they never enter the R3/R4 sims or finish probabilities.
    if completed_round >= 2 and completed_round <= 3:
        result = _filter_to_active_players(result, completed_round)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Tournament Simulation Engine
# ══════════════════════════════════════════════════════════════════════════════

def _active_round_expected_scores(player_names, model_preds, tournament_config):
    """Resolve absolute per-player baselines for the next simulated round.

    The prediction artifact's course baselines have already been checked against
    the authoritative Sheet map before this function is called. Single-course
    weeks use ``default_expected``; multi-course weeks use each player's verified
    ``course_score_adj``.
    """
    expected = np.full(
        len(player_names), float(tournament_config["default_expected"]), dtype=float
    )
    if model_preds is None or getattr(model_preds, "empty", True):
        return expected
    if "player_name" not in model_preds.columns:
        raise ValueError("active round predictions have no player_name column")
    if model_preds["player_name"].duplicated().any():
        raise ValueError("active round predictions contain duplicate player names")
    if "course_score_adj" not in model_preds.columns:
        return expected

    indexed = model_preds.set_index("player_name")
    missing = [player for player in player_names if player not in indexed.index]
    if missing:
        raise ValueError(
            f"active round predictions are missing {len(missing)} players: {missing[:5]}"
        )
    course_expected = pd.to_numeric(
        indexed.loc[player_names, "course_score_adj"], errors="coerce"
    ).to_numpy(dtype=float)
    available = np.isfinite(course_expected)
    expected[available] = course_expected[available]
    return expected


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
    centered_live_r4 = False
    live_r4_advantage = None
    live_r4_weather = None
    player_expected_r4 = None
    player_expected_active = _active_round_expected_scores(
        player_names, model_preds, tournament_config
    )

    # After R3, the live prediction file is authoritative for R4. It contains
    # zero-mean player advantages (skill + relative tee-time weather), while
    # the Sheet supplies the absolute active-field scoring expectation.
    if (completed_round == 3 and model_preds is not None
            and "scores_r4" in model_preds.columns
            and "centering_version" in model_preds.columns):
        versions = set(model_preds["centering_version"].dropna())
        if versions != {CENTERING_VERSION}:
            raise ValueError(f"Unsupported R4 centering versions: {versions}")
        groups = set(
            model_preds.get("centering_group", pd.Series(["field"])).dropna()
        )
        if len(groups) != 1:
            raise ValueError(f"Inconsistent R4 centering groups: {groups}")
        group_col = next(iter(groups))
        if group_col == "field":
            group_col = None
        validate_field_relative_predictions(
            model_preds,
            skill_col="my_pred4",
            score_col="scores_r4",
            weather_col="weather_sg_r4",
            group_col=group_col,
        )
        if model_preds["player_name"].duplicated().any():
            raise ValueError("R4 predictions contain duplicate player names")

        indexed_r4 = model_preds.set_index("player_name")
        missing_r4 = [p for p in player_names if p not in indexed_r4.index]
        if missing_r4:
            raise ValueError(
                f"R4 predictions are missing {len(missing_r4)} active players: "
                f"{missing_r4[:5]}"
            )
        live_r4_advantage = pd.to_numeric(
            indexed_r4.loc[player_names, "scores_r4"], errors="coerce"
        ).to_numpy(dtype=float)
        if np.isnan(live_r4_advantage).any():
            raise ValueError("R4 predictions contain missing centered advantages")

        if "weather_sg_r4" in indexed_r4.columns:
            live_r4_weather = pd.to_numeric(
                indexed_r4.loc[player_names, "weather_sg_r4"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
        else:
            live_r4_weather = np.zeros(n_players, dtype=float)

        player_expected_r4 = player_expected_active.copy()

        centered_live_r4 = True
        print(
            f"    Centered live R4 input: player mean="
            f"{live_r4_advantage.mean():+.8f}, field avg="
            f"{player_expected_r4.mean():.3f}"
        )

    # Get base predictions for each player
    my_pred_base = np.array([player_preds_base.get(p, 0.0) for p in player_names])

    # Per-player expected score for R2 (multi-course aware)
    player_expected_r2 = (
        player_expected_active.copy()
        if completed_round == 1
        else np.full(n_players, default_par, dtype=float)
    )
    if completed_round == 1:
        if np.unique(player_expected_r2).size > 1:
            print(f"    Multi-course R2: expected scores = {dict(zip(*np.unique(player_expected_r2, return_counts=True)))}")

    # ─── Fixture dump hook (HANDOFF step 2: round_sim production verification) ───
    # SIMS_DUMP_FIXTURE=<path.npz> freezes the fully-assembled kernel inputs so the
    # Rust kernel and the Python cascade can be compared on IDENTICAL real arrays.
    _dump_path = os.getenv("SIMS_DUMP_FIXTURE")
    if _dump_path:
        _kw = {}
        for _r in (1, 2, 3):
            if completed_round >= _r and _r in known_strokes:
                _kw[f"known_strokes_r{_r}"] = np.asarray(known_strokes[_r], dtype=np.int64)
                _kw[f"known_cats_r{_r}"] = np.asarray(
                    known_categories.get(_r, np.zeros((n_players, 4))), dtype=float)
        np.savez_compressed(
            _dump_path,
            completed_round=completed_round, default_par=default_par,
            mu=np.stack([m for (m, s) in player_cf_params]),
            std=np.stack([s for (m, s) in player_cf_params]),
            eff_skew=np.asarray(effective_skew, dtype=float),
            l_corr=np.asarray(L_corr, dtype=float),
            my_pred_base=np.asarray(my_pred_base, dtype=float),
            expected_r2=np.asarray(player_expected_r2, dtype=float),
            cut_line=CUT_LINE, use_10_shot_rule=USE_10_SHOT_RULE, num_sims=num_sims,
            player_names=np.asarray(player_names, dtype=object),
            **_kw,
        )
        print(f"  [fixture] dumped kernel inputs -> {_dump_path}")

    # ─── Rust kernel (default; --use-python forces the legacy Python cascade) ───
    # All inputs are assembled above; sims_kernel.run_remaining_rounds (seed 42)
    # returns the same (final_scores, made_cut_mask). On --use-python or any Rust
    # error we fall through to the Python cascade below.
    # The current Rust ABI accepts an absolute baseline only for R2. Until its
    # R3/R4 inputs are extended, using it for a non-par active R3/R4 would
    # silently produce different outright math than the round-score engine.
    rust_supports_active_baseline = (
        completed_round <= 1
        or np.allclose(player_expected_active, float(default_par), atol=1e-12)
    )
    if not _USE_PYTHON and not centered_live_r4 and rust_supports_active_baseline:
        try:
            import sims_kernel as _sk
            _A = np.ascontiguousarray
            _cv1 = lambda c: [c['ott'], c['putt'], c['residual'], c['residual2']]
            _cv2 = lambda c: [c['residual'], c['residual2'], c['residual3'], c['avg_ott'],
                              c['avg_putt'], c['avg_app'], c['avg_arg'], c['delta_app']]
            _cv3 = lambda c: [c['sg_ott_avg'], c['sg_putt_avg'], c['sg_app_avg'], c['sg_arg_avg'],
                              c.get('pos_6_10', 0.0)]
            _mu = np.stack([m for (m, s) in player_cf_params])
            _std = np.stack([s for (m, s) in player_cf_params])

            def _ks(r):
                return _A(known_strokes[r].astype(np.int64)) if (completed_round >= r and r in known_strokes) else None

            def _kc(r):
                if completed_round >= r and r in known_strokes:
                    return _A(np.asarray(known_categories.get(r, np.zeros((n_players, 4))), dtype=float))
                return None

            _fs, _mc, _win, _r2, _r3 = _sk.run_remaining_rounds(
                int(completed_round), float(default_par),
                _A(_mu), _A(_std), _A(effective_skew), _A(L_corr),
                _A(my_pred_base.astype(float)), _A(player_expected_r2.astype(float)),
                _ks(1), _kc(1), _ks(2), _kc(2), _ks(3), _kc(3),
                _cv1(coefficients_r1_high), _cv1(coefficients_r1_midh),
                _cv1(coefficients_r1_midl), _cv1(coefficients_r1_low),
                _cv2(coefficients_r2), _cv2(coefficients_r2_6_30), _cv2(coefficients_r2_30_up),
                _cv3(coefficients_r3), _cv3(coefficients_r3_mid), _cv3(coefficients_r3_high),
                int(CUT_LINE), bool(USE_10_SHOT_RULE), int(num_sims), 42,
            )
            print(f"  [rust] tournament cascade via sims_kernel.run_remaining_rounds "
                  f"(completed_round={completed_round}, {num_sims:,} sims)")
            return (np.ascontiguousarray(_fs).astype(int), np.ascontiguousarray(_mc),
                    np.ascontiguousarray(_r2).astype(int), np.ascontiguousarray(_r3).astype(int))
        except Exception as _rust_err:
            _handle_rust_kernel_failure("run_remaining_rounds", _rust_err)
    elif centered_live_r4 and not _USE_PYTHON:
        print("  [rust] Bypassing legacy cascade so centered live R4 inputs are honored")
    elif not rust_supports_active_baseline and not _USE_PYTHON:
        print(
            "  [rust] Bypassing legacy cascade so the active decimal R3/R4 "
            "scoring baseline is honored"
        )

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

    # Field skill (mean base pred) added to the R1 residual to match live_stats
    # _residuals_r1 (sg_total_adj = sg_total + pred_avg) and new_sim's cascade.
    field_skill = float(np.mean(my_pred_base))
    resid_r1 = sg_r1_actual + field_skill - my_pred_base[:, None]
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
    # User risk rule 2026-08: residual adjustments capped at +/-0.5 everywhere
    # (former -0.75 band retired); parity with live_stats_engine / Rust kernel.
    tot_resid_adj_r1 = np.maximum(
        np.minimum(np.where(mask_bad, 0.2, tot_resid_adj_r1), 0.5), -0.5)

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
        # Real cut already happened: load_known_rounds filtered the field to
        # active players (tee-time based), so everyone here made the cut and
        # the all-True mask is correct. The Rust kernel assumes the same.
        pass

    # R2 -> R3 skill update. Residual capped at +6 before the cubic: beyond
    # training support it explodes positive (parity: live_stats_engine.py
    # RESID_FIX_CAP and the Rust kernel).
    resid_r2 = np.minimum(sg_r2 - updated_skill_r2, 6.0)
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
    # +/-0.5 clip (user risk rule 2026-08): parity with live_stats_engine / Rust
    tot_resid_adj_r2 = np.minimum(np.maximum(
        ensure_array(adj_sum.get('residual_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('residual2_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('residual3_adj', 0.0), shape2),
        -0.5,
    ), 0.5)
    tot_sg_adj_r2 = (
        ensure_array(adj_sum.get('avg_ott_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('avg_putt_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('avg_app_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('avg_arg_adj', 0.0), shape2) +
        ensure_array(adj_sum.get('delta_app_adj', 0.0), shape2)
    )

    # R2's fresh adjustments REPLACE R1's entirely — sg AND residual (R1's
    # residual predicts nothing beyond R2; horizon regression 2026-08). Parity
    # with live_stats_engine reset-to-base: updated_skill_r3 = base + R2 fresh.
    total_adjustment_r2 = (tot_resid_adj_r2 + tot_sg_adj_r2) - ensure_array(total_adjustment_r1, shape2)
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
        expected_r3 = (
            player_expected_active
            if completed_round == 2
            else np.full(n_players, default_par, dtype=float)
        )
        strokes_r3 = np.clip(
            np.rint(expected_r3[:, None] - sg_r3),
            (expected_r3 - 12)[:, None],
            (expected_r3 + 12)[:, None],
        ).astype(int)

    r1_r3_scores = r1_r2_scores + strokes_r3

    # R3 -> R4 skill update (SG-only, no residual)
    avg_ott_r3 = 0.66 * avg_ott_r2 + 0.34 * cats_r3[:, :, 0]
    avg_app_r3 = 0.66 * avg_app_r2 + 0.34 * cats_r3[:, :, 1]
    avg_arg_r3 = 0.66 * avg_arg_r2 + 0.34 * cats_r3[:, :, 2]
    avg_putt_r3 = 0.66 * avg_putt_r2 + 0.34 * cats_r3[:, :, 3]

    pos_lt_6_mask_r3 = np.zeros((n_players, num_sims), dtype=bool)
    pos_6_20_mask_r3 = np.zeros((n_players, num_sims), dtype=bool)
    pos_gt_20_mask_r3 = np.zeros((n_players, num_sims), dtype=bool)
    pos_6_10_mask_r3 = np.zeros((n_players, num_sims), dtype=bool)

    for j in range(num_sims):
        pos = rank_positions_from_strokes(r1_r3_scores[:, j])
        pos_lt_6_mask_r3[:, j] = (pos < 6)
        pos_6_20_mask_r3[:, j] = (pos >= 6) & (pos <= 20)
        pos_gt_20_mask_r3[:, j] = (pos > 20)
        pos_6_10_mask_r3[:, j] = (pos >= 6) & (pos <= 10)

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
    # pos_6_10 LEVEL term (parity with the Rust kernel): positions 6-10 into R4 only
    tot_sg_adj_r3 = tot_sg_adj_r3 + np.where(
        pos_6_10_mask_r3, coefficients_r3_mid.get('pos_6_10', 0.0), 0.0
    )

    # Undo R2 adjustments, apply R3 adjustments
    updated_skill_r4 = updated_skill_r3 - (tot_sg_adj_r2 + tot_resid_adj_r2) + tot_sg_adj_r3

    # R4 simulation — category-first draws, per-sim-path skill mean
    cats_r4 = np.empty((n_players, num_sims, 4), dtype=float)
    sg_r4 = np.empty((n_players, num_sims), dtype=float)
    if centered_live_r4:
        for i, (mu, std_c) in enumerate(player_cf_params):
            pure_skill = live_r4_advantage[i] - live_r4_weather[i]
            shift = (pure_skill - mu.sum()) / 4.0
            cat_mu = mu + shift + live_r4_weather[i] * WEATHER_CAT_SPLIT
            Z = RNG_CF.standard_normal(size=(num_sims, 4))
            corr_z = Z @ L_corr.T
            for j in range(4):
                corr_z[:, j] = _apply_skew(
                    corr_z[:, j], effective_skew[i, j]
                )
            cats_r4[i] = np.clip(
                cat_mu + corr_z * std_c, CLIP_CAT[0], CLIP_CAT[1]
            )
            sg_r4[i] = cats_r4[i].sum(axis=1)
        strokes_r4 = np.clip(
            np.rint(player_expected_r4[:, None] - sg_r4),
            (player_expected_r4 - 12)[:, None],
            (player_expected_r4 + 12)[:, None],
        ).astype(int)
    else:
        for i, (mu, std_c) in enumerate(player_cf_params):
            cats_r4[i], sg_r4[i] = _catfirst_draw(
                mu, std_c, effective_skew[i], updated_skill_r4[i],
                L_corr, RNG, num_sims
            )
        expected_r4 = (
            player_expected_active
            if completed_round == 3
            else np.full(n_players, default_par, dtype=float)
        )
        strokes_r4 = np.clip(
            np.rint(expected_r4[:, None] - sg_r4),
            (expected_r4 - 12)[:, None],
            (expected_r4 + 12)[:, None],
        ).astype(int)

    # Missed-cut penalty
    r3_r4 = strokes_r3 + strokes_r4
    r3_r4[~made_cut_mask] = 200

    # Final 72-hole totals
    final_scores = r1_r2_scores + r3_r4

    # r1_r2_scores / r1_r3_scores are RAW cumulative standings (cut NOT applied) so
    # the caller can rank end-of-R2 / end-of-R3 for Kalshi leader/top-N markets.
    return final_scores, made_cut_mask, r1_r2_scores, r1_r3_scores


def compute_finish_probabilities(final_scores, player_names, made_cut_mask, num_sims):
    """
    Compute win and top-N probabilities from simulated 72-hole totals.

    Returns DataFrame with columns: player_name, simulated_win_prob, top_5, top_10, top_20
    """
    n_players = len(player_names)

    # ─── Rust kernel (default; --use-python forces the legacy pandas aggregation) ───
    # sims_kernel.aggregate_round replaces the 15.6M-row groupby/iterrows loop below
    # (the old 4-minute bottleneck). Validated integer-exact vs pandas by
    # validate_rust_python.py: prob_u / top_N_nodh exact, top_N dead-heat ~1e-15.
    # Win prob is the vectorized exact expectation of the random-tiebreak winner the
    # pandas loop estimated stochastically. On --use-python or any Rust error we fall
    # through to the pandas reference below.
    if not _USE_PYTHON:
        try:
            import sims_kernel as _sk
            _fs = np.ascontiguousarray(np.asarray(final_scores, dtype=np.int64))
            _prob_raw, _top_dh, _top_nodh = _sk.aggregate_round(_fs)
            _prob_raw = np.ascontiguousarray(_prob_raw)
            _top_dh = np.ascontiguousarray(_top_dh)
            _top_nodh = np.ascontiguousarray(_top_nodh)

            _mins = _fs.min(axis=0)
            _is_min = (_fs == _mins[None, :])
            _tie_ct = _is_min.sum(axis=0)
            _win_prob = (_is_min / _tie_ct).sum(axis=1) / num_sims

            finish_probs = pd.DataFrame({
                "player_name": list(player_names),
                "simulated_win_prob": _win_prob,
                "top_5": _top_dh[:, 0], "top_10": _top_dh[:, 1], "top_20": _top_dh[:, 2],
                "top_5_nodh": _top_nodh[:, 0], "top_10_nodh": _top_nodh[:, 1],
                "top_20_nodh": _top_nodh[:, 2],
            })
            _rows, _cols = np.nonzero(_prob_raw)
            rank_probs = pd.DataFrame({
                "player_name": np.asarray(player_names)[_rows],
                "rank": (_cols + 1).astype(int),
                "prob_u": _prob_raw[_rows, _cols],
            })
            print(f"  [rust] finish probs via sims_kernel.aggregate_round "
                  f"({n_players} players x {num_sims:,} sims)")
            return finish_probs, rank_probs
        except Exception as _rust_err:
            _handle_rust_kernel_failure("aggregate_round", _rust_err)

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


def prob_to_american(p):
    """Convert a probability to fair American odds (no vig)."""
    if pd.isna(p) or p <= 0:
        return None
    if p >= 1:
        return -100
    return int(round(-100 * p / (1 - p))) if p > 0.5 else int(round(100 * (1 - p) / p))


def write_full_finish_equity(finish_probs, tourney):
    """Write COMPLETE finish-position equity for every simmed player across
    win / top-5 / top-10 / top-20 — no book merge, no edge filter.

    This is the full reference table the sparse finish_equity_live file is not:
    that one inner-joins to posted book odds and keeps only edges > threshold, so
    it thins out (especially post-cut). Here we dump straight from finish_probs.

    Two files mirror the per-book dead-heat split that already exists in
    finish_probs: the 'dh' file uses the dead-heat-adjusted top-N columns; the
    'nodh' file uses the *_nodh columns (Kalshi / NoVig pay on raw finish, no DH).
    'win' is dead-heat-agnostic, so it appears in both. Returns paths written.
    """
    if finish_probs is None or finish_probs.empty:
        return []

    variants = [
        ("dh",   {"win": "simulated_win_prob", "top_5": "top_5",
                  "top_10": "top_10", "top_20": "top_20"}),
        ("nodh", {"win": "simulated_win_prob", "top_5": "top_5_nodh",
                  "top_10": "top_10_nodh", "top_20": "top_20_nodh"}),
    ]
    paths = []
    for tag, colmap in variants:
        missing = [src for src in colmap.values() if src not in finish_probs.columns]
        if missing:
            print(f"    [full-equity] skipping {tag}: missing columns {missing}")
            continue
        out = pd.DataFrame({"player_name": finish_probs["player_name"]})
        for market, src in colmap.items():
            p = finish_probs[src].astype(float)
            out[f"{market}_prob"] = p
            out[f"{market}_american"] = p.apply(prob_to_american)
        out = out.sort_values("win_prob", ascending=False).reset_index(drop=True)
        path = f"finish_equity_full_{tag}_{tourney}.csv"
        out.to_csv(path, index=False)
        paths.append(path)
        print(f"    Saved {path} ({len(out)} players)")
    return paths


def write_live_finish_equity(outrights_combined, tourney):
    """Persist the final live finish board after every book is merged.

    DataGolf books are priced first, then Kalshi and NoVig taker rows are
    appended.  Keeping this write in one finalization step prevents the
    dashboard CSV from capturing only the pre-exchange snapshot.
    """
    if outrights_combined is None or outrights_combined.empty:
        return None

    path = f"finish_equity_live_{tourney}.csv"
    outrights_combined.to_csv(path, index=False)
    books = (
        outrights_combined["bookmaker"]
        .astype(str)
        .str.lower()
        .value_counts()
        .sort_index()
        .to_dict()
        if "bookmaker" in outrights_combined.columns
        else {}
    )
    print(f"    Saved {path} ({len(outrights_combined)} rows; books={books})")
    return path


def format_units(stake_dollars):
    """Format a $ stake as units (1u = $200). Returns '—' for non-positive.
    Sub-0.3u stakes show 2 decimals so 0.05u doesn't render as '0.1u'."""
    if pd.isna(stake_dollars) or not stake_dollars or stake_dollars <= 0:
        return "—"
    u = float(stake_dollars) / 200.0
    return f"{u:.2f}u" if u < 0.3 else f"{u:.1f}u"


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


def _kalshi_outright_player(title):
    """Extract a player from legacy and current Kalshi outright titles."""
    import re

    value = str(title or "").strip()
    match = re.match(
        r".*:\s*Will (.+?) (?:finish|make|miss|lead|win)", value, re.I
    )
    if match:
        return match.group(1).strip()
    match = re.match(r"Will (.+?) win the ", value, re.I)
    if match:
        return match.group(1).strip()
    # Current top-N shape (August 2026): "X finishes top N".  Tournament
    # identity now lives in rules/event metadata rather than the title.
    match = re.match(
        r"^(.+?)\s+finishes?\s+(?:in\s+the\s+)?top\s+\d+"
        r"(?:\s+(?:at|in)\s+the\s+.+?)?\??$",
        value,
        re.I,
    )
    return match.group(1).strip() if match else ""


def _kalshi_outright_tournament(market):
    """Resolve tournament identity from title or Kalshi resolution metadata."""
    import re

    title = str((market or {}).get("title") or "").strip()
    match = re.search(r"(?:at|in|win) the (.+?)\?", title, re.I)
    if match:
        return match.group(1).strip()
    match = re.match(r"(.+?):\s*Will", title, re.I)
    if match:
        return match.group(1).strip()

    # Current top-N rules end with "... in the 2026 TOUR Championship,
    # then the market resolves ...".  Use the final "in the" clause so the
    # earlier "finishes in the top 5" clause cannot be mistaken for an event.
    rules = str((market or {}).get("rules_primary") or "").strip()
    resolution = re.split(
        r",\s*then the market", rules, maxsplit=1, flags=re.I
    )[0]
    lower = resolution.lower()
    for marker in (" in the ", " wins the "):
        index = lower.rfind(marker)
        if index < 0:
            continue
        candidate = resolution[index + len(marker):].strip()
        candidate = re.sub(r"^\d{4}\s+", "", candidate).strip()
        if candidate and not re.match(r"top\s+\d+", candidate, re.I):
            return candidate
    return ""


def _scope_kalshi_outright_markets(markets, configured_tourney):
    """Tag and retain Kalshi markets belonging to the configured tournament.

    Tournament metadata is read per market.  When it is absent, contracts in
    sibling series are joined by the stable suffix of ``event_ticker`` (for
    example TOP5-TOC26 and TOUR-TOC26), preventing blank-title top-N markets
    from leaking across simultaneously open tournaments.
    """
    import re
    from collections import Counter, defaultdict

    tagged = [dict(market) for market in (markets or [])]

    def event_code(market):
        ticker = str(market.get("event_ticker") or "")
        return ticker.split("-", 1)[1] if "-" in ticker else ""

    code_names = defaultdict(dict)
    for market in tagged:
        detected = _kalshi_outright_tournament(market)
        market["_kalshi_tournament"] = detected
        code = event_code(market)
        if code and detected:
            code_names[code][detected.lower()] = detected

    for market in tagged:
        if market["_kalshi_tournament"]:
            continue
        code = event_code(market)
        # A suffix proves identity only when every labelled sibling agrees.
        # Ambiguous or unseen suffixes remain unresolved and are rejected below.
        names_for_code = code_names.get(code, {})
        if code and len(names_for_code) == 1:
            market["_kalshi_tournament"] = next(iter(names_for_code.values()))

    tournament_counts = Counter(
        market["_kalshi_tournament"]
        for market in tagged
        if market["_kalshi_tournament"]
    )
    if not tournament_counts:
        return [], [], "no Kalshi market supplied resolvable tournament metadata"

    generic = {
        "the", "a", "an", "of", "at", "in", "tournament", "championship",
        "open", "classic", "invitational", "cup", "pga", "tour",
    }
    target_words = {
        word for word in re.split(r"[\s_]+", str(configured_tourney).lower())
        if word
    }
    distinct_words = target_words - generic

    def matches_target(name):
        value = name.lower()
        compact = re.sub(r"\s+", "", value)
        if distinct_words:
            return any(word in value or word in compact for word in distinct_words)
        return all(word in value or word in compact for word in target_words)

    matched = [name for name in tournament_counts if matches_target(name)]
    if matched:
        accepted = {name.lower() for name in matched}
        return (
            [
                market for market in tagged
                if market["_kalshi_tournament"].lower() in accepted
            ],
            matched,
            "",
        )

    known = ", ".join(sorted(tournament_counts))
    return (
        [],
        [],
        f"configured tournament '{configured_tourney}' does not match "
        f"open Kalshi tournament(s): {known}",
    )


def price_kalshi_outrights(finish_probs, pred_lookup, sample_lookup):
    """Price Kalshi outright markets using live API with orderbook-aware liquidity filtering.

    Two-stage approach (matches new_sim.py):
      Stage 1: Pre-filter on bid/ask presence, spread ≤ 10c, mid edge > 0.5%
      Stage 2: Fetch orderbook, walk for flat 1000-contract VWAP, gate taker rows on MIN_DEPTH=100

    Taker rows are the walked VWAP (incl. fee) for a flat 1000-contract order — only emitted
    when fillable depth ≥ MIN_DEPTH.
    Mid rows are the bid/ask midpoint (maker, no fee) — emitted for any stage-1 survivor.
    Downstream splits on `pricing` (taker → outrights pipeline; mid → maker-opportunity email).
    """
    import time as _time
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
            # Retry on 429 (back-to-back page requests throttle) + accumulate partial
            # pages, so a transient rate-limit no longer silently drops a whole series.
            data = None
            for attempt in range(4):
                resp = _client.get(f"{_KALSHI_API}/markets", params=params)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 0) or 0) or 0.5 * (attempt + 1)
                    _time.sleep(min(wait, 5.0))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            if data is None:
                print(f"  [warn] {series_ticker}: throttled (429) — returning {len(all_mkts)} partial markets")
                break
            mkts = data.get("markets", [])
            all_mkts.extend(mkts)
            cursor = data.get("cursor")
            if not cursor or len(mkts) < 200:
                break
            _time.sleep(0.05)   # inter-page courtesy sleep
        return all_mkts

    def _get_orderbook(ticker):
        # Retry on 429 (per-market orderbook reads throttle on bursts) before giving up.
        data = None
        for attempt in range(4):
            resp = _client.get(f"{_KALSHI_API}/markets/{ticker}/orderbook")
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 0) or 0) or 0.5 * (attempt + 1)
                _time.sleep(min(wait, 5.0))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        if data is None:
            return {"yes": [], "no": []}   # throttled out -> empty book
        ob = data.get("orderbook_fp", data.get("orderbook", data))   # _fp-first (safe)
        def parse_levels(raw):
            out = []
            for item in (raw or []):
                if isinstance(item, list) and len(item) == 2:
                    price = float(item[0])
                    if price > 1:        # bare-cents integer arrays -> dollars
                        price /= 100.0
                    out.append((price, int(float(item[1]))))
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

    all_markets, matched_tourneys, scope_rejection = (
        _scope_kalshi_outright_markets(all_markets, tourney)
    )
    if matched_tourneys:
        print(
            f"  Tournament: {', '.join(matched_tourneys)} "
            f"({len(all_markets)} markets, matched on tourney='{tourney}')"
        )
    elif scope_rejection:
        print(f"  [warn] Kalshi outrights rejected: {scope_rejection}")

    if not all_markets:
        return pd.DataFrame()

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

    FLAT_STAKE = 1000
    MIN_DEPTH = 300
    MIN_EDGE_STAGE1 = 0.5  # pct points on mid — loose gate

    def _walk_book(ob_levels, sim_prob, target=FLAT_STAKE):
        """Walk opposite-side levels to compute VWAP fill for a flat target order.

        Returns dict with effective_price, target, filled, depth_at_best (qty
        available at the headline price — i.e. the max take without slippage).
        """
        fill_levels = sorted([(1.0 - p, qty) for p, qty in ob_levels])
        if not fill_levels:
            return None
        best = fill_levels[0][0]
        eff_best = best + _kalshi_taker_fee(best)
        if eff_best <= 0 or eff_best >= 1 or eff_best >= sim_prob:
            return None
        depth_at_best = sum(qty for price, qty in fill_levels if price == best)
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
        return {
            "effective_price": vwap + avg_fee,
            "target": target,
            "filled": filled,
            "depth_at_best": int(depth_at_best),
        }

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

        player_raw = _kalshi_outright_player(title)
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
        yf = _walk_book(ob.get("no", []), s["sim_yes"])
        if yf and yf["filled"] >= MIN_DEPTH:
            eff = yf["effective_price"]
            rows.append({
                **base_row, "side": "yes", "pricing": "taker",
                "american_odds": implied_to_american(min(max(eff, 1e-4), 1 - 1e-4)),
                "sim_prob": s["sim_yes"], "implied_prob": eff,
                "edge": round((s["sim_yes"] - eff) * 100, 1),
                "my_fair": implied_to_american(min(max(s["sim_yes"], 1e-4), 1 - 1e-4)),
                "target": yf["target"], "filled": yf["filled"],
                "depth_at_best": yf.get("depth_at_best", 0),
            })

        # NO taker: walk YES levels (sellers of NO)
        nf = _walk_book(ob.get("yes", []), s["sim_no"])
        if nf and nf["filled"] >= MIN_DEPTH:
            eff = nf["effective_price"]
            rows.append({
                **base_row, "side": "no", "pricing": "taker",
                "american_odds": implied_to_american(min(max(eff, 1e-4), 1 - 1e-4)),
                "sim_prob": s["sim_no"], "implied_prob": eff,
                "edge": round((s["sim_no"] - eff) * 100, 1),
                "my_fair": implied_to_american(min(max(s["sim_no"], 1e-4), 1 - 1e-4)),
                "target": nf["target"], "filled": nf["filled"],
                "depth_at_best": nf.get("depth_at_best", 0),
            })

        # Maker rows (no fees): post at ask-1c, ONLY when the yes spread is tight
        # (< 4c) — matches the ancillary table's maker rule. `pricing` stays "mid"
        # so the downstream maker/taker split is unchanged.
        _spread = s["ask"] - s["bid"]
        if 0 < _spread < 0.04:
            yes_cost = s["ask"] - 0.01   # post YES at ask - 1c
            if 0 < yes_cost < 1:
                rows.append({
                    **base_row, "side": "yes", "pricing": "mid",
                    "american_odds": implied_to_american(min(max(yes_cost, 1e-4), 1 - 1e-4)),
                    "sim_prob": s["sim_yes"], "implied_prob": yes_cost,
                    "edge": round((s["sim_yes"] - yes_cost) * 100, 1),
                    "my_fair": implied_to_american(min(max(s["sim_yes"], 1e-4), 1 - 1e-4)),
                    "yes_bid": s["bid"], "yes_ask": s["ask"],
                })
            no_cost = (1 - s["bid"]) - 0.01   # post NO at its ask - 1c
            if 0 < no_cost < 1:
                rows.append({
                    **base_row, "side": "no", "pricing": "mid",
                    "american_odds": implied_to_american(min(max(no_cost, 1e-4), 1 - 1e-4)),
                    "sim_prob": s["sim_no"], "implied_prob": no_cost,
                    "edge": round((s["sim_no"] - no_cost) * 100, 1),
                    "my_fair": implied_to_american(min(max(s["sim_no"], 1e-4), 1 - 1e-4)),
                    "yes_bid": s["bid"], "yes_ask": s["ask"],
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
# Sim cache (for --sim-only / --price-only / --reprice)
# ══════════════════════════════════════════════════════════════════════════════

def save_sim_cache(sim_dict, sim_round, expected_avg, pred_lookup, wx_lookup=None,
                   health_manifest=None):
    """Persist sim_dict to parquet + JSON sidecar so --price-only / --reprice
    can skip re-running the simulation.

    Writes:
      {tourney}/sim_cache_r{N}.parquet  — players × num_sims integer scores
      {tourney}/sim_cache_r{N}_meta.json — round, expected_avg, pred_lookup, wx_lookup
    """
    import json as _json
    if (
        not health_manifest
        or (health_manifest.get("approval") or {}).get("status") != "approved"
        or not (health_manifest.get("checks") or {}).get("passed")
    ):
        raise SimulationHealthError(
            "Refusing to save a reusable sim cache without an approved health manifest"
        )
    out_dir = f"./{tourney}"
    os.makedirs(out_dir, exist_ok=True)

    players = sorted(sim_dict.keys())
    data = np.column_stack([sim_dict[p] for p in players])  # (num_sims, n_players)
    df = pd.DataFrame(data.T, index=players)  # (n_players, num_sims)
    df.index.name = "player_name"

    parquet_path = os.path.join(out_dir, f"sim_cache_r{sim_round}.parquet")
    df.to_parquet(parquet_path)

    meta = {
        "sim_round": sim_round,
        "expected_avg": expected_avg,
        "num_sims": len(next(iter(sim_dict.values()))),
        "num_players": len(players),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pred_lookup": {k: round(float(v), 4) for k, v in pred_lookup.items()},
        "wx_lookup": {k: round(float(v), 6) for k, v in (wx_lookup or {}).items()},
        "health_manifest": health_manifest,
    }
    meta_path = os.path.join(out_dir, f"sim_cache_r{sim_round}_meta.json")
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    print(f"  Saved sim cache: {parquet_path} ({len(players)} players × {meta['num_sims']:,} sims)")
    return parquet_path


def load_sim_cache(sim_round):
    """Load a previously cached sim. Returns (sim_dict, meta). Raises FileNotFoundError if absent."""
    import json as _json
    out_dir = f"./{tourney}"
    parquet_path = os.path.join(out_dir, f"sim_cache_r{sim_round}.parquet")
    meta_path = os.path.join(out_dir, f"sim_cache_r{sim_round}_meta.json")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No sim cache found at {parquet_path}. Run --sim-only first.")

    df = pd.read_parquet(parquet_path)
    sim_dict = {player: df.loc[player].values for player in df.index}

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = _json.load(f)

    print(f"  Loaded sim cache: {parquet_path} ({len(sim_dict)} players × {len(next(iter(sim_dict.values()))):,} sims)")
    print(f"  Cache saved at: {meta.get('saved_at', 'unknown')}")
    return sim_dict, meta


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
    return sim_dict, {}


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
    player_cf_params, effective_skew, L_corr = _load_catfirst_dists(
        list(model_preds["player_name"]),
        allow_player_subset=sim_round > 1,
    )

    has_course_adj = "course_score_adj" in model_preds.columns

    # ─── Rust kernel (default; --use-python forces the legacy Python loop) ───
    # Build per-(valid)player input arrays and draw via sims_kernel.run_single_round
    # (seed 789, weather split). Returns the same {player: int scores} + cat_mu_lookup.
    if not _USE_PYTHON:
        try:
            import sims_kernel as _sk
            _players, _mu, _std, _skew, _skill, _wx, _pavg = [], [], [], [], [], [], []
            for idx, (_, row) in enumerate(model_preds.iterrows()):
                player = row["player_name"]
                scores_rn = row[scores_col]
                if pd.isna(scores_rn):
                    continue
                if has_course_adj and pd.notna(row.get("course_score_adj")):
                    player_avg = row["course_score_adj"]
                else:
                    player_avg = expected_avg
                mu, std_c = player_cf_params[idx]
                wx_delta = wx_lookup.get(player, 0.0)
                _players.append(player)
                _mu.append(mu); _std.append(std_c); _skew.append(effective_skew[idx])
                _skill.append(float(scores_rn) - wx_delta); _wx.append(wx_delta)
                _pavg.append(float(player_avg))
            if _players:
                _A = np.ascontiguousarray
                _scores, _catmu = _sk.run_single_round(
                    _A(np.stack(_mu)), _A(np.stack(_std)), _A(np.stack(_skew)), _A(L_corr),
                    _A(np.asarray(_skill, dtype=float)), _A(np.asarray(_wx, dtype=float)),
                    _A(np.asarray(_pavg, dtype=float)), int(num_sims), 789,
                )
                sim_dict = {p: np.ascontiguousarray(_scores[k]).astype(int)
                            for k, p in enumerate(_players)}
                cat_mu_lookup = {p: np.ascontiguousarray(_catmu[k]).copy()
                                 for k, p in enumerate(_players)}
                print(f"  [rust] catfirst {len(sim_dict)} players × {num_sims:,} via run_single_round")
                return sim_dict, cat_mu_lookup
        except Exception as _rust_err:
            _handle_rust_kernel_failure("run_single_round", _rust_err)

    sim_dict = {}
    cat_mu_lookup = {}
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
        cat_mu_lookup[player] = cat_mu.copy()

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
    return sim_dict, cat_mu_lookup


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
        # Stash for the single aggregated summary; the per-pricer print was pure
        # post-cut noise (missed-cut players dominate) so it's dropped.
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

    from reprice_core import matchup_settlement_probabilities
    prob_p1, prob_p2, market_kind = matchup_settlement_probabilities(df)
    df["market_kind"] = market_kind

    # Edge = (prob × (decimal − 1) − (1 − prob)) × 100
    df["edge_p1"] = (prob_p1 * (df["p1_dec"] - 1) - (1 - prob_p1)) * 100
    df["edge_p2"] = (prob_p2 * (df["p2_dec"] - 1) - (1 - prob_p2)) * 100

    # Fair American odds (ties push)
    fair_p1_prob = df["my_odds_p1"].where(market_kind != "half_shot", prob_p1)
    fair_p2_prob = df["my_odds_p2"].where(market_kind != "half_shot", prob_p2)
    df["Fair_p1"] = fair_p1_prob.apply(
        lambda p: implied_to_american(p) if pd.notna(p) else None
    )
    df["Fair_p2"] = fair_p2_prob.apply(
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
    from reprice_core import actionable_matchup_mask
    quarantined = ~actionable_matchup_mask(df)
    if quarantined.any():
        print(
            f"  Quarantined {int(quarantined.sum())} ambiguous/unsupported "
            "matchup line(s) before alerts/storage"
        )
        df = df.loc[~quarantined].copy()

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

    # --- Sharp: keep each separately actionable sharp-book quote ---
    sharp = combined[combined["Bookmaker"].str.lower().isin(SHARP_BOOKS)].copy()
    from reprice_core import retain_unique_actionable_quotes
    sharp = retain_unique_actionable_quotes(sharp, player_count=2)

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
        "P1 Line", "P2 Line", "line_verified", "market_kind",
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


# Books whose 3-ball edges are trusted enough to surface in the email / Telegram
# alert. FanDuel / DraftKings 3-balls are still TRACKED and stored to the sheet —
# just not alerted on — per the "email books = kalshi/betonline/betcris" decision.
EMAIL_3BALL_BOOKS = {"kalshi", "betonline", "betcris"}


def price_3balls(tb_df, sim_dict):
    """Attach fair 3-ball probabilities — P(each player is lowest of the trio).

    The 3-ball analogue of price_matchups(). Two modes per side:
        my_pN     — dead-heat: ties for low split credit evenly (3-ball books
                    settle ties by dead-heat rules; matches the feed's "dead heat")
        my_pN_tl  — ties-loss: only a STRICT sole-low counts (books that void ties)

    Fairs come straight from the sim score arrays (sim_dict), so there's no
    dependency on round_3ball_rN.parquet or any cross-repo name normalization —
    the three players are looked up in sim_dict exactly like a pairwise matchup.
    Name mismatches are stashed in tb_df.attrs["name_mismatches"].
    """
    dh = {1: [], 2: [], 3: []}
    tl = {1: [], 2: [], 3: []}
    name_mismatches = defaultdict(set)

    for _, row in tb_df.iterrows():
        players = [row["Player 1"], row["Player 2"], row["Player 3"]]
        book = row.get("Bookmaker", "unknown")
        if any(p not in sim_dict for p in players):
            for p in players:
                if p not in sim_dict:
                    name_mismatches[p].add(book)
            for i in (1, 2, 3):
                dh[i].append(None)
                tl[i].append(None)
            continue

        arr = np.vstack([sim_dict[players[0]], sim_dict[players[1]], sim_dict[players[2]]])  # (3, n_sims)
        mn = arr.min(axis=0)
        is_min = arr == mn                        # (3, n) True where a player holds/ties the low
        ntied = is_min.sum(axis=0)                # (n,) 1 = sole low .. 3 = all tied
        dh_credit = (is_min / ntied).mean(axis=1)         # dead-heat expected credit per player
        strict = (is_min & (ntied == 1)).mean(axis=1)     # sole-low only
        for i in (1, 2, 3):
            dh[i].append(float(dh_credit[i - 1]))
            tl[i].append(float(strict[i - 1]))

    for i in (1, 2, 3):
        tb_df[f"my_p{i}"] = dh[i]
        tb_df[f"my_p{i}_tl"] = tl[i]

    if name_mismatches:
        tb_df.attrs["name_mismatches"] = dict(name_mismatches)
    return tb_df


def calculate_3ball_edges(df):
    """Per-player edges + fair odds for 3-ball rows (all books).

    3-balls settle by dead-heat, so the dead-heat fair (my_pN) drives the edge;
    my_pN_tl is kept for reference / any ties-void book. Same edge formula as
    matchups: edge = (prob*(dec-1) - (1-prob)) * 100.
    """
    df = df.dropna(subset=["my_p1", "my_p2", "my_p3"]).copy()
    for i in (1, 2, 3):
        oc = f"P{i} Odds"
        df[f"p{i}_dec"] = np.where(df[oc] > 0, df[oc] / 100 + 1, 100 / df[oc].abs() + 1)
        prob = df[f"my_p{i}"]
        df[f"edge_p{i}"] = (prob * (df[f"p{i}_dec"] - 1) - (1 - prob)) * 100
        df[f"Fair_p{i}"] = df[f"my_p{i}"].apply(
            lambda p: implied_to_american(p) if pd.notna(p) else None)
        df[f"p{i}_implied"] = df[oc].apply(
            lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None)
    return df


def build_3ball_outputs(df, sim_round, pred_lookup, sample_lookup, wx_lookup=None):
    """Filter/annotate 3-ball rows into (combined, email) frames.

    combined = all tracked books passing the same edge/sample/pred gate as
    matchups. email = the subset limited to EMAIL_3BALL_BOOKS, with only literal
    feed duplicates collapsed (FanDuel/DraftKings are tracked but never alerted).
    """
    if df is None or df.empty:
        print("  No 3-ball lines matched the active simulation field.")
        return pd.DataFrame(), pd.DataFrame()

    for i in (1, 2, 3):
        df[f"p{i}_pred"] = df[f"Player {i}"].map(pred_lookup)
        df[f"Sample_P{i}"] = df[f"Player {i}"].map(sample_lookup)
    df["Round"] = f"r{sim_round}"

    edges = df[["edge_p1", "edge_p2", "edge_p3"]]
    df["edge_on"] = edges.max(axis=1).round(1)
    best = edges.values.argmax(axis=1)  # 0/1/2 -> which player carries the edge
    players = df[["Player 1", "Player 2", "Player 3"]].values
    preds = df[["p1_pred", "p2_pred", "p3_pred"]].values
    samples = df[["Sample_P1", "Sample_P2", "Sample_P3"]].values
    idx = range(len(df))
    df["bet_on"] = [players[r, best[r]] for r in idx]
    df["pred_on"] = [preds[r, best[r]] for r in idx]
    df["sample_on"] = [samples[r, best[r]] for r in idx]

    if wx_lookup:
        df["wx_on"] = df["bet_on"].map(wx_lookup).fillna(0)

        def _wx_diff(r):
            others = [p for p in (r["Player 1"], r["Player 2"], r["Player 3"]) if p != r["bet_on"]]
            ow = [wx_lookup.get(p, 0) for p in others]
            return r["wx_on"] - (sum(ow) / len(ow) if ow else 0)

        df["wx_diff"] = df.apply(_wx_diff, axis=1)

    # --- Combined: same gate as matchups ---
    combined = df[df["edge_on"] > 3].copy()
    combined = combined[combined["sample_on"].fillna(0) >= 20]
    combined = combined[
        ((combined["pred_on"] > 0) & (combined["edge_on"] > 7))
        | (combined["pred_on"] > 1)
    ]
    combined = combined[~((combined["edge_on"] < 5) & (combined["pred_on"] < 1))]

    # --- Email: trusted books only; every distinct quote remains actionable ---
    email = combined[combined["Bookmaker"].str.lower().isin(EMAIL_3BALL_BOOKS)].copy()
    if not email.empty:
        from reprice_core import retain_unique_actionable_quotes
        email = retain_unique_actionable_quotes(email, player_count=3)

    for out in (combined, email):
        for i in (1, 2, 3):
            if f"p{i}_pred" in out.columns:
                out[f"p{i}_pred"] = out[f"p{i}_pred"].round(2)
            out[f"edge_p{i}"] = out[f"edge_p{i}"].round(1)

    display_cols = [
        "Player 1", "Player 2", "Player 3", "Round", "Bookmaker", "Ties",
        "P1 Odds", "P2 Odds", "P3 Odds", "Fair_p1", "Fair_p2", "Fair_p3",
        "edge_p1", "edge_p2", "edge_p3", "edge_on", "bet_on",
        "p1_pred", "p2_pred", "p3_pred", "pred_on",
        "Sample_P1", "Sample_P2", "Sample_P3", "sample_on",
    ]
    if wx_lookup:
        display_cols += ["wx_on", "wx_diff"]
    combined = combined[[c for c in display_cols if c in combined.columns]]
    email = email[[c for c in display_cols if c in email.columns]]

    print(f"  Combined 3-balls:  {len(combined)} rows")
    print(f"  Email 3-balls:     {len(email)} rows ({'/'.join(sorted(EMAIL_3BALL_BOOKS))})")
    return combined, email


def build_betonline_all_matchups_csv(matchup_df, sim_round, out_dir):
    """
    Extract ALL BetOnline matchup rows (no edge/sample/pred filters) and save as CSV.

    Includes book odds, fair odds, edges, pred, and sample for every matchup
    BetOnline prices — even negative edges and low-sample players.
    Sorted by edge_on descending (highest edge first).
    """
    from reprice_core import prepare_matchup_attachment_rows
    matchup_df = prepare_matchup_attachment_rows(matchup_df)
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
        "P1 Line", "P2 Line", "Spread", "line_verified", "market_kind",
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
    from reprice_core import prepare_matchup_attachment_rows
    matchup_df = prepare_matchup_attachment_rows(matchup_df)
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
        "P1 Line", "P2 Line", "Spread", "line_verified", "market_kind",
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

        settlement_scores, settlement_probs = fractional_settlement_pmf(scores)
        row = {"Player": player, "Pred": round(pred, 2)}
        for line in lines:
            # Cached score-est reprices may carry a uniform decimal offset.
            # Recover fractional probability mass from the inferred latent
            # rounding bins, matching the odds-board UI exactly. Integer and
            # whole-stroke-shifted caches retain point-mass parity.
            under_pct = float(settlement_probs[settlement_scores < line].sum())
            fair_under = implied_to_american(under_pct)
            row[str(line)] = fair_under

        rows.append(row)

    card = pd.DataFrame(rows)
    card = card.sort_values("Pred", ascending=False)
    print(f"  Score card: {len(card)} players × {len(lines)} lines ({lines[0]}–{lines[-1]})")
    return card


def build_round_score_probs(sim_dict, expected_avg_lookup, cat_mu_lookup=None):
    """
    Aggregate sim_dict into per-score probabilities for the dashboard.

    Parameters
    ----------
    sim_dict : {player_name: np.ndarray of int scores}
    expected_avg_lookup : {player_name: float} per-player expected score (course-adjusted)
                         OR a single float applied to all players.
    cat_mu_lookup : {player_name: np.ndarray([ott, app, arg, putt])} or None

    Returns
    -------
    DataFrame with columns: player_name, score, prob, sg_ott, sg_app, sg_arg,
    sg_putt, expected_avg. Rows are one-per-(player, score).
    """
    cat_mu_lookup = cat_mu_lookup or {}
    rows = []
    for player, scores in sim_dict.items():
        if scores is None or len(scores) == 0:
            continue
        # A fractional shift redistributes mass deterministically between the
        # adjacent integer settlement buckets. This is the same uniform-bin
        # approximation used by the deployed board frontend, so a 0.3-stroke
        # move changes strict half-stroke O/U fairs without Monte Carlo noise.
        vals, probs = fractional_settlement_pmf(scores)

        if isinstance(expected_avg_lookup, dict):
            p_exp = expected_avg_lookup.get(player)
        else:
            p_exp = expected_avg_lookup

        cat = cat_mu_lookup.get(player)
        sg_ott = float(cat[0]) if cat is not None else np.nan
        sg_app = float(cat[1]) if cat is not None else np.nan
        sg_arg = float(cat[2]) if cat is not None else np.nan
        sg_putt = float(cat[3]) if cat is not None else np.nan

        for s, p in zip(vals, probs):
            rows.append({
                "player_name": player,
                "score": int(s),
                "prob": float(p),
                "sg_ott": sg_ott,
                "sg_app": sg_app,
                "sg_arg": sg_arg,
                "sg_putt": sg_putt,
                "expected_avg": float(p_exp) if p_exp is not None else np.nan,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3b: Price Score Lines vs FanDuel Market Odds
# ══════════════════════════════════════════════════════════════════════════════

def load_score_lines(sim_round=None):
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

    def _round_ok(d):
        """Reject a round_scores file stamped for a different round than we price.
        The board now stamps `round` on round_scores; an unstamped legacy file passes."""
        fr = d.get("round")
        if fr is not None and sim_round is not None and int(fr) != int(sim_round):
            print(f"  Score lines file is R{fr}, not target R{sim_round} — skipping")
            return False
        return True

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
                elif _round_ok(data):
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
            if not _round_ok(data):
                continue
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
    """Find the best prediction column for display/filtering.

    Absolute skill (my_pred{N}) first: the pred_on/my_pred bet gates were
    calibrated on absolute SG. scores_r{N} is the FIELD-RELATIVE centered
    advantage post-ba777c2 (skill - field mean + relative weather) — serving
    it here loosened every >0/>0.75/>1 gate by the field-mean offset.
    """
    candidates = [
        f"my_pred{sim_round}" if sim_round > 1 else "my_pred",
        f"updated_pred_r{sim_round}",
        "updated_pred",
        "pred",
        "my_pred",
        f"scores_r{sim_round}",              # last resort: centered advantage
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
                             wx_lookup=None, score_edges=None, kalshi_mids=None,
                             ancillary_df=None, matchup_book_counts=None,
                             threeball_df=None):
    """
    Build HTML email body with a table of sharp matchup picks, finish position edges,
    and outright win edge tables (top positive + top negative).

    Filters sharp_df to rows where:
        - bet_on player's pred > EMAIL_MIN_PRED
        - bet_on player's sample > EMAIL_MIN_SAMPLE
    """
    from kalshi_ancillary import kelly_contracts  # 0.4-Kelly $50k contract sizing
    from reprice_core import selected_spread_line

    # --- Sharp-book matchup line-count banner (top of email) ---
    # Confirms how many matchup lines were priced from each sharp book so a book
    # that's offline (0) or barely posting (< threshold) is obvious at a glance.
    book_lines_html = ""
    if matchup_book_counts is not None:
        _disp = {"pinnacle": "Pinnacle", "betonline": "BetOnline", "betcris": "BetCris"}
        _cells, _alert = [], False
        for _b in SHARP_BOOKS:
            _n = int(matchup_book_counts.get(_b, 0))
            if _n == 0:
                _c, _tag, _alert = "#c0392b", " &#9888; NO LINES", True
            elif _n < MATCHUP_LINE_WARN_THRESHOLD:
                _c, _tag, _alert = "#e67e22", " &#9888; few", True
            else:
                _c, _tag = "#1e7e34", ""
            _cells.append(
                f'<span style="color:{_c}; font-weight:bold;">{_disp.get(_b, _b.title())}: {_n}{_tag}</span>'
            )
        _bg = "#fdecea" if _alert else "#eef7ee"
        _border = "#c0392b" if _alert else "#1e7e34"
        book_lines_html = (
            f'<div style="background:{_bg}; border-left:4px solid {_border}; '
            f'padding:8px 12px; margin:8px 0; font-size:13px;">'
            f'<strong>Matchup lines priced</strong> &nbsp; '
            + ' &nbsp;&middot;&nbsp; '.join(_cells)
            + '</div>'
        )

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
                try:
                    spread_line = float(selected_spread_line(row))
                except (TypeError, ValueError, OverflowError):
                    spread_line = np.nan
                edge = row["edge_on"]
                pred = row["pred_on"]
                sample = int(row["sample_on"])
                archetype = row.get("type_on", "")
                archetype_a = row.get("type_a", "")
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
                spread_str = (
                    f"{spread_line:+.1f}"
                    if pd.notna(spread_line) and abs(spread_line) > 1e-9
                    else "—"
                )

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{bet_player}</td>
                    <td style="padding:6px 10px; color:#666;">vs {opponent}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype_a}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{ties}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:600;">{spread_str}</td>
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
                    <th style="padding:6px 10px; text-align:center;">Type_a</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Ties</th>
                    <th style="padding:6px 10px; text-align:center;">Spread</th>
                    <th style="padding:6px 10px; text-align:center;">Odds</th>
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

    # Build 3-ball picks section. threeball_df is already the email-book subset
    # (kalshi/betonline/betcris); FanDuel/DraftKings are tracked but never alerted.
    threeball_html = ""
    if threeball_df is not None and not threeball_df.empty:
        f3 = threeball_df.copy()
        f3["sample_on"] = f3["bet_on"].map(sample_lookup).fillna(0)
        f3 = f3[
            (
                (f3["pred_on"] > EMAIL_MIN_PRED)
                | ((f3["pred_on"] > 0) & (f3["edge_on"] > 7))
            )
            & (f3["sample_on"] >= EMAIL_MIN_SAMPLE)
        ]
        if not f3.empty:
            f3 = f3.sort_values("edge_on", ascending=False)
            rows_html = ""
            for _, row in f3.iterrows():
                players = [row["Player 1"], row["Player 2"], row["Player 3"]]
                bet_on = row["bet_on"]
                si = players.index(bet_on) + 1 if bet_on in players else 1
                others = [p for p in players if p != bet_on]
                book = row.get("Bookmaker", "")
                book_odds = row.get(f"P{si} Odds")
                fair_odds = row.get(f"Fair_p{si}")
                edge = row["edge_on"]
                pred = row["pred_on"] if pd.notna(row["pred_on"]) else 0.0
                sample = int(row["sample_on"])
                archetype = row.get("type_on", "")
                edge_color = "#d4edda" if edge >= 7 else ("#fff3cd" if edge >= 4 else "#ffffff")
                pred_color = "#d4edda" if pred > 1.5 else "#ffffff"
                book_str = f"{int(book_odds):+d}" if pd.notna(book_odds) else ""
                fair_str = f"{int(fair_odds):+d}" if pd.notna(fair_odds) else ""
                vs_str = " / ".join(p.title() for p in others)
                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{bet_on.title()}</td>
                    <td style="padding:6px 10px; color:#666;">vs {vs_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                    <td style="padding:6px 10px; text-align:center;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{book_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred:.2f}</td>
                    <td style="padding:6px 10px; text-align:center;">{sample}</td>
                </tr>"""
            threeball_html = f"""
            <h3 style="color:#2c5282; margin:20px 0 8px 0;">
                3-Ball Picks ({'/'.join(sorted(EMAIL_3BALL_BOOKS))}; pred &gt; {EMAIL_MIN_PRED}, sample &gt; {EMAIL_MIN_SAMPLE})
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Bet On</th>
                    <th style="padding:6px 10px; text-align:left;">Group</th>
                    <th style="padding:6px 10px; text-align:center;">Type</th>
                    <th style="padding:6px 10px; text-align:center;">Book</th>
                    <th style="padding:6px 10px; text-align:center;">Line</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Sample</th>
                </tr>
                {rows_html}
            </table>"""

    # Build finish position edges section
    outrights_html = ""
    if outrights_sharp is not None and not outrights_sharp.empty:
        # Filter to positive pred players
        filtered_out = outrights_sharp[outrights_sharp['my_pred'].fillna(-1) > 0].copy()
        if not filtered_out.empty:
            # Sort by Kelly edge = (fair - cost)/cost = edge / implied_prob (the
            # 'edge' term in Kelly's f* = edge/odds), NOT the raw percentage-point
            # edge. Tilts toward high-ROI cheap contracts.
            _ip = filtered_out['implied_prob'].where(filtered_out['implied_prob'] > 0)
            filtered_out = (filtered_out.assign(_kelly_edge=filtered_out['edge'] / _ip)
                            .sort_values('_kelly_edge', ascending=False, na_position='last')
                            .drop(columns='_kelly_edge').head(20))

            rows_html = ""
            for _, row in filtered_out.iterrows():
                player = row['player_name'].title()
                market = row['market_type'].replace('_', ' ').title()
                book = str(row.get('bookmaker', '') or '')
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
                book_color = "#dd6b20" if book == "kalshi" else "#6b46c1" if book == "novig" else "#000000"

                odds_str = f"{int(odds):+d}" if pd.notna(odds) else ""
                fair_str = f"{int(fair):+d}" if pd.notna(fair) else ""
                pred_str = f"{pred:.2f}" if pd.notna(pred) else ""
                stake_str = f"${stake:.0f}" if pd.notna(stake) and stake > 0 else ""

                # Cent pricing for exchange rows (Kalshi): effective fill price incl. fee
                is_exchange = book.lower() in ("kalshi", "novig")
                imp_prob = row.get("implied_prob")
                sim_prob_row = row.get("sim_prob")
                line_c_str = f"{imp_prob * 100:.1f}&cent;" if is_exchange and pd.notna(imp_prob) else ""
                fair_c_str = f"{sim_prob_row * 100:.1f}&cent;" if is_exchange and pd.notna(sim_prob_row) else ""

                if book == "kalshi":
                    _filled = int(row.get('filled', 0) or 0)
                    fill_str = f"{_filled:,}c" if _filled else "—"
                    _depth_best = int(row.get('depth_at_best', 0) or 0)
                    top_str = f"{_depth_best:,}c" if _depth_best else "—"
                else:
                    fill_str = "—"
                    top_str = "—"

                # Kalshi/NoVig: Stake = 0.4-Kelly contracts on a $50k bankroll,
                # Units = kelly$ / $500. Non-exchange rows keep the $ stake / $200 units.
                if is_exchange and pd.notna(imp_prob) and pd.notna(sim_prob_row):
                    _contracts, _dollars, _units = kelly_contracts(sim_prob_row, imp_prob)
                    stake_str = f"{_contracts:,}c" if _contracts else "—"
                    units_str = f"{_units:.1f}u" if _units > 0 else "—"
                else:
                    _stake_dollars = float(stake) if pd.notna(stake) else 0.0
                    units_str = format_units(_stake_dollars)

                rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center;">{market}</td>
                    <td style="padding:6px 10px; text-align:center; color:{book_color}; font-weight:500;">{book}</td>
                    <td style="padding:6px 10px; text-align:center; background:{side_color}; font-weight:600;">{side_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-size:11px; color:#555;">{archetype}</td>
                    <td style="padding:6px 10px; text-align:center;">{odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; color:#2c5282;">{line_c_str}</td>
                    <td style="padding:6px 10px; text-align:center; color:#2c5282; font-weight:500;">{fair_c_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{stake_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{fill_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{top_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{units_str}</td>
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
                    <th style="padding:6px 10px; text-align:center;">Fill</th>
                    <th style="padding:6px 10px; text-align:center;">Top</th>
                    <th style="padding:6px 10px; text-align:center;">Units</th>
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
            book = str(row.get('bookmaker', '') or '')
            odds = row.get('american_odds', '')
            fair = row.get('my_fair', '')
            edge = row.get('edge', 0)
            sim_prob = row.get('simulated_win_prob', 0)
            kelly = row.get('kelly', 0)

            edge_color = "#d4edda" if edge > 10 else "#fff3cd" if edge > 5 else "#ffffff"
            book_color = "#dd6b20" if book == "kalshi" else "#6b46c1" if book == "novig" else "#000000"

            if book == "kalshi":
                _filled = int(row.get('filled', 0) or 0)
                fill_str = f"{_filled:,}c" if _filled else "—"
                _depth_best = int(row.get('depth_at_best', 0) or 0)
                top_str = f"{_depth_best:,}c" if _depth_best else "—"
            else:
                fill_str = "—"
                top_str = "—"

            odds_str = f"{int(odds):+d}" if pd.notna(odds) else ""
            fair_str = f"{int(fair):+d}" if pd.notna(fair) else ""
            sim_str = f"{sim_prob*100:.2f}%" if pd.notna(sim_prob) else ""
            kelly_str = f"${kelly:.0f}" if pd.notna(kelly) and kelly > 0 else "$0"
            units_str = format_units(kelly)

            rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center; color:{book_color}; font-weight:500;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center;">{sim_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{kelly_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{fill_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{top_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{units_str}</td>
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
                    <th style="padding:6px 10px; text-align:center;">Fill</th>
                    <th style="padding:6px 10px; text-align:center;">Top</th>
                    <th style="padding:6px 10px; text-align:center;">Units</th>
                </tr>
                {rows_html}
            </table>"""

    # Build outright win — top negative edges (fades) table
    win_neg_html = ""
    if win_negative_top10 is not None and not win_negative_top10.empty:
        rows_html = ""
        for _, row in win_negative_top10.iterrows():
            player = row['player_name'].title()
            book = str(row.get('bookmaker', '') or '')
            odds = row.get('american_odds', '')
            fair = row.get('my_fair', '')
            edge = row.get('edge', 0)
            sim_prob = row.get('simulated_win_prob', 0)
            implied = row.get('implied_prob', 0)

            edge_color = "#f8d7da" if edge < -10 else "#fff3cd" if edge < -5 else "#ffffff"
            book_color = "#dd6b20" if book == "kalshi" else "#6b46c1" if book == "novig" else "#000000"

            if book == "kalshi":
                _filled = int(row.get('filled', 0) or 0)
                fill_str = f"{_filled:,}c" if _filled else "—"
                _depth_best = int(row.get('depth_at_best', 0) or 0)
                top_str = f"{_depth_best:,}c" if _depth_best else "—"
            else:
                fill_str = "—"
                top_str = "—"

            odds_str = f"{int(odds):+d}" if pd.notna(odds) else ""
            fair_str = f"{int(fair):+d}" if pd.notna(fair) else ""
            sim_str = f"{sim_prob*100:.2f}%" if pd.notna(sim_prob) else ""
            impl_str = f"{implied*100:.2f}%" if pd.notna(implied) else ""

            rows_html += f"""
                <tr>
                    <td style="padding:6px 10px; font-weight:600;">{player}</td>
                    <td style="padding:6px 10px; text-align:center; color:{book_color}; font-weight:500;">{book}</td>
                    <td style="padding:6px 10px; text-align:center;">{odds_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
                    <td style="padding:6px 10px; text-align:center;">{sim_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{impl_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{fill_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{top_str}</td>
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
                    <th style="padding:6px 10px; text-align:center;">Fill</th>
                    <th style="padding:6px 10px; text-align:center;">Top</th>
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

                # Cent pricing (Kalshi maker = post at ask-1c, no fees)
                imp_prob = row.get("implied_prob")
                sim_prob_row = row.get("sim_prob")
                line_c_str = f"{imp_prob * 100:.1f}&cent;" if pd.notna(imp_prob) else ""
                fair_c_str = f"{sim_prob_row * 100:.1f}&cent;" if pd.notna(sim_prob_row) else ""
                _mk_contracts, _mk_dollars, _mk_units = kelly_contracts(sim_prob_row, imp_prob)
                mk_stake_str = f"{_mk_contracts:,}c" if _mk_contracts else "—"

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
                    <td style="padding:6px 10px; text-align:center; font-weight:600;">{mk_stake_str}</td>
                    <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred_str}</td>
                    <td style="padding:6px 10px; text-align:center;">{wx_str}</td>
                </tr>"""

            kalshi_mids_html = f"""
            <h3 style="color:#2c5282; margin:30px 0 8px 0;">
                Kalshi Maker Opportunities (Post at Ask&minus;1&cent;, spread &lt; 4&cent;, No Fees)
            </h3>
            <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
                <tr style="background:#343a40; color:white;">
                    <th style="padding:6px 10px; text-align:left;">Player</th>
                    <th style="padding:6px 10px; text-align:center;">Market</th>
                    <th style="padding:6px 10px; text-align:center;">Side</th>
                    <th style="padding:6px 10px; text-align:center;">Post</th>
                    <th style="padding:6px 10px; text-align:center;">Fair</th>
                    <th style="padding:6px 10px; text-align:center;">Post&cent;</th>
                    <th style="padding:6px 10px; text-align:center;">Fair&cent;</th>
                    <th style="padding:6px 10px; text-align:center;">Edge</th>
                    <th style="padding:6px 10px; text-align:center;">Stake</th>
                    <th style="padding:6px 10px; text-align:center;">Pred</th>
                    <th style="padding:6px 10px; text-align:center;">Wx SG</th>
                </tr>
                {rows_html}
            </table>"""

    # ── Ancillary Kalshi edges (round leaders/top-N, playoff) + round H2H ──
    # Round head-to-head (KXPGAH2H) gets its OWN table; everything else stays in
    # the general ancillary table.
    ancillary_html = ""
    h2h_html = ""
    anc_df = h2h_df = None
    if ancillary_df is not None and not ancillary_df.empty:
        _mt = ancillary_df["market_type"].astype(str)
        h2h_df = ancillary_df[_mt == "h2h"]
        anc_df = ancillary_df[_mt != "h2h"]

    if anc_df is not None and not anc_df.empty:
        anc_rows = ""
        for _, r in anc_df.iterrows():
            fill = "" if pd.isna(r.get("fill")) else int(r["fill"])
            pricing = str(r["pricing"])
            side = str(r["side"]).upper()
            _stake = int(r.get("stake", 0) or 0)
            stake_cell = f"{_stake:,}c" if _stake else "—"
            _opp = r.get("opponents")
            _opp = "" if (_opp is None or (isinstance(_opp, float) and pd.isna(_opp))) else str(_opp)
            anc_rows += (
                "<tr>"
                f"<td style='padding:4px 10px;'>{r['market_type']}</td>"
                f"<td style='padding:4px 10px;'>{str(r['player_name']).title()}</td>"
                f"<td style='padding:4px 10px; color:#666; font-size:11px;'>{_opp}</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{side}</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{pricing}</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{r['fair_prob']*100:.1f}%</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{r['cost']*100:.1f}&cent;</td>"
                f"<td style='padding:4px 10px; text-align:center; color:#0a7;'>+{r['edge_pp']:.1f}</td>"
                f"<td style='padding:4px 10px; text-align:center; font-weight:600;'>{stake_cell}</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{fill}</td>"
                "</tr>"
            )
        ancillary_html = f"""
        <h3 style="margin-bottom:4px;">Ancillary Kalshi Edges ({len(anc_df)})</h3>
        <p style="color:#666; font-size:11px; margin-top:0;">
            Taker = can fill 300 @ ask + fee and still +edge |
            Maker = post at ask&minus;1&cent; (no fee), yes spread &lt; 4&cent;
        </p>
        <table style="border-collapse:collapse; font-size:13px; margin-bottom:20px;">
            <tr style="background:#f0f0f0;">
                <th style="padding:6px 10px; text-align:left;">Market</th>
                <th style="padding:6px 10px; text-align:left;">Player</th>
                <th style="padding:6px 10px; text-align:left;">Opponents</th>
                <th style="padding:6px 10px;">Side</th>
                <th style="padding:6px 10px;">Pricing</th>
                <th style="padding:6px 10px;">Fair</th>
                <th style="padding:6px 10px;">Cost</th>
                <th style="padding:6px 10px;">Edge</th>
                <th style="padding:6px 10px;">Stake</th>
                <th style="padding:6px 10px;">Fill</th>
            </tr>
            {anc_rows}
        </table>"""

    if h2h_df is not None and not h2h_df.empty:
        h2h_rows = ""
        for _, r in h2h_df.iterrows():
            fill = "" if pd.isna(r.get("fill")) else int(r["fill"])
            pricing = str(r["pricing"])
            side = str(r["side"]).upper()
            _stake = int(r.get("stake", 0) or 0)
            stake_cell = f"{_stake:,}c" if _stake else "—"
            _opp = r.get("opponents")
            _opp = "" if (_opp is None or (isinstance(_opp, float) and pd.isna(_opp))) else str(_opp)
            h2h_rows += (
                "<tr>"
                f"<td style='padding:4px 10px;'>{str(r['player_name']).title()}</td>"
                f"<td style='padding:4px 10px;'>{_opp}</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{side}</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{pricing}</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{r['fair_prob']*100:.1f}%</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{r['cost']*100:.1f}&cent;</td>"
                f"<td style='padding:4px 10px; text-align:center; color:#0a7;'>+{r['edge_pp']:.1f}</td>"
                f"<td style='padding:4px 10px; text-align:center; font-weight:600;'>{stake_cell}</td>"
                f"<td style='padding:4px 10px; text-align:center;'>{fill}</td>"
                "</tr>"
            )
        h2h_html = f"""
        <h3 style="margin-bottom:4px;">Kalshi Round Matchups — R{sim_round} H2H ({len(h2h_df)})</h3>
        <p style="color:#666; font-size:11px; margin-top:0;">
            Priced off the single-round score sim (P(beats opponent in R{sim_round}), ties pushed) —
            not tournament-finish odds. Taker = fill 300 @ ask + fee and still +edge |
            Maker = post at ask&minus;1&cent; (no fee), yes spread &lt; 4&cent;
        </p>
        <table style="border-collapse:collapse; font-size:13px; margin-bottom:20px;">
            <tr style="background:#f0f0f0;">
                <th style="padding:6px 10px; text-align:left;">Player</th>
                <th style="padding:6px 10px; text-align:left;">Opponent</th>
                <th style="padding:6px 10px;">Side</th>
                <th style="padding:6px 10px;">Pricing</th>
                <th style="padding:6px 10px;">Fair</th>
                <th style="padding:6px 10px;">Cost</th>
                <th style="padding:6px 10px;">Edge</th>
                <th style="padding:6px 10px;">Stake</th>
                <th style="padding:6px 10px;">Fill</th>
            </tr>
            {h2h_rows}
        </table>"""

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:960px; margin:0 auto; padding:20px;">
        <h2 style="margin-bottom:4px;">R{sim_round} Round Sim - {tourney.replace('_', ' ').title()}</h2>
        <p style="color:#666; margin-top:0;">{datetime.now().strftime('%B %d, %Y %I:%M %p')}</p>

        {book_lines_html}

        {matchups_html}

        {threeball_html}

        {score_edges_html}

        {outrights_html}

        {kalshi_mids_html}

        {ancillary_html}

        {h2h_html}

        {win_pos_html}

        {win_neg_html}

        <p style="color:#999; font-size:11px; margin-top:30px;">
            Fair = our no-vig price | Edge = expected return % |
            Wx = weather SG advantage vs opponent (positive = favorable) |
            Pred = model SG prediction | Stake = suggested Kelly stake
        </p>
        <p style="color:#999; font-size:11px;">
            Attachments: fair score card (CSV), full matchup workbook (XLSX),
            full finish equity — dead-heat &amp; no-dead-heat (CSV)
        </p>
    </body>
    </html>"""

    return html


def send_round_sim_email(sharp_df, sim_round, sample_lookup,
                         excel_path=None, card_csv_path=None, outrights_sharp=None,
                         win_edges_csv_path=None, bol_matchups_csv_path=None,
                         all_books_csv_path=None,
                         finish_equity_csv_path=None,
                         finish_equity_full_paths=None,
                         win_positive_top10=None, win_negative_top10=None,
                         wx_lookup=None, score_edges=None, kalshi_mids=None,
                         ancillary_df=None, ancillary_csv_path=None,
                         matchup_book_counts=None, threeball_df=None,
                         required=False):
    """
    Send round sim email with:
        - HTML body: filtered sharp matchup table + finish position edges
                     + outright win positive/negative edge tables
        - Attachment 1: full matchup + score card Excel workbook
        - Attachment 2: BetOnline all matchups CSV (unfiltered)

    Returns True only after SMTP accepts the message.  Interactive runs remain
    best-effort by default; production automation passes ``required=True`` so a
    missing/failed report stops the run before bet storage.
    """
    password = os.getenv("EMAIL_PASSWORD")
    recipients = [str(address).strip() for address in EMAIL_TO if str(address).strip()]
    if not password or not EMAIL_FROM or not recipients:
        message = "Round sim email credentials or recipients are not configured"
        if required:
            raise EmailDeliveryError(message)
        print(f"  Warning: {message}. Skipping email.")
        return False

    try:
        html = build_matchup_email_html(sharp_df, sim_round, sample_lookup, outrights_sharp,
                                        win_positive_top10=win_positive_top10,
                                        win_negative_top10=win_negative_top10,
                                        wx_lookup=wx_lookup,
                                        score_edges=score_edges,
                                        kalshi_mids=kalshi_mids,
                                        ancillary_df=ancillary_df,
                                        matchup_book_counts=matchup_book_counts,
                                        threeball_df=threeball_df)

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"R{sim_round} Round Sim — {tourney.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(recipients)

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

        # Attach ancillary Kalshi edges CSV
        if ancillary_csv_path and os.path.exists(ancillary_csv_path):
            with open(ancillary_csv_path, "rb") as f:
                att = MIMEApplication(f.read(), _subtype="csv")
                att.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(ancillary_csv_path),
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

        # Attach full finish-position equity CSVs (every player, all markets;
        # dead-heat + no-dead-heat variants)
        for full_path in (finish_equity_full_paths or []):
            if full_path and os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    att = MIMEApplication(f.read(), _subtype="csv")
                    att.add_header(
                        "Content-Disposition", "attachment",
                        filename=os.path.basename(full_path),
                    )
                    msg.attach(att)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, password)
            server.sendmail(EMAIL_FROM, recipients, msg.as_string())

        print("  Round sim email sent")
        return True

    except Exception as e:
        if required:
            raise EmailDeliveryError("Round sim email delivery failed") from e
        print(f"  Warning: Email failed: {e}")
        print("    (Sim outputs still saved — email is non-blocking)")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Reprice: dedup + store + alert helpers (for --reprice mode)
# ══════════════════════════════════════════════════════════════════════════════

def _canonical_mu_key(p1, p2, book, o1, o2, bet_on):
    """Compatibility wrapper around the cache-free canonical bet identity."""
    from reprice_core import _canonical_mu_key as canonical_mu_key
    return canonical_mu_key(p1, p2, book, o1, o2, bet_on)


def _dedup_round_matchups(combined, spreadsheet, event_id, sim_round):
    """Use the hardened cache-free dedup contract for legacy ``--reprice``."""
    from reprice_core import dedup_round_matchups
    return dedup_round_matchups(combined, spreadsheet, event_id, sim_round)


def _dedup_score_edges(score_edges, spreadsheet, event_id, sim_round):
    """Return rows from `score_edges` that aren't already in the Score Edges sheet."""
    if score_edges is None or score_edges.empty:
        return score_edges

    from sheets_storage import (
        TAB_SCORE_EDGES,
        SCORE_EDGES_HEADERS,
        _get_or_create_tab,
        _norm_key_cell,
        is_excluded_or_invalid_result,
    )
    ws = _get_or_create_tab(spreadsheet, TAB_SCORE_EDGES, SCORE_EDGES_HEADERS)
    existing = ws.get_all_records()

    existing_keys = set()
    for row in existing:
        if is_excluded_or_invalid_result(row.get("result", "")):
            continue
        if str(row.get("event_id", "")) == str(event_id) and str(row.get("round", "")) == str(sim_round):
            key = (
                _norm_key_cell(row.get("player", "")),
                _norm_key_cell(row.get("line", "")),
                _norm_key_cell(row.get("book", "")),
                _norm_key_cell(row.get("best_side", "")),
                _norm_key_cell(row.get("mkt_under", "")),
                _norm_key_cell(row.get("mkt_over", "")),
            )
            existing_keys.add(key)

    if not existing_keys:
        return score_edges

    mask = []
    for _, r in score_edges.iterrows():
        key = (
            _norm_key_cell(r.get("Player", "")),
            _norm_key_cell(r.get("Line", "")),
            _norm_key_cell(r.get("Book", "")),
            _norm_key_cell(r.get("Best_Side", "")),
            _norm_key_cell(r.get("Mkt_Under", "")),
            _norm_key_cell(r.get("Mkt_Over", "")),
        )
        mask.append(key not in existing_keys)

    new_rows = score_edges[mask].copy()
    print(f"  [reprice] Score edges: {len(score_edges)} total, {len(new_rows)} new (deduped {len(score_edges) - len(new_rows)})")
    return new_rows


def _dedup_finish_positions(outrights_df, spreadsheet, event_id):
    """Return rows from `outrights_df` not already in the Finish Positions sheet.

    Dedup key: (player_name, market_type, sportsbook, american_odds) for this
    event. Finish Positions are tournament-level (no round column), so the
    dedup is across the whole event.
    """
    if outrights_df is None or outrights_df.empty:
        return outrights_df

    from sheets_storage import (
        TAB_FINISH_POS,
        FINISH_POS_HEADERS,
        _get_or_create_tab,
        is_excluded_or_invalid_result,
    )
    ws = _get_or_create_tab(spreadsheet, TAB_FINISH_POS, FINISH_POS_HEADERS)
    existing = ws.get_all_records()

    existing_keys = set()
    for row in existing:
        if is_excluded_or_invalid_result(row.get("result", "")):
            continue
        if str(row.get("event_id", "")) == str(event_id):
            key = (
                str(row.get("player_name", "")).lower().strip(),
                str(row.get("market_type", "")).lower().strip(),
                str(row.get("sportsbook", "")).lower().strip(),
                str(row.get("american_odds", "")),
            )
            existing_keys.add(key)

    if not existing_keys:
        return outrights_df

    mask = []
    for _, r in outrights_df.iterrows():
        key = (
            str(r.get("player_name", "")).lower().strip(),
            str(r.get("market_type", r.get("market", ""))).lower().strip(),
            str(r.get("bookmaker", r.get("book", r.get("sportsbook", "")))).lower().strip(),
            str(r.get("american_odds", "")),
        )
        mask.append(key not in existing_keys)

    new_rows = outrights_df[mask].copy()
    print(f"  [reprice] Outrights: {len(outrights_df)} total, {len(new_rows)} new (deduped {len(outrights_df) - len(new_rows)})")
    return new_rows


def _send_reprice_alert(new_mu, new_se, sim_round, tourney_name, new_op=None):
    """Strictly deliver newly-priced bets before any corresponding storage.

    Unlike the diagnostic ``_send_telegram`` helper, this path validates
    credentials, HTTP status, and Telegram's JSON ``ok`` response.  Long alerts
    are split before delivery.  Any rejected chunk raises and the caller must
    leave all pending rows unstored so the workflow retry can alert them.
    """
    lines = []
    lines.append(f"<b>R{sim_round} Reprice — {tourney_name.replace('_', ' ').title()}</b>")
    lines.append("")

    def _fmt_odds(v):
        if isinstance(v, (int, float)) and not pd.isna(v):
            return f"+{int(v)}" if v > 0 else str(int(v))
        return str(v)

    if new_mu is not None and not new_mu.empty:
        lines.append(f"<b>New Matchups ({len(new_mu)}):</b>")
        for _, r in new_mu.iterrows():
            bet = r.get("bet_on", "?")
            edge = r.get("edge_on", "?")
            book = r.get("Bookmaker", "?")
            is_p1 = str(r.get("bet_on", "")).lower() == str(r.get("Player 1", "")).lower()
            opp = r.get("Player 2", "") if is_p1 else r.get("Player 1", "")
            mkt_odds = r.get("P1 Odds", "?") if is_p1 else r.get("P2 Odds", "?")
            fair_odds = r.get("Fair_p1", "?") if is_p1 else r.get("Fair_p2", "?")
            lines.append(f"  {bet} vs {opp}")
            lines.append(f"    {book} {_fmt_odds(mkt_odds)} (fair {_fmt_odds(fair_odds)}) edge={edge}%")
        lines.append("")

    if new_se is not None and not new_se.empty:
        lines.append(f"<b>New Score Edges ({len(new_se)}):</b>")
        for _, r in new_se.iterrows():
            player = r.get("Player", "?")
            line_val = r.get("Line", "?")
            side = r.get("Best_Side", "?")
            edge = r.get("Best_Edge", "?")
            book = r.get("Book", "?")
            if side == "Under":
                mkt_odds = r.get("Mkt_Under", "?")
                fair_odds = r.get("Fair_Under", "?")
            else:
                mkt_odds = r.get("Mkt_Over", "?")
                fair_odds = r.get("Fair_Over", "?")
            lines.append(f"  {player} {side} {line_val}")
            lines.append(f"    {book} {_fmt_odds(mkt_odds)} (fair {_fmt_odds(fair_odds)}) edge={edge}%")
        lines.append("")

    if new_op is not None and not new_op.empty:
        lines.append(f"<b>New Outright Edges ({len(new_op)}):</b>")
        for _, r in new_op.iterrows():
            player = str(r.get("player_name", "?")).title()
            market = r.get("market_type", r.get("market", "?"))
            side = r.get("side", "")
            book = r.get("bookmaker", r.get("book", "?"))
            edge = r.get("edge", "?")
            mkt_odds = r.get("american_odds", "?")
            fair_odds = r.get("my_fair", "?")
            side_str = f" {side}" if side else ""
            lines.append(f"  {player} {market}{side_str}")
            lines.append(f"    {book} {_fmt_odds(mkt_odds)} (fair {_fmt_odds(fair_odds)}) edge={edge}%")
        lines.append("")

    if ((new_mu is None or new_mu.empty)
            and (new_se is None or new_se.empty)
            and (new_op is None or new_op.empty)):
        return 0

    import reprice_core as _reprice_core

    messages = []
    chunk = []
    for line in lines:
        candidate = "\n".join(chunk + [line])
        if chunk and len(candidate) > _reprice_core.TELEGRAM_MESSAGE_MAX_CHARS:
            messages.append("\n".join(chunk))
            chunk = [line]
        else:
            chunk.append(line)
    if chunk:
        messages.append("\n".join(chunk))

    for message in messages:
        _reprice_core.send_telegram(message, required=True)
    print(f"  [reprice] Telegram delivered {len(messages)} required message(s).")
    return len(messages)


def _reprice_store_and_alert(combined, score_edges, sim_round, tourney_name, event_id,
                              outrights_combined=None, health_check=None):
    """Dedup, strictly deliver every required alert, then store new rows.

    Alert-required rows are never written first.  A delivery failure propagates
    to the command entrypoint and keeps the rows retryable.
    """
    from sheets_storage import (
        get_spreadsheet,
        store_round_matchups,
        store_score_edges,
        store_finish_positions,
        load_dg_id_lookup,
    )

    spreadsheet = get_spreadsheet()

    new_mu, seen_alert_keys = _dedup_round_matchups(combined, spreadsheet, event_id, sim_round)
    new_se = _dedup_score_edges(score_edges, spreadsheet, event_id, sim_round)
    new_op = _dedup_finish_positions(outrights_combined, spreadsheet, event_id)

    try:
        dg_id_lookup = load_dg_id_lookup(tourney_name, name_replacements)
    except Exception:
        dg_id_lookup = {}

    # Telegram filter: only sharp books, and only edges NOT previously surfaced —
    # a price/edge-size move on an already-seen pairing+side re-stores silently;
    # only a new pairing (or the edge flipping to the other player) pings.
    from reprice_core import partition_matchup_alert_rows
    _tg_mu, _ = partition_matchup_alert_rows(
        new_mu, seen_alert_keys=seen_alert_keys
    )
    if _tg_mu is not None and _tg_mu.empty:
        _tg_mu = None

    # Telegram filter for outrights: only include exchange (kalshi/novig) taker
    # rows with non-trivial edge to keep alerts actionable.
    _tg_op = None
    if new_op is not None and not new_op.empty:
        _book_col = "bookmaker" if "bookmaker" in new_op.columns else "book"
        _pricing = (
            new_op["pricing"]
            if "pricing" in new_op.columns
            else pd.Series("taker", index=new_op.index)
        )
        _tg_op = new_op[
            new_op[_book_col].astype(str).str.lower().isin({"kalshi", "novig"})
            & (_pricing.astype(str).str.lower() == "taker")
            & (pd.to_numeric(new_op["edge"], errors="coerce").fillna(0) > EDGE_THRESHOLD_TOPN)
        ].copy()
        if _tg_op.empty:
            _tg_op = None

    # This is the commit point: every alertable row is accepted by Telegram
    # before the first Sheet/ledger mutation.  Failure raises and stores nothing.
    if health_check is not None:
        health_check()
    _send_reprice_alert(_tg_mu, new_se, sim_round, tourney_name, new_op=_tg_op)

    # Recheck after delivery as well: an accepted alert cannot authorize storage
    # if the simulation inputs/artifacts changed while Telegram was in flight.
    if health_check is not None:
        health_check()

    if new_mu is not None and not new_mu.empty:
        store_round_matchups(
            new_mu, sim_round, tourney_name, event_id,
            dg_id_lookup=dg_id_lookup, spreadsheet=spreadsheet,
        )
    else:
        print("  [reprice] No new matchup rows to store.")

    if new_se is not None and not new_se.empty:
        store_score_edges(
            new_se, sim_round, tourney_name, event_id,
            spreadsheet=spreadsheet,
        )
    else:
        print("  [reprice] No new score edge rows to store.")

    if new_op is not None and not new_op.empty:
        store_finish_positions(
            new_op, tourney_name, event_id,
            dg_id_lookup=dg_id_lookup, spreadsheet=spreadsheet,
        )
    else:
        print("  [reprice] No new outright rows to store.")

    n_mu = len(new_mu) if new_mu is not None and not new_mu.empty else 0
    n_se = len(new_se) if new_se is not None and not new_se.empty else 0
    n_op = len(new_op) if new_op is not None and not new_op.empty else 0
    print(f"  [reprice] Done. Stored {n_mu} matchups + {n_se} score edges + {n_op} outrights after required alert delivery.")


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
    parser.add_argument("--no-store", action="store_true",
                        help="Send the email but skip bet storage + dashboard push "
                             "(for test emails without writing bets)")
    parser.add_argument(
        "--require-complete-live-publish",
        action="store_true",
        help=("Publish one strict complete-live package (including "
              "sim_release_manifest.json) instead of the ordinary best-effort "
              "odds-board package"),
    )
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy N(mu, STD_DEV) draws instead of category-first (default: catfirst)")
    parser.add_argument("--use-python", action="store_true",
                        help="Use the legacy Python sim draws instead of the Rust sims_kernel "
                             "(the Rust cascade + score-card draws are the production default)")
    parser.add_argument("--no-skew-cal", action="store_true",
                        help="Skip the round-score skew calibration top-up (skew_calibration.py)")
    parser.add_argument("--sim-only", action="store_true",
                        help="Run round score sim, save sim_cache parquet, then exit (no pricing/email)")
    parser.add_argument("--price-only", action="store_true",
                        help="Load cached sim arrays, skip simulation, run pricing + email")
    parser.add_argument("--reprice", action="store_true",
                        help="Like --price-only but dedup against Sheets and Telegram-alert only new bets (no email)")
    parser.add_argument("--score-est", type=float, default=None,
                        help="Override expected score estimate (only valid with --price-only/--reprice). "
                             "Fractionally shifts cached round-score mass and re-prices; during a live "
                             "event also requires a fresh remaining-tournament/finish rerun.")
    parser.add_argument(
        "--health-approved-by",
        default=None,
        help="Named operator approving a CLI-configured betting run. Required for "
             "non-dry-run --cli runs; never bypasses failed health checks.",
    )
    args = parser.parse_args()

    required_matchup_books = resolve_required_matchup_books()
    if _round_email_required():
        print(
            "  Required matchup books for complete delivery: "
            + ", ".join(required_matchup_books)
        )

    global _USE_PYTHON
    _USE_PYTHON = args.use_python

    if args.score_est is not None and not args.price_only and not args.reprice:
        parser.error("--score-est requires --price-only or --reprice")

    # --reprice implies --price-only
    if args.reprice:
        args.price_only = True
    if args.sim_only and args.price_only:
        parser.error("Cannot use --sim-only and --price-only together")
    if args.require_complete_live_publish and (
        args.dry_run
        or args.no_store
        or args.sim_only
        or args.price_only
        or args.skip_tournament_sim
    ):
        parser.error(
            "--require-complete-live-publish requires a full stored round and "
            "remaining-tournament simulation"
        )

    # ── Config ───────────────────────────────────────────────────────────
    sheet_config = None
    expected_avg_authority = "cli" if args.cli else "sheet"
    if not args.cli:
        try:
            sheet_config = _cfg  # reuse config loaded at import time
            round_num = sheet_config["round_num"]
            sim_round = round_num + 1 if round_num < 4 else 4
            # Warning-only DOW divergence check (sheet stays the human-controlled
            # source so a past round can still be re-priced manually).
            try:
                from api_utils import sim_round_from_dow
                _dow_round = sim_round_from_dow()
                if _dow_round != sim_round:
                    print(f"  [warn] Sheet round R{sim_round} != day-of-week round R{_dow_round}; "
                          f"using SHEET (update round_config if this is wrong)")
            except Exception:
                pass
            expected_avg = sheet_config.get("expected_score_1")
            if expected_avg is None:
                expected_avg = PAR
                expected_avg_authority = "sheet_missing_fallback"
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
            expected_avg_authority = "cli"
    else:
        if args.sim_round is None:
            parser.error("--sim-round is required in CLI mode")
        sim_round = args.sim_round
        if args.expected_avg is None:
            _sheet_expected = _cfg.get("expected_score_1")
            if _sheet_expected is None:
                expected_avg = PAR
                expected_avg_authority = "sheet_missing_fallback"
            else:
                expected_avg = (
                    float(_sheet_expected)
                    if abs(float(_sheet_expected)) > 50
                    else PAR + float(_sheet_expected)
                )
                expected_avg_authority = "sheet"
                print(f"  CLI round uses Sheet expected_score_1={expected_avg}")
        else:
            expected_avg = float(args.expected_avg)
            expected_avg_authority = "cli"
        round_num = sim_round - 1

    # Preserve the authoritative target before an in-memory --score-est shift.
    # The gate later requires the shifted tape to equal this value exactly.
    configured_expected_avg = float(expected_avg)

    # Event identity has two weekly sources (Sheet runtime config and
    # sim_inputs data/model config). They must agree before any cache can be
    # approved; otherwise a correctly-shaped old field can be mislabeled as
    # the new event.
    import sim_inputs as _health_sim_inputs
    _source_tourney = str(getattr(_health_sim_inputs, "tourney", "")).strip().lower()
    _source_event_ids = getattr(_health_sim_inputs, "event_ids", []) or []
    _source_event_id = _source_event_ids[0] if _source_event_ids else None
    if _source_tourney != str(tourney).strip().lower() or str(_source_event_id) != str(_event_id):
        raise SimulationHealthError(
            "BLOCKED — Sheet/sim_inputs event split-brain: "
            f"sheet={tourney}/{_event_id}, sim_inputs={_source_tourney}/{_source_event_id}"
        )

    # ── Load sim cache early if --price-only / --reprice ─────────────────
    cached_sim_dict = None
    cached_meta = None
    cached_parent_health = None
    score_shift_delta = 0.0
    if args.price_only:
        try:
            cached_sim_dict, cached_meta = load_sim_cache(sim_round)
        except FileNotFoundError as e:
            if args.reprice:
                print(f"  [reprice] {e}")
                print(f"  [reprice] Nothing to reprice without a cache. Exiting cleanly.")
                return
            raise

        cached_parent_health = (cached_meta or {}).get("health_manifest")
        if not cached_parent_health:
            raise SimulationHealthError(
                "BLOCKED — sim cache has no sealed health manifest; run a fresh full simulation"
            )

        if args.score_est is not None:
            cached_avg = sealed_cache_expected_avg(cached_meta)
            score_shift_delta = float(args.score_est) - cached_avg
            print(f"  [score-est] Overriding expected_avg: {cached_avg} -> {args.score_est} "
                  f"(delta = {score_shift_delta:+.3f}); cached scores will be shifted")
            expected_avg = float(args.score_est)

    # ── Load predictions (optional in --price-only if cache has them) ─────
    pred_file = f"model_predictions_r{sim_round}.csv"
    pred_file_path = None
    if os.path.exists(pred_file):
        pred_file_path = pred_file
    else:
        # dashboard_data/ copies are tracked in git and can be a PREVIOUS
        # week's vintage on a fresh checkout; only use one when the sync
        # manifest attributes this exact file to the current event (the same
        # guard sync_event_files applies — 2026-08 audit: the unguarded copy
        # could sim a new event on last week's field/skill state).
        alt = os.path.join("dashboard_data", pred_file)
        if os.path.exists(alt):
            from sync_event_files import _manifest_allowed
            if pred_file in _manifest_allowed(str(tourney).lower()):
                print(f"  [resolve] {pred_file} not in root, using {alt} "
                      f"(manifest-verified for {tourney})")
                pred_file_path = alt
            else:
                print(f"  [resolve] {pred_file} exists in dashboard_data/ but the "
                      f"sync manifest does not attribute it to '{tourney}' — "
                      f"refusing stale fallback (rerun live_stats_engine)")

    if pred_file_path:
        model_preds = pd.read_csv(pred_file_path)
        model_preds["player_name"] = (
            model_preds["player_name"].str.lower().str.strip().replace(name_replacements)
        )
        if "centering_version" not in model_preds.columns or "centering_group" not in model_preds.columns:
            raise SimulationHealthError(
                f"BLOCKED — {pred_file_path} is a legacy/unmarked prediction file; "
                "rebuild it with live_stats_engine.py"
            )
        if model_preds[["centering_version", "centering_group"]].isna().any().any():
            raise SimulationHealthError(
                f"BLOCKED — {pred_file_path} has partially missing centering metadata"
            )
        versions = set(model_preds["centering_version"].astype(str))
        if versions != {CENTERING_VERSION}:
            raise SimulationHealthError(
                f"BLOCKED — unsupported score centering in {pred_file_path}: {versions}"
            )
        groups = set(model_preds["centering_group"].astype(str))
        if len(groups) != 1:
            raise SimulationHealthError(
                f"BLOCKED — inconsistent centering groups in {pred_file_path}: {groups}"
            )
        group_col = next(iter(groups))
        if group_col == "field":
            group_col = None
        skill_col = "my_pred" if sim_round == 1 else f"my_pred{sim_round}"
        validate_field_relative_predictions(
            model_preds,
            skill_col=skill_col,
            score_col=f"scores_r{sim_round}",
            weather_col=f"weather_sg_r{sim_round}",
            group_col=group_col,
        )
        print(
            f"  Verified {CENTERING_VERSION}: scores_r{sim_round} "
            f"is zero-mean by {group_col or 'field'}"
        )
        pred_col = find_pred_col(model_preds, sim_round)
        pred_lookup = dict(zip(model_preds["player_name"], model_preds[pred_col]))
    elif args.price_only and cached_meta and cached_meta.get("pred_lookup"):
        print(f"  [price-only] {pred_file} not found — using pred_lookup from cache meta")
        model_preds = pd.DataFrame(columns=["player_name"])
        pred_col = None
        pred_lookup = {k: float(v) for k, v in cached_meta["pred_lookup"].items()}
    else:
        raise FileNotFoundError(
            f"{pred_file} not found. Run live_stats_engine.py first."
        )

    sample_lookup = load_sample_data()

    _mode = "PRICE-ONLY" if args.price_only else ("SIM-ONLY" if args.sim_only else "FULL")
    print(f"\n{'='*60}")
    print(f"  ROUND {sim_round} {'RE-PRICING' if args.price_only else 'SIMULATION'} - {tourney} [{_mode}]")
    print(f"{'='*60}")
    print(f"  Predictions:  {pred_file_path or '(from cache)'} ({len(model_preds)} players)")
    print(f"  Expected avg: {expected_avg}")
    print(f"  Draw mode:    {'legacy (STD_DEV=' + str(STD_DEV) + ')' if args.legacy else 'category-first'}")
    print(f"  Simulations:  {NUM_SIMULATIONS:,}")
    print(f"  Pred column:  {pred_col}")

    # ── Build weather lookup from model predictions ─────────────────────
    _wind_col = f"wind_adj{sim_round}"
    _dew_col = f"dew_adj{sim_round}"
    _weather_col = f"weather_sg_r{sim_round}"
    _wx_lookup = {}
    if not model_preds.empty and _weather_col in model_preds.columns:
        _wx_lookup = dict(zip(
            model_preds["player_name"],
            pd.to_numeric(model_preds[_weather_col], errors="coerce").fillna(0.0),
        ))
        print(
            f"  Weather lookup: {len(_wx_lookup)} players "
            f"(explicit centered column={_weather_col})"
        )
    elif (not model_preds.empty
            and _wind_col in model_preds.columns
            and _dew_col in model_preds.columns):
        # Must mirror live_stats_engine's scores_rN weather term EXACTLY
        # (scores_rN = pred + avg_wind - wind_adj - dew_adj), since the sim
        # recovers pure skill as scores_rN - wx_delta and re-adds wx_delta
        # per category. Both adj columns are score-space costs.
        _avg_wind = model_preds[_wind_col].mean()
        _wx_lookup = dict(zip(
            model_preds["player_name"],
            _avg_wind - model_preds[_wind_col] - model_preds[_dew_col],
        ))
        print(
            f"  Weather lookup: {len(_wx_lookup)} players "
            f"(legacy wind col={_wind_col}, dew col={_dew_col})"
        )
    elif args.price_only and cached_meta and cached_meta.get("wx_lookup"):
        _wx_lookup = {k: float(v) for k, v in cached_meta["wx_lookup"].items()}
        print(f"  Weather lookup: {len(_wx_lookup)} players (from cache meta)")
    else:
        print(f"  No weather columns ({_wind_col}, {_dew_col}) — weather decomposition disabled")

    # Behavioral centering authority comes directly from the active Sheet course
    # map. The prediction CSV's course_score_adj is checked against that map but
    # is never allowed to define its own target (a stale 69.7 column must not be
    # able to approve a 69.7 tape merely labelled 68.7).
    sheet_course_averages = configured_round_scoring_baselines(_cfg)
    if expected_avg_authority == "cli" and args.expected_avg is not None:
        if len(sheet_course_averages) > 1:
            raise SimulationHealthError(
                "BLOCKED — a scalar CLI expected average is ambiguous for a multi-course "
                "event; update every Sheet course baseline and rerun"
            )
        active_course_averages = {"field": float(expected_avg)}
    else:
        active_course_averages = sheet_course_averages
    expected_field_mean = float(expected_avg)
    expected_player_means = {
        str(player): float(expected_avg) for player in model_preds.get("player_name", [])
    }
    if not model_preds.empty:
        expected_field_mean, expected_player_means = derive_authoritative_scoring_targets(
            model_preds,
            sim_round=sim_round,
            skill_col=pred_col,
            configured_expected_avg=configured_expected_avg,
            course_averages=active_course_averages,
        )
        print(f"  Centering target: {expected_field_mean:.4f} strokes (player-input implied)")

    # ── Step 1: Simulate scores (or load from cache) ─────────────────────
    active_health_manifest = None
    approved_cache_saved = False
    if args.price_only:
        sim_dict = cached_sim_dict
        cat_mu_lookup = {}
        print(f"\n  [price-only] Skipped sim — using {len(sim_dict)} cached player arrays")
        if score_shift_delta != 0.0:
            sim_dict = uniformly_shift_score_tape(sim_dict, score_shift_delta)
            print(f"  [score-est] Shifted {len(sim_dict)} cached arrays by {score_shift_delta:+.3f} strokes")

        parent_health = cached_parent_health
        parent_model = parent_health.get("model") or {}
        selected_model = parent_model.get("selected", "category_first")
        overlay = collect_overlay_provenance(
            tourney=tourney,
            event_id=_event_id,
            dists_path=(parent_model.get("shot_dispersion_overlay") or {}).get("distribution_file"),
            selected_model=selected_model,
        )
        # Validate the unmodified parent tape before authorising any derived
        # score-est repricing; a derived manifest can never launder a stale or
        # mismatched cache into a fresh approval.
        require_simulation_healthy(
            parent_health,
            tourney=tourney,
            event_id=_event_id,
            sim_round=sim_round,
            configured_expected_avg=(parent_health.get("scoring") or {}).get("expected_avg"),
            sim_dict=cached_sim_dict,
            model_players=list((cached_meta or {}).get("pred_lookup") or cached_sim_dict),
            current_overlay=overlay,
        )
        if score_shift_delta != 0.0:
            parent_course_averages = {
                str(code): float(value)
                for code, value in (
                    (parent_health.get("scoring") or {}).get(
                        "configured_course_averages"
                    ) or {}
                ).items()
            }
            shifted_course_averages = {
                code: value + score_shift_delta
                for code, value in parent_course_averages.items()
            }
            if (
                set(shifted_course_averages) != set(active_course_averages)
                or any(
                    abs(shifted_course_averages[code] - active_course_averages[code]) > 1e-6
                    for code in active_course_averages
                )
            ):
                raise SimulationHealthError(
                    "BLOCKED — a uniform --score-est shift does not match every active "
                    "course baseline; run a fresh full simulation"
                )
            parent_player_means = {
                str(name): float(value)
                for name, value in (
                    (parent_health.get("scoring") or {}).get("player_expected_means")
                    or {}
                ).items()
            }
            shifted_player_means = {
                name: value + score_shift_delta
                for name, value in parent_player_means.items()
            }
            if (
                set(shifted_player_means) != set(expected_player_means)
                or any(
                    abs(shifted_player_means[name] - expected_player_means[name]) > 1e-6
                    for name in expected_player_means
                )
            ):
                raise SimulationHealthError(
                    "BLOCKED — active player skill/weather inputs changed since the "
                    "cached simulation; run a fresh full round simulation instead of "
                    "a score-est-only shift"
                )
            active_health_manifest = build_simulation_manifest(
                sim_dict,
                tourney=tourney,
                event_id=_event_id,
                sim_round=sim_round,
                expected_avg=expected_avg,
                model_players=list(pred_lookup),
                expected_avg_authority=(
                    "sheet_score_est"
                    if expected_avg_authority == "sheet"
                    else expected_avg_authority
                ),
                expected_field_mean=expected_field_mean,
                expected_player_means=expected_player_means,
                configured_course_averages=active_course_averages,
                selected_model=selected_model,
                skew_calibrated=bool(parent_model.get("skew_calibrated")),
                overlay=overlay,
                prediction_path=pred_file_path,
                manual_approved_by=args.health_approved_by,
                manual_approval_required=(args.cli or expected_avg_authority == "cli"),
                parent_manifest=parent_health,
                derivation={
                    "method": FRACTIONAL_SCORE_REPRICE_METHOD,
                    "from_expected_avg": cached_avg,
                    "to_expected_avg": float(expected_avg),
                    "delta": score_shift_delta,
                },
            )
        else:
            active_health_manifest = parent_health
        require_simulation_healthy(
            active_health_manifest,
            tourney=tourney,
            event_id=_event_id,
            sim_round=sim_round,
            configured_expected_avg=configured_expected_avg,
            configured_course_averages=active_course_averages,
            sim_dict=sim_dict,
            model_players=list(pred_lookup),
            current_overlay=overlay,
        )
    else:
        print(f"\n  Simulating R{sim_round} scores...")
        # Category-first is the default; --legacy falls back to N(mu, STD_DEV) draws
        sim_dict_cf, cat_mu_lookup = simulate_round_scores_catfirst(
            model_preds, sim_round, expected_avg, _wx_lookup
        )

        if args.legacy:
            sim_dict_legacy, _ = simulate_round_scores(model_preds, sim_round, expected_avg)
            print_catfirst_comparison(sim_dict_legacy, sim_dict_cf, model_preds, pred_col, expected_avg)
            print("  [legacy] Using standard N(mu, STD_DEV) draws for matchup pricing")
            sim_dict = sim_dict_legacy
        else:
            sim_dict = sim_dict_cf
            # Top up within-player score skew to the empirical target (+0.26):
            # the category sum CLT-washes skew to ~+0.12, leaving the sim's
            # median ~0.05 too pessimistic vs reality. Applied before the
            # cache save so --price-only / --reprice inherit calibrated draws.
            if not args.no_skew_cal:
                from skew_calibration import calibrate_round_skew
                sim_dict = calibrate_round_skew(sim_dict)

        selected_model = "legacy" if args.legacy else "category_first"
        overlay = collect_overlay_provenance(
            tourney=tourney,
            event_id=_event_id,
            dists_path=DISTS_FILE_V2,
            selected_model=selected_model,
        )
        active_health_manifest = build_simulation_manifest(
            sim_dict,
            tourney=tourney,
            event_id=_event_id,
            sim_round=sim_round,
            expected_avg=expected_avg,
            model_players=model_preds["player_name"].tolist(),
            expected_avg_authority=expected_avg_authority,
            expected_field_mean=expected_field_mean,
            expected_player_means=expected_player_means,
            configured_course_averages=active_course_averages,
            selected_model=selected_model,
            skew_calibrated=not args.no_skew_cal,
            overlay=overlay,
            prediction_path=pred_file_path,
            manual_approved_by=args.health_approved_by,
            manual_approval_required=(args.cli or expected_avg_authority == "cli"),
        )
        try:
            require_simulation_healthy(
                active_health_manifest,
                tourney=tourney,
                event_id=_event_id,
                sim_round=sim_round,
                configured_expected_avg=configured_expected_avg,
                configured_course_averages=active_course_averages,
                sim_dict=sim_dict,
                model_players=model_preds["player_name"].tolist(),
                current_overlay=overlay,
            )
        except SimulationHealthError:
            if not args.dry_run:
                raise
            print("  [sim-health] Dry-run diagnostics continue; rejected cache will not be written")
        else:
            # Only approved tapes may become reusable/publishable caches.
            save_sim_cache(
                sim_dict, sim_round, expected_avg, pred_lookup, _wx_lookup,
                health_manifest=active_health_manifest,
            )
            approved_cache_saved = True

        if args.sim_only:
            if approved_cache_saved:
                print("\n  Sim complete (--sim-only). Approved cache saved for --price-only / --reprice.")
            else:
                print("\n  Sim diagnostics complete (--sim-only). No reusable cache was written.")
            print(f"{'='*60}\n")
            return

    must_refresh_live_tournament = score_est_requires_live_refresh(
        price_only=args.price_only,
        score_shift_delta=score_shift_delta,
        completed_round=round_num,
    )
    if must_refresh_live_tournament:
        if args.skip_tournament_sim:
            raise SimulationHealthError(
                "BLOCKED — a live --score-est refresh must rebuild the remaining-"
                "tournament/finish tape; remove --skip-tournament-sim"
            )
        if not pred_file_path or model_preds.empty:
            raise SimulationHealthError(
                f"BLOCKED — a live --score-est refresh requires fresh centered "
                f"{pred_file}; rerun live_stats_engine.py, then retry"
            )

    # ── Step 2: Matchup pricing ──────────────────────────────────────────
    print(f"\n  Fetching matchup odds (scraped -> DataGolf fallback)...")
    matchup_book_counts = {b: 0 for b in SHARP_BOOKS}  # sharp-book line counts for email banner
    matchup_pricing_error = None
    matchup_name_mismatches = {}
    try:
        from odds_loader import load_matchup_odds
        matchup_df = load_matchup_odds("round_matchups", api_key=API_KEY, round=sim_round)
        matchup_df = price_matchups(matchup_df, sim_dict)
        matchup_name_mismatches = dict(
            matchup_df.attrs.get("name_mismatches") or {}
        )
        matchup_df = calculate_edges(matchup_df)
        if "Bookmaker" in matchup_df.columns and not matchup_df.empty:
            from reprice_core import actionable_matchup_mask
            _actionable_matchups = matchup_df.loc[
                actionable_matchup_mask(matchup_df)
            ]
            _vc = _actionable_matchups["Bookmaker"].astype(str).str.lower().value_counts()
            matchup_book_counts = {b: int(_vc.get(b, 0)) for b in SHARP_BOOKS}
        print("  Matchup lines priced per sharp book: "
              + ", ".join(f"{b}={matchup_book_counts[b]}" for b in SHARP_BOOKS))
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
        matchup_pricing_error = e
        print(f"  Warning: Matchup pricing failed: {e}")
        combined = pd.DataFrame()
        sharp = pd.DataFrame()
        bol_matchups_csv = None
        all_books_csv = None

    # ── Step 2b: 3-ball pricing (scraped feed -> fair vs sim) ─────────────
    # 3-ball fairs are computed straight from sim_dict (P of lowest, dead-heat),
    # priced against the scraped board feed (soft books + betcris/betonline/kalshi).
    # combined_3b = all tracked books; email_3b = the alert subset (kalshi/betonline/betcris).
    threeball_pricing_error = None
    try:
        from odds_loader import load_3ball_odds
        tb_df = load_3ball_odds(round=sim_round)
        if tb_df is None or tb_df.empty:
            print("  No 3-ball lines available yet.")
            combined_3b = pd.DataFrame()
            email_3b = pd.DataFrame()
        else:
            tb_df = price_3balls(tb_df, sim_dict)
            tb_df = calculate_3ball_edges(tb_df)
            combined_3b, email_3b = build_3ball_outputs(
                tb_df, sim_round, pred_lookup, sample_lookup, wx_lookup=_wx_lookup
            )
    except Exception as e:
        threeball_pricing_error = e
        print(f"  Warning: 3-ball pricing failed: {e}")
        combined_3b = pd.DataFrame()
        email_3b = pd.DataFrame()

    # ── Step 3: Score card ───────────────────────────────────────────────
    # Multi-course: build separate score cards per course
    score_cards_by_course = {}
    if (not model_preds.empty
            and "course_score_adj" in model_preds.columns
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

    # ── Step 3a: Pre-aggregated score distributions for dashboard ────────
    # Rebuild these in --price-only too.  A --score-est override shifts the
    # cached arrays in memory, and publishing the old parquet would otherwise
    # put the odds board on a different scoring average than the score card.
    # Category means are optional metadata, so their absence in price-only is
    # harmless.
    score_pmf_health = None
    score_pmf_health_files = {}
    try:
        # Per-player expected avg (course-adjusted when multi-course), else scalar
        if "course_score_adj" in model_preds.columns:
            course_expected = model_preds["course_score_adj"].fillna(expected_avg)
            exp_lookup = dict(zip(model_preds["player_name"], course_expected))
        else:
            exp_lookup = expected_avg

        score_probs = build_round_score_probs(sim_dict, exp_lookup, cat_mu_lookup)
        if not score_probs.empty:
            require_round_score_probability_table(score_probs, active_health_manifest)
            out_dir_probs = f"./{tourney}"
            os.makedirs(out_dir_probs, exist_ok=True)
            probs_path = os.path.join(out_dir_probs, f"round_score_probs_r{sim_round}.parquet")
            score_probs.to_parquet(probs_path, index=False)
            score_pmf_health_path = os.path.join(
                out_dir_probs, f"round_score_probs_r{sim_round}_health.json"
            )
            score_pmf_health_files = {"score_pmf": probs_path}
            score_pmf_health = write_bound_artifact_manifest(
                score_pmf_health_path,
                kind="round_score_pmf",
                simulation_manifest=active_health_manifest,
                files=score_pmf_health_files,
                extra={
                    "reprice_method": FRACTIONAL_SCORE_REPRICE_METHOD,
                    "num_players": int(score_probs["player_name"].nunique()),
                    "num_rows": int(len(score_probs)),
                },
            )
            print(f"  Saved {probs_path} ({len(score_probs)} rows, {score_probs['player_name'].nunique()} players)")
    except Exception as e:
        if must_refresh_live_tournament:
            raise SimulationHealthError(
                f"BLOCKED — decimal score refresh could not publish a bound score PMF: {e}"
            ) from e
        print(f"  Warning: round_score_probs write failed: {e}")

    # ── Step 3b: Price score lines vs market ─────────────────────────────
    score_edges = pd.DataFrame()
    score_line_pricing_error = None
    try:
        market_lines = load_score_lines(sim_round)
        if market_lines:
            score_edges = price_score_lines(score_card, market_lines)
    except Exception as e:
        score_line_pricing_error = e
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
    finish_equity_full_paths = []
    win_positive_top10 = pd.DataFrame()
    win_negative_top10 = pd.DataFrame()
    ancillary_edges = pd.DataFrame()
    ancillary_csv_path = None
    tournament_health = None
    tournament_health_path = f"tournament_live_{tourney}_health.json"
    tournament_health_files = {}
    tournament_field_players = []
    tournament_pipeline_error = None

    if (
        not args.skip_tournament_sim
        and (not args.price_only or must_refresh_live_tournament)
        and round_num >= 1
    ):
        _refresh_label = " [required decimal refresh]" if must_refresh_live_tournament else ""
        print(f"\n  Running tournament simulation (R{round_num} complete -> R4){_refresh_label}...")
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
            # The active next-round target is authoritative. In particular,
            # do not let a stale per-round fallback inside the tournament config
            # keep live finish tapes on the pre-update scoring baseline.
            tourn_config["default_expected"] = float(expected_avg)

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
                player_cf_params, effective_skew, L_corr = _load_catfirst_dists(
                    player_names,
                    allow_player_subset=round_num >= 1,
                )
                print(f"    Loaded catfirst distribution parameters")

                # Simulate remaining rounds
                print(f"    Simulating remaining rounds ({TOURNAMENT_SIMULATIONS:,} sims)...")
                final_scores, made_cut_mask, r1_r2_standings, r1_r3_standings = simulate_remaining_rounds(
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

                # Persist final_scores so the Kalshi maker can price H2H matchups
                # against post-round joint outcomes (P(p1 beats p2)). Without this,
                # the maker has to either skip matchups mid-event or use the stale
                # pre-tournament final_scores from new_sim.py.
                np.save(f"final_scores_live_{tourney}.npy", final_scores)
                # Name sidecar: npy rows are in player_names order (r1_live_model
                # order, post filtering) — NOT new_sim's alphabetical
                # player_names.json. kalshi_maker must pair the live npy with
                # THIS file or its H2H probs index the wrong players.
                import json as _json
                with open(f"player_names_live_{tourney}.json", "w") as _f:
                    _json.dump(list(player_names), _f)
                print(f"    Saved final_scores_live_{tourney}.npy "
                      f"({final_scores.shape[0]} players x {final_scores.shape[1]:,} sims) "
                      f"+ player_names_live_{tourney}.json")

                # End-of-R2 / end-of-R3 standings (RAW; cut applied downstream via
                # made_cut_mask) for Kalshi round-leader / round-top-N pricing.
                np.save(f"standings_r2_live_{tourney}.npy", r1_r2_standings)
                np.save(f"standings_r3_live_{tourney}.npy", r1_r3_standings)
                np.save(f"made_cut_live_{tourney}.npy", made_cut_mask)

                # Bind the exact live outright tape, its row ordering, and the
                # probability table to the same approved round simulation. A
                # later --reprice run may use these files only if every byte
                # still matches this detached manifest.
                tournament_field_players = list(player_names)
                tournament_health_files = {
                    "final_scores": f"final_scores_live_{tourney}.npy",
                    "player_names": f"player_names_live_{tourney}.json",
                    "made_cut": f"made_cut_live_{tourney}.npy",
                    "finish_probs": "simulated_probs_live.csv",
                    "finish_probs_event": f"top_finish_probs_live_{tourney}.csv",
                }
                tournament_health = write_bound_artifact_manifest(
                    tournament_health_path,
                    kind="live_tournament_tape",
                    simulation_manifest=active_health_manifest,
                    files=tournament_health_files,
                    extra={
                        "field_player_set_sha256": names_sha256(player_names),
                        "num_players": len(player_names),
                        "num_sims": int(final_scores.shape[1]),
                    },
                )
                print(f"    Saved {tournament_health_path} (content-bound live tape)")

                _tournament_report = require_bound_artifact(
                    tournament_health,
                    kind="live_tournament_tape",
                    files=tournament_health_files,
                    tourney=tourney,
                    event_id=_event_id,
                    sim_round=sim_round,
                    configured_expected_avg=configured_expected_avg,
                    configured_course_averages=active_course_averages,
                    current_overlay=overlay,
                )
                require_live_tournament_alignment(
                    final_scores_path=tournament_health_files["final_scores"],
                    player_names_path=tournament_health_files["player_names"],
                    made_cut_path=tournament_health_files["made_cut"],
                    finish_probs=finish_probs,
                    artifact_manifest=tournament_health,
                )
                require_market_outputs_healthy(
                    finish_probs=finish_probs,
                    expected_players=tournament_field_players,
                    bound_artifact_report=_tournament_report,
                )
                if must_refresh_live_tournament:
                    # Promote the decimal-shifted round cache only after the
                    # corresponding live tournament/finish tape is complete and
                    # content-bound to the exact same derived manifest.
                    save_sim_cache(
                        sim_dict,
                        sim_round,
                        expected_avg,
                        pred_lookup,
                        _wx_lookup,
                        health_manifest=active_health_manifest,
                    )
                    approved_cache_saved = True
                    print(
                        "    [score-est] Promoted derived round cache only after "
                        "live tournament refresh passed"
                    )

                # ── Price ancillary Kalshi markets (round leaders/top-N, playoff) ──
                # Graceful: never let this break the existing pipeline.
                try:
                    import kalshi_ancillary as _ka
                    print(f"    Pricing ancillary Kalshi markets...")
                    ancillary_edges = _ka.price_ancillary_markets(
                        {
                            "player_names": player_names,
                            "standings_r2": r1_r2_standings,
                            "standings_r3": r1_r3_standings,
                            "made_cut": made_cut_mask,
                            "final_scores": final_scores,
                            "pred_lookup": pred_lookup,
                            "sim_dict": sim_dict,
                            "completed_round": round_num,
                        },
                        tourney, name_replacements,
                        edge_threshold=EDGE_THRESHOLD_TOPN,
                    )
                    if not ancillary_edges.empty:
                        ancillary_csv_path = f"kalshi_ancillary_edges_{tourney}.csv"
                        ancillary_edges.to_csv(ancillary_csv_path, index=False)
                        print(f"    Saved {ancillary_csv_path} "
                              f"({len(ancillary_edges)} fillable ancillary edges)")
                    else:
                        print(f"    No fillable ancillary edges "
                              f"(taker 300@ask+fee / maker ask-1c, spread<4c)")
                except Exception as _anc_err:
                    import traceback as _tb
                    print(f"    Warning: ancillary Kalshi pricing failed: {_anc_err!r}")
                    _tb.print_exc()

                # Price outrights against market
                print(f"    Fetching outright odds and calculating edges...")
                priced_markets = price_outrights(finish_probs, pred_lookup, sample_lookup)

                # Build outputs
                outrights_combined, outrights_sharp = build_finish_outputs(
                    priced_markets, pred_lookup, sample_lookup
                )

                # Full finish-position equity (every player, all markets, no edge
                # filter) — dead-heat and no-dead-heat variants emailed as CSVs.
                finish_equity_full_paths = write_full_finish_equity(finish_probs, tourney)

                # Price Kalshi outrights (no dead-heat)
                print(f"\n    Pricing Kalshi outrights (no dead-heat)...")
                kalshi_mids = pd.DataFrame()
                try:
                    kalshi_edges = price_kalshi_outrights(
                        finish_probs, pred_lookup, sample_lookup
                    )
                    if not kalshi_edges.empty:
                        # Split taker vs mid pricing
                        kalshi_taker = kalshi_edges[
                            kalshi_edges["pricing"] == "taker"
                        ].copy()
                        kalshi_mids = kalshi_edges[
                            kalshi_edges["pricing"] == "mid"
                        ].copy()

                        # Merge taker rows into outrights_combined/sharp (existing flow)
                        if not kalshi_taker.empty:
                            outrights_combined = pd.concat(
                                [outrights_combined, kalshi_taker], ignore_index=True
                            )
                            kalshi_sharp = kalshi_taker[
                                kalshi_taker["edge"] > EDGE_THRESHOLD_TOPN
                            ].copy()
                            if not kalshi_sharp.empty:
                                outrights_sharp = pd.concat(
                                    [outrights_sharp, kalshi_sharp], ignore_index=True
                                )
                                outrights_sharp = outrights_sharp.sort_values(
                                    "edge", ascending=False
                                )
                except Exception as e:
                    # Exchange pricing is optional to the DataGolf/NoVig live
                    # finish board. One unexpected Kalshi failure must not abort
                    # their pricing or prevent the combined CSV from publishing.
                    kalshi_edges = pd.DataFrame()
                    kalshi_mids = pd.DataFrame()
                    print(f"    Warning: Kalshi outright pricing failed: {e}")

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

                # Persist the dashboard snapshot only after exchange taker rows
                # have joined the DataGolf-book rows.
                finish_equity_csv_path = write_live_finish_equity(
                    outrights_combined, tourney
                )
                if finish_equity_csv_path:
                    print(
                        f"    Outrights: {len(outrights_combined)} rows found, "
                        f"{len(outrights_sharp)} sharp"
                    )
                else:
                    print(f"    No outright edges above threshold")

                # --- Win market edge CSVs ---
                print(f"\n    Building outright win edge CSVs...")
                out_dir = f"./{tourney}"
                os.makedirs(out_dir, exist_ok=True)
                _, win_edges_csv_path, win_all_edges = build_win_edges_csv(finish_probs, pred_lookup, sample_lookup, out_dir)
                build_betonline_negative_edges_csv(finish_probs, pred_lookup, sample_lookup, out_dir)

                # Build top-10 positive and negative win edge tables for email.
                # Merge Kalshi + NoVig winner taker edges (YES side) into the DG-based
                # win edges so exchange winner prices show up in the dedicated table.
                win_combined = win_all_edges.copy() if not win_all_edges.empty else pd.DataFrame()

                def _attach_exchange_winner(src_df):
                    if src_df is None or src_df.empty:
                        return pd.DataFrame()
                    w = src_df[
                        (src_df.get("market_type") == "winner")
                        & (src_df.get("pricing", "taker") == "taker")
                        & (src_df.get("side") == "yes")
                    ].copy()
                    if w.empty:
                        return w
                    w["simulated_win_prob"] = w["sim_prob"].astype(float)
                    w["decimal_odds"] = 1.0 / w["implied_prob"].clip(lower=1e-6)
                    p = w["simulated_win_prob"]
                    b = w["decimal_odds"] - 1.0
                    q = 1.0 - p
                    f_star = ((b * p - q) / b.clip(lower=1e-6)).clip(lower=0)
                    w["kelly"] = (BANKROLL * KELLY_FRACTION * f_star).astype(float)
                    return w

                k_win = _attach_exchange_winner(kalshi_edges)
                n_win = _attach_exchange_winner(novig_edges)
                if not k_win.empty:
                    win_combined = pd.concat([win_combined, k_win], ignore_index=True)
                if not n_win.empty:
                    win_combined = pd.concat([win_combined, n_win], ignore_index=True)

                if not win_combined.empty:
                    pos = win_combined[win_combined['edge'] > 0].copy()
                    pos = pos.sort_values('kelly', ascending=False).drop_duplicates('player_name', keep='first').head(10)
                    win_positive_top10 = pos

                    neg = win_combined[(win_combined['edge'] < 0) & (win_combined['decimal_odds'] < 31.0)].copy()
                    neg = neg.sort_values('edge', ascending=True).drop_duplicates('player_name', keep='first').head(10)
                    win_negative_top10 = neg

            else:
                if must_refresh_live_tournament:
                    raise SimulationHealthError(
                        "BLOCKED — decimal score refresh found no authoritative known-round "
                        "players; refresh DataGolf round data and retry"
                    )
                print(f"    No player data found for tournament sim")

        except SimulationHealthError:
            raise
        except Exception as e:
            tournament_pipeline_error = e
            if must_refresh_live_tournament:
                raise SimulationHealthError(
                    "BLOCKED — decimal score refresh could not rebuild the live "
                    f"tournament/finish tape: {e}"
                ) from e
            print(f"    Warning: Tournament simulation failed: {e}")
            import traceback
            traceback.print_exc()
    elif args.skip_tournament_sim:
        print(f"\n  Skipping tournament simulation (--skip-tournament-sim)")
    elif args.price_only and round_num >= 1:
        # In --price-only / --reprice: skip the expensive tournament sim, but
        # still reprice Kalshi + Novig outrights against fresh API odds using
        # the cached finish_probs from the previous full run.
        print(f"\n  [reprice-outrights] Loading cached finish probs for Kalshi + Novig pricing...")
        _fp_path = None
        for _cand in ("simulated_probs_live.csv", f"top_finish_probs_live_{tourney}.csv"):
            if os.path.exists(_cand):
                _fp_path = _cand
                break
        if _fp_path is None:
            tournament_pipeline_error = FileNotFoundError(
                "cached finish probabilities are missing"
            )
            print(f"  [reprice-outrights] No cached finish_probs found — skipping Kalshi/Novig pricing.")
        else:
            try:
                import json as _health_json
                if not os.path.exists(tournament_health_path):
                    raise SimulationHealthError(
                        "BLOCKED — cached live outright probabilities have no content-bound health manifest"
                    )
                with open(tournament_health_path, encoding="utf-8") as _hf:
                    tournament_health = _health_json.load(_hf)
                tournament_health_files = {
                    "final_scores": f"final_scores_live_{tourney}.npy",
                    "player_names": f"player_names_live_{tourney}.json",
                    "made_cut": f"made_cut_live_{tourney}.npy",
                    "finish_probs": "simulated_probs_live.csv",
                    "finish_probs_event": f"top_finish_probs_live_{tourney}.csv",
                }
                with open(tournament_health_files["player_names"], encoding="utf-8") as _nf:
                    tournament_field_players = _health_json.load(_nf)
                require_exact_simulation_source(
                    tournament_health,
                    active_health_manifest,
                    artifact_label="cached live outright tape",
                )
                require_bound_artifact(
                    tournament_health,
                    kind="live_tournament_tape",
                    files=tournament_health_files,
                    tourney=tourney,
                    event_id=_event_id,
                    sim_round=sim_round,
                    configured_expected_avg=configured_expected_avg,
                    configured_course_averages=active_course_averages,
                    current_overlay=overlay,
                )
                finish_probs = pd.read_csv(_fp_path)
                finish_probs["player_name"] = (
                    finish_probs["player_name"].astype(str).str.lower().str.strip().replace(name_replacements)
                )
                print(f"  [reprice-outrights] Loaded {len(finish_probs)} players from {_fp_path}")

                # Price Kalshi outrights (no dead-heat)
                print(f"\n    Pricing Kalshi outrights (no dead-heat)...")
                kalshi_edges = price_kalshi_outrights(finish_probs, pred_lookup, sample_lookup)
                if not kalshi_edges.empty:
                    kalshi_taker = kalshi_edges[kalshi_edges["pricing"] == "taker"].copy()
                    kalshi_mids = kalshi_edges[kalshi_edges["pricing"] == "mid"].copy()
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
            except SimulationHealthError:
                raise
            except Exception as e:
                tournament_pipeline_error = e
                print(f"  [reprice-outrights] Failed: {e}")
                import traceback
                traceback.print_exc()
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
    try:
        if (ancillary_edges is not None and hasattr(ancillary_edges, "attrs")
                and ancillary_edges.attrs.get("name_mismatches")):
            for name in ancillary_edges.attrs["name_mismatches"]:
                _all_mismatches.setdefault(name, set()).add("kalshi_ancillary")
    except NameError:
        pass

    if _all_mismatches and not args.dry_run:
        # Suppress players who aren't in our live field at all (missed-cut /
        # wrong-event scraped lines) — those aren't actionable name_replacements
        # fixes, just post-cut noise. Keep only names we DO model but failed to join.
        try:
            _field = set(model_preds["player_name"].astype(str)
                         .str.lower().str.strip().replace(name_replacements))
        except Exception:
            _field = set()
        _actionable = ({n: b for n, b in _all_mismatches.items() if n in _field}
                       if _field else dict(_all_mismatches))
        _suppressed = len(_all_mismatches) - len(_actionable)
        if _actionable:
            _mm_lines = [f"<b>R{sim_round} Name Mismatches — {tourney.replace('_', ' ').title()}</b>", ""]
            _mm_lines.append(f"{len(_actionable)} in-field players not joined (add to name_replacements):")
            for name, books in sorted(_actionable.items())[:12]:
                book_str = ", ".join(sorted(books)) if isinstance(books, set) else str(books)
                _mm_lines.append(f"  • {name}  ({book_str})")
            if len(_actionable) > 12:
                _mm_lines.append(f"  …and {len(_actionable) - 12} more")
            if _suppressed:
                _mm_lines.append(f"\n({_suppressed} missed-cut/non-field players suppressed)")
            _send_telegram("\n".join(_mm_lines))
            print(f"  Name mismatches: {len(_actionable)} actionable, {_suppressed} suppressed (missed-cut)")
        else:
            print(f"  Name mismatches: 0 actionable ({_suppressed} missed-cut/non-field suppressed)")

    # ── Kalshi ancillary pickup-sanity tripwire ─────────────────────────────
    # raw>0 but matched==0 on a series = the series HAD markets but none matched our
    # event (tournament-tag drift / matcher break) — the signal that distinguishes a
    # broken pickup from genuine "no liquidity". Fires loud + Telegram.
    try:
        _pr = (ancillary_edges.attrs.get("pickup_report")
               if ancillary_edges is not None and hasattr(ancillary_edges, "attrs") else None)
    except NameError:
        _pr = None
    if _pr:
        _drift = [r for r in _pr if (r.get("raw") or 0) > 0 and r.get("matched", 0) == 0]
        _total_matched = sum(r.get("matched", 0) for r in _pr)
        _any_raw = any((r.get("raw") or 0) > 0 for r in _pr)
        _pickup_failed = _any_raw and _total_matched == 0
        if _drift or _pickup_failed:
            _pl = [f"<b>R{sim_round} Kalshi Pickup Sanity — {tourney.replace('_', ' ').title()}</b>", ""]
            if _pickup_failed:
                _pl.append("PICKUP FAILURE: 0 markets matched our event across ALL series "
                           "(matcher broke or wrong slug).")
            for r in _drift:
                _pl.append(f"TAG DRIFT: {r['series']} fetched {r['raw']} markets, "
                           f"matched 0 for tourney={tourney}")
            _pl += ["", "Verify sim_inputs.tourney / kalshi_event_tags vs Kalshi's live event tags."]
            print("  [ancillary] PICKUP SANITY ALERT:\n    " + "\n    ".join(_pl[2:]))
            if not args.dry_run:
                _send_telegram("\n".join(_pl))
                print("  Sent Telegram pickup-sanity alert")

    # ── Compute archetypes (before export + email so type_on appears everywhere) ──
    try:
        from sg_diagnostic import compute_rolling_archetypes, load_archetype_map
        _field = model_preds['player_name'].unique().tolist() if model_preds is not None else []
        try:
            _arch_map = load_archetype_map(_event_id)
        except Exception as _csv_err:
            print(f"  Archetype CSV load failed ({_csv_err}) — trying live db")
            _arch_map = {}
        if _arch_map:
            print(f"  Loaded {len(_arch_map)} archetypes from weekly CSV")
        else:
            _arch_df = compute_rolling_archetypes(_event_id, _field)
            _arch_map = dict(zip(_arch_df['player_name'], _arch_df['archetype']))
            print(f"  Computed archetypes for {len(_arch_map)} players (live db)")
        if not combined.empty:
            combined['type_on'] = (
                combined['bet_on'].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
            )
        for _df3b in (combined_3b, email_3b):
            if _df3b is not None and not _df3b.empty:
                _df3b['type_on'] = (
                    _df3b['bet_on'].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
                )
        if not sharp.empty:
            sharp['type_on'] = (
                sharp['bet_on'].astype(str).str.lower().str.strip().map(_arch_map).fillna("")
            )
            # type_a = archetype of the opponent (the player we're betting AGAINST)
            _opp = sharp['Player 2'].where(
                sharp['bet_on'].astype(str) == sharp['Player 1'].astype(str),
                sharp['Player 1'],
            )
            sharp['type_a'] = (
                _opp.astype(str).str.lower().str.strip().map(_arch_map).fillna("")
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

    def _require_betting_health():
        """Re-check exact artifacts immediately before any bet side effect."""
        require_pricing_pipeline_healthy(
            matchup_error=matchup_pricing_error,
            threeball_error=threeball_pricing_error,
            score_line_error=score_line_pricing_error,
            matchup_book_counts=matchup_book_counts,
            matchup_name_mismatches=matchup_name_mismatches,
            require_complete_email=_round_email_required(),
            require_live_tournament=(round_num >= 1 and not args.skip_tournament_sim),
            tournament_error=tournament_pipeline_error,
            finish_probs=finish_probs,
            required_matchup_books=required_matchup_books,
        )
        require_simulation_healthy(
            active_health_manifest,
            tourney=tourney,
            event_id=_event_id,
            sim_round=sim_round,
            configured_expected_avg=configured_expected_avg,
            configured_course_averages=active_course_averages,
            sim_dict=sim_dict,
            model_players=list(pred_lookup),
            current_overlay=overlay,
        )
        if not score_pmf_health or not score_pmf_health_files:
            raise SimulationHealthError(
                "BLOCKED — score prices are not bound to the active simulation manifest"
            )
        require_exact_simulation_source(
            score_pmf_health,
            active_health_manifest,
            artifact_label="round score PMF",
        )
        require_bound_artifact(
            score_pmf_health,
            kind="round_score_pmf",
            files=score_pmf_health_files,
            tourney=tourney,
            event_id=_event_id,
            sim_round=sim_round,
            configured_expected_avg=configured_expected_avg,
            configured_course_averages=active_course_averages,
            current_overlay=overlay,
        )
        if finish_probs is not None and not finish_probs.empty:
            if not tournament_health or not tournament_health_files:
                raise SimulationHealthError(
                    "BLOCKED — outright prices are not bound to an approved live tournament tape"
                )
            require_exact_simulation_source(
                tournament_health,
                active_health_manifest,
                artifact_label="live outright tape",
            )
            _bound_report = require_bound_artifact(
                tournament_health,
                kind="live_tournament_tape",
                files=tournament_health_files,
                tourney=tourney,
                event_id=_event_id,
                sim_round=sim_round,
                configured_expected_avg=configured_expected_avg,
                configured_course_averages=active_course_averages,
                current_overlay=overlay,
            )
            if names_sha256(tournament_field_players) != (
                (tournament_health.get("extra") or {}).get("field_player_set_sha256")
            ):
                raise SimulationHealthError(
                    "BLOCKED — live tournament player ordering/field does not match its manifest"
                )
            require_live_tournament_alignment(
                final_scores_path=tournament_health_files["final_scores"],
                player_names_path=tournament_health_files["player_names"],
                made_cut_path=tournament_health_files["made_cut"],
                finish_probs=finish_probs,
                artifact_manifest=tournament_health,
            )
            require_market_outputs_healthy(
                finish_probs=finish_probs,
                expected_players=tournament_field_players,
                bound_artifact_report=_bound_report,
            )

    if args.dry_run:
        try:
            _require_betting_health()
        except SimulationHealthError as _health_err:
            print(f"  [sim-health] Dry-run side-effect verdict: {_health_err}")

    # ── Step 6a: --reprice exits here (dedup + store + Telegram + email) ──
    if args.reprice:
        _require_betting_health()
        if args.dry_run:
            print("\n  [reprice] Dry run: skipping dedup, alerts, and all bet storage.")
        else:
            print(f"\n  [reprice] Dedup + strict alert-before-store...")
            _reprice_store_and_alert(combined, score_edges, sim_round, tourney, _event_id,
                                      outrights_combined=outrights_combined,
                                      health_check=_require_betting_health)

        if not args.dry_run:
            _require_betting_health()
            print(f"\n  [reprice] Sending email...")
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
                finish_equity_full_paths=finish_equity_full_paths,
                win_positive_top10=win_positive_top10,
                win_negative_top10=win_negative_top10,
                wx_lookup=_wx_lookup,
                score_edges=score_edges,
                kalshi_mids=kalshi_mids,
                matchup_book_counts=matchup_book_counts,
                threeball_df=email_3b,
                required=_round_email_required(),
            )

        print(f"\n{'='*60}\n  Done (--reprice).\n{'='*60}")
        return

    # ── Step 6: Email ────────────────────────────────────────────────────
    if not args.dry_run:
        _require_betting_health()
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
            finish_equity_full_paths=finish_equity_full_paths,
            win_positive_top10=win_positive_top10,
            win_negative_top10=win_negative_top10,
            wx_lookup=_wx_lookup,
            score_edges=score_edges,
            kalshi_mids=kalshi_mids,
            ancillary_df=ancillary_edges,
            ancillary_csv_path=ancillary_csv_path,
            matchup_book_counts=matchup_book_counts,
            threeball_df=email_3b,
            required=_round_email_required(),
        )
    else:
        print(f"\n  [dry-run] Skipping email")

    # ── Storage ──────────────────────────────────────────────────────────────
    if not args.dry_run and not args.no_store:
        _require_betting_health()
        from sheets_storage import (
            is_valid_run_time,
            get_spreadsheet,
            store_round_matchups,
            store_round_3balls,
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

                # 1b. All filtered 3-balls (tracks every book; alerting is separate)
                store_round_3balls(
                    combined_3b, sim_round, tourney, _event_id,
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
            except SimulationHealthError:
                raise
            except Exception as e:
                # Storage is a required output, not post-run diagnostics.  Each
                # store_* call is idempotent, so propagating the failure lets the
                # workflow retry any missing Sheet/ledger half without turning a
                # partial dual-write into a green run.
                print(f"[storage] FAILED: {e}")
                import traceback; traceback.print_exc()
                raise
        else:
            print("[storage] Skipped - before Monday 3 PM EST cutoff.")
    else:
        print(f"  [dry-run] Skipping bet storage")
    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")

    # Push dashboard data to Render (skip on dry-run / no-store)
    if not args.dry_run and not args.no_store:
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

        # Publish sim fairs + round samples (round scores / matchups / 3-balls) for the odds board
        try:
            import publish_sim_fairs
            print(f"\n{'='*60}")
            print("  Publishing sim fairs + round samples for the odds board...")
            publish_sim_fairs.publish(
                require_complete_live=args.require_complete_live_publish,
                expected_round=(
                    sim_round if args.require_complete_live_publish else None
                ),
            )
            print(f"{'='*60}")
        except SimulationHealthError:
            raise
        except Exception as e:
            print(f"  [publish_sim_fairs] Warning: {e}")
            if args.require_complete_live_publish or (
                os.environ.get("REQUIRE_SIM_FAIRS_PUBLISH") or ""
            ).strip().lower() in (
                "1", "true", "yes"
            ):
                raise


if __name__ == "__main__":
    main()

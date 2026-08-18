"""
Round-score skew calibration.

July 2026 analysis (dg_historical.db, 2017-2025, 159k player-rounds):
within-player single-round score distributions are right-skewed — moment
skew +0.26, mean-median gap +0.11 strokes (blow-up rounds drag the mean UP;
a player's median round is ~0.11 better than his mean). The shape is stable
across courses once field-composition mixture is removed (3M/TPC Twin
Cities within-player: +0.25, indistinguishable from tour).

The category-first sim composes to only ~+0.12 total-score skew: summing
four partially-correlated skewed category draws CLT-washes the skew, and
inflating the per-category Cornish-Fisher targets enough to compensate
(OTT ~ -1.9 SG-space) would push the CF quadratic into its non-monotone
fold. So instead: a monotone per-player CF top-up applied to the simulated
round-score draws AFTER the category sim (works identically on the Rust and
Python paths).

Properties:
  - Preserves each player's mean and std exactly (re-standardized), so the
    field-average calibration to expected_avg is untouched.
  - Monotone (top-up gamma <= 0.35 -> fold at |x| > 8.6 sigma), so the
    joint copula across players is preserved — H2H / 3-ball / rank
    structure unaffected beyond the intended marginal reshape.
  - Only ever ADDS positive (score-space) skew: players whose draws already
    meet the target are left untouched.
"""
import numpy as np
from scipy.stats import skew as _moment_skew

# Empirical within-player round-score skew target (strokes space, positive =
# right tail of bad scores). Tour-wide 2017-2025: +0.263; 3M-specific: +0.250.
TARGET_ROUND_SKEW = 0.26
MIN_TOPUP = 0.02   # don't bother below this increment
MAX_TOPUP = 0.35   # safety cap (monotone far beyond any draw at this level)
_SEED = 20260726   # fixed seed: calibration is deterministic run-to-run


def _cf(x, gamma):
    """First-order Cornish-Fisher reshape of standardized draws."""
    return x + (gamma / 6.0) * (x * x - 1.0)


def calibrate_round_skew(sim_dict, target=TARGET_ROUND_SKEW, verbose=True):
    """
    Top up each player's simulated round-score skew to the empirical target.

    The draws are INTEGER strokes, so a naive transform-then-rint round-trip
    quantizes the ~0.1-sigma reshape back to the original integers and the
    PMF never moves. Instead: de-quantize with a uniform dither, apply the
    CF top-up continuously, then STOCHASTIC-round back to integers — the
    fractional shifts become fractional probability mass moved between
    adjacent strokes, which is exactly what score-line prices respond to.

    Args:
        sim_dict: {player_name: np.array of int round scores}
        target: within-player moment-skew target in score space
        verbose: print a one-line summary

    Returns:
        The same dict with adjusted integer score arrays (players already at
        or above target, or with degenerate draws, pass through unchanged).
    """
    rng = np.random.default_rng(_SEED)
    skews_before, skews_after = [], []
    n_adj = 0
    for player, s in sim_dict.items():
        s_int = np.asarray(s)
        s_f = s_int.astype(float)
        if s_f.std() < 1e-9:
            continue
        # De-quantize: uniform dither undoes integer rounding to first order
        v = s_f + rng.uniform(-0.5, 0.5, size=s_f.shape)
        m, sd = v.mean(), v.std()
        x = (v - m) / sd
        g0 = float(_moment_skew(x))
        skews_before.append(g0)
        need = target - g0
        if need < MIN_TOPUP:
            skews_after.append(g0)
            continue

        # Two-pass: apply the naive increment, measure what it actually
        # bought on this (already non-normal) distribution, rescale once.
        gamma = min(need, MAX_TOPUP)
        achieved = float(_moment_skew(_cf(x, gamma))) - g0
        if achieved > 1e-6:
            gamma = float(np.clip(gamma * need / achieved, 0.0, MAX_TOPUP))
        y = _cf(x, gamma)
        y = (y - y.mean()) / y.std()
        # Variance compensation: stochastic rounding will add ~E[f(1-f)]
        # ~= 1/6 of rounding-noise variance; shrink the continuous spread so
        # the FINAL integer draws come back at the original std.
        var_target = max(sd * sd - 1.0 / 6.0, 0.25)
        out_c = m + y * np.sqrt(var_target)

        # Stochastic rounding: unbiased, moves fractional mass between
        # adjacent integers instead of snapping back
        fl = np.floor(out_c)
        out = fl + (rng.uniform(size=out_c.shape) < (out_c - fl))
        # bound the reshaped tail near the original support (integer strokes)
        out = np.clip(out, s_f.min() - 3, s_f.max() + 3)
        sim_dict[player] = out.astype(s_int.dtype
                                      if np.issubdtype(s_int.dtype, np.integer)
                                      else int)
        skews_after.append(float(_moment_skew((out - out.mean()) / out.std())))
        n_adj += 1

    if verbose and skews_before:
        print(f"  [skew-cal] {n_adj}/{len(skews_before)} players topped up: "
              f"avg round skew {np.mean(skews_before):+.3f} -> {np.mean(skews_after):+.3f} "
              f"(target {target:+.2f})")
    return sim_dict


# Empirical within-player 72-hole score skew target (2026-08 joint fix analysis
# on dg_historical 2017-2025 player-weeks; cut-conditional at cut events). The
# week-level latent recovers part of this organically; this top-up is
# SELF-CALIBRATING (measures realized skew, adds only the residual), so
# running it LAST makes the joint fit automatic — never apply it before the
# latent is in the cascade.
TARGET_TOTAL_SKEW = 0.23


def calibrate_total_skew(final_scores, made_cut_mask=None,
                         target=TARGET_TOTAL_SKEW, verbose=True):
    """
    Top up each player's simulated 72-HOLE total-score skew to the empirical
    target. Same machinery as calibrate_round_skew (dither -> Cornish-Fisher
    reshape -> stochastic rounding; monotone per player, preserves per-player
    mean/std and the cross-player copula), applied to the (n, sims) integer
    totals matrix IN PLACE.

    At cut events the reshape is cut-conditional: only sims where the player
    made the cut are transformed (missed-cut rows carry the +200 penalty and
    are left untouched), matching how the empirical target is measured.

    Args:
        final_scores: (n_players, sims) integer 72-hole totals (penalty incl.)
        made_cut_mask: (n_players, sims) bool, or None for no-cut events
        target: within-player moment-skew target for 72-hole totals

    Returns:
        The same array, adjusted in place (also returned for chaining).
    """
    rng = np.random.default_rng(_SEED + 1)
    n_players = final_scores.shape[0]
    skews_before, skews_after = [], []
    n_adj = 0
    for i in range(n_players):
        if made_cut_mask is not None:
            sel = made_cut_mask[i]
            if sel.sum() < 100:
                continue
        else:
            sel = slice(None)
        s_int = final_scores[i, sel]
        s_f = s_int.astype(float)
        if s_f.std() < 1e-9:
            continue
        v = s_f + rng.uniform(-0.5, 0.5, size=s_f.shape)
        m, sd = v.mean(), v.std()
        x = (v - m) / sd
        g0 = float(_moment_skew(x))
        skews_before.append(g0)
        need = target - g0
        if need < MIN_TOPUP:
            skews_after.append(g0)
            continue
        gamma = min(need, MAX_TOPUP)
        achieved = float(_moment_skew(_cf(x, gamma))) - g0
        if achieved > 1e-6:
            gamma = float(np.clip(gamma * need / achieved, 0.0, MAX_TOPUP))
        y = _cf(x, gamma)
        y = (y - y.mean()) / y.std()
        var_target = max(sd * sd - 1.0 / 6.0, 0.25)
        out_c = m + y * np.sqrt(var_target)
        fl = np.floor(out_c)
        out = fl + (rng.uniform(size=out_c.shape) < (out_c - fl))
        out = np.clip(out, s_f.min() - 4, s_f.max() + 4)
        final_scores[i, sel] = out.astype(final_scores.dtype)
        skews_after.append(float(_moment_skew((out - out.mean()) / out.std())))
        n_adj += 1

    if verbose and skews_before:
        print(f"  [skew-cal] totals: {n_adj}/{len(skews_before)} players topped up: "
              f"avg 72h skew {np.mean(skews_before):+.3f} -> {np.mean(skews_after):+.3f} "
              f"(target {target:+.2f})")
    return final_scores

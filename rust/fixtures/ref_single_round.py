"""Standalone Python reference for round_sim.py `simulate_round_scores_catfirst`.

Faithful copy of round_sim.py:1884-1944 (seed-789 single-round category-first
score draw), refactored to a pure `run(inp, seed)` function. Statistical oracle
for test_single_round_parity.py and executable spec of the Rust run_single_round
input contract. Keep aligned with round_sim.py. See RUST_MIGRATION_PLAN.md §6.2.
"""

import numpy as np

CLIP_CAT = (-8.0, 8.0)
WEATHER_CAT_SPLIT = np.array([0.35, 0.35, 0.15, 0.15])


def _cf_calibration_multiplier(gamma):
    ag = abs(gamma)
    return 1.0 if ag < 0.2 else 1.0 + 0.0234 * ag ** 2 + 0.0125 * ag ** 3


def _apply_skew(z, gamma):
    if abs(gamma) < 0.01:
        return z
    ga = gamma * _cf_calibration_multiplier(gamma)
    zs = z + (ga / 6.0) * (z ** 2 - 1.0)
    return zs / np.sqrt(1.0 + ga ** 2 / 18.0)


def run(inp, seed=789):
    """inp keys: mu (n,4), std_course (n,4), eff_skew (n,4), l_corr (4,4),
    skill (n,), wx_delta (n,), player_avg (n,), sims. Returns (scores (n,sims) int,
    cat_mu (n,4))."""
    RNG_CF = np.random.default_rng(seed)
    mu = inp["mu"]
    std_course = inp["std_course"]
    eff_skew = inp["eff_skew"]
    L = inp["l_corr"]
    skill = inp["skill"]
    wx = inp["wx_delta"]
    pavg = inp["player_avg"]
    SIMS = inp["sims"]
    n = mu.shape[0]

    scores = np.empty((n, SIMS), dtype=np.int64)
    cat_mu_out = np.empty((n, 4))
    for i in range(n):
        m = mu[i]
        shift = (skill[i] - m.sum()) / 4.0
        cat_mu = m + shift + wx[i] * WEATHER_CAT_SPLIT
        cat_mu_out[i] = cat_mu
        Z = RNG_CF.standard_normal(size=(SIMS, 4))
        corr_z = Z @ L.T
        for j in range(4):
            corr_z[:, j] = _apply_skew(corr_z[:, j], eff_skew[i, j])
        draws = np.clip(cat_mu + corr_z * std_course[i], *CLIP_CAT)
        sg_total = draws.sum(axis=1)
        sc = np.rint(pavg[i] - sg_total).astype(int)
        lo = int(round(pavg[i])) - 12
        hi = int(round(pavg[i])) + 12
        scores[i] = np.clip(sc, lo, hi)
    return scores, cat_mu_out

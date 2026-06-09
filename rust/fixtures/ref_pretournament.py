"""Standalone Python reference for the new_sim.py pre-tournament cascade.

This is a faithful, near-verbatim copy of new_sim.py:589-912 (the seed-456 draw
kernel), refactored into a pure `run(inp, seed)` function. It serves two roles:

  1. The executable spec of the §11 "frozen-input contract" — it takes ONLY
     precomputed arrays + a seed and returns `final_scores` (+ win_prob), proving
     the kernel boundary is a pure `(inputs, seed) -> outputs` mapping with no
     hidden in-kernel data loading.
  2. The statistical oracle for `test_cascade_parity.py`: the Rust kernel uses a
     different RNG stream, so per-cell parity is impossible; instead we compare
     aggregate distributions within Monte-Carlo SE.

Keep this aligned with new_sim.py. When the production cascade changes, update
here too (and re-run the parity test). See RUST_MIGRATION_PLAN.md §7.
"""

import numpy as np

WEATHER_CAT_SPLIT = np.array([0.35, 0.35, 0.15, 0.15])
CLIP_CAT = (-8.0, 8.0)
PAR = 72.0


def _cf_calibration_multiplier(gamma):
    ag = abs(gamma)
    if ag < 0.2:
        return 1.0
    return 1.0 + 0.0234 * ag ** 2 + 0.0125 * ag ** 3


def _apply_skew(z, gamma):
    if abs(gamma) < 0.01:
        return z
    gamma_adj = gamma * _cf_calibration_multiplier(gamma)
    z_skewed = z + (gamma_adj / 6.0) * (z ** 2 - 1.0)
    z_skewed /= np.sqrt(1.0 + gamma_adj ** 2 / 18.0)
    return z_skewed


def _rank_min(strokes_col):
    import pandas as pd
    return pd.Series(strokes_col).rank(method="min").astype(int).to_numpy()


def run(inp, seed=456):
    """inp: dict of numpy arrays + coeff dicts (see test). Returns (final_scores, win_prob)."""
    RNG = np.random.default_rng(seed)
    mu = inp["mu"]                       # (n,4)
    std_course = inp["std_course"]       # (n,4)
    eff_skew = inp["eff_skew"]           # (n,4)
    L = inp["l_corr"]                    # (4,4)
    my_pred = inp["my_pred_base"]        # (n,)
    r2_mu, r3_mu, r4_mu = inp["r2_mu"], inp["r3_mu"], inp["r4_mu"]
    wd1, wd2 = inp["weather_delta_r1"], inp["weather_delta_r2"]
    SIMS = inp["sims"]
    CUT_LINE = inp["cut_line"]
    USE_10 = inp["use_10_shot_rule"]
    n = mu.shape[0]

    def coeff_vec_r1(c):
        return np.array([c["ott"], 0.0, 0.0, c["putt"], c["residual"], c["residual2"]])

    # ---- R1 draws ----
    cats_r1 = np.empty((n, SIMS, 4))
    sg_r1 = np.empty((n, SIMS))
    for i in range(n):
        cat_mu = mu[i] - wd1[i] * WEATHER_CAT_SPLIT
        Z = RNG.standard_normal(size=(SIMS, 4))
        corr_z = Z @ L.T
        for j in range(4):
            corr_z[:, j] = _apply_skew(corr_z[:, j], eff_skew[i, j])
        draws = cat_mu + corr_z * std_course[i]
        cats_r1[i] = np.clip(draws, *CLIP_CAT)
        sg_r1[i] = cats_r1[i].sum(axis=1)
    strokes_r1 = np.rint(PAR - sg_r1).astype(int)

    # ---- R1 -> R2 update ----
    resid_r1 = sg_r1 - my_pred[:, None]
    resid2_r1 = resid_r1 ** 2
    high_m = my_pred > 1.0
    midh_m = (my_pred > 0.5) & (my_pred <= 1.0)
    midl_m = (my_pred > -0.5) & (my_pred <= 0.5)
    low_m = my_pred <= -0.5
    C = np.zeros((n, 6))
    C[high_m] = coeff_vec_r1(inp["r1_high"])
    C[midh_m] = coeff_vec_r1(inp["r1_midh"])
    C[midl_m] = coeff_vec_r1(inp["r1_midl"])
    C[low_m] = coeff_vec_r1(inp["r1_low"])
    tot_resid_adj_r1 = resid_r1 * C[:, [4]] + resid2_r1 * C[:, [5]]
    mask_bad = (resid_r1 < 0) & (tot_resid_adj_r1 > 0.2)
    tot_resid_adj_r1 = np.minimum(np.where(mask_bad, 0.2, tot_resid_adj_r1), 0.5)
    sg_adj_r1 = cats_r1[:, :, 0] * C[:, [0]] + cats_r1[:, :, 3] * C[:, [3]]
    total_adjustment_r1 = tot_resid_adj_r1 + sg_adj_r1
    updated_skill_r2 = my_pred[:, None] + total_adjustment_r1
    sg_r2_mean = updated_skill_r2 + (r2_mu - my_pred)[:, None]

    # ---- R2 draws ----
    cats_r2 = np.empty((n, SIMS, 4))
    sg_r2 = np.empty((n, SIMS))
    for i in range(n):
        cat_mu = mu[i] - wd2[i] * WEATHER_CAT_SPLIT
        base_total_mu = mu[i].sum() - wd2[i]
        skill_shift = sg_r2_mean[i] - base_total_mu
        cat_mu_shifted = cat_mu + skill_shift[:, None] / 4.0
        Z = RNG.standard_normal(size=(SIMS, 4))
        corr_z = Z @ L.T
        for j in range(4):
            corr_z[:, j] = _apply_skew(corr_z[:, j], eff_skew[i, j])
        draws = cat_mu_shifted + corr_z * std_course[i]
        cats_r2[i] = np.clip(draws, *CLIP_CAT)
        sg_r2[i] = cats_r2[i].sum(axis=1)
    strokes_r2 = np.rint(PAR - sg_r2).astype(int)
    r1_r2 = strokes_r1 + strokes_r2

    # ---- cut ----
    made_cut = np.ones_like(r1_r2, dtype=bool)
    if CUT_LINE < n:
        for s in range(SIMS):
            sc = r1_r2[:, s]
            cut_score = np.sort(sc)[CUT_LINE - 1]
            top_cut = sc <= cut_score
            made_cut[:, s] = (top_cut | (sc <= sc.min() + 10)) if USE_10 else top_cut

    # ---- R2 -> R3 update ----
    resid_r2 = sg_r2 - sg_r2_mean
    resid2_r2 = resid_r2 ** 2
    resid3_r2 = resid_r2 ** 3
    avg_ott = 0.5 * (cats_r1[:, :, 0] + cats_r2[:, :, 0])
    avg_app = 0.5 * (cats_r1[:, :, 1] + cats_r2[:, :, 1])
    avg_arg = 0.5 * (cats_r1[:, :, 2] + cats_r2[:, :, 2])
    avg_putt = 0.5 * (cats_r1[:, :, 3] + cats_r2[:, :, 3])
    delta_app = cats_r2[:, :, 1] - cats_r1[:, :, 1]
    pos = np.empty((n, SIMS), dtype=int)
    for s in range(SIMS):
        pos[:, s] = _rank_min(r1_r2[:, s])

    def r2_adj(cd, mask):
        return mask * (
            resid_r2 * cd["residual"] + resid2_r2 * cd["residual2"] + resid3_r2 * cd["residual3"]
        ), mask * (
            avg_ott * cd["avg_ott"] + avg_putt * cd["avg_putt"] + avg_app * cd["avg_app"]
            + avg_arg * cd["avg_arg"] + delta_app * cd["delta_app"]
        )

    m_lt6 = pos < 6
    m_6_30 = (pos >= 6) & (pos <= 30)
    m_30up = pos > 30
    tr1, ts1 = r2_adj(inp["r2_lt6"], m_lt6)
    tr2, ts2 = r2_adj(inp["r2_6_30"], m_6_30)
    tr3, ts3 = r2_adj(inp["r2_30up"], m_30up)
    tot_resid_adj_r2 = tr1 + tr2 + tr3
    tot_sg_adj_r2 = ts1 + ts2 + ts3
    total_adjustment_r2 = (tot_resid_adj_r2 + tot_sg_adj_r2) - sg_adj_r1
    updated_skill_r3 = updated_skill_r2 + total_adjustment_r2
    sg_r3_mean = updated_skill_r3 + (r3_mu - my_pred)[:, None]

    # ---- R3 draws ----
    cats_r3 = np.empty((n, SIMS, 4))
    sg_r3 = np.empty((n, SIMS))
    for i in range(n):
        base_total_mu = mu[i].sum()
        skill_shift = sg_r3_mean[i] - base_total_mu
        cat_mu_shifted = mu[i] + skill_shift[:, None] / 4.0
        Z = RNG.standard_normal(size=(SIMS, 4))
        corr_z = Z @ L.T
        for j in range(4):
            corr_z[:, j] = _apply_skew(corr_z[:, j], eff_skew[i, j])
        draws = cat_mu_shifted + corr_z * std_course[i]
        cats_r3[i] = np.clip(draws, *CLIP_CAT)
        sg_r3[i] = cats_r3[i].sum(axis=1)
    strokes_r3 = np.rint(PAR - sg_r3).astype(int)
    r1_r3 = r1_r2 + strokes_r3

    # ---- R3 -> R4 update ----
    a_ott = 0.66 * (0.5 * (cats_r1[:, :, 0] + cats_r2[:, :, 0])) + 0.34 * cats_r3[:, :, 0]
    a_app = 0.66 * (0.5 * (cats_r1[:, :, 1] + cats_r2[:, :, 1])) + 0.34 * cats_r3[:, :, 1]
    a_arg = 0.66 * (0.5 * (cats_r1[:, :, 2] + cats_r2[:, :, 2])) + 0.34 * cats_r3[:, :, 2]
    a_putt = 0.66 * (0.5 * (cats_r1[:, :, 3] + cats_r2[:, :, 3])) + 0.34 * cats_r3[:, :, 3]
    pos3 = np.empty((n, SIMS), dtype=int)
    for s in range(SIMS):
        pos3[:, s] = _rank_min(r1_r3[:, s])

    def r3_adj(cd, mask):
        return mask * (
            a_ott * cd["sg_ott_avg"] + a_putt * cd["sg_putt_avg"]
            + a_app * cd["sg_app_avg"] + a_arg * cd["sg_arg_avg"]
        )

    ts_r3 = (
        r3_adj(inp["r3_lt6"], pos3 < 6)
        + r3_adj(inp["r3_6_20"], (pos3 >= 6) & (pos3 <= 20))
        + r3_adj(inp["r3_30up"], pos3 > 20)
    )
    updated_skill_r4 = updated_skill_r3 - (tot_sg_adj_r2 + tot_resid_adj_r2) + ts_r3
    sg_r4_mean = updated_skill_r4 + (r4_mu - my_pred)[:, None]

    # ---- R4 draws ----
    cats_r4 = np.empty((n, SIMS, 4))
    sg_r4 = np.empty((n, SIMS))
    for i in range(n):
        base_total_mu = mu[i].sum()
        skill_shift = sg_r4_mean[i] - base_total_mu
        cat_mu_shifted = mu[i] + skill_shift[:, None] / 4.0
        Z = RNG.standard_normal(size=(SIMS, 4))
        corr_z = Z @ L.T
        for j in range(4):
            corr_z[:, j] = _apply_skew(corr_z[:, j], eff_skew[i, j])
        draws = cat_mu_shifted + corr_z * std_course[i]
        cats_r4[i] = np.clip(draws, *CLIP_CAT)
        sg_r4[i] = cats_r4[i].sum(axis=1)
    strokes_r4 = np.rint(PAR - sg_r4).astype(int)

    r3_r4 = strokes_r3 + strokes_r4
    r3_r4[~made_cut] = 200
    final_scores = r1_r2 + r3_r4

    # win column
    win_counts = np.zeros(n)
    for s in range(SIMS):
        sc = final_scores[:, s]
        tied = np.where(sc == sc.min())[0]
        win_counts[RNG.choice(tied)] += 1
    win_prob = win_counts / SIMS

    return final_scores.astype(np.int64), win_prob

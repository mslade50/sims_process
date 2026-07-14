"""Standalone Python reference for round_sim.py `simulate_remaining_rounds`.

Faithful near-verbatim copy of round_sim.py:508-852 (seed-42 tournament cascade +
compute_finish_probabilities), refactored to a pure `run(inp, seed)` function. Same
two roles as ref_pretournament.py: executable spec of the frozen-input contract and
the statistical oracle for test_round_cascade_parity.py.

Keep aligned with round_sim.py. See RUST_MIGRATION_PLAN.md §6.2/§7.
"""

import numpy as np
import pandas as pd

CLIP_CAT = (-8.0, 8.0)


def _cf_calibration_multiplier(gamma):
    ag = abs(gamma)
    return 1.0 if ag < 0.2 else 1.0 + 0.0234 * ag ** 2 + 0.0125 * ag ** 3


def _apply_skew(z, gamma):
    if abs(gamma) < 0.01:
        return z
    ga = gamma * _cf_calibration_multiplier(gamma)
    zs = z + (ga / 6.0) * (z ** 2 - 1.0)
    return zs / np.sqrt(1.0 + ga ** 2 / 18.0)


def _catfirst_draw(mu, std_c, eff_skew, skill_mean, L, rng, num_sims):
    skill_shift = skill_mean - mu.sum()
    if np.ndim(skill_shift) == 0:
        cat_mu = mu + skill_shift / 4.0
    else:
        cat_mu = mu + skill_shift[:, None] / 4.0
    Z = rng.standard_normal(size=(num_sims, 4))
    corr_z = Z @ L.T
    for j in range(4):
        corr_z[:, j] = _apply_skew(corr_z[:, j], eff_skew[j])
    draws = cat_mu + corr_z * std_c
    cats = np.clip(draws, *CLIP_CAT)
    return cats, cats.sum(axis=1)


def coeff_vec_r1(c):
    return np.array([c["ott"], 0.0, 0.0, c["putt"], c["residual"], c["residual2"]])


def _rank_min(col):
    return pd.Series(col).rank(method="min").astype(int).to_numpy()


def run(inp, seed=42):
    RNG = np.random.default_rng(seed)
    completed = inp["completed_round"]
    par = inp["default_par"]
    mu = inp["mu"]
    std_course = inp["std_course"]
    eff_skew = inp["eff_skew"]
    L = inp["l_corr"]
    my_pred = inp["my_pred_base"]
    expected_r2 = inp["expected_r2"]
    known_strokes = inp["known_strokes"]   # dict {round: (n,) int}
    known_categories = inp["known_categories"]
    SIMS = inp["sims"]
    CUT_LINE = inp["cut_line"]
    USE_10 = inp["use_10_shot_rule"]
    n = mu.shape[0]
    params = list(zip(mu, std_course))

    # ---- R1 ----
    if completed >= 1 and 1 in known_strokes:
        strokes_r1 = np.tile(known_strokes[1][:, None], (1, SIMS))
        cats_r1 = np.tile(known_categories.get(1, np.zeros((n, 4)))[:, None, :], (1, SIMS, 1))
    else:
        cats_r1 = np.empty((n, SIMS, 4)); sg_r1 = np.empty((n, SIMS))
        for i, (m, sc) in enumerate(params):
            cats_r1[i], sg_r1[i] = _catfirst_draw(m, sc, eff_skew[i], my_pred[i], L, RNG, SIMS)
        strokes_r1 = np.clip(np.rint(par - sg_r1), par - 12, par + 12).astype(int)

    # ---- R1 -> R2 update ----
    sg_r1_actual = par - strokes_r1.astype(float)
    # field_skill offset added to production (round_sim.py:731-732, mirroring
    # live_stats _residuals_r1) after this spec was frozen 2026-06; kernel has it
    field_skill = float(np.mean(my_pred))
    resid_r1 = sg_r1_actual + field_skill - my_pred[:, None]
    resid2_r1 = resid_r1 ** 2
    C = np.zeros((n, 6))
    C[my_pred > 1.0] = coeff_vec_r1(inp["r1_high"])
    C[(my_pred > 0.5) & (my_pred <= 1.0)] = coeff_vec_r1(inp["r1_midh"])
    C[(my_pred > -0.5) & (my_pred <= 0.5)] = coeff_vec_r1(inp["r1_midl"])
    C[my_pred <= -0.5] = coeff_vec_r1(inp["r1_low"])
    tot_resid_adj_r1 = resid_r1 * C[:, [4]] + resid2_r1 * C[:, [5]]
    mask_bad = (resid_r1 < 0) & (tot_resid_adj_r1 > 0.2)
    tot_resid_adj_r1 = np.minimum(np.where(mask_bad, 0.2, tot_resid_adj_r1), 0.5)
    sg_adj_r1 = cats_r1[:, :, 0] * C[:, [0]] + cats_r1[:, :, 3] * C[:, [3]]
    updated_skill_r2 = my_pred[:, None] + tot_resid_adj_r1 + sg_adj_r1

    # ---- R2 ----
    if completed >= 2 and 2 in known_strokes:
        strokes_r2 = np.tile(known_strokes[2][:, None], (1, SIMS))
        cats_r2 = np.tile(known_categories.get(2, np.zeros((n, 4)))[:, None, :], (1, SIMS, 1))
        sg_r2 = par - strokes_r2.astype(float)
    else:
        cats_r2 = np.empty((n, SIMS, 4)); sg_r2 = np.empty((n, SIMS))
        for i, (m, sc) in enumerate(params):
            cats_r2[i], sg_r2[i] = _catfirst_draw(m, sc, eff_skew[i], updated_skill_r2[i], L, RNG, SIMS)
        strokes_r2 = np.clip(np.rint(expected_r2[:, None] - sg_r2),
                             (expected_r2 - 12)[:, None], (expected_r2 + 12)[:, None]).astype(int)

    r1_r2 = strokes_r1 + strokes_r2

    # ---- cut ----
    made_cut = np.ones_like(r1_r2, dtype=bool)
    if completed < 2:
        for j in range(SIMS):
            sc = r1_r2[:, j]
            cut_score = np.sort(sc)[min(CUT_LINE - 1, len(sc) - 1)]
            top_cut = sc <= cut_score
            made_cut[:, j] = (top_cut | (sc <= sc.min() + 10)) if USE_10 else top_cut

    # ---- R2 -> R3 update ----
    resid_r2 = sg_r2 - updated_skill_r2
    resid2_r2 = resid_r2 ** 2
    resid3_r2 = resid_r2 ** 3
    avg_ott2 = 0.5 * (cats_r1[:, :, 0] + cats_r2[:, :, 0])
    avg_app2 = 0.5 * (cats_r1[:, :, 1] + cats_r2[:, :, 1])
    avg_arg2 = 0.5 * (cats_r1[:, :, 2] + cats_r2[:, :, 2])
    avg_putt2 = 0.5 * (cats_r1[:, :, 3] + cats_r2[:, :, 3])
    delta_app2 = cats_r2[:, :, 1] - cats_r1[:, :, 1]
    pos = np.empty((n, SIMS), dtype=int)
    for j in range(SIMS):
        pos[:, j] = _rank_min(r1_r2[:, j])

    def r2adj(cd, mask):
        tr = mask * (resid_r2 * cd["residual"] + resid2_r2 * cd["residual2"] + resid3_r2 * cd["residual3"])
        ts = mask * (avg_ott2 * cd["avg_ott"] + avg_putt2 * cd["avg_putt"] + avg_app2 * cd["avg_app"]
                     + avg_arg2 * cd["avg_arg"] + delta_app2 * cd["delta_app"])
        return tr, ts

    tr1, ts1 = r2adj(inp["r2_lt6"], pos < 6)
    tr2, ts2 = r2adj(inp["r2_6_30"], (pos >= 6) & (pos <= 30))
    tr3, ts3 = r2adj(inp["r2_30up"], pos > 30)
    tot_resid_adj_r2 = tr1 + tr2 + tr3
    tot_sg_adj_r2 = ts1 + ts2 + ts3
    updated_skill_r3 = updated_skill_r2 + (tot_resid_adj_r2 + tot_sg_adj_r2) - sg_adj_r1

    # ---- R3 ----
    if completed >= 3 and 3 in known_strokes:
        strokes_r3 = np.tile(known_strokes[3][:, None], (1, SIMS))
        cats_r3 = np.tile(known_categories.get(3, np.zeros((n, 4)))[:, None, :], (1, SIMS, 1))
    else:
        cats_r3 = np.empty((n, SIMS, 4)); sg_r3 = np.empty((n, SIMS))
        for i, (m, sc) in enumerate(params):
            cats_r3[i], sg_r3[i] = _catfirst_draw(m, sc, eff_skew[i], updated_skill_r3[i], L, RNG, SIMS)
        strokes_r3 = np.clip(np.rint(par - sg_r3), par - 12, par + 12).astype(int)

    r1_r3 = r1_r2 + strokes_r3

    # ---- R3 -> R4 update ----
    a_ott = 0.66 * avg_ott2 + 0.34 * cats_r3[:, :, 0]
    a_app = 0.66 * avg_app2 + 0.34 * cats_r3[:, :, 1]
    a_arg = 0.66 * avg_arg2 + 0.34 * cats_r3[:, :, 2]
    a_putt = 0.66 * avg_putt2 + 0.34 * cats_r3[:, :, 3]
    pos3 = np.empty((n, SIMS), dtype=int)
    for j in range(SIMS):
        pos3[:, j] = _rank_min(r1_r3[:, j])

    def r3adj(cd, mask):
        return mask * (a_ott * cd["sg_ott_avg"] + a_putt * cd["sg_putt_avg"]
                       + a_app * cd["sg_app_avg"] + a_arg * cd["sg_arg_avg"])

    tot_sg_adj_r3 = (r3adj(inp["r3_lt6"], pos3 < 6)
                     + r3adj(inp["r3_6_20"], (pos3 >= 6) & (pos3 <= 20))
                     + r3adj(inp["r3_30up"], pos3 > 20))
    # pos_6_10 LEVEL term: flat add for positions 6-10 into R4 (mid bucket only)
    tot_sg_adj_r3 = tot_sg_adj_r3 + ((pos3 >= 6) & (pos3 <= 10)) * inp["r3_6_20"].get("pos_6_10", 0.0)
    updated_skill_r4 = updated_skill_r3 - (tot_sg_adj_r2 + tot_resid_adj_r2) + tot_sg_adj_r3

    # ---- R4 (always simulated) ----
    cats_r4 = np.empty((n, SIMS, 4)); sg_r4 = np.empty((n, SIMS))
    for i, (m, sc) in enumerate(params):
        cats_r4[i], sg_r4[i] = _catfirst_draw(m, sc, eff_skew[i], updated_skill_r4[i], L, RNG, SIMS)
    strokes_r4 = np.clip(np.rint(par - sg_r4), par - 12, par + 12).astype(int)

    r3_r4 = strokes_r3 + strokes_r4
    r3_r4[~made_cut] = 200
    final_scores = (r1_r2 + r3_r4).astype(np.int64)

    # win column
    win = np.zeros(n)
    for j in range(SIMS):
        sc = final_scores[:, j]
        tied = np.where(sc == sc.min())[0]
        win[RNG.choice(tied)] += 1
    return final_scores, made_cut, win / SIMS

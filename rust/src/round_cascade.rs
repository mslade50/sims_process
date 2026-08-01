//! `run_remaining_rounds` — round_sim.py seed-42 tournament cascade.
//!
//! Pure Rust port of round_sim.py `simulate_remaining_rounds` (L508-793). Simulates
//! from round (completed_round+1) through R4 using category-first draws, seeding
//! completed rounds with their KNOWN strokes/categories.
//!
//! Differences from new_sim's `run_pretournament` (see cascade.rs):
//!  * Known-round seeding: rounds <= completed_round use tiled actual strokes/cats.
//!  * Stroke clipping: `clip(rint(expected - sg), expected-12, expected+12)`.
//!  * Multi-course R2: per-player `expected_r2` (course_score_adj), default_par else.
//!  * No weather split in the draw (tournament cascade); skill shift spread /4.
//!  * No r2_mu/r3_mu/r4_mu offsets — updated_skill_* is the draw skill mean directly.
//!  * R1 residual uses sg recomputed from ROUNDED strokes (`default_par - strokes_r1`).
//!
//! Parity invariants identical to cascade.rs: round_ties_even, exact min-rank for
//! bucket routing, serial seed-42 stream, injected L_corr.

use crate::cascade::{CoeffR1, CoeffR2, CoeffR3};
use crate::ops::{sum4, Skew};
use crate::rng::NormalStream;

const CLIP_CAT: (f64, f64) = (-8.0, 8.0);
const MISSED_CUT_PENALTY: i64 = 200;
const STROKE_CLIP: f64 = 12.0;
// Phase 5 single-round draw splits weather across categories (added back here).
const WEATHER_CAT_SPLIT: [f64; 4] = [0.35, 0.35, 0.15, 0.15];

/// A completed round's known data (constant across sims, tiled).
pub struct KnownRound {
    pub strokes: Vec<i64>,   // (n,)
    pub cats: Vec<[f64; 4]>, // (n,4)
}

pub struct Inputs {
    pub n: usize,
    pub sims: usize,
    pub seed: u64, // 42
    pub completed_round: usize,
    pub default_par: f64,
    pub mu: Vec<[f64; 4]>,
    pub std_course: Vec<[f64; 4]>,
    pub eff_skew: Vec<[f64; 4]>,
    pub l_corr: [[f64; 4]; 4],
    pub my_pred_base: Vec<f64>,
    /// Per-player R2 expected score (course_score_adj; default_par where absent).
    pub expected_r2: Vec<f64>,
    /// Known rounds (Some when round <= completed_round). R4 is never known.
    pub known_r1: Option<KnownRound>,
    pub known_r2: Option<KnownRound>,
    pub known_r3: Option<KnownRound>,
    pub r1_high: CoeffR1,
    pub r1_midh: CoeffR1,
    pub r1_midl: CoeffR1,
    pub r1_low: CoeffR1,
    pub r2_lt6: CoeffR2,
    pub r2_6_30: CoeffR2,
    pub r2_30up: CoeffR2,
    pub r3_lt6: CoeffR3,
    pub r3_6_20: CoeffR3,
    pub r3_30up: CoeffR3,
    pub cut_line: usize,
    pub use_10_shot_rule: bool,
}

pub struct Output {
    pub n: usize,
    pub sims: usize,
    pub final_scores: Vec<i64>, // (n*sims) row-major
    pub made_cut: Vec<bool>,    // (n*sims) row-major
    pub win_prob: Vec<f64>,     // (n,)
    /// End-of-R2 cumulative strokes (n*sims) row-major. Full field (cut not applied).
    /// Feeds Kalshi "Round 2 Leader / Top-N" pricing.
    pub r1_r2: Vec<i64>,
    /// End-of-R3 cumulative strokes (n*sims) row-major. RAW (cut NOT applied here);
    /// the caller masks missed-cut sims via `made_cut` before ranking R3 standings.
    pub r1_r3: Vec<i64>,
}

#[inline]
fn idx(i: usize, s: usize, sims: usize) -> usize {
    i * sims + s
}

/// Clip-rint to integer strokes: `clip(rint(expected - sg), expected-12, expected+12)`
/// then truncate to i64 (matches numpy astype(int) on the float-clipped value).
#[inline]
fn strokes_from_sg(expected: f64, sg: f64) -> i64 {
    let r = (expected - sg).round_ties_even();
    let c = r.clamp(expected - STROKE_CLIP, expected + STROKE_CLIP);
    c as i64
}

/// Draw one simulated round for all players (no weather; skill shift spread /4).
/// `skill_mean[i*sims+s]` is the per-(player,sim) skill target; `expected[i]` the
/// per-player expected score used for stroke conversion + clipping.
#[allow(clippy::too_many_arguments)]
fn draw_round(
    stream: &mut NormalStream,
    n: usize,
    sims: usize,
    mu: &[[f64; 4]],
    std_course: &[[f64; 4]],
    skew: &[[Skew; 4]],
    l_corr: &[[f64; 4]; 4],
    skill_mean: &[f64],
    expected: &[f64],
    cats: &mut [[f64; 4]],
    sg: &mut [f64],
    strokes: &mut [i64],
) {
    for i in 0..n {
        let m = mu[i];
        let base_sum = m[0] + m[1] + m[2] + m[3];
        let sc = std_course[i];
        let sk = &skew[i];
        let exp_i = expected[i];
        for s in 0..sims {
            let z = stream.next_row4();
            let k = idx(i, s, sims);
            let shift = (skill_mean[k] - base_sum) / 4.0;
            let mut c = [0.0f64; 4];
            for j in 0..4 {
                let l = l_corr[j];
                let corr_z = z[0] * l[0] + z[1] * l[1] + z[2] * l[2] + z[3] * l[3];
                let skewed = sk[j].apply(corr_z);
                let draw = (m[j] + shift) + skewed * sc[j];
                c[j] = draw.clamp(CLIP_CAT.0, CLIP_CAT.1);
            }
            let total = sum4(c);
            cats[k] = c;
            sg[k] = total;
            strokes[k] = strokes_from_sg(exp_i, total);
        }
    }
}

/// Fill a known round: tile constant strokes/cats across sims; sg = par - strokes.
fn fill_known(
    known: &KnownRound,
    n: usize,
    sims: usize,
    sg_par: f64,
    cats: &mut [[f64; 4]],
    sg: &mut [f64],
    strokes: &mut [i64],
) {
    for i in 0..n {
        let st = known.strokes[i];
        let ct = known.cats[i];
        let sgv = sg_par - st as f64;
        for s in 0..sims {
            let k = idx(i, s, sims);
            cats[k] = ct;
            strokes[k] = st;
            sg[k] = sgv;
        }
    }
}

fn min_rank_col(strokes: &[i64], n: usize, sims: usize, s: usize, pos: &mut [i64], order: &mut [usize]) {
    for (k, o) in order.iter_mut().enumerate() {
        *o = k;
    }
    order.sort_by(|&a, &b| strokes[idx(a, s, sims)].cmp(&strokes[idx(b, s, sims)]));
    let mut i = 0usize;
    while i < n {
        let v = strokes[idx(order[i], s, sims)];
        let group_rank = (i as i64) + 1;
        let mut k = i;
        while k < n && strokes[idx(order[k], s, sims)] == v {
            pos[order[k]] = group_rank;
            k += 1;
        }
        i = k;
    }
}

pub fn run_remaining_rounds(inp: &Inputs) -> Output {
    let n = inp.n;
    let sims = inp.sims;
    let ns = n * sims;
    let par = inp.default_par;

    let skew: Vec<[Skew; 4]> = (0..n)
        .map(|i| {
            let e = inp.eff_skew[i];
            [Skew::new(e[0]), Skew::new(e[1]), Skew::new(e[2]), Skew::new(e[3])]
        })
        .collect();
    let par_expected: Vec<f64> = vec![par; n];

    let mut stream = NormalStream::new(inp.seed);

    // ---- R1 (known or simulated) ----
    let mut cats_r1 = vec![[0.0f64; 4]; ns];
    let mut sg_r1 = vec![0.0f64; ns];
    let mut strokes_r1 = vec![0i64; ns];
    if let Some(k1) = &inp.known_r1 {
        fill_known(k1, n, sims, par, &mut cats_r1, &mut sg_r1, &mut strokes_r1);
    } else {
        // skill_mean = my_pred_base[i] broadcast across sims
        let sm: Vec<f64> = (0..ns).map(|k| inp.my_pred_base[k / sims]).collect();
        draw_round(&mut stream, n, sims, &inp.mu, &inp.std_course, &skew, &inp.l_corr,
            &sm, &par_expected, &mut cats_r1, &mut sg_r1, &mut strokes_r1);
    }

    // ---- R1 -> R2 skill update ----
    // sg_r1_actual = default_par - strokes_r1 (rounded/known), NOT raw sg.
    // Field skill = mean(my_pred) over the field, added to the R1 residual to match
    // live_stats _residuals_r1 (residual = sg_total + pred_avg - pred) and new_sim's
    // run_pretournament baseline. resid = sg_r1 + field_skill - my_pred.
    let field_skill = inp.my_pred_base.iter().sum::<f64>() / n as f64;
    let mut sg_adj_r1 = vec![0.0f64; ns];
    let mut updated_skill_r2 = vec![0.0f64; ns];
    for i in 0..n {
        let mp = inp.my_pred_base[i];
        let c = if mp > 1.0 {
            inp.r1_high
        } else if mp > 0.5 {
            inp.r1_midh
        } else if mp > -0.5 {
            inp.r1_midl
        } else {
            inp.r1_low
        };
        for s in 0..sims {
            let k = idx(i, s, sims);
            let sg_actual = par - strokes_r1[k] as f64;
            let resid = sg_actual + field_skill - mp;
            let resid2 = resid * resid;
            let mut tr = resid * c.residual + resid2 * c.residual2;
            if resid < 0.0 && tr > 0.2 {
                tr = 0.2;
            }
            if tr > 0.5 {
                tr = 0.5;
            }
            // Floor -0.75 for resid [-8,-6), -0.5 elsewhere: parity with
            // live_stats_engine _totals_r1.
            // User risk rule 2026-08: +/-0.5 everywhere (former -0.75 band retired).
            let floor = -0.5;
            if tr < floor {
                tr = floor;
            }
            let sg_adj = cats_r1[k][0] * c.ott + cats_r1[k][3] * c.putt;
            sg_adj_r1[k] = sg_adj;
            updated_skill_r2[k] = mp + tr + sg_adj;
        }
    }

    // ---- R2 (known or simulated) ----
    let mut cats_r2 = vec![[0.0f64; 4]; ns];
    let mut sg_r2 = vec![0.0f64; ns];
    let mut strokes_r2 = vec![0i64; ns];
    if let Some(k2) = &inp.known_r2 {
        fill_known(k2, n, sims, par, &mut cats_r2, &mut sg_r2, &mut strokes_r2);
    } else {
        draw_round(&mut stream, n, sims, &inp.mu, &inp.std_course, &skew, &inp.l_corr,
            &updated_skill_r2, &inp.expected_r2, &mut cats_r2, &mut sg_r2, &mut strokes_r2);
    }

    let mut r1_r2 = vec![0i64; ns];
    for k in 0..ns {
        r1_r2[k] = strokes_r1[k] + strokes_r2[k];
    }

    // ---- Cut after 36 (only simulated when completed_round < 2) ----
    let mut made_cut = vec![true; ns];
    if inp.completed_round < 2 && inp.cut_line >= 1 {
        let cl = inp.cut_line.min(n);
        let mut order = vec![0usize; n];
        let mut colvals = vec![0i64; n];
        for s in 0..sims {
            for i in 0..n {
                colvals[i] = r1_r2[idx(i, s, sims)];
                order[i] = i;
            }
            order.sort_by(|&a, &b| colvals[a].cmp(&colvals[b]));
            let cut_score = colvals[order[cl - 1]];
            let min_score = colvals[order[0]];
            for i in 0..n {
                let sc = colvals[i];
                let top_cut = sc <= cut_score;
                let made = if inp.use_10_shot_rule {
                    top_cut || sc <= min_score + 10
                } else {
                    top_cut
                };
                made_cut[idx(i, s, sims)] = made;
            }
        }
    }

    // ---- R2 -> R3 update ----
    let mut tot_resid_adj_r2 = vec![0.0f64; ns];
    let mut tot_sg_adj_r2 = vec![0.0f64; ns];
    let mut updated_skill_r3 = vec![0.0f64; ns];
    {
        let mut pos = vec![0i64; n];
        let mut order = vec![0usize; n];
        for s in 0..sims {
            min_rank_col(&r1_r2, n, sims, s, &mut pos, &mut order);
            for i in 0..n {
                let k = idx(i, s, sims);
                let c = if pos[i] < 6 {
                    inp.r2_lt6
                } else if pos[i] <= 30 {
                    inp.r2_6_30
                } else {
                    inp.r2_30up
                };
                // Cap at +6 before the cubic (parity: Python RESID_FIX_CAP).
                let resid = (sg_r2[k] - updated_skill_r2[k]).min(6.0);
                let resid2 = resid * resid;
                let resid3 = resid2 * resid;
                let avg_ott = 0.5 * (cats_r1[k][0] + cats_r2[k][0]);
                let avg_app = 0.5 * (cats_r1[k][1] + cats_r2[k][1]);
                let avg_arg = 0.5 * (cats_r1[k][2] + cats_r2[k][2]);
                let avg_putt = 0.5 * (cats_r1[k][3] + cats_r2[k][3]);
                let delta_app = cats_r2[k][1] - cats_r1[k][1];
                // +/-0.5 clip (user risk rule 2026-08): parity with live_stats_engine.
                let tr = (resid * c.residual + resid2 * c.residual2 + resid3 * c.residual3)
                    .max(-0.5)
                    .min(0.5);
                let ts = avg_ott * c.avg_ott
                    + avg_putt * c.avg_putt
                    + avg_app * c.avg_app
                    + avg_arg * c.avg_arg
                    + delta_app * c.delta_app;
                tot_resid_adj_r2[k] = tr;
                tot_sg_adj_r2[k] = ts;
                updated_skill_r3[k] = updated_skill_r2[k] + (tr + ts) - sg_adj_r1[k];
            }
        }
    }

    // ---- R3 (known or simulated) ----
    let mut cats_r3 = vec![[0.0f64; 4]; ns];
    let mut sg_r3 = vec![0.0f64; ns];
    let mut strokes_r3 = vec![0i64; ns];
    if let Some(k3) = &inp.known_r3 {
        fill_known(k3, n, sims, par, &mut cats_r3, &mut sg_r3, &mut strokes_r3);
    } else {
        draw_round(&mut stream, n, sims, &inp.mu, &inp.std_course, &skew, &inp.l_corr,
            &updated_skill_r3, &par_expected, &mut cats_r3, &mut sg_r3, &mut strokes_r3);
    }

    let mut r1_r3 = vec![0i64; ns];
    for k in 0..ns {
        r1_r3[k] = r1_r2[k] + strokes_r3[k];
    }

    // ---- R3 -> R4 update (avg-SG only) ----
    let mut updated_skill_r4 = vec![0.0f64; ns];
    {
        let mut pos = vec![0i64; n];
        let mut order = vec![0usize; n];
        for s in 0..sims {
            min_rank_col(&r1_r3, n, sims, s, &mut pos, &mut order);
            for i in 0..n {
                let k = idx(i, s, sims);
                let c = if pos[i] < 6 {
                    inp.r3_lt6
                } else if pos[i] <= 20 {
                    inp.r3_6_20
                } else {
                    inp.r3_30up
                };
                let avg_ott = 0.66 * (0.5 * (cats_r1[k][0] + cats_r2[k][0])) + 0.34 * cats_r3[k][0];
                let avg_app = 0.66 * (0.5 * (cats_r1[k][1] + cats_r2[k][1])) + 0.34 * cats_r3[k][1];
                let avg_arg = 0.66 * (0.5 * (cats_r1[k][2] + cats_r2[k][2])) + 0.34 * cats_r3[k][2];
                let avg_putt = 0.66 * (0.5 * (cats_r1[k][3] + cats_r2[k][3])) + 0.34 * cats_r3[k][3];
                let mut ts = avg_ott * c.sg_ott_avg
                    + avg_putt * c.sg_putt_avg
                    + avg_app * c.sg_app_avg
                    + avg_arg * c.sg_arg_avg;
                // level term gated to positions 6-10 into R4 (subset of the 6-20 bucket)
                if pos[i] >= 6 && pos[i] <= 10 {
                    ts += c.pos_6_10;
                }
                updated_skill_r4[k] = updated_skill_r3[k] - (tot_sg_adj_r2[k] + tot_resid_adj_r2[k]) + ts;
            }
        }
    }

    // ---- R4 (always simulated) ----
    let mut cats_r4 = vec![[0.0f64; 4]; ns];
    let mut sg_r4 = vec![0.0f64; ns];
    let mut strokes_r4 = vec![0i64; ns];
    draw_round(&mut stream, n, sims, &inp.mu, &inp.std_course, &skew, &inp.l_corr,
        &updated_skill_r4, &par_expected, &mut cats_r4, &mut sg_r4, &mut strokes_r4);

    // ---- finalize ----
    let mut final_scores = vec![0i64; ns];
    for k in 0..ns {
        let r3_r4 = if made_cut[k] {
            strokes_r3[k] + strokes_r4[k]
        } else {
            MISSED_CUT_PENALTY
        };
        final_scores[k] = r1_r2[k] + r3_r4;
    }

    // ---- win column (seed-42 choice tiebreak, after all draws) ----
    let mut win_counts = vec![0.0f64; n];
    let mut tied: Vec<usize> = Vec::with_capacity(8);
    for s in 0..sims {
        let mut min_score = i64::MAX;
        for i in 0..n {
            let v = final_scores[idx(i, s, sims)];
            if v < min_score {
                min_score = v;
            }
        }
        tied.clear();
        for i in 0..n {
            if final_scores[idx(i, s, sims)] == min_score {
                tied.push(i);
            }
        }
        let winner = if tied.len() == 1 {
            tied[0]
        } else {
            tied[stream.gen_index(tied.len())]
        };
        win_counts[winner] += 1.0;
    }
    for w in win_counts.iter_mut() {
        *w /= sims as f64;
    }

    Output {
        n,
        sims,
        final_scores,
        made_cut,
        win_prob: win_counts,
        r1_r2,
        r1_r3,
    }
}

// ============================================================
// Phase 5: single-round score card (seed 789, simulate_round_scores_catfirst)
// ============================================================

/// Inputs for the single-round category-first draw (round_sim.py:1884). One round,
/// per player. KEY DIFF from the cascade: weather IS split here and ADDED per
/// category (cat_mu = mu + shift + wx_delta*WEATHER_CAT_SPLIT), and the stroke clip
/// is to int(round(player_avg)) +/- 12. The glue filters NaN-score players out.
pub struct SingleRoundInputs {
    pub n: usize,
    pub sims: usize,
    pub seed: u64, // 789
    pub mu: Vec<[f64; 4]>,
    pub std_course: Vec<[f64; 4]>,
    pub eff_skew: Vec<[f64; 4]>,
    pub l_corr: [[f64; 4]; 4],
    /// scores_rN - wx_delta (round SG target with weather removed).
    pub skill: Vec<f64>,
    pub wx_delta: Vec<f64>,
    /// expected score for stroke conversion (course_score_adj or global avg).
    pub player_avg: Vec<f64>,
}

pub struct SingleRoundOutput {
    pub n: usize,
    pub sims: usize,
    pub scores: Vec<i64>, // (n*sims) row-major int round scores
    pub cat_mu: Vec<f64>, // (n*4) per-player re-centered category means (cat_mu_lookup)
}

/// rint(player_avg - sg) -> int, then clip to int(round(player_avg)) +/- 12.
/// Matches numpy `np.rint(...).astype(int)` then `np.clip(_, round(avg)-12, +12)`.
#[inline]
fn score_single(player_avg: f64, sg: f64) -> i64 {
    let center = player_avg.round_ties_even() as i64;
    let sc = (player_avg - sg).round_ties_even() as i64;
    sc.clamp(center - 12, center + 12)
}

pub fn run_single_round(inp: &SingleRoundInputs) -> SingleRoundOutput {
    let n = inp.n;
    let sims = inp.sims;
    let ns = n * sims;
    let skew: Vec<[Skew; 4]> = (0..n)
        .map(|i| {
            let e = inp.eff_skew[i];
            [Skew::new(e[0]), Skew::new(e[1]), Skew::new(e[2]), Skew::new(e[3])]
        })
        .collect();
    let mut stream = NormalStream::new(inp.seed);
    let mut scores = vec![0i64; ns];
    let mut cat_mu_out = vec![0.0f64; n * 4];
    for i in 0..n {
        let m = inp.mu[i];
        let base_sum = m[0] + m[1] + m[2] + m[3];
        let shift = (inp.skill[i] - base_sum) / 4.0;
        let wd = inp.wx_delta[i];
        let mut cat_mu = [0.0f64; 4];
        for j in 0..4 {
            cat_mu[j] = m[j] + shift + wd * WEATHER_CAT_SPLIT[j];
        }
        cat_mu_out[i * 4..i * 4 + 4].copy_from_slice(&cat_mu);
        let sc = inp.std_course[i];
        let sk = &skew[i];
        let pa = inp.player_avg[i];
        for s in 0..sims {
            let z = stream.next_row4();
            let mut c = [0.0f64; 4];
            for j in 0..4 {
                let l = inp.l_corr[j];
                let corr_z = z[0] * l[0] + z[1] * l[1] + z[2] * l[2] + z[3] * l[3];
                let skewed = sk[j].apply(corr_z);
                let draw = cat_mu[j] + skewed * sc[j];
                c[j] = draw.clamp(CLIP_CAT.0, CLIP_CAT.1);
            }
            let total = sum4(c);
            scores[idx(i, s, sims)] = score_single(pa, total);
        }
    }
    SingleRoundOutput { n, sims, scores, cat_mu: cat_mu_out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident_l() -> [[f64; 4]; 4] {
        let mut l = [[0.0f64; 4]; 4];
        for i in 0..4 {
            l[i][i] = 1.0;
        }
        l
    }

    fn base_inputs(n: usize, sims: usize, completed: usize) -> Inputs {
        let z1 = CoeffR1 { ott: 0.0, putt: 0.0, residual: 0.0, residual2: 0.0 };
        let z2 = CoeffR2 {
            residual: 0.0, residual2: 0.0, residual3: 0.0, avg_ott: 0.0,
            avg_putt: 0.0, avg_app: 0.0, avg_arg: 0.0, delta_app: 0.0,
        };
        let z3 = CoeffR3 { sg_ott_avg: 0.0, sg_putt_avg: 0.0, sg_app_avg: 0.0, sg_arg_avg: 0.0, pos_6_10: 0.0 };
        let mk_known = |val: i64| KnownRound {
            strokes: vec![val; n],
            cats: vec![[0.1, 0.1, 0.0, 0.0]; n],
        };
        Inputs {
            n, sims, seed: 42, completed_round: completed, default_par: 72.0,
            mu: vec![[0.1, 0.1, 0.0, 0.0]; n],
            std_course: vec![[1.0, 1.0, 0.8, 0.8]; n],
            eff_skew: vec![[0.0; 4]; n],
            l_corr: ident_l(),
            my_pred_base: (0..n).map(|i| i as f64 * 0.05 - 0.2).collect(),
            expected_r2: vec![72.0; n],
            known_r1: if completed >= 1 { Some(mk_known(71)) } else { None },
            known_r2: if completed >= 2 { Some(mk_known(70)) } else { None },
            known_r3: if completed >= 3 { Some(mk_known(72)) } else { None },
            r1_high: z1, r1_midh: z1, r1_midl: z1, r1_low: z1,
            r2_lt6: z2, r2_6_30: z2, r2_30up: z2,
            r3_lt6: z3, r3_6_20: z3, r3_30up: z3,
            cut_line: 65, use_10_shot_rule: true,
        }
    }

    #[test]
    fn pos_6_10_level_hits_only_ranks_6_to_10() {
        let n = 15usize;
        let sims = 4000usize;
        let mut inp = base_inputs(n, sims, 3);
        // distinct known strokes -> deterministic ranks 1..n into R4 in every sim
        for round in [&mut inp.known_r1, &mut inp.known_r2, &mut inp.known_r3] {
            if let Some(k) = round {
                k.strokes = (0..n).map(|i| 70 + i as i64).collect();
            }
        }
        inp.cut_line = n; // everyone survives the cut
        let base = run_remaining_rounds(&inp);
        inp.r3_6_20.pos_6_10 = -0.5;
        let with_term = run_remaining_rounds(&inp);
        // same seed + level-add consumes no RNG -> paired sims isolate the term:
        // affected players' skill drops 0.5 -> mean strokes rise ~0.5; others unmoved
        for i in 0..n {
            let mean = |o: &Output| {
                (0..sims).map(|s| o.final_scores[idx(i, s, sims)] as f64).sum::<f64>() / sims as f64
            };
            let delta = mean(&with_term) - mean(&base);
            let rank = i as i64 + 1;
            if (6..=10).contains(&rank) {
                assert!((delta - 0.5).abs() < 0.15, "rank {rank}: delta {delta} != ~0.5");
            } else {
                assert!(delta.abs() < 0.15, "rank {rank}: delta {delta} != ~0");
            }
        }
    }

    #[test]
    fn runs_round0_and_shapes() {
        let out = run_remaining_rounds(&base_inputs(20, 500, 0));
        assert_eq!(out.final_scores.len(), 20 * 500);
        assert_eq!(out.win_prob.len(), 20);
        let total: f64 = out.win_prob.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn deterministic_same_seed() {
        let a = run_remaining_rounds(&base_inputs(15, 300, 0));
        let b = run_remaining_rounds(&base_inputs(15, 300, 0));
        assert_eq!(a.final_scores, b.final_scores);
    }

    #[test]
    fn known_rounds_seeded_constant() {
        // completed=2: R1,R2 known and constant -> r1_r2 identical across sims.
        let out = run_remaining_rounds(&base_inputs(10, 200, 2));
        // R1=71, R2=70 known for all -> r1_r2 = 141 for everyone; only R3/R4 vary.
        // final = 141 + strokes_r3 + strokes_r4 (made cut, cut not simulated at completed>=2)
        for i in 0..10 {
            for s in 0..200 {
                let v = out.final_scores[idx(i, s, 200)];
                assert!(v >= 141 + 2 * (72 - 12) && v <= 141 + 2 * (72 + 12), "score {v}");
            }
        }
        // made_cut all true at completed>=2
        assert!(out.made_cut.iter().all(|&m| m));
    }

    #[test]
    fn r4_consumes_rng_when_completed3() {
        // completed=3: only R4 simulated; runs without panic, win sums to 1.
        let out = run_remaining_rounds(&base_inputs(12, 400, 3));
        let t: f64 = out.win_prob.iter().sum();
        assert!((t - 1.0).abs() < 1e-9);
    }

    #[test]
    fn single_round_runs_and_clips() {
        let n = 20;
        let sims = 500;
        let inp = SingleRoundInputs {
            n, sims, seed: 789,
            mu: vec![[0.1, 0.1, 0.0, 0.0]; n],
            std_course: vec![[1.0, 1.0, 0.8, 0.8]; n],
            eff_skew: vec![[0.0; 4]; n],
            l_corr: ident_l(),
            skill: (0..n).map(|i| i as f64 * 0.05 - 0.2).collect(),
            wx_delta: vec![0.3; n],
            player_avg: vec![71.0; n],
        };
        let out = run_single_round(&inp);
        assert_eq!(out.scores.len(), n * sims);
        assert_eq!(out.cat_mu.len(), n * 4);
        // round(71) +/- 12 = [59, 83]
        for &v in &out.scores {
            assert!((59..=83).contains(&v), "score {v} out of clip band");
        }
    }

    #[test]
    fn single_round_deterministic() {
        let mk = || SingleRoundInputs {
            n: 12, sims: 300, seed: 789,
            mu: vec![[0.1, 0.1, 0.0, 0.0]; 12],
            std_course: vec![[1.0, 1.0, 0.8, 0.8]; 12],
            eff_skew: vec![[0.0; 4]; 12],
            l_corr: ident_l(),
            skill: vec![0.2; 12],
            wx_delta: vec![0.0; 12],
            player_avg: vec![70.5; 12],
        };
        assert_eq!(run_single_round(&mk()).scores, run_single_round(&mk()).scores);
    }
}

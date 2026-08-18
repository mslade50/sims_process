//! `run_pretournament` — the new_sim.py seed-456 draw cascade (R1->R4).
//!
//! Pure Rust port of new_sim.py:589-912. All glue (config/pred loading, the 4x4
//! Cholesky, weather/tee APIs, file IO) stays in Python and is passed in as fixed
//! arrays; `L_corr` is injected (plan §3/§4). The only output that depends on the
//! RNG is the integer `final_scores` matrix (+ the win-tiebreak choice).
//!
//! Parity invariants honored here:
//!  * np.rint -> round_ties_even (ops::rint_i64)
//!  * rank(method='min') -> exact competition ranking (routes coeff buckets)
//!  * serial single RNG stream for all draws; choice tiebreak after
//!  * weather distributed via WEATHER_CAT_SPLIT; skill shift spread /4 evenly
//!
//! Statistical target: the RNG stream differs from numpy, so `final_scores`
//! differs sim-by-sim, but the aggregate distributions match within MC SE.

use crate::ops::{rint_i64, sum4, Skew};
use crate::rng::NormalStream;

const PAR: f64 = 72.0;
const WEATHER_CAT_SPLIT: [f64; 4] = [0.35, 0.35, 0.15, 0.15];
const CLIP_CAT: (f64, f64) = (-8.0, 8.0);
const MISSED_CUT_PENALTY: i64 = 200;

/// R1 skill-update coefficients (one of 4 my_pred buckets).
/// Mirrors `coeff_vec_r1`: only ott/putt/residual/residual2 are non-zero.
#[derive(Clone, Copy)]
pub struct CoeffR1 {
    pub ott: f64,
    pub putt: f64,
    pub residual: f64,
    pub residual2: f64,
}

/// R2 skill-update coefficients (one of 3 position buckets).
#[derive(Clone, Copy)]
pub struct CoeffR2 {
    pub residual: f64,
    pub residual2: f64,
    pub residual3: f64,
    pub avg_ott: f64,
    pub avg_putt: f64,
    pub avg_app: f64,
    pub avg_arg: f64,
    pub delta_app: f64,
}

/// R3 skill-update coefficients (one of 3 position buckets; avg-SG only).
/// pos_6_10 is a LEVEL add applied only when position-into-R4 is 6..=10
/// (live only in the 6-20 bucket dict; 0.0 elsewhere).
#[derive(Clone, Copy)]
pub struct CoeffR3 {
    pub sg_ott_avg: f64,
    pub sg_putt_avg: f64,
    pub sg_app_avg: f64,
    pub sg_arg_avg: f64,
    pub pos_6_10: f64,
}

/// Fully-precomputed kernel inputs (everything RNG-independent comes from Python).
pub struct Inputs {
    pub n: usize,
    pub sims: usize,
    pub seed: u64,
    /// (n,4) re-centered category means (sums to my_pred). Row-major.
    pub mu: Vec<[f64; 4]>,
    /// (n,4) per-player category stds * course mults.
    pub std_course: Vec<[f64; 4]>,
    /// (n,4) per-player effective skew.
    pub eff_skew: Vec<[f64; 4]>,
    /// 4x4 lower Cholesky (injected; not ported).
    pub l_corr: [[f64; 4]; 4],
    pub my_pred_base: Vec<f64>,
    pub r2_mu: Vec<f64>,
    pub r3_mu: Vec<f64>,
    pub r4_mu: Vec<f64>,
    pub weather_delta_r1: Vec<f64>,
    pub weather_delta_r2: Vec<f64>,
    // R1 buckets, selected by my_pred_base: >1.0 / (0.5,1.0] / (-0.5,0.5] / <=-0.5
    pub r1_high: CoeffR1,
    pub r1_midh: CoeffR1,
    pub r1_midl: CoeffR1,
    pub r1_low: CoeffR1,
    // R2 buckets, by rank: <6 / 6..=30 / >30
    pub r2_lt6: CoeffR2,
    pub r2_6_30: CoeffR2,
    pub r2_30up: CoeffR2,
    // R3 buckets, by rank: <6 / 6..=20 / >20
    pub r3_lt6: CoeffR3,
    pub r3_6_20: CoeffR3,
    pub r3_30up: CoeffR3,
    pub cut_line: usize,
    pub use_10_shot_rule: bool,
    /// Week-level form latent sigma (SG/round). One shared draw per
    /// (player, sim) added as +w/4 to every round's category means, with
    /// idiosyncratic category stds shrunk so per-round total variance is
    /// unchanged. 0.0 disables — bit-identical to the pre-latent cascade
    /// (the latent uses its own RNG stream, never the main seed stream).
    pub week_latent_sd: f64,
}

pub struct Output {
    pub n: usize,
    pub sims: usize,
    /// (n*sims) row-major integer 72-hole totals.
    pub final_scores: Vec<i64>,
    /// (n,) simulated win probability (RNG.choice tiebreak among min-score ties).
    pub win_prob: Vec<f64>,
    /// (n*4) row-major per-player per-category SG means over sims, per round.
    /// Feeds new_sim's avg_expected_cat_sg_{tourney}.csv (mean over ALL sims of
    /// the raw category draws; the missed-cut penalty does NOT touch these).
    pub cat_means_r1: Vec<f64>,
    pub cat_means_r2: Vec<f64>,
    pub cat_means_r3: Vec<f64>,
    pub cat_means_r4: Vec<f64>,
    /// (n*sims) row-major made-the-cut mask, pairing with final_scores draw-for-draw.
    /// The +200 missed-cut penalty rows are exactly `!made_cut`. Published so the
    /// board's portfolio can price make_cut off the SAME joint as winner/top-N
    /// instead of an independent copula draw.
    pub made_cut: Vec<bool>,
}

#[inline]
fn idx(i: usize, s: usize, sims: usize) -> usize {
    i * sims + s
}

/// Per-player per-category mean over sims of one round's category draws.
/// Returns (n*4) row-major, matching numpy `cats_rN.mean(axis=1)`.
fn cat_means(cats: &[[f64; 4]], n: usize, sims: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * 4];
    let inv = 1.0 / sims as f64;
    for i in 0..n {
        let mut acc = [0.0f64; 4];
        for s in 0..sims {
            let c = cats[idx(i, s, sims)];
            acc[0] += c[0];
            acc[1] += c[1];
            acc[2] += c[2];
            acc[3] += c[3];
        }
        for j in 0..4 {
            out[i * 4 + j] = acc[j] * inv;
        }
    }
    out
}

/// Draw one round for all players into `cats`/`sg`/`strokes`.
///
/// `base_cat[i][j]` is the per-player category mean BEFORE the per-sim skill
/// shift; `shift[i*sims+s]` is the per-(player,sim) total skill shift (spread /4
/// across categories). R1 passes an all-zero shift. RNG is consumed serially in
/// player-major, sim order to keep the stream deterministic.
#[allow(clippy::too_many_arguments)]
fn draw_round(
    stream: &mut NormalStream,
    n: usize,
    sims: usize,
    base_cat: &[[f64; 4]],
    std_course: &[[f64; 4]],
    skew: &[[Skew; 4]],
    l_corr: &[[f64; 4]; 4],
    shift: Option<&[f64]>,
    cats: &mut [[f64; 4]],
    sg: &mut [f64],
    strokes: &mut [i64],
) {
    for i in 0..n {
        let bc = base_cat[i];
        let sc = std_course[i];
        let sk = &skew[i];
        for s in 0..sims {
            let z = stream.next_row4();
            let sh = match shift {
                Some(arr) => arr[idx(i, s, sims)] / 4.0,
                None => 0.0,
            };
            let mut c = [0.0f64; 4];
            for j in 0..4 {
                let l = l_corr[j];
                let corr_z = z[0] * l[0] + z[1] * l[1] + z[2] * l[2] + z[3] * l[3];
                let skewed = sk[j].apply(corr_z);
                let draw = (bc[j] + sh) + skewed * sc[j];
                c[j] = draw.clamp(CLIP_CAT.0, CLIP_CAT.1);
            }
            let total = sum4(c);
            let k = idx(i, s, sims);
            cats[k] = c;
            sg[k] = total;
            strokes[k] = rint_i64(PAR - total);
        }
    }
}

/// Competition min-rank of one sim's strokes column (1-based, ties share min).
/// Writes into `pos` (length n). Reuses `order` scratch.
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

pub fn run_pretournament(inp: &Inputs) -> Output {
    let n = inp.n;
    let sims = inp.sims;
    let ns = n * sims;

    // Precompute per-(player,category) skew transforms once.
    let skew: Vec<[Skew; 4]> = (0..n)
        .map(|i| {
            let e = inp.eff_skew[i];
            [Skew::new(e[0]), Skew::new(e[1]), Skew::new(e[2]), Skew::new(e[3])]
        })
        .collect();

    let mut stream = NormalStream::new(inp.seed);

    // ---- Week-level form latent (2026-08 joint distributional fix) ----
    // Separate RNG stream so sd=0 leaves the main seed-456 stream bit-identical.
    // std_eff = std_course * shrink_i with shrink_i^2 = 1 - sd^2/round_var_i,
    // round_var_i = std_i^T (L L^T) std_i — per-round total variance unchanged.
    let (w_latent, std_eff): (Option<Vec<f64>>, Vec<[f64; 4]>) = if inp.week_latent_sd > 0.0 {
        let sd = inp.week_latent_sd;
        let mut lat_stream = NormalStream::new(inp.seed.wrapping_add(789));
        let mut w = vec![0.0f64; ns];
        for v in w.iter_mut() {
            *v = sd * lat_stream.next_normal();
        }
        let mut se = Vec::with_capacity(n);
        for i in 0..n {
            let s = inp.std_course[i];
            // round_var = |L^T s|^2  (R = L L^T)
            let mut rv = 0.0f64;
            for k in 0..4 {
                let mut dot = 0.0f64;
                for a in 0..4 {
                    dot += inp.l_corr[a][k] * s[a];
                }
                rv += dot * dot;
            }
            let shrink = (1.0 - sd * sd / rv).clamp(0.05, 1.0).sqrt();
            se.push([s[0] * shrink, s[1] * shrink, s[2] * shrink, s[3] * shrink]);
        }
        (Some(w), se)
    } else {
        (None, inp.std_course.clone())
    };

    // ---- R1 ----
    // base_cat = mu - weather_delta_r1 * WEATHER_CAT_SPLIT
    let base_cat_r1: Vec<[f64; 4]> = (0..n)
        .map(|i| {
            let m = inp.mu[i];
            let w = inp.weather_delta_r1[i];
            [
                m[0] - w * WEATHER_CAT_SPLIT[0],
                m[1] - w * WEATHER_CAT_SPLIT[1],
                m[2] - w * WEATHER_CAT_SPLIT[2],
                m[3] - w * WEATHER_CAT_SPLIT[3],
            ]
        })
        .collect();
    let mut cats_r1 = vec![[0.0f64; 4]; ns];
    let mut sg_r1 = vec![0.0f64; ns];
    let mut strokes_r1 = vec![0i64; ns];
    draw_round(&mut stream, n, sims, &base_cat_r1, &std_eff, &skew, &inp.l_corr,
        w_latent.as_deref(), &mut cats_r1, &mut sg_r1, &mut strokes_r1);

    // ---- R1 -> R2 skill update ----
    // adj_r1 (R1's FULL fresh adjustment: sg + residual), updated_skill_r2,
    // sg_r2_mean (per player-sim). The R2 update replaces R1's adjustment
    // entirely — parity with live_stats_engine reset-to-base (2026-08).
    let mut adj_r1 = vec![0.0f64; ns];
    let mut updated_skill_r2 = vec![0.0f64; ns];
    let mut sg_r2_mean = vec![0.0f64; ns];
    // Field-average skill = mean(my_pred) over the field. Added to the R1 residual
    // to match ETR's rust_sim baseline: resid = sg_r1 + field_skill - my_pred.
    let field_skill = inp.my_pred_base.iter().sum::<f64>() / n as f64;
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
        let r2off = inp.r2_mu[i] - mp;
        for s in 0..sims {
            let k = idx(i, s, sims);
            let resid = sg_r1[k] + field_skill - mp;
            let resid2 = resid * resid;
            let mut tot_resid_adj = resid * c.residual + resid2 * c.residual2;
            if resid < 0.0 && tot_resid_adj > 0.2 {
                tot_resid_adj = 0.2;
            }
            if tot_resid_adj > 0.5 {
                tot_resid_adj = 0.5;
            }
            // Floor -0.75 for resid [-8,-6), -0.5 elsewhere: parity with
            // live_stats_engine _totals_r1.
            // User risk rule 2026-08: +/-0.5 everywhere (former -0.75 band retired).
            let floor = -0.5;
            if tot_resid_adj < floor {
                tot_resid_adj = floor;
            }
            let ott_adj = cats_r1[k][0] * c.ott;
            let putt_adj = cats_r1[k][3] * c.putt;
            let sg_adj = ott_adj + putt_adj;
            let total_adj = tot_resid_adj + sg_adj;
            adj_r1[k] = total_adj;
            let usk2 = mp + total_adj;
            updated_skill_r2[k] = usk2;
            sg_r2_mean[k] = usk2 + r2off;
        }
    }

    // ---- R2 draws ----
    // base_cat = mu - weather_delta_r2*WCS ; shift = sg_r2_mean - (mu.sum()-wd_r2)
    let base_cat_r2: Vec<[f64; 4]> = (0..n)
        .map(|i| {
            let m = inp.mu[i];
            let w = inp.weather_delta_r2[i];
            [
                m[0] - w * WEATHER_CAT_SPLIT[0],
                m[1] - w * WEATHER_CAT_SPLIT[1],
                m[2] - w * WEATHER_CAT_SPLIT[2],
                m[3] - w * WEATHER_CAT_SPLIT[3],
            ]
        })
        .collect();
    let mut shift_r2 = vec![0.0f64; ns];
    for i in 0..n {
        let base_total = inp.mu[i].iter().sum::<f64>() - inp.weather_delta_r2[i];
        for s in 0..sims {
            let k = idx(i, s, sims);
            shift_r2[k] = sg_r2_mean[k] - base_total;
        }
    }
    if let Some(w) = &w_latent {
        for k in 0..ns {
            shift_r2[k] += w[k];
        }
    }
    let mut cats_r2 = vec![[0.0f64; 4]; ns];
    let mut sg_r2 = vec![0.0f64; ns];
    let mut strokes_r2 = vec![0i64; ns];
    draw_round(&mut stream, n, sims, &base_cat_r2, &std_eff, &skew, &inp.l_corr,
        Some(&shift_r2), &mut cats_r2, &mut sg_r2, &mut strokes_r2);

    // r1_r2_scores
    let mut r1_r2 = vec![0i64; ns];
    for k in 0..ns {
        r1_r2[k] = strokes_r1[k] + strokes_r2[k];
    }

    // ---- Cut logic after 36 ----
    let mut made_cut = vec![true; ns];
    if inp.cut_line < n {
        let mut order = vec![0usize; n];
        let mut colvals = vec![0i64; n];
        for s in 0..sims {
            for i in 0..n {
                colvals[i] = r1_r2[idx(i, s, sims)];
                order[i] = i;
            }
            order.sort_by(|&a, &b| colvals[a].cmp(&colvals[b]));
            let cut_score = colvals[order[inp.cut_line - 1]];
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

    // ---- R2 -> R3 update (position buckets on r1_r2) ----
    let mut tot_resid_adj_r2 = vec![0.0f64; ns];
    let mut tot_sg_adj_r2 = vec![0.0f64; ns];
    let mut updated_skill_r3 = vec![0.0f64; ns];
    let mut sg_r3_mean = vec![0.0f64; ns];
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
                let resid = (sg_r2[k] - sg_r2_mean[k]).min(6.0);
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
                let total_adj = (tr + ts) - adj_r1[k];
                let usk3 = updated_skill_r2[k] + total_adj;
                updated_skill_r3[k] = usk3;
                sg_r3_mean[k] = usk3 + (inp.r3_mu[i] - inp.my_pred_base[i]);
            }
        }
    }

    // ---- R3 draws (no weather) ----
    let mut shift_r3 = vec![0.0f64; ns];
    for i in 0..n {
        let base_total = inp.mu[i].iter().sum::<f64>();
        for s in 0..sims {
            let k = idx(i, s, sims);
            shift_r3[k] = sg_r3_mean[k] - base_total;
        }
    }
    if let Some(w) = &w_latent {
        for k in 0..ns {
            shift_r3[k] += w[k];
        }
    }
    let mut cats_r3 = vec![[0.0f64; 4]; ns];
    let mut sg_r3 = vec![0.0f64; ns];
    let mut strokes_r3 = vec![0i64; ns];
    draw_round(&mut stream, n, sims, &inp.mu, &std_eff, &skew, &inp.l_corr,
        Some(&shift_r3), &mut cats_r3, &mut sg_r3, &mut strokes_r3);

    let mut r1_r3 = vec![0i64; ns];
    for k in 0..ns {
        r1_r3[k] = r1_r2[k] + strokes_r3[k];
    }

    // ---- R3 -> R4 update (avg-SG only; position buckets on r1_r3) ----
    let mut sg_r4_mean = vec![0.0f64; ns];
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
                // undo prior R2 adjustment, add R3 adjustment
                let usk4 = updated_skill_r3[k] - (tot_sg_adj_r2[k] + tot_resid_adj_r2[k]) + ts;
                sg_r4_mean[k] = usk4 + (inp.r4_mu[i] - inp.my_pred_base[i]);
            }
        }
    }

    // ---- R4 draws (no weather) ----
    let mut shift_r4 = vec![0.0f64; ns];
    for i in 0..n {
        let base_total = inp.mu[i].iter().sum::<f64>();
        for s in 0..sims {
            let k = idx(i, s, sims);
            shift_r4[k] = sg_r4_mean[k] - base_total;
        }
    }
    if let Some(w) = &w_latent {
        for k in 0..ns {
            shift_r4[k] += w[k];
        }
    }
    let mut cats_r4 = vec![[0.0f64; 4]; ns];
    let mut sg_r4 = vec![0.0f64; ns];
    let mut strokes_r4 = vec![0i64; ns];
    draw_round(&mut stream, n, sims, &inp.mu, &std_eff, &skew, &inp.l_corr,
        Some(&shift_r4), &mut cats_r4, &mut sg_r4, &mut strokes_r4);

    // ---- finalize: missed-cut penalty + totals ----
    let mut final_scores = vec![0i64; ns];
    for k in 0..ns {
        let r3_r4 = if made_cut[k] {
            strokes_r3[k] + strokes_r4[k]
        } else {
            MISSED_CUT_PENALTY
        };
        final_scores[k] = r1_r2[k] + r3_r4;
    }

    // ---- win column: per-sim min, RNG.choice among ties (serial, after draws) ----
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
    let inv = sims as f64;
    for w in win_counts.iter_mut() {
        *w /= inv;
    }

    // ---- per-category means over sims (for avg_expected_cat_sg) ----
    let cat_means_r1 = cat_means(&cats_r1, n, sims);
    let cat_means_r2 = cat_means(&cats_r2, n, sims);
    let cat_means_r3 = cat_means(&cats_r3, n, sims);
    let cat_means_r4 = cat_means(&cats_r4, n, sims);

    Output {
        n,
        sims,
        final_scores,
        win_prob: win_counts,
        cat_means_r1,
        cat_means_r2,
        cat_means_r3,
        cat_means_r4,
        made_cut,
    }
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

    fn tiny_inputs(n: usize, sims: usize) -> Inputs {
        let z1 = CoeffR1 { ott: 0.0, putt: 0.0, residual: 0.0, residual2: 0.0 };
        let z2 = CoeffR2 {
            residual: 0.0, residual2: 0.0, residual3: 0.0, avg_ott: 0.0,
            avg_putt: 0.0, avg_app: 0.0, avg_arg: 0.0, delta_app: 0.0,
        };
        let z3 = CoeffR3 { sg_ott_avg: 0.0, sg_putt_avg: 0.0, sg_app_avg: 0.0, sg_arg_avg: 0.0, pos_6_10: 0.0 };
        Inputs {
            n,
            sims,
            seed: 456,
            mu: vec![[0.1, 0.1, 0.0, 0.0]; n],
            std_course: vec![[1.0, 1.0, 0.8, 0.8]; n],
            eff_skew: vec![[0.0, 0.0, 0.0, 0.0]; n],
            l_corr: ident_l(),
            my_pred_base: (0..n).map(|i| i as f64 * 0.1 - 0.2).collect(),
            r2_mu: vec![0.0; n],
            r3_mu: vec![0.0; n],
            r4_mu: vec![0.0; n],
            weather_delta_r1: vec![0.0; n],
            weather_delta_r2: vec![0.0; n],
            r1_high: z1, r1_midh: z1, r1_midl: z1, r1_low: z1,
            r2_lt6: z2, r2_6_30: z2, r2_30up: z2,
            r3_lt6: z3, r3_6_20: z3, r3_30up: z3,
            cut_line: 65,
            use_10_shot_rule: true,
            week_latent_sd: 0.0,
        }
    }

    #[test]
    fn runs_and_shapes() {
        let inp = tiny_inputs(20, 500);
        let out = run_pretournament(&inp);
        assert_eq!(out.final_scores.len(), 20 * 500);
        assert_eq!(out.win_prob.len(), 20);
    }

    #[test]
    fn win_prob_sums_to_one() {
        let inp = tiny_inputs(20, 2000);
        let out = run_pretournament(&inp);
        let total: f64 = out.win_prob.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "win prob sum {total}");
    }

    #[test]
    fn week_latent_raises_total_variance_ratio() {
        // No-cut config (cut_line >= n): final = sum of 4 rounds' strokes.
        // With ident corr and std [1,1,.8,.8], round_var = 3.28. At sd=0.6 the
        // per-player total variance should rise by ~1 + 3*sd^2/round_var = 1.33
        // (per-round variance is preserved by the shrink, so the increase is
        // pure cross-round covariance).
        let n = 30usize;
        let sims = 6000usize;
        let mut inp0 = tiny_inputs(n, sims);
        inp0.cut_line = n; // no cut
        let mut inp1 = tiny_inputs(n, sims);
        inp1.cut_line = n;
        inp1.week_latent_sd = 0.6;
        let out0 = run_pretournament(&inp0);
        let out1 = run_pretournament(&inp1);
        let var = |fs: &[i64]| -> f64 {
            let mut acc = 0.0f64;
            for i in 0..n {
                let row = &fs[i * sims..(i + 1) * sims];
                let m = row.iter().sum::<i64>() as f64 / sims as f64;
                let v = row.iter().map(|&x| { let d = x as f64 - m; d * d }).sum::<f64>()
                    / sims as f64;
                acc += v;
            }
            acc / n as f64
        };
        let ratio = var(&out1.final_scores) / var(&out0.final_scores);
        assert!(ratio > 1.20 && ratio < 1.45,
            "latent variance ratio {ratio} outside expected (1.20, 1.45)");
        let total: f64 = out1.win_prob.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "win prob sum with latent {total}");
    }

    #[test]
    fn deterministic_same_seed() {
        let a = run_pretournament(&tiny_inputs(15, 300));
        let b = run_pretournament(&tiny_inputs(15, 300));
        assert_eq!(a.final_scores, b.final_scores);
        assert_eq!(a.win_prob, b.win_prob);
    }

    #[test]
    fn scores_in_plausible_range() {
        // Exercise the cut: cut_line < n so some sims miss (r1_r2 + 200 penalty),
        // some make it (~4*par). All totals should be finite and in a sane band.
        let mut inp = tiny_inputs(30, 1000);
        inp.cut_line = 15;
        let out = run_pretournament(&inp);
        let mn = *out.final_scores.iter().min().unwrap();
        let mx = *out.final_scores.iter().max().unwrap();
        // made-cut 4-round totals ~270-310; missed-cut ~ r1_r2(~144) + 200 ~ 340.
        assert!(mn > 230 && mn < 320, "min total {mn}");
        assert!(mx > 300 && mx < 420, "max total {mx}");
        // both regimes present: at least one missed-cut (>= some big value).
        let missed = out.final_scores.iter().filter(|&&v| v > 320).count();
        assert!(missed > 0, "expected some missed-cut sims");
    }
}

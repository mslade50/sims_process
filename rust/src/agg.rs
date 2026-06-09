//! Post-sim aggregation — RNG-free pure functions of the integer `final_scores`
//! matrix. Ports new_sim.py:1013-1188 (rank_probs `prob_u`/`prob_ndh`, top-N
//! dead-heat, the O(n^2) h2h loop).
//!
//! Because these consume only the already-quantized integer scores, they are
//! INTEGER-EXACT vs pandas — validated array-exact, not statistically (plan §7).
//! This is the biggest safe speedup (plan §5): rayon over RNG-free work.

use numpy::ndarray::ArrayView2;
use rayon::prelude::*;

/// Per-sim ranking scratch: for each player, its 1-based min-rank `pos` and the
/// size `tie_ct` of its tie group. Computed in one sorted pass (equivalent to
/// pandas `rank(method='min')` + a tie-count groupby, but without re-scanning).
fn rank_and_ties(col: &[i64], pos: &mut [i64], tie_ct: &mut [i64], order: &mut [usize]) {
    let n = col.len();
    for (k, o) in order.iter_mut().enumerate() {
        *o = k;
    }
    order.sort_by(|&a, &b| col[a].cmp(&col[b]));
    let mut i = 0usize;
    while i < n {
        let v = col[order[i]];
        let start = i;
        let mut k = i;
        while k < n && col[order[k]] == v {
            k += 1;
        }
        let group_rank = (start as i64) + 1;
        let tc = (k - start) as i64;
        for &idx in &order[start..k] {
            pos[idx] = group_rank;
            tie_ct[idx] = tc;
        }
        i = k;
    }
}

/// Dead-heat overlap factor — port of `dead_heat_factor` (new_sim.py:1018).
#[inline]
fn dead_heat_factor(position: i64, tie_count: i64, threshold: i64) -> f64 {
    let start = position;
    let end = position + tie_count - 1;
    let overlap_start = start.max(1);
    let overlap_end = end.min(threshold);
    let overlap_count = (overlap_end - overlap_start + 1).max(0);
    overlap_count as f64 / tie_count as f64
}

/// Dense aggregation output. Ranks are 1-based; row = player index (input order),
/// column = rank-1. `prob_u` is dead-heat-split, `prob_ndh` is raw min-rank.
pub struct Aggregates {
    pub n: usize,
    /// (n*n) row-major: prob_u[player][rank-1]
    pub prob_u: Vec<f64>,
    /// (n*n) row-major: prob_ndh[player][rank-1]
    pub prob_ndh: Vec<f64>,
    /// (n*3) row-major: [top5, top10, top20] per player (dead-heat adjusted)
    pub top_finish: Vec<f64>,
    /// (n*3) row-major: [top5, top10, top20] NO dead-heat (pos<=thr counts as 1).
    /// round_sim's Kalshi finish markets (compute_finish_probabilities `*_nodh`).
    pub top_finish_nodh: Vec<f64>,
}

/// Per-thread accumulator (dense), folded across sims then reduced.
struct Acc {
    prob_u: Vec<f64>,
    prob_ndh: Vec<f64>,
    top: Vec<f64>,
    top_nodh: Vec<f64>,
}
impl Acc {
    fn zero(n: usize) -> Self {
        Acc {
            prob_u: vec![0.0; n * n],
            prob_ndh: vec![0.0; n * n],
            top: vec![0.0; n * 3],
            top_nodh: vec![0.0; n * 3],
        }
    }
    fn merge(mut self, other: Acc) -> Acc {
        for (a, b) in self.prob_u.iter_mut().zip(&other.prob_u) {
            *a += b;
        }
        for (a, b) in self.prob_ndh.iter_mut().zip(&other.prob_ndh) {
            *a += b;
        }
        for (a, b) in self.top.iter_mut().zip(&other.top) {
            *a += b;
        }
        for (a, b) in self.top_nodh.iter_mut().zip(&other.top_nodh) {
            *a += b;
        }
        self
    }
}

/// Aggregate `final_scores` (n, sims) into dense rank/top-N probability tables.
/// Each output is divided by `sims` so it is a probability.
pub fn aggregate(final_scores: &ArrayView2<i64>) -> Aggregates {
    let n = final_scores.nrows();
    let sims = final_scores.ncols();

    // Fold across sims in parallel; each task owns scratch + a dense Acc.
    let acc = (0..sims)
        .into_par_iter()
        .fold(
            || (Acc::zero(n), vec![0i64; n], vec![0i64; n], vec![0i64; n], vec![0usize; n]),
            |(mut acc, mut col, mut pos, mut tie_ct, mut order), j| {
                for i in 0..n {
                    col[i] = final_scores[(i, j)];
                }
                rank_and_ties(&col, &mut pos, &mut tie_ct, &mut order);
                for i in 0..n {
                    let p = pos[i];
                    let tc = tie_ct[i];
                    // prob_ndh: raw min-rank, +1 at column p-1
                    acc.prob_ndh[i * n + (p as usize - 1)] += 1.0;
                    // prob_u: spread weight 1/tc across ranks p..p+tc-1
                    let w = 1.0 / tc as f64;
                    for off in 0..tc {
                        let r = (p + off) as usize - 1;
                        acc.prob_u[i * n + r] += w;
                    }
                    // top-N dead-heat credit
                    acc.top[i * 3] += dead_heat_factor(p, tc, 5);
                    acc.top[i * 3 + 1] += dead_heat_factor(p, tc, 10);
                    acc.top[i * 3 + 2] += dead_heat_factor(p, tc, 20);
                    // top-N no-dead-heat: min-rank pos within threshold counts fully
                    if p <= 5 {
                        acc.top_nodh[i * 3] += 1.0;
                    }
                    if p <= 10 {
                        acc.top_nodh[i * 3 + 1] += 1.0;
                    }
                    if p <= 20 {
                        acc.top_nodh[i * 3 + 2] += 1.0;
                    }
                }
                (acc, col, pos, tie_ct, order)
            },
        )
        .map(|(acc, ..)| acc)
        .reduce(|| Acc::zero(n), |a, b| a.merge(b));

    // Divide (not multiply-by-reciprocal) to match pandas `.div(SIMULATIONS)`.
    // prob_ndh accumulates only integer +1.0s (order-independent, exact f64), so
    // count/sims is bit-identical to pandas. prob_u/top carry fractional 1/tc
    // weights whose rayon fold/reduce sum order differs from pandas — those stay
    // FP-order-equivalent (gated numerically, not bit-exact; plan §7 statistical).
    let s = sims as f64;
    let mut out = Aggregates {
        n,
        prob_u: acc.prob_u,
        prob_ndh: acc.prob_ndh,
        top_finish: acc.top,
        top_finish_nodh: acc.top_nodh,
    };
    out.prob_u.iter_mut().for_each(|x| *x /= s);
    out.prob_ndh.iter_mut().for_each(|x| *x /= s);
    out.top_finish.iter_mut().for_each(|x| *x /= s);
    out.top_finish_nodh.iter_mut().for_each(|x| *x /= s);
    out
}

/// Pairwise head-to-head — port of new_sim.py:1166-1185.
///
/// For each pair (i<j): wins = #(s_i < s_j), losses = #(s_i > s_j), ties = rest.
/// Skips pairs with `wins+losses == 0` (matches Python's `continue`). Returns
/// (idx_a, idx_b, prob_a, tie_pct) aligned. RNG-free, integer-exact, rayon over i.
pub fn h2h(final_scores: &ArrayView2<i64>) -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>) {
    let n = final_scores.nrows();
    let sims = final_scores.ncols();

    let mut records: Vec<(i64, i64, f64, f64)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut local = Vec::new();
            for jj in (i + 1)..n {
                let mut wins = 0i64;
                let mut losses = 0i64;
                for s in 0..sims {
                    let a = final_scores[(i, s)];
                    let b = final_scores[(jj, s)];
                    if a < b {
                        wins += 1;
                    } else if a > b {
                        losses += 1;
                    }
                }
                let denom = wins + losses;
                if denom == 0 {
                    continue;
                }
                let ties = sims as i64 - denom;
                let total = sims as i64;
                local.push((
                    i as i64,
                    jj as i64,
                    wins as f64 / denom as f64,
                    if total != 0 { ties as f64 / total as f64 } else { 0.0 },
                ));
            }
            local.into_iter()
        })
        .collect();

    // flat_map over i is already i-ordered? par flat_map_iter preserves order.
    records.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    let mut idx_a = Vec::with_capacity(records.len());
    let mut idx_b = Vec::with_capacity(records.len());
    let mut prob_a = Vec::with_capacity(records.len());
    let mut tie_pct = Vec::with_capacity(records.len());
    for (a, b, p, t) in records {
        idx_a.push(a);
        idx_b.push(b);
        prob_a.push(p);
        tie_pct.push(t);
    }
    (idx_a, idx_b, prob_a, tie_pct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn ranks_and_ties_basic() {
        let col = [71i64, 68, 70, 70];
        let mut pos = [0i64; 4];
        let mut tc = [0i64; 4];
        let mut order = [0usize; 4];
        rank_and_ties(&col, &mut pos, &mut tc, &mut order);
        // 68->1(tc1), 70,70->2(tc2), 71->4(tc1)
        assert_eq!(pos, [4, 1, 2, 2]);
        assert_eq!(tc, [1, 1, 2, 2]);
    }

    #[test]
    fn dead_heat_factor_cases() {
        // solo 3rd, threshold 5 -> fully inside
        assert_eq!(dead_heat_factor(3, 1, 5), 1.0);
        // tie of 2 starting at 5, threshold 5 -> only pos5 counts -> 0.5
        assert_eq!(dead_heat_factor(5, 2, 5), 0.5);
        // tie of 2 starting at 6, threshold 5 -> none -> 0
        assert_eq!(dead_heat_factor(6, 2, 5), 0.0);
    }

    #[test]
    fn aggregate_two_player_two_sim() {
        // player0 wins sim0, tie sim1; player1 wins... let's hand-check.
        // scores (n=2, sims=2): p0=[70,71], p1=[71,71]
        // sim0: p0=70(rank1,tc1), p1=71(rank2,tc1)
        // sim1: p0=71,p1=71 tie -> both rank1 tc2
        let fs = array![[70i64, 71], [71, 71]];
        let a = aggregate(&fs.view());
        // prob_ndh[p0]: rank1 in sim0, rank1 in sim1 -> 2/2 = 1.0 at rank1
        assert_eq!(a.prob_ndh[0 * 2 + 0], 1.0); // p0 rank1
        assert_eq!(a.prob_ndh[0 * 2 + 1], 0.0);
        // prob_ndh[p1]: rank2 sim0, rank1 sim1 -> 0.5 each
        assert_eq!(a.prob_ndh[1 * 2 + 0], 0.5);
        assert_eq!(a.prob_ndh[1 * 2 + 1], 0.5);
        // prob_u[p1]: sim0 solo rank2 -> +1 at r2; sim1 tie tc2 at rank1 -> 0.5 at r1 & r2
        // total /2: r1 = 0.5/2=0.25 ; r2 = (1+0.5)/2 = 0.75
        assert!((a.prob_u[1 * 2 + 0] - 0.25).abs() < 1e-12);
        assert!((a.prob_u[1 * 2 + 1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn h2h_basic() {
        // p0 beats p1 in sim0, ties sim1 -> wins1 loss0 -> prob_a=1.0, tie_pct=0.5
        let fs = array![[70i64, 71], [71, 71]];
        let (ia, ib, pa, tp) = h2h(&fs.view());
        assert_eq!(ia, vec![0]);
        assert_eq!(ib, vec![1]);
        assert_eq!(pa, vec![1.0]);
        assert_eq!(tp, vec![0.5]);
    }
}

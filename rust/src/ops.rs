//! Numerical op primitives ported from `new_sim.py`.
//!
//! These are the *load-bearing* pieces called out in RUST_MIGRATION_PLAN.md §3:
//! a `*.5`-stroke flip cascades through ranks -> cut -> coefficient buckets, so
//! `round_ties_even` and exact min-rank are MANDATORY even under the statistical
//! parity target. Each function here mirrors a specific Python construct and is
//! validated array-exact against frozen Python fixtures (see tests/).

/// Cornish-Fisher calibration multiplier.
///
/// Port of `_cf_calibration_multiplier` (new_sim.py:197). Polynomial fit that
/// compensates for first-order CF skewness saturation at |gamma| > ~0.5.
#[inline]
pub fn cf_calibration_multiplier(gamma: f64) -> f64 {
    let ag = gamma.abs();
    if ag < 0.2 {
        1.0
    } else {
        1.0 + 0.0234 * ag.powi(2) + 0.0125 * ag.powi(3)
    }
}

/// Precomputed skew transform for a fixed `gamma`.
///
/// In the Python kernel `_apply_skew` is called once per (player, category) with
/// a scalar `gamma`, then applied to the whole SIMS-length column. Splitting the
/// per-gamma constants out lets the hot loop avoid recomputing `cf_*` and `sqrt`.
#[derive(Clone, Copy, Debug)]
pub enum Skew {
    /// |gamma| < 0.01 -> identity (Python returns `z` unchanged).
    Identity,
    Active {
        /// gamma_adj / 6.0
        coef: f64,
        /// sqrt(1 + gamma_adj^2 / 18). Python divides by this (`z_skewed /= ...`);
        /// we divide too (not multiply by a reciprocal) so the result is
        /// bit-identical, not 1-ULP off.
        scale: f64,
    },
}

impl Skew {
    /// Port of `_apply_skew` setup (new_sim.py:211). Mirrors the |gamma| < 0.01
    /// short-circuit exactly so identity columns are bit-identical to Python.
    pub fn new(gamma: f64) -> Self {
        if gamma.abs() < 0.01 {
            return Skew::Identity;
        }
        let gamma_adj = gamma * cf_calibration_multiplier(gamma);
        Skew::Active {
            coef: gamma_adj / 6.0,
            scale: (1.0 + gamma_adj * gamma_adj / 18.0).sqrt(),
        }
    }

    /// Apply the skew to a single draw. Replicates the Python op order:
    /// `z + (gamma_adj/6)*(z^2 - 1)`, then divide by the variance-correction scale.
    #[inline]
    pub fn apply(&self, z: f64) -> f64 {
        match *self {
            Skew::Identity => z,
            // Matches Python op order exactly: add the skew term, THEN divide.
            Skew::Active { coef, scale } => (z + coef * (z * z - 1.0)) / scale,
        }
    }
}

/// Port of `np.rint(x).astype(int)`.
///
/// numpy `rint` is round-half-to-even (banker's). Rust's `f64::round` is
/// half-away-from-zero, which would systematically shift scores — MANDATORY to
/// use `round_ties_even` here (plan §3).
#[inline]
pub fn rint_i64(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Correlated draw + skew + scale + clip + sum for one player's 4 SG categories.
///
/// Mirrors the inner body of the draw loop (new_sim.py:597-603):
///   corr_z = Z @ L_corr.T   (per simulation row, 4-wide)
///   corr_z[j] = apply_skew(corr_z[j], skew[j])
///   draws = cat_mu[j] + corr_z[j] * std_c[j]
///   cats  = clip(draws, lo, hi)
///   sg    = cats.sum()
///
/// `z` is one row of standard normals (length 4). `l_corr` is the injected lower
/// Cholesky (4x4) — NOT ported from LAPACK (plan §3). Returns (clipped_cats, sg).
#[inline]
pub fn draw_one_sim(
    z: [f64; 4],
    l_corr: &[[f64; 4]; 4],
    skew: &[Skew; 4],
    cat_mu: [f64; 4],
    std_c: [f64; 4],
    clip: (f64, f64),
) -> ([f64; 4], f64) {
    // corr_z = Z @ L_corr.T : corr_z[j] = sum_k Z[k] * L_corr[j][k]
    // Hand-written 4-wide loop (no BLAS), explicit accumulation order.
    let mut cats = [0.0f64; 4];
    for j in 0..4 {
        let l = l_corr[j];
        let corr_z = z[0] * l[0] + z[1] * l[1] + z[2] * l[2] + z[3] * l[3];
        let skewed = skew[j].apply(corr_z);
        let draw = cat_mu[j] + skewed * std_c[j];
        cats[j] = draw.clamp(clip.0, clip.1);
    }
    (cats, sum4(cats))
}

/// Port of `cats.sum(axis=1)` over 4 categories.
///
/// numpy's pairwise summation falls back to a sequential accumulate for n<=8, so
/// the realized order is left-to-right `((c0+c1)+c2)+c3`. We match that exactly;
/// the difference vs `(c0+c1)+(c2+c3)` only matters within ~1e-13 of a `*.5`
/// boundary, but min-rank routing makes even that worth pinning. Validated
/// against `np.sum` in fixtures.
#[inline]
pub fn sum4(c: [f64; 4]) -> f64 {
    ((c[0] + c[1]) + c[2]) + c[3]
}

/// Port of `pandas.Series.rank(method='min')` on ascending integer strokes.
///
/// Competition ranking, 1-based: tied values all receive the minimum (1-based)
/// position of the group. This routes which R2/R3 skill-update coefficient bucket
/// (`<6` / `6-30` / `>30`) applies, so it is load-bearing (plan §3), not display.
///
/// Returns ranks aligned to the input order.
pub fn min_rank(values: &[i64]) -> Vec<i64> {
    let n = values.len();
    // order = indices sorted by value ascending (stable, mirrors pandas).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].cmp(&values[b]));

    let mut ranks = vec![0i64; n];
    let mut i = 0usize;
    while i < n {
        let v = values[order[i]];
        // min rank of this tie group = its first 1-based position.
        let group_rank = (i as i64) + 1;
        let mut k = i;
        while k < n && values[order[k]] == v {
            ranks[order[k]] = group_rank;
            k += 1;
        }
        i = k;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rint_is_half_even() {
        // Banker's rounding: .5 rounds to nearest even integer.
        assert_eq!(rint_i64(0.5), 0);
        assert_eq!(rint_i64(1.5), 2);
        assert_eq!(rint_i64(2.5), 2);
        assert_eq!(rint_i64(3.5), 4);
        assert_eq!(rint_i64(-0.5), 0);
        assert_eq!(rint_i64(-1.5), -2);
        assert_eq!(rint_i64(-2.5), -2);
        // non-tie cases
        assert_eq!(rint_i64(71.17), 71);
        assert_eq!(rint_i64(70.83), 71);
    }

    #[test]
    fn cf_mult_threshold() {
        assert_eq!(cf_calibration_multiplier(0.1), 1.0);
        assert_eq!(cf_calibration_multiplier(-0.19), 1.0);
        // at 0.2 the polynomial kicks in
        let m = cf_calibration_multiplier(-0.93);
        assert!(m > 1.0 && m < 1.1, "got {m}");
    }

    #[test]
    fn skew_identity_below_threshold() {
        let s = Skew::new(0.005);
        assert!(matches!(s, Skew::Identity));
        assert_eq!(s.apply(1.234), 1.234);
        let s2 = Skew::new(-0.009);
        assert_eq!(s2.apply(-2.5), -2.5);
    }

    #[test]
    fn skew_matches_python_formula() {
        // Independent recompute of the Python formula for gamma=-0.93, z=1.5.
        let gamma = -0.93f64;
        let z = 1.5f64;
        let ag = gamma.abs();
        let mult = 1.0 + 0.0234 * ag.powi(2) + 0.0125 * ag.powi(3);
        let gamma_adj = gamma * mult;
        let expected =
            (z + (gamma_adj / 6.0) * (z * z - 1.0)) / (1.0 + gamma_adj * gamma_adj / 18.0).sqrt();
        let s = Skew::new(gamma);
        assert!((s.apply(z) - expected).abs() < 1e-15);
    }

    #[test]
    fn min_rank_basic() {
        // values ascending strokes; ties share the minimum 1-based rank.
        // sorted: 68,70,70,71 -> ranks 1,2,2,4
        let v = vec![71, 68, 70, 70];
        let r = min_rank(&v);
        assert_eq!(r, vec![4, 1, 2, 2]);
    }

    #[test]
    fn min_rank_all_tied() {
        let v = vec![72, 72, 72];
        assert_eq!(min_rank(&v), vec![1, 1, 1]);
    }

    #[test]
    fn min_rank_strict() {
        let v = vec![5, 4, 3, 2, 1];
        assert_eq!(min_rank(&v), vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn sum4_order() {
        let c = [0.1, 0.2, 0.3, 0.4];
        // exact left-to-right accumulation
        let expected = ((0.1f64 + 0.2) + 0.3) + 0.4;
        assert_eq!(sum4(c), expected);
    }
}

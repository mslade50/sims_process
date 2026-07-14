//! `sims_kernel` — Rust Monte Carlo kernel for new_sim/round_sim.
//!
//! Phase 0/1 scaffold (RUST_MIGRATION_PLAN.md §9): RNG + op primitives plus the
//! PyO3 plumbing and a self-test, so the build/CI machinery and the in-process
//! Python<->Rust diff loop are proven before the full kernel lands in Phase 2.

pub mod agg;
pub mod cascade;
pub mod ops;
pub mod rng;
pub mod round_cascade;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Crate version string — proves the wheel imported and links.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Cheap import-time self-test: exercises each op primitive and returns true.
/// Lets `nightly-round-sim.yml` fail fast if a broken wheel was installed.
#[pyfunction]
fn selftest() -> bool {
    // round_ties_even
    if ops::rint_i64(2.5) != 2 || ops::rint_i64(3.5) != 4 {
        return false;
    }
    // min-rank
    if ops::min_rank(&[71, 68, 70, 70]) != vec![4, 1, 2, 2] {
        return false;
    }
    // skew identity short-circuit
    if !matches!(ops::Skew::new(0.0), ops::Skew::Identity) {
        return false;
    }
    // rng determinism
    let mut a = rng::NormalStream::new(456);
    let mut b = rng::NormalStream::new(456);
    a.next_normal() == b.next_normal()
}

// ---- Primitives exposed for Python-side fixture validation (Phase 1 gate) ----

/// Vectorized `np.rint(x).astype(int)` equivalent for array-exact diffing.
#[pyfunction]
fn rint_array<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<i64>> {
    let out: Vec<i64> = x.as_slice().unwrap().iter().map(|&v| ops::rint_i64(v)).collect();
    out.into_pyarray_bound(py)
}

/// Vectorized `pandas.Series.rank(method='min')` for array-exact diffing.
#[pyfunction]
fn min_rank_array<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, i64>) -> Bound<'py, PyArray1<i64>> {
    ops::min_rank(x.as_slice().unwrap()).into_pyarray_bound(py)
}

/// Vectorized `_apply_skew(z, gamma)` for one scalar gamma over a z-column.
#[pyfunction]
fn apply_skew_array<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    gamma: f64,
) -> Bound<'py, PyArray1<f64>> {
    let s = ops::Skew::new(gamma);
    let out: Vec<f64> = z.as_slice().unwrap().iter().map(|&v| s.apply(v)).collect();
    out.into_pyarray_bound(py)
}

/// Row-wise `cats.sum(axis=1)` over a (n,4) array flattened row-major.
#[pyfunction]
fn sum4_rows<'py>(py: Python<'py>, flat: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let s = flat.as_slice().unwrap();
    let out: Vec<f64> = s
        .chunks_exact(4)
        .map(|c| ops::sum4([c[0], c[1], c[2], c[3]]))
        .collect();
    out.into_pyarray_bound(py)
}

// ---- Post-sim aggregation (RNG-free; integer-exact vs pandas) ----

/// Aggregate `final_scores` (n, sims) i64 into dense rank/top-N tables.
/// Returns (prob_u (n,n), prob_ndh (n,n), top_finish (n,3)) — all probabilities.
#[pyfunction]
fn aggregate<'py>(
    py: Python<'py>,
    final_scores: PyReadonlyArray2<'py, i64>,
) -> (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
) {
    let view = final_scores.as_array();
    let a = agg::aggregate(&view);
    let n = a.n;
    let prob_u = numpy::ndarray::Array2::from_shape_vec((n, n), a.prob_u).unwrap();
    let prob_ndh = numpy::ndarray::Array2::from_shape_vec((n, n), a.prob_ndh).unwrap();
    let top = numpy::ndarray::Array2::from_shape_vec((n, 3), a.top_finish).unwrap();
    (
        prob_u.into_pyarray_bound(py),
        prob_ndh.into_pyarray_bound(py),
        top.into_pyarray_bound(py),
    )
}

/// Pairwise head-to-head from `final_scores` (n, sims) i64.
/// Returns (idx_a, idx_b, prob_a, tie_pct), pairs with no decisive sims dropped.
#[pyfunction]
fn h2h<'py>(
    py: Python<'py>,
    final_scores: PyReadonlyArray2<'py, i64>,
) -> (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let view = final_scores.as_array();
    let (ia, ib, pa, tp) = agg::h2h(&view);
    (
        ia.into_pyarray_bound(py),
        ib.into_pyarray_bound(py),
        pa.into_pyarray_bound(py),
        tp.into_pyarray_bound(py),
    )
}

// ---- Full pre-tournament cascade (seed-456 draw kernel) ----

/// Convert a (n,4) numpy view into Vec<[f64;4]>.
fn rows4(a: &numpy::ndarray::ArrayView2<f64>) -> Vec<[f64; 4]> {
    (0..a.nrows())
        .map(|i| [a[(i, 0)], a[(i, 1)], a[(i, 2)], a[(i, 3)]])
        .collect()
}

/// Run the R1->R4 pre-tournament cascade. All inputs are precomputed in Python
/// (config/preds/Cholesky/weather); `l_corr` is injected (plan §4). Coefficient
/// buckets are passed as fixed-length sequences:
///   r1_*: [ott, putt, residual, residual2]
///   r2_*: [residual, residual2, residual3, avg_ott, avg_putt, avg_app, avg_arg, delta_app]
///   r3_*: [sg_ott_avg, sg_putt_avg, sg_app_avg, sg_arg_avg, pos_6_10]
/// Returns (final_scores (n,sims) i64, win_prob (n,) f64,
///          cat_means_r1, cat_means_r2, cat_means_r3, cat_means_r4 — each (n,4) f64,
///          per-player per-category SG means over sims; feed avg_expected_cat_sg,
///          made_cut (n,sims) bool — pairs with final_scores draw-for-draw).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_pretournament<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<'py, f64>,
    std_course: PyReadonlyArray2<'py, f64>,
    eff_skew: PyReadonlyArray2<'py, f64>,
    l_corr: PyReadonlyArray2<'py, f64>,
    my_pred_base: PyReadonlyArray1<'py, f64>,
    r2_mu: PyReadonlyArray1<'py, f64>,
    r3_mu: PyReadonlyArray1<'py, f64>,
    r4_mu: PyReadonlyArray1<'py, f64>,
    weather_delta_r1: PyReadonlyArray1<'py, f64>,
    weather_delta_r2: PyReadonlyArray1<'py, f64>,
    r1_high: [f64; 4], r1_midh: [f64; 4], r1_midl: [f64; 4], r1_low: [f64; 4],
    r2_lt6: [f64; 8], r2_6_30: [f64; 8], r2_30up: [f64; 8],
    r3_lt6: [f64; 5], r3_6_20: [f64; 5], r3_30up: [f64; 5],
    cut_line: usize,
    use_10_shot_rule: bool,
    sims: usize,
    seed: u64,
) -> (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<bool>>,
) {
    use cascade::{CoeffR1, CoeffR2, CoeffR3, Inputs};
    let mu_v = mu.as_array();
    let n = mu_v.nrows();
    let c1 = |a: [f64; 4]| CoeffR1 { ott: a[0], putt: a[1], residual: a[2], residual2: a[3] };
    let c2 = |a: [f64; 8]| CoeffR2 {
        residual: a[0], residual2: a[1], residual3: a[2], avg_ott: a[3],
        avg_putt: a[4], avg_app: a[5], avg_arg: a[6], delta_app: a[7],
    };
    let c3 = |a: [f64; 5]| CoeffR3 {
        sg_ott_avg: a[0], sg_putt_avg: a[1], sg_app_avg: a[2], sg_arg_avg: a[3],
        pos_6_10: a[4],
    };
    let lc_v = l_corr.as_array();
    let mut lc = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            lc[i][j] = lc_v[(i, j)];
        }
    }
    let inp = Inputs {
        n,
        sims,
        seed,
        mu: rows4(&mu_v),
        std_course: rows4(&std_course.as_array()),
        eff_skew: rows4(&eff_skew.as_array()),
        l_corr: lc,
        my_pred_base: my_pred_base.as_slice().unwrap().to_vec(),
        r2_mu: r2_mu.as_slice().unwrap().to_vec(),
        r3_mu: r3_mu.as_slice().unwrap().to_vec(),
        r4_mu: r4_mu.as_slice().unwrap().to_vec(),
        weather_delta_r1: weather_delta_r1.as_slice().unwrap().to_vec(),
        weather_delta_r2: weather_delta_r2.as_slice().unwrap().to_vec(),
        r1_high: c1(r1_high), r1_midh: c1(r1_midh), r1_midl: c1(r1_midl), r1_low: c1(r1_low),
        r2_lt6: c2(r2_lt6), r2_6_30: c2(r2_6_30), r2_30up: c2(r2_30up),
        r3_lt6: c3(r3_lt6), r3_6_20: c3(r3_6_20), r3_30up: c3(r3_30up),
        cut_line,
        use_10_shot_rule,
    };
    let out = cascade::run_pretournament(&inp);
    let no = out.n;
    let fs = numpy::ndarray::Array2::from_shape_vec((no, out.sims), out.final_scores).unwrap();
    let cm1 = numpy::ndarray::Array2::from_shape_vec((no, 4), out.cat_means_r1).unwrap();
    let cm2 = numpy::ndarray::Array2::from_shape_vec((no, 4), out.cat_means_r2).unwrap();
    let cm3 = numpy::ndarray::Array2::from_shape_vec((no, 4), out.cat_means_r3).unwrap();
    let cm4 = numpy::ndarray::Array2::from_shape_vec((no, 4), out.cat_means_r4).unwrap();
    let mc = numpy::ndarray::Array2::from_shape_vec((no, out.sims), out.made_cut).unwrap();
    (
        fs.into_pyarray_bound(py),
        out.win_prob.into_pyarray_bound(py),
        cm1.into_pyarray_bound(py),
        cm2.into_pyarray_bound(py),
        cm3.into_pyarray_bound(py),
        cm4.into_pyarray_bound(py),
        mc.into_pyarray_bound(py),
    )
}

/// round_sim finish-prob aggregation. Returns (prob_raw (n,n), top_dh (n,3),
/// top_nodh (n,3)). NOTE the naming inversion: round_sim's `prob_u` is the RAW
/// min-rank prob = this `prob_raw` (= new_sim's prob_ndh).
#[pyfunction]
fn aggregate_round<'py>(
    py: Python<'py>,
    final_scores: PyReadonlyArray2<'py, i64>,
) -> (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
) {
    let view = final_scores.as_array();
    let a = agg::aggregate(&view);
    let n = a.n;
    let praw = numpy::ndarray::Array2::from_shape_vec((n, n), a.prob_ndh).unwrap();
    let tdh = numpy::ndarray::Array2::from_shape_vec((n, 3), a.top_finish).unwrap();
    let tnodh = numpy::ndarray::Array2::from_shape_vec((n, 3), a.top_finish_nodh).unwrap();
    (
        praw.into_pyarray_bound(py),
        tdh.into_pyarray_bound(py),
        tnodh.into_pyarray_bound(py),
    )
}

/// Run the round_sim seed-42 tournament cascade from `completed_round`.
/// Known rounds (strokes/cats) are passed for rounds <= completed_round; pass
/// None otherwise. Returns (final_scores (n,sims) i64, made_cut (n,sims) bool,
/// win_prob (n,) f64).
#[pyfunction]
#[pyo3(signature = (
    completed_round, default_par, mu, std_course, eff_skew, l_corr, my_pred_base, expected_r2,
    known_strokes_r1=None, known_cats_r1=None, known_strokes_r2=None, known_cats_r2=None,
    known_strokes_r3=None, known_cats_r3=None,
    r1_high=[0.0;4], r1_midh=[0.0;4], r1_midl=[0.0;4], r1_low=[0.0;4],
    r2_lt6=[0.0;8], r2_6_30=[0.0;8], r2_30up=[0.0;8],
    r3_lt6=[0.0;5], r3_6_20=[0.0;5], r3_30up=[0.0;5],
    cut_line=0, use_10_shot_rule=true, sims=0, seed=42
))]
#[allow(clippy::too_many_arguments)]
fn run_remaining_rounds<'py>(
    py: Python<'py>,
    completed_round: usize,
    default_par: f64,
    mu: PyReadonlyArray2<'py, f64>,
    std_course: PyReadonlyArray2<'py, f64>,
    eff_skew: PyReadonlyArray2<'py, f64>,
    l_corr: PyReadonlyArray2<'py, f64>,
    my_pred_base: PyReadonlyArray1<'py, f64>,
    expected_r2: PyReadonlyArray1<'py, f64>,
    known_strokes_r1: Option<PyReadonlyArray1<'py, i64>>,
    known_cats_r1: Option<PyReadonlyArray2<'py, f64>>,
    known_strokes_r2: Option<PyReadonlyArray1<'py, i64>>,
    known_cats_r2: Option<PyReadonlyArray2<'py, f64>>,
    known_strokes_r3: Option<PyReadonlyArray1<'py, i64>>,
    known_cats_r3: Option<PyReadonlyArray2<'py, f64>>,
    r1_high: [f64; 4], r1_midh: [f64; 4], r1_midl: [f64; 4], r1_low: [f64; 4],
    r2_lt6: [f64; 8], r2_6_30: [f64; 8], r2_30up: [f64; 8],
    r3_lt6: [f64; 5], r3_6_20: [f64; 5], r3_30up: [f64; 5],
    cut_line: usize,
    use_10_shot_rule: bool,
    sims: usize,
    seed: u64,
) -> (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
) {
    use cascade::{CoeffR1, CoeffR2, CoeffR3};
    use round_cascade::{Inputs, KnownRound};
    let mu_v = mu.as_array();
    let n = mu_v.nrows();
    let c1 = |a: [f64; 4]| CoeffR1 { ott: a[0], putt: a[1], residual: a[2], residual2: a[3] };
    let c2 = |a: [f64; 8]| CoeffR2 {
        residual: a[0], residual2: a[1], residual3: a[2], avg_ott: a[3],
        avg_putt: a[4], avg_app: a[5], avg_arg: a[6], delta_app: a[7],
    };
    let c3 = |a: [f64; 5]| CoeffR3 {
        sg_ott_avg: a[0], sg_putt_avg: a[1], sg_app_avg: a[2], sg_arg_avg: a[3],
        pos_6_10: a[4],
    };
    let mk_known = |st: Option<PyReadonlyArray1<i64>>, ct: Option<PyReadonlyArray2<f64>>| {
        match (st, ct) {
            (Some(s), Some(c)) => Some(KnownRound {
                strokes: s.as_slice().unwrap().to_vec(),
                cats: rows4(&c.as_array()),
            }),
            _ => None,
        }
    };
    let lc_v = l_corr.as_array();
    let mut lc = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            lc[i][j] = lc_v[(i, j)];
        }
    }
    let inp = Inputs {
        n, sims, seed, completed_round, default_par,
        mu: rows4(&mu_v),
        std_course: rows4(&std_course.as_array()),
        eff_skew: rows4(&eff_skew.as_array()),
        l_corr: lc,
        my_pred_base: my_pred_base.as_slice().unwrap().to_vec(),
        expected_r2: expected_r2.as_slice().unwrap().to_vec(),
        known_r1: mk_known(known_strokes_r1, known_cats_r1),
        known_r2: mk_known(known_strokes_r2, known_cats_r2),
        known_r3: mk_known(known_strokes_r3, known_cats_r3),
        r1_high: c1(r1_high), r1_midh: c1(r1_midh), r1_midl: c1(r1_midl), r1_low: c1(r1_low),
        r2_lt6: c2(r2_lt6), r2_6_30: c2(r2_6_30), r2_30up: c2(r2_30up),
        r3_lt6: c3(r3_lt6), r3_6_20: c3(r3_6_20), r3_30up: c3(r3_30up),
        cut_line, use_10_shot_rule,
    };
    let out = round_cascade::run_remaining_rounds(&inp);
    let (no, ns2) = (out.n, out.sims);
    let fs = numpy::ndarray::Array2::from_shape_vec((no, ns2), out.final_scores).unwrap();
    let mc = numpy::ndarray::Array2::from_shape_vec((no, ns2), out.made_cut).unwrap();
    let r2 = numpy::ndarray::Array2::from_shape_vec((no, ns2), out.r1_r2).unwrap();
    let r3 = numpy::ndarray::Array2::from_shape_vec((no, ns2), out.r1_r3).unwrap();
    (
        fs.into_pyarray_bound(py),
        mc.into_pyarray_bound(py),
        out.win_prob.into_pyarray_bound(py),
        r2.into_pyarray_bound(py),
        r3.into_pyarray_bound(py),
    )
}

/// Single-round category-first score draw (round_sim simulate_round_scores_catfirst,
/// seed 789). Weather IS split per category and added back here. Returns
/// (scores (n,sims) i64, cat_mu (n,4) f64). The caller must pass only players with a
/// valid scores_rN (NaN players are filtered in Python) and map rows back by order.
#[pyfunction]
#[pyo3(signature = (mu, std_course, eff_skew, l_corr, skill, wx_delta, player_avg, sims, seed=789))]
#[allow(clippy::too_many_arguments)]
fn run_single_round<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<'py, f64>,
    std_course: PyReadonlyArray2<'py, f64>,
    eff_skew: PyReadonlyArray2<'py, f64>,
    l_corr: PyReadonlyArray2<'py, f64>,
    skill: PyReadonlyArray1<'py, f64>,
    wx_delta: PyReadonlyArray1<'py, f64>,
    player_avg: PyReadonlyArray1<'py, f64>,
    sims: usize,
    seed: u64,
) -> (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>) {
    use round_cascade::{run_single_round as rsr, SingleRoundInputs};
    let mu_v = mu.as_array();
    let n = mu_v.nrows();
    let lc_v = l_corr.as_array();
    let mut lc = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            lc[i][j] = lc_v[(i, j)];
        }
    }
    let inp = SingleRoundInputs {
        n, sims, seed,
        mu: rows4(&mu_v),
        std_course: rows4(&std_course.as_array()),
        eff_skew: rows4(&eff_skew.as_array()),
        l_corr: lc,
        skill: skill.as_slice().unwrap().to_vec(),
        wx_delta: wx_delta.as_slice().unwrap().to_vec(),
        player_avg: player_avg.as_slice().unwrap().to_vec(),
    };
    let out = rsr(&inp);
    let sc = numpy::ndarray::Array2::from_shape_vec((out.n, out.sims), out.scores).unwrap();
    let cm = numpy::ndarray::Array2::from_shape_vec((out.n, 4), out.cat_mu).unwrap();
    (sc.into_pyarray_bound(py), cm.into_pyarray_bound(py))
}

#[pymodule]
fn sims_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(selftest, m)?)?;
    m.add_function(wrap_pyfunction!(rint_array, m)?)?;
    m.add_function(wrap_pyfunction!(min_rank_array, m)?)?;
    m.add_function(wrap_pyfunction!(apply_skew_array, m)?)?;
    m.add_function(wrap_pyfunction!(sum4_rows, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_round, m)?)?;
    m.add_function(wrap_pyfunction!(h2h, m)?)?;
    m.add_function(wrap_pyfunction!(run_pretournament, m)?)?;
    m.add_function(wrap_pyfunction!(run_remaining_rounds, m)?)?;
    m.add_function(wrap_pyfunction!(run_single_round, m)?)?;
    Ok(())
}

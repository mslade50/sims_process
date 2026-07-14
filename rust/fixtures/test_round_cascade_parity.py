"""Statistical parity: Rust `run_remaining_rounds` vs the Python reference.

RUST_MIGRATION_PLAN.md §6.2/§7. Like test_cascade_parity.py but for round_sim's
seed-42 tournament cascade, swept across completed_round (0=pre-event .. 3=only R4
left) to exercise the known-round seeding paths. Rust uses a different RNG stream,
so we compare aggregate finish distributions within Monte-Carlo SE.

Usage:  python rust/fixtures/test_round_cascade_parity.py
"""

import sys
import numpy as np

import sims_kernel as k
from ref_remaining_rounds import run as run_py

SEED = 42
N = 60
SIMS = 60_000
K = 5.0

COEFFS = dict(
    r1_high=dict(ott=0.0, putt=0.0, residual=0.0460, residual2=0.0126),
    r1_midh=dict(ott=0.0, putt=0.0, residual=0.0415, residual2=-0.0057),
    r1_midl=dict(ott=0.11, putt=-0.03, residual=0.0571, residual2=-0.0052),
    r1_low=dict(ott=0.09, putt=-0.07, residual=0.0578, residual2=-0.0092),
    r2_lt6=dict(residual=0.1641, residual2=-0.0809, residual3=0.0076, avg_ott=0.11,
                avg_putt=-0.03, avg_app=-0.0128, avg_arg=-0.15, delta_app=0.06),
    r2_6_30=dict(residual=0.0029, residual2=-0.0209, residual3=0.0038, avg_ott=0.07,
                 avg_putt=-0.01, avg_app=-0.03, avg_arg=-0.03, delta_app=0.01),
    r2_30up=dict(residual=0.0603, residual2=-0.0010, residual3=0.0015, avg_ott=0.166,
                 avg_putt=0.00, avg_app=0.015, avg_arg=-0.06, delta_app=0.004),
    r3_lt6=dict(sg_ott_avg=0.035, sg_putt_avg=-0.14, sg_app_avg=-0.02, sg_arg_avg=-0.3),
    r3_6_20=dict(sg_ott_avg=0.089, sg_putt_avg=-0.02, sg_app_avg=-0.07, sg_arg_avg=-0.13, pos_6_10=-0.13),
    r3_30up=dict(sg_ott_avg=0.15, sg_putt_avg=-0.00, sg_app_avg=0.05, sg_arg_avg=-0.01),
)


def make_inputs(rng, completed):
    n = N
    my_pred = np.linspace(-1.0, 2.0, n)
    base = rng.normal(0.0, 0.15, size=(n, 4)); base -= base.mean(axis=1, keepdims=True)
    mu = base + my_pred[:, None] / 4.0
    std_course = np.abs(rng.normal(1.0, 0.1, size=(n, 4))).clip(0.3) * np.array([1.2, 1.0, 0.9, 1.05])
    eff_skew = np.tile(np.array([-0.93, -0.21, -0.18, -0.05]), (n, 1))
    # Fixed mild-correlation PD matrix (injected identically into ref + rust, so the
    # exact structure is irrelevant to parity; just needs to be valid).
    R = np.full((4, 4), 0.2) + np.diag([0.8, 0.8, 0.8, 0.8])
    L = np.linalg.cholesky(R)
    # known rounds: integer strokes near par + small known SG categories
    known_strokes, known_categories = {}, {}
    for rnd in range(1, completed + 1):
        known_strokes[rnd] = (72 - np.rint(rng.normal(0, 2.5, n))).astype(np.int64)
        kc = rng.normal(0, 0.4, size=(n, 4)); known_categories[rnd] = kc
    inp = dict(
        completed_round=completed, default_par=72.0, mu=mu, std_course=std_course,
        eff_skew=eff_skew, l_corr=L, my_pred_base=my_pred.copy(),
        expected_r2=np.full(n, 72.0), known_strokes=known_strokes,
        known_categories=known_categories, cut_line=35, use_10_shot_rule=True, sims=SIMS,
        **COEFFS,
    )
    return inp


def c1(d): return [d["ott"], d["putt"], d["residual"], d["residual2"]]
def c2(d): return [d["residual"], d["residual2"], d["residual3"], d["avg_ott"],
                   d["avg_putt"], d["avg_app"], d["avg_arg"], d["delta_app"]]
def c3(d): return [d["sg_ott_avg"], d["sg_putt_avg"], d["sg_app_avg"], d["sg_arg_avg"], d.get("pos_6_10", 0.0)]


def run_rust(inp):
    A = np.ascontiguousarray
    ks = inp["known_strokes"]; kc = inp["known_categories"]
    def ks_get(r): return A(ks[r].astype(np.int64)) if r in ks else None
    def kc_get(r): return A(kc[r]) if r in kc else None
    return k.run_remaining_rounds(
        inp["completed_round"], inp["default_par"], A(inp["mu"]), A(inp["std_course"]),
        A(inp["eff_skew"]), A(inp["l_corr"]), A(inp["my_pred_base"]), A(inp["expected_r2"]),
        ks_get(1), kc_get(1), ks_get(2), kc_get(2), ks_get(3), kc_get(3),
        c1(inp["r1_high"]), c1(inp["r1_midh"]), c1(inp["r1_midl"]), c1(inp["r1_low"]),
        c2(inp["r2_lt6"]), c2(inp["r2_6_30"]), c2(inp["r2_30up"]),
        c3(inp["r3_lt6"]), c3(inp["r3_6_20"]), c3(inp["r3_30up"]),
        inp["cut_line"], inp["use_10_shot_rule"], inp["sims"], SEED,
    )


def se(p, n1, n2):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))


def gate(pr, pp, label, fails):
    z = np.abs(pr - pp) / np.maximum(se(0.5 * (pr + pp), SIMS, SIMS), 1e-12)
    nbad = int(np.sum(z > K))
    print(f"    {'OK ' if nbad == 0 else 'BAD'} {label:14s} worst z={float(np.max(z)):5.2f}  >{K:.0f}se: {nbad}")
    if nbad:
        fails.append(f"{label}: {nbad} cells > {K}se")


def verify(completed):
    rng = np.random.default_rng(1000 + completed)
    inp = make_inputs(rng, completed)
    print(f"[completed_round={completed}] n={N}, sims={SIMS:,}")
    fs_py, mc_py, win_py = run_py(inp, seed=SEED)
    fs_r, mc_r, win_r, *_extra = run_rust(inp)  # kernel also returns r1_r2/r1_r3 now

    praw_p, tdh_p, tnodh_p = k.aggregate_round(np.ascontiguousarray(fs_py))
    praw_r, tdh_r, tnodh_r = k.aggregate_round(np.ascontiguousarray(fs_r))
    fails = []
    gate(win_r, win_py, "win_prob", fails)
    gate(tdh_r[:, 0], tdh_p[:, 0], "top5", fails)
    gate(tdh_r[:, 1], tdh_p[:, 1], "top10", fails)
    gate(tnodh_r[:, 2], tnodh_p[:, 2], "top20_nodh", fails)
    gate(praw_r[:, :20].ravel(), praw_p[:, :20].ravel(), "rank[1..20]", fails)
    # made-cut rate sanity (only meaningful when completed<2; else all True)
    if completed < 2:
        gate(mc_r.mean(axis=1), mc_py.mean(axis=1), "made_cut_rate", fails)
    return fails


def main():
    all_fail = []
    for completed in (0, 1, 2):
        all_fail += verify(completed)
        print()
    if all_fail:
        print("ROUND CASCADE PARITY FAILED:")
        for m in all_fail:
            print(f"  - {m}")
        sys.exit(1)
    print(f"ROUND CASCADE STATISTICALLY EQUIVALENT vs Python reference (all < {K:.0f} SE).")


if __name__ == "__main__":
    main()

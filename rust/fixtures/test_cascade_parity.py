"""Statistical parity: Rust `run_pretournament` vs the Python reference cascade.

RUST_MIGRATION_PLAN.md §7 ("Probability" + "Decision" layers + "Seed-sweep").
The Rust kernel uses a different RNG stream than numpy (statistical target, plan
§3), so `final_scores` differs sim-by-sim. We instead compare AGGREGATE outputs:
win prob, top-5/10/20, and a sample of rank cells must agree within Monte-Carlo
standard error, and no decision-relevant quantity may diverge.

Both runs are independent Monte-Carlo samples of the SAME model, so a per-cell
difference is ~ N(0, se_combined) with se_combined = sqrt(p(1-p)*(1/N_r + 1/N_p)).
We gate at k=5 SE (Bonferroni-ish across many cells) and report the worst z.

Usage:  python rust/fixtures/test_cascade_parity.py
"""

import sys
import numpy as np

import sims_kernel as k
from ref_pretournament import run as run_py

SEED = 456
N = 60
SIMS = 60_000
K = 5.0  # SE multiplier gate


def make_inputs(rng):
    n = N
    # Skill spread across buckets (drives R1 my_pred bucket + position routing).
    my_pred = np.linspace(-1.2, 2.0, n)
    # Category means re-centered so they sum to my_pred (as production does).
    base = rng.normal(0.0, 0.15, size=(n, 4))
    base -= base.mean(axis=1, keepdims=True)
    mu = base + (my_pred[:, None] / 4.0)
    std_course = np.abs(rng.normal(1.0, 0.1, size=(n, 4))).clip(0.3) * np.array([1.2, 1.0, 0.9, 1.05])
    eff_skew = np.tile(np.array([-0.93, -0.21, -0.18, -0.05]), (n, 1))
    # PD correlation -> lower Cholesky (injected, as production does).
    A = rng.normal(0, 1, size=(4, 4))
    R = np.corrcoef(A @ A.T + np.eye(4))
    L = np.linalg.cholesky(R)
    inp = dict(
        mu=mu, std_course=std_course, eff_skew=eff_skew, l_corr=L,
        my_pred_base=my_pred.copy(),
        r2_mu=my_pred + rng.normal(0, 0.05, n),
        r3_mu=my_pred + rng.normal(0, 0.05, n),
        r4_mu=my_pred + rng.normal(0, 0.05, n),
        weather_delta_r1=rng.normal(0, 0.3, n),
        weather_delta_r2=rng.normal(0, 0.3, n),
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
        cut_line=35, use_10_shot_rule=True, sims=SIMS,
    )
    return inp


def coeff_arr_r1(d):
    return [d["ott"], d["putt"], d["residual"], d["residual2"]]


def coeff_arr_r2(d):
    return [d["residual"], d["residual2"], d["residual3"], d["avg_ott"],
            d["avg_putt"], d["avg_app"], d["avg_arg"], d["delta_app"]]


def coeff_arr_r3(d):
    return [d["sg_ott_avg"], d["sg_putt_avg"], d["sg_app_avg"], d["sg_arg_avg"], d.get("pos_6_10", 0.0)]


def run_rust(inp):
    return k.run_pretournament(
        np.ascontiguousarray(inp["mu"]), np.ascontiguousarray(inp["std_course"]),
        np.ascontiguousarray(inp["eff_skew"]), np.ascontiguousarray(inp["l_corr"]),
        np.ascontiguousarray(inp["my_pred_base"]), np.ascontiguousarray(inp["r2_mu"]),
        np.ascontiguousarray(inp["r3_mu"]), np.ascontiguousarray(inp["r4_mu"]),
        np.ascontiguousarray(inp["weather_delta_r1"]), np.ascontiguousarray(inp["weather_delta_r2"]),
        coeff_arr_r1(inp["r1_high"]), coeff_arr_r1(inp["r1_midh"]),
        coeff_arr_r1(inp["r1_midl"]), coeff_arr_r1(inp["r1_low"]),
        coeff_arr_r2(inp["r2_lt6"]), coeff_arr_r2(inp["r2_6_30"]), coeff_arr_r2(inp["r2_30up"]),
        coeff_arr_r3(inp["r3_lt6"]), coeff_arr_r3(inp["r3_6_20"]), coeff_arr_r3(inp["r3_30up"]),
        inp["cut_line"], inp["use_10_shot_rule"], inp["sims"], SEED,
    )


def se_combined(p, n1, n2):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))


def worst_z(p_rust, p_py, label, failures):
    se = se_combined(0.5 * (p_rust + p_py), SIMS, SIMS)
    z = np.abs(p_rust - p_py) / np.maximum(se, 1e-12)
    wz = float(np.max(z))
    nbad = int(np.sum(z > K))
    status = "OK " if nbad == 0 else "BAD"
    print(f"  {status} {label:18s} worst z={wz:5.2f}  cells>{K:.0f}se: {nbad}")
    if nbad > 0:
        failures.append(f"{label}: {nbad} cells exceed {K}se (worst z={wz:.2f})")


def main():
    rng = np.random.default_rng(20260605)
    inp = make_inputs(rng)
    print(f"[cascade-parity] n={N}, sims={SIMS:,}, gate=k{K:.0f} SE\n")

    fs_py, win_py = run_py(inp, seed=SEED)
    fs_rust, win_rust, *_ = run_rust(inp)

    # Aggregate both via the (validated) Rust aggregator for an apples-to-apples
    # comparison of the DISTRIBUTIONS produced by each kernel's final_scores.
    u_py, ndh_py, top_py = k.aggregate(np.ascontiguousarray(fs_py))
    u_rust, ndh_rust, top_rust = k.aggregate(np.ascontiguousarray(fs_rust))

    failures = []
    # Win prob (each kernel computes its own via choice-tiebreak).
    worst_z(win_rust, win_py, "win_prob", failures)
    # Top-N finish.
    worst_z(top_rust[:, 0], top_py[:, 0], "top5", failures)
    worst_z(top_rust[:, 1], top_py[:, 1], "top10", failures)
    worst_z(top_rust[:, 2], top_py[:, 2], "top20", failures)
    # Rank prob mass for ranks 1..20 (the betting-relevant head of the table).
    worst_z(u_rust[:, :20].ravel(), u_py[:, :20].ravel(), "rank_prob[1..20]", failures)

    # Decision sanity: win-prob ordering shouldn't flip materially. Report top-5.
    order_py = np.argsort(-win_py)[:5]
    order_rust = np.argsort(-win_rust)[:5]
    print(f"\n  top-5 by win prob (py):   {order_py.tolist()}")
    print(f"  top-5 by win prob (rust): {order_rust.tolist()}")

    if failures:
        print("\nCASCADE PARITY FAILED (statistical):")
        for m in failures:
            print(f"  - {m}")
        sys.exit(1)
    print(f"\nCASCADE STATISTICALLY EQUIVALENT vs Python reference (all cells < {K:.0f} SE).")


if __name__ == "__main__":
    main()

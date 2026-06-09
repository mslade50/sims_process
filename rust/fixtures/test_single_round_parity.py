"""Statistical parity: Rust `run_single_round` vs the Python reference.

Phase 5 (round_sim simulate_round_scores_catfirst, seed 789). cat_mu is RNG-free
so it must match (near-)exactly; the integer score distributions use a different
RNG stream, so compare per-player score CDFs within Monte-Carlo SE.

Usage:  python rust/fixtures/test_single_round_parity.py
"""

import sys
import numpy as np

import sims_kernel as k
from ref_single_round import run as run_py

SEED = 789
N = 60
SIMS = 60_000
K = 5.0


def make_inputs(rng):
    n = N
    skill = np.linspace(-1.5, 1.5, n)
    base = rng.normal(0.0, 0.15, size=(n, 4)); base -= base.mean(axis=1, keepdims=True)
    mu = base + skill[:, None] / 4.0
    std_course = np.abs(rng.normal(1.0, 0.1, size=(n, 4))).clip(0.3) * np.array([1.2, 1.0, 0.9, 1.05])
    eff_skew = np.tile(np.array([-0.93, -0.21, -0.18, -0.05]), (n, 1))
    R = np.full((4, 4), 0.2) + np.diag([0.8, 0.8, 0.8, 0.8])
    L = np.linalg.cholesky(R)
    wx_delta = rng.normal(0.0, 0.4, size=n)
    player_avg = 71.0 + rng.normal(0.0, 0.8, size=n)
    return dict(mu=mu, std_course=std_course, eff_skew=eff_skew, l_corr=L,
                skill=skill, wx_delta=wx_delta, player_avg=player_avg, sims=SIMS)


def run_rust(inp):
    A = np.ascontiguousarray
    return k.run_single_round(
        A(inp["mu"]), A(inp["std_course"]), A(inp["eff_skew"]), A(inp["l_corr"]),
        A(inp["skill"]), A(inp["wx_delta"]), A(inp["player_avg"]), inp["sims"], SEED,
    )


def se(p, n1, n2):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))


def cdf(scores, lines):
    # (n, len(lines)) P(score <= line)
    return np.stack([(scores <= L).mean(axis=1) for L in lines], axis=1)


def main():
    rng = np.random.default_rng(2025)
    inp = make_inputs(rng)
    print(f"[single-round-parity] n={N}, sims={SIMS:,}, seed={SEED}")

    sc_py, cm_py = run_py(inp, seed=SEED)
    sc_r, cm_r = run_rust(inp)
    sc_r = np.ascontiguousarray(sc_r)
    fails = []

    # 1. cat_mu is deterministic (RNG-free) -> must match
    cm_diff = float(np.abs(cm_py - np.asarray(cm_r)).max())
    if cm_diff < 1e-9:
        print(f"  OK  cat_mu            max|diff|={cm_diff:.2e} (deterministic, exact)")
    else:
        fails.append(f"cat_mu max|diff|={cm_diff:.2e}")
        print(f"  BAD cat_mu            max|diff|={cm_diff:.2e}")

    # 2. score CDF per player across a common line grid, within k*SE
    lo = int(min(sc_py.min(), sc_r.min())); hi = int(max(sc_py.max(), sc_r.max()))
    lines = np.arange(lo, hi + 1)
    cp = cdf(sc_py, lines); cr = cdf(sc_r, lines)
    z = np.abs(cr - cp) / np.maximum(se(0.5 * (cr + cp), SIMS, SIMS), 1e-12)
    nbad = int(np.sum(z > K))
    print(f"  {'OK ' if nbad == 0 else 'BAD'} score CDF[{lo}..{hi}]  worst z={float(np.max(z)):5.2f}  >{K:.0f}se: {nbad}")
    if nbad:
        fails.append(f"score CDF: {nbad} cells > {K}se")

    # 3. per-player mean score within k * (std/sqrt(sims)) combined SE
    mean_py = sc_py.mean(axis=1); mean_r = sc_r.mean(axis=1)
    se_mean = np.sqrt(sc_py.std(axis=1) ** 2 / SIMS + sc_r.std(axis=1) ** 2 / SIMS)
    zb = np.abs(mean_r - mean_py) / np.maximum(se_mean, 1e-12)
    nbad2 = int(np.sum(zb > K))
    print(f"  {'OK ' if nbad2 == 0 else 'BAD'} mean score        worst z={float(np.max(zb)):5.2f}  >{K:.0f}se: {nbad2}")
    if nbad2:
        fails.append(f"mean score: {nbad2} players > {K}se")

    print()
    if fails:
        print("SINGLE-ROUND PARITY FAILED:")
        for m in fails:
            print(f"  - {m}")
        sys.exit(1)
    print(f"SINGLE-ROUND STATISTICALLY EQUIVALENT vs Python reference (all < {K:.0f} SE).")


if __name__ == "__main__":
    main()

"""Freeze current-Python op-primitive outputs as the CI regression oracle.

RUST_MIGRATION_PLAN.md §7 (capture-then-validate): the kernel's Python source is a
temporary scaffold. Before any Python kernel block is deleted, its behavior must be
frozen to disk so the Rust port can be validated without live Python. This script
captures the *op-primitive layer* (Phase 1): np.rint half-even, pandas min-rank,
_apply_skew, and the 4-category row sum.

Inputs are deliberately stress-shaped:
  * rint: many exact `*.5` values (the half-even boundary that cascades into ranks)
  * min-rank: tie-heavy integer leaderboards (routes coefficient buckets)
  * skew: gammas spanning the |gamma|<0.01 identity branch, the |gamma|<0.2 cf
    branch, and saturated large-|gamma| values
  * sum4: random 4-tuples plus values engineered near a `*.5` total

Run from anywhere:  python rust/fixtures/capture_ops.py
Writes:             rust/fixtures/ops_fixture.npz

The reference implementations here are copied verbatim from new_sim.py so the
fixture is anchored to production behavior, not a re-derivation.
"""

import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops_fixture.npz")
RNG = np.random.default_rng(0)  # fixed seed -> reproducible fixture


# --- Reference ops: verbatim from new_sim.py -------------------------------

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


def rank_min(values_int):
    return pd.Series(values_int).rank(method="min").astype(int).to_numpy()


# --- Build stressed inputs --------------------------------------------------

# rint: blend of random floats and exact half-integers (both signs).
half_ints = np.arange(-5, 6) + 0.5            # -4.5 ... 5.5
half_ints = np.concatenate([half_ints, half_ints + 60.0])  # par-scale too
rint_in = np.concatenate([
    RNG.normal(72.0, 3.0, size=500),
    half_ints,
    RNG.uniform(-200.0, 80.0, size=200),       # missed-cut penalty scale
]).astype(float)

# min-rank: several tie-heavy leaderboards of realistic size.
rank_cases = []
for n in (10, 50, 147):
    # small integer spread forces many ties (like real stroke leaderboards)
    rank_cases.append(RNG.integers(130, 160, size=n).astype(np.int64))
rank_in = np.concatenate(rank_cases)
rank_lens = np.array([len(c) for c in rank_cases], dtype=np.int64)

# skew: representative gammas + a shared z-column.
gammas = np.array([
    0.0, 0.005, -0.009,        # identity branch
    0.05, -0.19,               # cf-mult == 1 branch
    -0.93, -0.21, -0.18, -0.05,  # BASELINE_CAT_SKEW (PGA)
    0.4, -1.2, 2.0,            # saturated
], dtype=float)
z_col = RNG.standard_normal(2000)
skew_out = np.stack([_apply_skew(z_col, g) for g in gammas])  # (len(gammas), 2000)

# sum4: random rows + rows engineered near a *.5 total.
sum_rows = RNG.standard_normal((1000, 4))
edge = np.array([
    [0.25, 0.25, 0.0, 0.0],    # sums to 0.5
    [0.1, 0.2, 0.3, -0.1],     # 0.5
    [18.0, 18.0, 18.0, 17.5],  # 71.5 at par scale
])
sum_in = np.concatenate([sum_rows, edge]).astype(float)

# --- Reference outputs ------------------------------------------------------

rint_out = np.rint(rint_in).astype(np.int64)
rank_out = np.concatenate([rank_min(c) for c in rank_cases]).astype(np.int64)
sum_out = sum_in.sum(axis=1)

np.savez(
    OUT,
    rint_in=rint_in, rint_out=rint_out,
    rank_in=rank_in, rank_lens=rank_lens, rank_out=rank_out,
    gammas=gammas, z_col=z_col, skew_out=skew_out,
    sum_in=sum_in, sum_out=sum_out,
)
print(f"wrote {OUT}")
print(f"  rint: {rint_in.size} values ({int((np.modf(rint_in)[0] == 0.5).sum() + (np.modf(rint_in)[0] == -0.5).sum())} exact half)")
print(f"  rank: {rank_in.size} values across {rank_lens.size} leaderboards")
print(f"  skew: {gammas.size} gammas x {z_col.size} draws")
print(f"  sum4: {sum_in.shape[0]} rows")

"""Step 1: verify the RNG-free aggregation half against REAL production output.

RUST_MIGRATION_PLAN.md §7. The aggregation layer (rank_probs `prob_u`/`prob_ndh`,
top-N, h2h) is a pure function of the integer `final_scores` matrix. Production
already wrote both the input (`<event>/final_scores.npy`) and its outputs
(`rank_probs_updated_<event>.parquet`, `h2h_matrix_<event>.parquet`) from the same
run, so we can diff Rust against genuine production output with NO code changes
and NO RNG dependence.

Expected: prob_ndh + h2h are integer/rational -> (near-)bit-exact; prob_u carries
fractional dead-heat weights whose sum order differs from pandas -> within ~1e-15.

Usage:  python rust/fixtures/verify_against_prod.py [event ...]
        (default: cjcup schwab)
"""

import json
import os
import sys
import numpy as np
import pandas as pd

import sims_kernel as k

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOL_U = 1e-12  # fractional dead-heat weights: FP sum-order, not bit-exact


def load_event(event):
    fs_path = os.path.join(ROOT, event, "final_scores.npy")
    pn_path = os.path.join(ROOT, event, "player_names.json")
    if not os.path.exists(fs_path):
        fs_path = os.path.join(ROOT, f"final_scores_{event}.npy")
    rp_path = os.path.join(ROOT, f"rank_probs_updated_{event}.parquet")
    h_path = os.path.join(ROOT, f"h2h_matrix_{event}.parquet")
    for p in (fs_path, pn_path, rp_path, h_path):
        if not os.path.exists(p):
            return None
    fs = np.load(fs_path).astype(np.int64)
    pn = json.load(open(pn_path))
    rp = pd.read_parquet(rp_path)
    h = pd.read_parquet(h_path)
    return fs, pn, rp, h


def verify(event):
    loaded = load_event(event)
    if loaded is None:
        print(f"[{event}] SKIP — missing one of final_scores/player_names/rank_probs/h2h")
        return True
    fs, pn, rp, h = loaded
    n, sims = fs.shape
    assert len(pn) == n, f"player_names ({len(pn)}) != final_scores rows ({n})"
    pidx = {p: i for i, p in enumerate(pn)}
    print(f"[{event}] n={n}, sims={sims:,}")

    fs = np.ascontiguousarray(fs)
    ru, rndh, rtop = k.aggregate(fs)
    ia, ib, pa, tp = k.h2h(fs)

    failures = []

    # --- Densify production rank_probs to (n,n) and compare full matrices ---
    py_u = np.zeros((n, n))
    py_ndh = np.zeros((n, n))
    for r in rp.itertuples(index=False):
        i = pidx[r.player_name]
        py_u[i, int(r.rank) - 1] = r.prob_u
        py_ndh[i, int(r.rank) - 1] = r.prob_ndh

    # prob_ndh = integer count / sims -> bit-exact vs production.
    if np.array_equal(rndh, py_ndh):
        print(f"  prob_ndh   OK  (full {n}x{n}, BIT-EXACT vs prod)")
    else:
        d = float(np.max(np.abs(rndh - py_ndh)))
        nbad = int(np.sum(rndh != py_ndh))
        failures.append(f"prob_ndh: {nbad} cells differ, max|diff|={d:.2e}")

    du = float(np.max(np.abs(ru - py_u)))
    if du < TOL_U:
        print(f"  prob_u     OK  (full {n}x{n}, max|diff|={du:.1e} < {TOL_U:.0e})")
    else:
        # locate worst cell for diagnostics
        ij = np.unravel_index(int(np.argmax(np.abs(ru - py_u))), ru.shape)
        failures.append(f"prob_u: max|diff|={du:.2e} at player={pn[ij[0]]} rank={ij[1]+1} "
                        f"(rust={ru[ij]:.6g} prod={py_u[ij]:.6g})")

    # --- h2h: compare pair-by-pair keyed on (name_a, name_b) ---
    rust_h = {}
    for a, b, p, t in zip(ia.tolist(), ib.tolist(), pa.tolist(), tp.tolist()):
        rust_h[(pn[a], pn[b])] = (p, t)
    prod_h = {(r.player_a, r.player_b): (r.prob_a, r.tie_pct) for r in h.itertuples(index=False)}

    if set(rust_h) != set(prod_h):
        only_rust = len(set(rust_h) - set(prod_h))
        only_prod = len(set(prod_h) - set(rust_h))
        failures.append(f"h2h pair sets differ: rust-only={only_rust}, prod-only={only_prod}")
    else:
        max_p = max_t = 0.0
        exact_p = 0
        for key, (pp, pt) in prod_h.items():
            rp_, rt_ = rust_h[key]
            max_p = max(max_p, abs(rp_ - pp))
            max_t = max(max_t, abs(rt_ - pt))
            if rp_ == pp:
                exact_p += 1
        npairs = len(prod_h)
        if max_p < TOL_U and max_t < TOL_U:
            print(f"  h2h        OK  ({npairs} pairs; prob_a exact={exact_p}/{npairs}, "
                  f"max|dp|={max_p:.1e}, max|dt|={max_t:.1e})")
        else:
            failures.append(f"h2h: max|d prob_a|={max_p:.2e}, max|d tie_pct|={max_t:.2e}")

    if failures:
        print(f"  [{event}] FAIL:")
        for m in failures:
            print(f"    - {m}")
        return False
    return True


def main():
    events = sys.argv[1:] or ["cjcup", "schwab"]
    ok = True
    for ev in events:
        ok &= verify(ev)
        print()
    if not ok:
        print("PRODUCTION AGGREGATION PARITY FAILED.")
        sys.exit(1)
    print("RUST AGGREGATION MATCHES REAL PRODUCTION OUTPUT.")


if __name__ == "__main__":
    main()

"""Validate the Rust `sims_kernel` op primitives against the frozen Python oracle.

RUST_MIGRATION_PLAN.md §7, "Op unit" + "Bucket routing" layers: these must be
ARRAY-EXACT (not statistical) — they are the load-bearing pieces whose tie/`*.5`
semantics route coefficient buckets. Any mismatch fails the wheel.

Usage:
    python rust/fixtures/capture_ops.py          # (re)generate the oracle
    python rust/fixtures/test_ops_parity.py      # validate Rust vs oracle

Exit code 0 = parity holds. Intended to gate the wheel publish in CI.
"""

import os
import sys
import numpy as np

import sims_kernel as k

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops_fixture.npz")


def main():
    if not os.path.exists(FIX):
        sys.exit(f"missing fixture {FIX} — run capture_ops.py first")
    f = np.load(FIX)
    failures = []

    # --- rint (round-half-to-even) : array-exact ---
    got = k.rint_array(np.ascontiguousarray(f["rint_in"]))
    if not np.array_equal(got, f["rint_out"]):
        n_bad = int((got != f["rint_out"]).sum())
        failures.append(f"rint: {n_bad}/{got.size} mismatches")
    else:
        print(f"  rint  OK ({got.size} values, exact)")

    # --- min-rank : array-exact, per leaderboard ---
    offset = 0
    rank_ok = True
    for li, n in enumerate(f["rank_lens"]):
        n = int(n)
        chunk = np.ascontiguousarray(f["rank_in"][offset:offset + n])
        got = k.min_rank_array(chunk)
        exp = f["rank_out"][offset:offset + n]
        if not np.array_equal(got, exp):
            failures.append(f"min_rank: leaderboard {li} (n={n}) mismatch")
            rank_ok = False
        offset += n
    if rank_ok:
        print(f"  rank  OK ({offset} values across {f['rank_lens'].size} boards, exact)")

    # --- apply_skew : exact float match (same op order as Python) ---
    skew_ok = True
    z = np.ascontiguousarray(f["z_col"])
    for gi, gamma in enumerate(f["gammas"]):
        got = k.apply_skew_array(z, float(gamma))
        exp = f["skew_out"][gi]
        if not np.array_equal(got, exp):
            max_abs = float(np.max(np.abs(got - exp)))
            # identity + plain f64 arithmetic should be bit-identical; flag any drift.
            failures.append(f"apply_skew: gamma={gamma:+.3f} max|diff|={max_abs:.2e}")
            skew_ok = False
    if skew_ok:
        print(f"  skew  OK ({f['gammas'].size} gammas x {z.size} draws, exact)")

    # --- sum4 (4-category row sum) : exact float match ---
    flat = np.ascontiguousarray(f["sum_in"].ravel())
    got = k.sum4_rows(flat)
    if not np.array_equal(got, f["sum_out"]):
        max_abs = float(np.max(np.abs(got - f["sum_out"])))
        failures.append(f"sum4: max|diff|={max_abs:.2e}")
    else:
        print(f"  sum4  OK ({f['sum_in'].shape[0]} rows, exact)")

    if failures:
        print("\nPARITY FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)
    print("\nALL OP PRIMITIVES ARRAY-EXACT vs Python oracle.")


if __name__ == "__main__":
    main()

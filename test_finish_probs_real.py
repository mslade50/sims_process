#!/usr/bin/env python
"""Run the REAL round_sim.compute_finish_probabilities both ways on the same
final_scores and diff. Unlike validate_rust_python.py (which reimplements the two
paths), this imports round_sim and toggles round_sim._USE_PYTHON, so it exercises
the exact production function — Rust branch vs pandas branch — on identical input.
Exact by construction (no RNG-draw confound): same scores in, both aggregations out.
"""
import sys
import time

import numpy as np

import round_sim as rs  # NOTE: module import does one read-only Google Sheets config read


def synth_post_cut_scores(n_players, num_sims, seed=11):
    """Realistic 72-hole totals: clustered integers (frequent ties) with weaker
    players pinned at 200 per-sim (mirrors round_sim's r3_r4[~made_cut]=200)."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 3.0, n_players)
    fs = np.rint(280 + skill[:, None] + rng.normal(0, 4.0, (n_players, num_sims))).astype(np.int64)
    # per-sim missed cut: prob rises for weaker players (higher skill value = worse here)
    miss_p = np.clip((skill - skill.mean()) / (3.5 * skill.std()) + 0.30, 0.0, 0.85)
    missed = rng.random((n_players, num_sims)) < miss_p[:, None]
    fs[missed] = 200
    return fs, missed


def main():
    n, ns = 156, 100_000
    names = [f"player_{i:03d}" for i in range(n)]
    fs, made_cut_mask = synth_post_cut_scores(n, ns)
    print(f"Input: {n} players x {ns:,} sims; {(fs == 200).mean()*100:.1f}% of "
          f"player-sim slots are missed-cut (200)\n")

    rs._USE_PYTHON = False
    t = time.time()
    fp_r, rp_r = rs.compute_finish_probabilities(fs, names, made_cut_mask, ns)
    t_rust = time.time() - t

    print(f"\n  Running the slow pandas reference (--use-python path)... this takes minutes")
    rs._USE_PYTHON = True
    t = time.time()
    fp_p, rp_p = rs.compute_finish_probabilities(fs, names, made_cut_mask, ns)
    t_py = time.time() - t

    print(f"\n  Rust  compute_finish_probabilities: {t_rust:8.2f}s")
    print(f"  Python compute_finish_probabilities: {t_py:8.2f}s   "
          f"({t_py / max(t_rust, 1e-6):.0f}x slower)\n")

    ok = True

    # Finish-probs table: align on player_name.
    m = fp_p.merge(fp_r, on="player_name", how="outer", suffixes=("_py", "_rs")).fillna(0.0)
    det_cols = ["top_5", "top_10", "top_20", "top_5_nodh", "top_10_nodh", "top_20_nodh"]
    print("  Finish probs (deterministic — must be exact):")
    for c in det_cols:
        d = np.abs(m[f"{c}_py"] - m[f"{c}_rs"]).max()
        flag = "OK " if d <= 1e-9 else "FAIL"
        if d > 1e-9:
            ok = False
        print(f"    {c:14s} max|d| = {d:.3e}  [{flag}]")

    dwin = np.abs(m["simulated_win_prob_py"] - m["simulated_win_prob_rs"]).max()
    win_tol = max(5e-3, 4.0 / np.sqrt(ns))
    wflag = "OK " if dwin <= win_tol else "FAIL"
    if dwin > win_tol:
        ok = False
    print(f"    {'win_prob':14s} max|d| = {dwin:.3e}  [{wflag}  tol={win_tol:.0e}, MC noise]")

    # Rank-prob table: align on (player_name, rank).
    mr = rp_p.merge(rp_r, on=["player_name", "rank"], how="outer",
                    suffixes=("_py", "_rs")).fillna(0.0)
    drank = np.abs(mr["prob_u_py"] - mr["prob_u_rs"]).max()
    rflag = "OK " if drank <= 1e-9 else "FAIL"
    if drank > 1e-9:
        ok = False
    print(f"\n  Rank probs prob_u (deterministic — must be exact):")
    print(f"    {'prob_u':14s} max|d| = {drank:.3e}  [{rflag}]   "
          f"(rows: py={len(rp_p)}, rs={len(rp_r)})")

    print(f"\n{'='*58}\n  RESULT: {'EXACT MATCH — Rust == Python' if ok else 'MISMATCH'}\n{'='*58}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

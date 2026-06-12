#!/usr/bin/env python
"""Rust-vs-Python cross-validation harness for round_sim finish-prob aggregation.

The Rust kernel's draw path is already wired into round_sim.py. The slow part was
`compute_finish_probabilities` — a 15.6M-iteration pandas `groupby`+`iterrows`
loop. This harness proves `sims_kernel.aggregate_round` (Rust) reproduces the
pandas reference EXACTLY for the deterministic outputs, before/while it is wired
into production.

AGGREGATION PARITY (exact, this script). Feed the SAME integer `final_scores` to
both the pandas reference and the Rust kernel. The min-rank probs and top-N tables
are pure integer-counting functions of the scores -> they must match to FP
tolerance. This is the rigorous wiring gate for the change made tonight (only the
aggregation was swapped; the draw kernel was already Rust).

END-TO-END (statistical, run by hand). Run the live pipeline both ways and diff the
saved artifacts:
    python round_sim.py                # default = Rust  -> writes rank_probs_live_{t}.parquet
    cp rank_probs_live_{t}.parquet rank_probs_rust.parquet
    python round_sim.py --use-python   # pandas reference -> overwrites the parquet
    cp rank_probs_live_{t}.parquet rank_probs_py.parquet
Then diff the two parquets. Note: the Rust and Python DRAW kernels use different RNG
streams, so this is a distribution match within MC noise, not bit-exact. The exact
gate is this script (same scores in, both aggregations out).

Usage:
    python validate_rust_python.py                 # synthetic exact-parity gate
    python validate_rust_python.py --players 156 --sims 100000

Exit code 0 = all gates pass, 1 = a gate failed.
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd


# ── Production pandas reference (verbatim logic from round_sim.compute_finish_probabilities) ──
def dead_heat_factor(position, tie_count, threshold):
    start = position
    end = position + tie_count - 1
    overlap_start = max(start, 1)
    overlap_end = min(end, threshold)
    overlap_count = max(0, overlap_end - overlap_start + 1)
    return overlap_count / tie_count


def finish_probs_pandas(final_scores, player_names, num_sims, rng):
    """The reference implementation — what round_sim.py does under --use-python."""
    # Win probabilities (playoff tiebreaker: random winner)
    simulated_winners = []
    for j in range(num_sims):
        sc = final_scores[:, j]
        min_score = sc.min()
        tied = np.where(sc == min_score)[0]
        simulated_winners.append(player_names[rng.choice(tied)])
    win_counts = pd.Series(simulated_winners).value_counts(normalize=True)
    sim_win_probs = win_counts.rename_axis('player_name').reset_index(name='simulated_win_prob')

    df_long = pd.DataFrame(final_scores, index=player_names).T
    df_long['simulation_id'] = np.arange(num_sims)
    long_df = df_long.melt(id_vars='simulation_id', var_name='player_name', value_name='score')
    long_df['rank'] = long_df.groupby('simulation_id')['score'].rank(method='min')

    player_stats = {p: {
        "top_5": 0.0, "top_10": 0.0, "top_20": 0.0,
        "top_5_nodh": 0.0, "top_10_nodh": 0.0, "top_20_nodh": 0.0,
    } for p in player_names}
    for sim_id, group in long_df.groupby("simulation_id", sort=False):
        pos_counts = group['rank'].value_counts().to_dict()
        for _, row in group.iterrows():
            p = row['player_name']
            pos = int(row['rank'])
            tie_ct = pos_counts[pos]
            player_stats[p]["top_5"] += dead_heat_factor(pos, tie_ct, 5)
            player_stats[p]["top_10"] += dead_heat_factor(pos, tie_ct, 10)
            player_stats[p]["top_20"] += dead_heat_factor(pos, tie_ct, 20)
            player_stats[p]["top_5_nodh"] += 1.0 if pos <= 5 else 0.0
            player_stats[p]["top_10_nodh"] += 1.0 if pos <= 10 else 0.0
            player_stats[p]["top_20_nodh"] += 1.0 if pos <= 20 else 0.0

    topn_df = pd.DataFrame.from_dict(player_stats, orient='index')
    topn_df = topn_df.div(num_sims).reset_index().rename(columns={'index': 'player_name'})
    finish_probs = pd.merge(sim_win_probs, topn_df, on="player_name", how="outer").fillna(0)

    rank_probs = (long_df.groupby(['player_name', 'rank']).size()
                  .div(num_sims).rename('prob_u').reset_index())
    rank_probs['rank'] = rank_probs['rank'].astype(int)
    return finish_probs, rank_probs


# ── New Rust-backed path (mirror of what gets wired into round_sim.py) ──
def finish_probs_rust(final_scores, player_names, num_sims):
    import sims_kernel as sk
    fs = np.ascontiguousarray(final_scores.astype(np.int64))
    prob_raw, top_dh, top_nodh = sk.aggregate_round(fs)  # (n,n), (n,3), (n,3)
    prob_raw = np.ascontiguousarray(prob_raw)
    top_dh = np.ascontiguousarray(top_dh)
    top_nodh = np.ascontiguousarray(top_nodh)

    # Win prob = exact expectation of the random-tiebreak winner (vectorized).
    # This is the deterministic value the pandas RNG.choice loop estimates.
    mins = fs.min(axis=0)
    is_min = (fs == mins[None, :])
    tie_counts = is_min.sum(axis=0)
    win_prob = (is_min / tie_counts).sum(axis=1) / num_sims

    finish_probs = pd.DataFrame({
        "player_name": player_names,
        "simulated_win_prob": win_prob,
        "top_5": top_dh[:, 0], "top_10": top_dh[:, 1], "top_20": top_dh[:, 2],
        "top_5_nodh": top_nodh[:, 0], "top_10_nodh": top_nodh[:, 1], "top_20_nodh": top_nodh[:, 2],
    })

    rows, cols = np.nonzero(prob_raw)
    rank_probs = pd.DataFrame({
        "player_name": np.asarray(player_names)[rows],
        "rank": (cols + 1).astype(int),
        "prob_u": prob_raw[rows, cols],
    })
    return finish_probs, rank_probs


def synth_final_scores(n_players, num_sims, cut=None, seed=7):
    """Golf-like 72-hole totals: clustered integers so ties are common, plus
    missed-cut players pinned at 200 (mirrors round_sim's r3_r4[~made_cut]=200)."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 3.0, n_players)            # player strength spread
    base = 280 + skill[:, None]                       # ~par totals
    noise = rng.normal(0, 4.0, (n_players, num_sims))
    fs = np.rint(base + noise).astype(np.int64)
    if cut:
        # bottom `cut` fraction of players miss the cut in ~half the sims
        missed = rng.random((n_players, num_sims)) < (skill[:, None] > np.quantile(skill, 1 - cut)) * 0.5
        fs[missed] = 200
    return fs


def compare(name, ref, new, key_cols, val_cols, tol):
    """Merge ref/new on key_cols, return (max_abs_diff per val col, passed)."""
    m = ref.merge(new, on=key_cols, how="outer", suffixes=("_ref", "_new")).fillna(0.0)
    ok = True
    print(f"\n  {name}:")
    for c in val_cols:
        d = np.abs(m[f"{c}_ref"] - m[f"{c}_new"]).max()
        flag = "OK " if d <= tol else "FAIL"
        if d > tol:
            ok = False
        print(f"    {c:18s} max|d| = {d:.3e}   [{flag}  tol={tol:.0e}]")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=156)
    ap.add_argument("--sims", type=int, default=20000,
                    help="sims for the parity gate (default 20k — pandas ref is slow; "
                         "use 100000 for a full-scale check)")
    ap.add_argument("--cut", type=float, default=0.45, help="fraction of field that can miss cut")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    player_names = [f"player_{i:03d}" for i in range(args.players)]
    fs = synth_final_scores(args.players, args.sims, cut=args.cut, seed=args.seed)
    n_cut = int((fs == 200).all(axis=1).sum())
    print(f"Parity gate: {args.players} players x {args.sims:,} sims "
          f"({(fs == 200).mean() * 100:.1f}% missed-cut sim-slots)")

    t = time.time()
    ref_fp, ref_rp = finish_probs_pandas(fs, player_names, args.sims, np.random.default_rng(123))
    t_pandas = time.time() - t

    t = time.time()
    new_fp, new_rp = finish_probs_rust(fs, player_names, args.sims)
    t_rust = time.time() - t

    print(f"\n  pandas reference: {t_pandas:8.2f}s")
    print(f"  rust aggregate  : {t_rust:8.2f}s   ({t_pandas / max(t_rust, 1e-6):.0f}x faster)")

    # Tight gate: deterministic integer-counting outputs must match exactly.
    det_cols = ["top_5", "top_10", "top_20", "top_5_nodh", "top_10_nodh", "top_20_nodh"]
    ok_topn = compare("Top-N finish probs (deterministic)", ref_fp, new_fp,
                      ["player_name"], det_cols, tol=1e-9)
    ok_rank = compare("Rank probabilities prob_u (deterministic)", ref_rp, new_rp,
                      ["player_name", "rank"], ["prob_u"], tol=1e-9)
    # Loose gate: win prob — pandas is a stochastic MC estimate of the rust
    # deterministic expectation; agreement bounded by MC noise (~1/sqrt(sims)).
    win_tol = max(5e-3, 4.0 / np.sqrt(args.sims))
    ok_win = compare("Win prob (stochastic ref vs deterministic — MC noise expected)",
                     ref_fp, new_fp, ["player_name"], ["simulated_win_prob"], tol=win_tol)

    all_ok = ok_topn and ok_rank and ok_win
    print(f"\n{'='*60}\n  RESULT: {'ALL GATES PASS' if all_ok else 'GATE FAILURE'}\n{'='*60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

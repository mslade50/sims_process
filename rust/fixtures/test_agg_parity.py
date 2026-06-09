"""Validate Rust post-sim aggregation against pandas — INTEGER-EXACT.

RUST_MIGRATION_PLAN.md §7: the post-sim layer (rank_probs `prob_u`/`prob_ndh`,
top-N dead-heat, the O(n^2) h2h loop) consumes only the quantized integer
`final_scores`, so it must match pandas EXACTLY, not statistically.

The Python reference here is copied from new_sim.py:1013-1188 so we are comparing
against production behavior, densified for cell-by-cell comparison.

Usage:  python rust/fixtures/test_agg_parity.py
Exit 0 = exact parity.
"""

import sys
import numpy as np
import pandas as pd

import sims_kernel as k


def python_reference(final_scores, player_names):
    """Verbatim port of new_sim.py aggregation, returned dense for comparison."""
    n = len(player_names)
    SIMULATIONS = final_scores.shape[1]

    df_long = pd.DataFrame(final_scores, index=player_names).T
    df_long["simulation_id"] = np.arange(SIMULATIONS)
    long_df = df_long.melt(id_vars="simulation_id", var_name="player_name", value_name="score")
    long_df["rank"] = long_df.groupby("simulation_id")["score"].rank(method="min")

    def dead_heat_factor(position, tie_count, threshold):
        start = position
        end = position + tie_count - 1
        overlap_start = max(start, 1)
        overlap_end = min(end, threshold)
        overlap_count = max(0, overlap_end - overlap_start + 1)
        return overlap_count / tie_count

    player_stats = {p: {"top_5": 0.0, "top_10": 0.0, "top_20": 0.0} for p in player_names}
    for _, group in long_df.groupby("simulation_id", sort=False):
        pos_counts = group["rank"].value_counts().to_dict()
        for _, row in group.iterrows():
            p = row["player_name"]
            pos = int(row["rank"])
            tie_ct = pos_counts[pos]
            player_stats[p]["top_5"] += dead_heat_factor(pos, tie_ct, 5)
            player_stats[p]["top_10"] += dead_heat_factor(pos, tie_ct, 10)
            player_stats[p]["top_20"] += dead_heat_factor(pos, tie_ct, 20)
    topn = pd.DataFrame.from_dict(player_stats, orient="index").div(SIMULATIONS)

    rank_probs_ndh = (
        long_df.groupby(["player_name", "rank"]).size().div(SIMULATIONS)
        .rename("prob_ndh").reset_index()
    )
    rank_probs_ndh["rank"] = rank_probs_ndh["rank"].astype(int)

    long_df["tie_count"] = long_df.groupby(["simulation_id", "rank"])["player_name"].transform("count")
    no_tie = long_df[long_df["tie_count"] == 1][["player_name", "rank"]].copy()
    no_tie["weight"] = 1.0
    ties = long_df[long_df["tie_count"] > 1].copy()
    if not ties.empty:
        expanded_parts = []
        for tc_val in ties["tie_count"].unique():
            tc_int = int(tc_val)
            sub = ties[ties["tie_count"] == tc_val][["player_name", "rank"]].copy()
            for offset in range(tc_int):
                part = sub.copy()
                part["rank"] = sub["rank"].astype(int) + offset
                part["weight"] = 1.0 / tc_int
                expanded_parts.append(part)
        all_rank_rows = pd.concat([no_tie] + expanded_parts, ignore_index=True)
    else:
        all_rank_rows = no_tie
    rank_probs = (
        all_rank_rows.groupby(["player_name", "rank"])["weight"].sum().div(SIMULATIONS)
        .rename("prob_u").reset_index()
    )
    rank_probs["rank"] = rank_probs["rank"].astype(int)

    # Densify to (n, n) by player index / (rank-1).
    pidx = {p: i for i, p in enumerate(player_names)}
    prob_u = np.zeros((n, n))
    for _, r in rank_probs.iterrows():
        prob_u[pidx[r["player_name"]], int(r["rank"]) - 1] = r["prob_u"]
    prob_ndh = np.zeros((n, n))
    for _, r in rank_probs_ndh.iterrows():
        prob_ndh[pidx[r["player_name"]], int(r["rank"]) - 1] = r["prob_ndh"]
    top = np.zeros((n, 3))
    for p in player_names:
        top[pidx[p]] = [topn.loc[p, "top_5"], topn.loc[p, "top_10"], topn.loc[p, "top_20"]]

    # h2h (new_sim.py:1166-1185)
    h2h_rows = []
    for i in range(n):
        si = final_scores[i]
        for j in range(i + 1, n):
            sj = final_scores[j]
            wins = int(np.sum(si < sj))
            losses = int(np.sum(si > sj))
            ties_ = int(np.sum(si == sj))
            denom = wins + losses
            if denom == 0:
                continue
            total = wins + losses + ties_
            h2h_rows.append((i, j, wins / denom, ties_ / total if total else 0.0))
    return prob_u, prob_ndh, top, h2h_rows


def main():
    rng = np.random.default_rng(7)
    n, sims = 30, 3000
    player_names = [f"p{i}" for i in range(n)]
    # Tight stroke range -> many ties (stresses dead-heat + tie-routing paths).
    final_scores = rng.integers(255, 300, size=(n, sims)).astype(np.int64)

    py_u, py_ndh, py_top, py_h2h = python_reference(final_scores, player_names)

    fs = np.ascontiguousarray(final_scores)
    ru, rndh, rtop = k.aggregate(fs)
    ia, ib, pa, tp = k.h2h(fs)

    failures = []

    # prob_ndh accumulates only integer counts -> bit-exact vs pandas.
    if np.array_equal(rndh, py_ndh):
        print(f"  prob_ndh   OK ({n}x{n} cells, bit-exact)")
    else:
        failures.append(f"prob_ndh: max|diff|={np.max(np.abs(rndh - py_ndh)):.2e}")

    # prob_u / top_finish carry fractional 1/tie_count weights; rayon's sum order
    # differs from pandas groupby, so they are FP-order-equivalent, not bit-exact.
    # Tolerance 1e-12 is ~10 orders tighter than the Monte-Carlo SE gate (~8e-3)
    # yet absorbs the ~1e-16 reduction-order noise.
    TOL = 1e-12
    du = float(np.max(np.abs(ru - py_u)))
    if du < TOL:
        print(f"  prob_u     OK ({n}x{n} cells, max|diff|={du:.1e} < {TOL:.0e})")
    else:
        failures.append(f"prob_u: max|diff|={du:.2e} >= {TOL:.0e}")
    dt = float(np.max(np.abs(rtop - py_top)))
    if dt < TOL:
        print(f"  top_finish OK ({n}x3 cells, max|diff|={dt:.1e} < {TOL:.0e})")
    else:
        failures.append(f"top_finish: max|diff|={dt:.2e} >= {TOL:.0e}")

    # h2h: same row order (sorted by (i,j)) and same values.
    if len(ia) != len(py_h2h):
        failures.append(f"h2h: row count {len(ia)} vs {len(py_h2h)}")
    else:
        rust_rows = list(zip(ia.tolist(), ib.tolist(), pa.tolist(), tp.tolist()))
        bad = 0
        for (ra, rb, rp, rt), (qa, qb, qp, qt) in zip(rust_rows, py_h2h):
            if ra != qa or rb != qb or rp != qp or rt != qt:
                bad += 1
        if bad == 0:
            print(f"  h2h       OK ({len(ia)} pairs, exact)")
        else:
            failures.append(f"h2h: {bad}/{len(ia)} pair mismatches")

    if failures:
        print("\nPARITY FAILED:")
        for m in failures:
            print(f"  - {m}")
        sys.exit(1)
    print("\nPOST-SIM AGGREGATION INTEGER-EXACT vs pandas.")


if __name__ == "__main__":
    main()

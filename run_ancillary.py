#!/usr/bin/env python
"""Re-price ancillary Kalshi markets from the saved live sim arrays — no re-sim.

Loads the standings / final_scores / made_cut / player_names that round_sim wrote
for the current event and runs kalshi_ancillary.price_ancillary_markets against
live Kalshi. Handy for iterating on fairs/edges (the slow cascade already ran).

    python run_ancillary.py [tourney] [edge_threshold]
"""
import json
import sys

import numpy as np

from sim_inputs import name_replacements
import kalshi_ancillary as ka


def main():
    tourney = sys.argv[1] if len(sys.argv) > 1 else "rbc_canada"
    edge = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    import glob
    import pandas as pd

    with open(f"player_names_live_{tourney}.json") as f:
        player_names = json.load(f)

    # Single-round score sim (for 3-ball / H2H) from the most recent sim cache.
    sim_dict = None
    caches = sorted(glob.glob(f"{tourney}/sim_cache_r*.parquet"))
    if caches:
        cache = pd.read_parquet(caches[-1])
        sim_dict = {p: cache.loc[p].values for p in cache.index}
        print(f"Loaded sim_dict from {caches[-1]} ({len(sim_dict)} players)")

    sim_data = {
        "player_names": player_names,
        "standings_r2": np.load(f"standings_r2_live_{tourney}.npy"),
        "standings_r3": np.load(f"standings_r3_live_{tourney}.npy"),
        "made_cut": np.load(f"made_cut_live_{tourney}.npy"),
        "final_scores": np.load(f"final_scores_live_{tourney}.npy"),
        "sim_dict": sim_dict,
        "pred_lookup": {},
    }

    # Fair sanity: dead-heat leader probs must sum to ~1.0 across the field.
    f_r2 = ka.leader_topn_fairs(sim_data["standings_r2"], player_names)
    f_r3 = ka.leader_topn_fairs(sim_data["standings_r3"], player_names,
                                made_cut=sim_data["made_cut"])
    print(f"R2 leader fair sum = {f_r2['leader'].sum():.4f} (should be ~1.0)  "
          f"top10 sum = {f_r2['top10'].sum():.2f}")
    print(f"R3 leader fair sum = {f_r3['leader'].sum():.4f} (should be ~1.0)  "
          f"top10 sum = {f_r3['top10'].sum():.2f}")
    print(f"playoff P = {ka.playoff_prob(sim_data['final_scores']):.4f}\n")

    df = ka.price_ancillary_markets(sim_data, tourney, name_replacements, edge_threshold=edge)
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    if df.empty:
        print("\nNo fillable ancillary edges.")
    else:
        print(f"\n{len(df)} fillable ancillary edges:")
        print(df[["market_type", "player_name", "side", "pricing", "fair_prob",
                  "cost", "edge_pp", "fill", "yes_bid", "yes_ask"]].to_string(index=False))
        out = f"kalshi_ancillary_edges_{tourney}.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

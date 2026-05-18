"""
sg_category_variance_test.py — Empirical test of per-category SG variance decomposition.

Tests whether per-category (OTT/APP/ARG/PUTT) variance multipliers are:
  1. Meaningful: Do they vary across courses?
  2. Stable: Are they consistent year-to-year at the same course?
  3. Decomposable: Does sum(category variances) ≈ total variance?
  4. Independent: Are cross-category correlations low?

Uses all PGA events with 5+ years of data in the last 5 years.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

DG_HISTORICAL_DB = os.path.join(os.path.expanduser("~"), "OneDrive", "dg_historical.db")

CATEGORIES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
MIN_TRAILING = 50
CURRENT_YEAR = datetime.now().year
YEAR_RANGE = 5  # last 5 full years
MIN_YEAR = CURRENT_YEAR - YEAR_RANGE  # 2021+
REQUIRED_YEARS = 5


def find_qualifying_events(conn):
    """Find PGA event_ids that appear in at least REQUIRED_YEARS distinct years."""
    query = f"""
        SELECT event_id, COUNT(DISTINCT year) AS n_years,
               MIN(year) AS first_yr, MAX(year) AS last_yr,
               MIN(event_name) AS event_name
        FROM player_rounds
        WHERE tour = 'pga' AND year >= {MIN_YEAR}
          AND sg_total_adj IS NOT NULL
        GROUP BY event_id
        HAVING COUNT(DISTINCT year) >= {REQUIRED_YEARS}
        ORDER BY event_name
    """
    df = pd.read_sql_query(query, conn)
    return df


def load_player_year_combos(conn, event_ids):
    """Get all (player, year, event_start) combos for qualifying events."""
    ph = ",".join("?" * len(event_ids))
    query = f"""
        SELECT event_id, player_name, year, MIN(round_date) AS event_start
        FROM player_rounds
        WHERE event_id IN ({ph}) AND year >= {MIN_YEAR} AND tour = 'pga'
        GROUP BY event_id, player_name, year
    """
    df = pd.read_sql_query(query, conn, params=event_ids)
    df["event_start"] = pd.to_datetime(df["event_start"])
    return df


def load_all_player_rounds(conn, player_names):
    """Load all rounds for given players (all tours) with SG categories."""
    all_dfs = []
    batch_size = 500
    cols = ", ".join(["player_name", "round_date", "sg_total_adj"] + CATEGORIES)
    for i in range(0, len(player_names), batch_size):
        batch = player_names[i:i + batch_size]
        ph = ",".join("?" * len(batch))
        query = f"""
            SELECT {cols}
            FROM player_rounds
            WHERE player_name IN ({ph})
              AND sg_total_adj IS NOT NULL
            ORDER BY player_name, round_date
        """
        df = pd.read_sql_query(query, conn, params=batch)
        all_dfs.append(df)
    result = pd.concat(all_dfs, ignore_index=True)
    result["round_date"] = pd.to_datetime(result["round_date"])
    return result


def load_event_sg(conn, event_ids):
    """Load SG data at qualifying events."""
    ph = ",".join("?" * len(event_ids))
    cols = ", ".join(["event_id", "player_name", "year", "sg_total_adj"] + CATEGORIES)
    query = f"""
        SELECT {cols}
        FROM player_rounds
        WHERE event_id IN ({ph}) AND year >= {MIN_YEAR}
          AND sg_total_adj IS NOT NULL
    """
    return pd.read_sql_query(query, conn, params=event_ids)


def compute_trailing_stds(player_years, all_rounds):
    """
    For each (event_id, player, year), compute trailing 50-round std dev
    for sg_total_adj and each SG category.

    Returns dict: (event_id, player, year) -> {sg_total_adj: std, sg_ott: std, ...}
    """
    results = {}
    cols = ["sg_total_adj"] + CATEGORIES

    # Group all_rounds by player for fast lookup
    grouped = {name: group.sort_values("round_date") for name, group in all_rounds.groupby("player_name")}

    for _, row in player_years.iterrows():
        eid = row["event_id"]
        pname = row["player_name"]
        yr = row["year"]
        estart = row["event_start"]

        if pname not in grouped:
            continue

        player_data = grouped[pname]
        prior = player_data[player_data["round_date"] < estart]

        if len(prior) < MIN_TRAILING:
            continue

        last_n = prior.tail(MIN_TRAILING)

        # Need non-null category data for at least 30 of 50 rounds
        cat_valid = last_n[CATEGORIES].notna().all(axis=1).sum()
        if cat_valid < 30:
            continue

        stds = {}
        for col in cols:
            valid = last_n[col].dropna()
            if len(valid) >= 30:
                stds[col] = float(valid.std())
            else:
                stds[col] = np.nan
        results[(eid, pname, yr)] = stds

    return results


def run_analysis():
    conn = sqlite3.connect(DG_HISTORICAL_DB)

    # Step 1: Find qualifying events
    print("=" * 80)
    print("SG CATEGORY VARIANCE DECOMPOSITION — EMPIRICAL TEST")
    print(f"PGA events with {REQUIRED_YEARS}+ years, {MIN_YEAR}-{CURRENT_YEAR}")
    print("=" * 80)

    events_df = find_qualifying_events(conn)
    event_ids = events_df["event_id"].tolist()
    print(f"\n{len(events_df)} qualifying events:")
    for _, e in events_df.iterrows():
        print(f"  [{e['event_id']:>3}] {e['event_name']} ({e['first_yr']}-{e['last_yr']}, {e['n_years']}yr)")

    # Step 2: Load player-year combos
    print(f"\nLoading player-year combos...")
    player_years = load_player_year_combos(conn, event_ids)
    print(f"  {len(player_years)} player-year-event combos")

    # Step 3: Load all rounds
    unique_players = player_years["player_name"].unique().tolist()
    print(f"\nLoading rounds for {len(unique_players)} unique players...")
    all_rounds = load_all_player_rounds(conn, unique_players)
    print(f"  {len(all_rounds)} total rounds loaded")

    # Step 4: Load event SG data
    print(f"\nLoading event SG data...")
    event_sg = load_event_sg(conn, event_ids)
    print(f"  {len(event_sg)} event round records")

    conn.close()

    # Step 5: Compute trailing stds
    print(f"\nComputing trailing {MIN_TRAILING}-round std devs (this takes a minute)...")
    trailing = compute_trailing_stds(player_years, all_rounds)
    print(f"  {len(trailing)} qualified player-event-years")

    # Step 6: Per event-year analysis
    cols_all = ["sg_total_adj"] + CATEGORIES
    results = []

    for eid in event_ids:
        ename = events_df[events_df["event_id"] == eid]["event_name"].iloc[0]
        years = sorted(player_years[player_years["event_id"] == eid]["year"].unique())

        for yr in years:
            # Qualified players this event-year
            qual_keys = [(e, p, y) for (e, p, y) in trailing.keys() if e == eid and y == yr]
            qual_players = [k[1] for k in qual_keys]
            n_qual = len(qual_players)

            if n_qual < 20:
                continue

            # Actual std devs at this event (qualified players only)
            yr_event = event_sg[
                (event_sg["event_id"] == eid) &
                (event_sg["year"] == yr) &
                (event_sg["player_name"].isin(qual_players))
            ]
            if len(yr_event) < 20:
                continue

            row = {"event_id": eid, "event_name": ename, "year": yr, "n_qual": n_qual}

            for col in cols_all:
                valid_event = yr_event[col].dropna()
                if len(valid_event) < 20:
                    row[f"actual_{col}"] = np.nan
                    row[f"field_exp_{col}"] = np.nan
                    row[f"mult_{col}"] = np.nan
                    continue

                actual_std = float(valid_event.std())

                # Field expected: mean of qualified players' trailing stds for this col
                trailing_vals = [trailing[k].get(col, np.nan) for k in qual_keys]
                trailing_vals = [v for v in trailing_vals if not np.isnan(v)]
                field_exp = float(np.mean(trailing_vals)) if trailing_vals else np.nan

                row[f"actual_{col}"] = round(actual_std, 4)
                row[f"field_exp_{col}"] = round(field_exp, 4)
                if field_exp and field_exp > 0:
                    row[f"mult_{col}"] = round(actual_std / field_exp, 4)
                else:
                    row[f"mult_{col}"] = np.nan

            # Variance decomposition check:
            # sum of category actual variances vs total actual variance
            cat_var_sum = 0
            for cat in CATEGORIES:
                a = row.get(f"actual_{cat}", np.nan)
                if not np.isnan(a):
                    cat_var_sum += a ** 2
            total_var = row.get(f"actual_sg_total_adj", np.nan)
            if not np.isnan(total_var):
                total_var = total_var ** 2
                row["cat_var_sum"] = round(cat_var_sum, 4)
                row["total_var"] = round(total_var, 4)
                row["decomp_ratio"] = round(cat_var_sum / total_var, 4) if total_var > 0 else np.nan
            else:
                row["cat_var_sum"] = np.nan
                row["total_var"] = np.nan
                row["decomp_ratio"] = np.nan

            results.append(row)

    df = pd.DataFrame(results)
    print(f"\n  {len(df)} event-year observations across {df['event_id'].nunique()} events")

    # ======================================================================
    # ANALYSIS 1: Variance Decomposition — does sum(cat vars) ≈ total var?
    # ======================================================================
    print("\n" + "=" * 80)
    print("TEST 1: VARIANCE DECOMPOSITION")
    print("Does sum(OTT^2 + APP^2 + ARG^2 + PUTT^2) ~= Total^2?")
    print("Ratio = 1.0 means perfect decomposition, >1 means positive covariance")
    print("=" * 80)

    valid_decomp = df["decomp_ratio"].dropna()
    print(f"\n  N obs:   {len(valid_decomp)}")
    print(f"  Mean:    {valid_decomp.mean():.4f}")
    print(f"  Median:  {valid_decomp.median():.4f}")
    print(f"  Std:     {valid_decomp.std():.4f}")
    print(f"  Min:     {valid_decomp.min():.4f}")
    print(f"  Max:     {valid_decomp.max():.4f}")
    print(f"  P25:     {valid_decomp.quantile(0.25):.4f}")
    print(f"  P75:     {valid_decomp.quantile(0.75):.4f}")

    # Per-event average decomp ratio
    event_decomp = df.groupby(["event_id", "event_name"])["decomp_ratio"].mean().sort_values()
    print(f"\n  Per-event avg decomp ratio (bottom/top 5):")
    for (eid, ename), val in event_decomp.head(5).items():
        print(f"    {val:.3f}  {ename}")
    print(f"    ...")
    for (eid, ename), val in event_decomp.tail(5).items():
        print(f"    {val:.3f}  {ename}")

    # ======================================================================
    # ANALYSIS 2: Do category multipliers vary across courses?
    # ======================================================================
    print("\n" + "=" * 80)
    print("TEST 2: CROSS-COURSE VARIATION IN CATEGORY MULTIPLIERS")
    print("Higher std = more differentiation across courses")
    print("=" * 80)

    # Average multiplier per event first, then look at cross-event variation
    mult_cols = [f"mult_{cat}" for cat in CATEGORIES] + ["mult_sg_total_adj"]
    event_avg_mults = df.groupby(["event_id", "event_name"])[mult_cols].mean()

    print(f"\n  {'Category':>12} | {'Mean Mult':>9} | {'Std Mult':>8} | {'Min':>6} | {'Max':>6} | {'Range':>6}")
    print(f"  {'-'*12}-+-{'-'*9}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for cat in CATEGORIES + ["sg_total_adj"]:
        col = f"mult_{cat}"
        vals = event_avg_mults[col].dropna()
        label = cat.replace("sg_", "").upper()
        print(f"  {label:>12} | {vals.mean():>9.4f} | {vals.std():>8.4f} | "
              f"{vals.min():>6.3f} | {vals.max():>6.3f} | {vals.max()-vals.min():>6.3f}")

    # ======================================================================
    # ANALYSIS 3: Year-to-year stability within same event
    # ======================================================================
    print("\n" + "=" * 80)
    print("TEST 3: YEAR-TO-YEAR STABILITY (within-event std of multipliers)")
    print("Lower = more stable/reliable course fingerprint")
    print("=" * 80)

    within_event_std = df.groupby(["event_id", "event_name"])[mult_cols].std()

    print(f"\n  {'Category':>12} | {'Mean Within-Evt Std':>19} | {'Median':>8}")
    print(f"  {'-'*12}-+-{'-'*19}-+-{'-'*8}")
    for cat in CATEGORIES + ["sg_total_adj"]:
        col = f"mult_{cat}"
        vals = within_event_std[col].dropna()
        label = cat.replace("sg_", "").upper()
        print(f"  {label:>12} | {vals.mean():>19.4f} | {vals.median():>8.4f}")

    # Signal-to-noise: cross-event std / within-event std
    print(f"\n  Signal-to-noise (cross-event variation / within-event variation):")
    print(f"  {'Category':>12} | {'Cross-Evt Std':>13} | {'Within-Evt Std':>14} | {'S/N Ratio':>9}")
    print(f"  {'-'*12}-+-{'-'*13}-+-{'-'*14}-+-{'-'*9}")
    for cat in CATEGORIES + ["sg_total_adj"]:
        col = f"mult_{cat}"
        cross = event_avg_mults[col].dropna().std()
        within = within_event_std[col].dropna().mean()
        sn = cross / within if within > 0 else np.nan
        label = cat.replace("sg_", "").upper()
        print(f"  {label:>12} | {cross:>13.4f} | {within:>14.4f} | {sn:>9.3f}")

    # ======================================================================
    # ANALYSIS 4: Cross-category correlations (at event level)
    # ======================================================================
    print("\n" + "=" * 80)
    print("TEST 4: CROSS-CATEGORY CORRELATION OF MULTIPLIERS (event-level)")
    print("High correlation = categories move together (less independent)")
    print("=" * 80)

    cat_mult_cols = [f"mult_{cat}" for cat in CATEGORIES]
    # Use event-year level data
    corr = df[cat_mult_cols].corr()
    labels = [c.replace("sg_", "").upper() for c in CATEGORIES]

    print(f"\n  {'':>8}", end="")
    for l in labels:
        print(f" {l:>8}", end="")
    print()
    for i, l1 in enumerate(labels):
        print(f"  {l1:>8}", end="")
        for j, l2 in enumerate(labels):
            val = corr.iloc[i, j]
            print(f" {val:>8.3f}", end="")
        print()

    # ======================================================================
    # ANALYSIS 5: Top/bottom events per category
    # ======================================================================
    print("\n" + "=" * 80)
    print("TEST 5: MOST/LEAST AMPLIFYING EVENTS PER CATEGORY")
    print("=" * 80)

    for cat in CATEGORIES:
        col = f"mult_{cat}"
        label = cat.replace("sg_", "").upper()
        sorted_evts = event_avg_mults[col].dropna().sort_values()

        print(f"\n  {label}:")
        print(f"    Lowest (compresses):")
        for (eid, ename), val in sorted_evts.head(5).items():
            print(f"      {val:.3f}  {ename}")
        print(f"    Highest (amplifies):")
        for (eid, ename), val in sorted_evts.tail(5).items():
            print(f"      {val:.3f}  {ename}")

    # ======================================================================
    # ANALYSIS 6: Do events have distinct "profiles"?
    # ======================================================================
    print("\n" + "=" * 80)
    print("TEST 6: EVENT VARIANCE PROFILES (event-avg multipliers)")
    print("Events where one category stands out from the rest")
    print("=" * 80)

    cat_labels = ["OTT", "APP", "ARG", "PUTT"]
    cat_mult_only = [f"mult_{cat}" for cat in CATEGORIES]
    profiles = event_avg_mults[cat_mult_only].copy()
    profiles.columns = cat_labels

    # For each event, compute how much each category deviates from that event's mean multiplier
    profiles["event_mean"] = profiles[cat_labels].mean(axis=1)
    for cat in cat_labels:
        profiles[f"{cat}_dev"] = profiles[cat] - profiles["event_mean"]

    # Find events with the largest spread (std across categories)
    profiles["cat_spread"] = profiles[cat_labels].std(axis=1)

    print(f"\n  {'Event':<40} | {'OTT':>6} | {'APP':>6} | {'ARG':>6} | {'PUTT':>6} | {'Spread':>6}")
    print(f"  {'-'*40}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for (eid, ename), row in profiles.sort_values("cat_spread", ascending=False).head(15).iterrows():
        name_trunc = ename[:40]
        print(f"  {name_trunc:<40} | {row['OTT']:>6.3f} | {row['APP']:>6.3f} | "
              f"{row['ARG']:>6.3f} | {row['PUTT']:>6.3f} | {row['cat_spread']:>6.3f}")

    print(f"\n  ... Bottom 5 (most uniform across categories):")
    for (eid, ename), row in profiles.sort_values("cat_spread").head(5).iterrows():
        name_trunc = ename[:40]
        print(f"  {name_trunc:<40} | {row['OTT']:>6.3f} | {row['APP']:>6.3f} | "
              f"{row['ARG']:>6.3f} | {row['PUTT']:>6.3f} | {row['cat_spread']:>6.3f}")

    # Save full results for further analysis
    output_path = os.path.join(os.path.dirname(__file__), "sg_category_variance_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Full results saved to {output_path}")

    profiles_out = os.path.join(os.path.dirname(__file__), "sg_category_event_profiles.csv")
    profiles.reset_index().to_csv(profiles_out, index=False)
    print(f"  Event profiles saved to {profiles_out}")


if __name__ == "__main__":
    run_analysis()

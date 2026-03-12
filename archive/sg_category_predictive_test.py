"""
sg_category_predictive_test.py — Does per-category variance predict player-level outcomes?

Core test: for each player-round, compare two predictions of that player's
expected deviation from their mean:
  1. Simple: trailing 50-round sg_total_adj std dev
  2. Category-weighted: sqrt(sum(cat_std_i^2 * course_mult_i^2))

Course multipliers computed leave-one-year-out to avoid data leakage.

If category-weighted is a better predictor, the theory has player-level signal.
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
MIN_YEAR = CURRENT_YEAR - 5  # 2021+
REQUIRED_YEARS = 5


def find_qualifying_events(conn):
    query = f"""
        SELECT event_id, COUNT(DISTINCT year) AS n_years, MIN(event_name) AS event_name
        FROM player_rounds
        WHERE tour = 'pga' AND year >= {MIN_YEAR} AND sg_total_adj IS NOT NULL
        GROUP BY event_id
        HAVING COUNT(DISTINCT year) >= {REQUIRED_YEARS}
        ORDER BY event_name
    """
    return pd.read_sql_query(query, conn)


def load_all_data(conn, event_ids):
    """Load all player-round data needed in bulk."""
    ph = ",".join("?" * len(event_ids))
    cols = ", ".join(["event_id", "player_name", "year", "round_num",
                      "round_date", "sg_total_adj"] + CATEGORIES)

    # Event rounds
    event_query = f"""
        SELECT {cols}
        FROM player_rounds
        WHERE event_id IN ({ph}) AND year >= {MIN_YEAR} AND sg_total_adj IS NOT NULL
    """
    event_df = pd.read_sql_query(event_query, conn, params=event_ids)
    event_df["round_date"] = pd.to_datetime(event_df["round_date"])

    # Event start dates per player
    starts = event_df.groupby(["event_id", "player_name", "year"])["round_date"].min()
    starts = starts.reset_index().rename(columns={"round_date": "event_start"})

    # All player rounds (all tours) for trailing computation
    unique_players = event_df["player_name"].unique().tolist()
    all_rounds = []
    batch_size = 500
    trail_cols = ", ".join(["player_name", "round_date", "sg_total_adj"] + CATEGORIES)
    for i in range(0, len(unique_players), batch_size):
        batch = unique_players[i:i + batch_size]
        bph = ",".join("?" * len(batch))
        q = f"""
            SELECT {trail_cols}
            FROM player_rounds
            WHERE player_name IN ({bph}) AND sg_total_adj IS NOT NULL
            ORDER BY player_name, round_date
        """
        all_rounds.append(pd.read_sql_query(q, conn, params=batch))

    rounds_df = pd.concat(all_rounds, ignore_index=True)
    rounds_df["round_date"] = pd.to_datetime(rounds_df["round_date"])

    return event_df, starts, rounds_df


def compute_trailing_stats(player_years, all_rounds):
    """
    For each (event_id, player, year), compute trailing 50-round:
      - mean and std of sg_total_adj
      - std of each SG category

    Returns dict: (eid, player, year) -> {total_mean, total_std, sg_ott_std, ...}
    """
    results = {}
    grouped = {}
    for name, group in all_rounds.groupby("player_name"):
        grouped[name] = group.sort_values("round_date")

    for _, row in player_years.iterrows():
        eid = row["event_id"]
        pname = row["player_name"]
        yr = row["year"]
        estart = row["event_start"]

        if pname not in grouped:
            continue

        prior = grouped[pname]
        prior = prior[prior["round_date"] < estart]

        if len(prior) < MIN_TRAILING:
            continue

        last_n = prior.tail(MIN_TRAILING)

        # Need category data for at least 30 of 50
        cat_valid = last_n[CATEGORIES].notna().all(axis=1).sum()
        if cat_valid < 30:
            continue

        stats = {
            "total_mean": float(last_n["sg_total_adj"].mean()),
            "total_std": float(last_n["sg_total_adj"].std()),
        }
        for cat in CATEGORIES:
            valid = last_n[cat].dropna()
            if len(valid) >= 30:
                stats[f"{cat}_std"] = float(valid.std())
            else:
                stats[f"{cat}_std"] = np.nan

        # Skip if any category is missing
        if any(np.isnan(stats.get(f"{cat}_std", np.nan)) for cat in CATEGORIES):
            continue

        results[(eid, pname, yr)] = stats

    return results


def compute_course_multipliers_loo(event_df, trailing, event_ids):
    """
    Compute per-category course multipliers using leave-one-year-out.

    For each (event_id, year), use all OTHER years at that event to compute
    the category multipliers.

    Returns dict: (event_id, hold_out_year) -> {sg_ott_mult, sg_app_mult, ...}
    """
    results = {}
    cols_all = ["sg_total_adj"] + CATEGORIES

    for eid in event_ids:
        eid_keys = [(e, p, y) for (e, p, y) in trailing.keys() if e == eid]
        years_at_event = sorted(set(y for _, _, y in eid_keys))

        for hold_year in years_at_event:
            # Use all other years to compute multipliers
            train_keys = [(e, p, y) for (e, p, y) in eid_keys if y != hold_year]

            if len(train_keys) < 20:
                continue

            # Get train-year event data (qualified players only)
            train_players_by_year = {}
            for (e, p, y) in train_keys:
                if y not in train_players_by_year:
                    train_players_by_year[y] = []
                train_players_by_year[y].append(p)

            mults = {}
            for col in cols_all:
                # Compute actual std from event rounds (train years, qualified players)
                actual_stds = []
                field_exp_stds = []

                for yr, players in train_players_by_year.items():
                    yr_event = event_df[
                        (event_df["event_id"] == eid) &
                        (event_df["year"] == yr) &
                        (event_df["player_name"].isin(players))
                    ]
                    valid = yr_event[col].dropna()
                    if len(valid) < 20:
                        continue

                    actual_stds.append(float(valid.std()))

                    yr_keys = [(e, p, y) for (e, p, y) in train_keys if y == yr]
                    trailing_vals = [trailing[k].get(f"{col}_std" if col != "sg_total_adj" else "total_std")
                                     for k in yr_keys]
                    trailing_vals = [v for v in trailing_vals if v is not None and not np.isnan(v)]
                    if trailing_vals:
                        field_exp_stds.append(float(np.mean(trailing_vals)))

                if actual_stds and field_exp_stds:
                    avg_actual = np.mean(actual_stds)
                    avg_field_exp = np.mean(field_exp_stds)
                    if avg_field_exp > 0:
                        mults[col] = avg_actual / avg_field_exp
                    else:
                        mults[col] = 1.0
                else:
                    mults[col] = 1.0

            results[(eid, hold_year)] = mults

    return results


def build_test_dataset(event_df, trailing, course_mults):
    """
    Build the test dataset: one row per player-round with both predictions.

    For each player-round:
      - actual_dev: |sg_total_adj - trailing_mean| (absolute deviation from expected)
      - simple_pred: trailing total std
      - cat_pred: sqrt(sum(cat_std^2 * course_mult^2))
    """
    rows = []

    for (eid, pname, yr), stats in trailing.items():
        # Get course multipliers for this event-year (leave-one-out)
        if (eid, yr) not in course_mults:
            continue
        mults = course_mults[(eid, yr)]

        # Simple prediction
        simple_pred = stats["total_std"]

        # Category-weighted prediction
        cat_var = 0
        for cat in CATEGORIES:
            cat_std = stats[f"{cat}_std"]
            cat_mult = mults.get(cat, 1.0)
            cat_var += (cat_std * cat_mult) ** 2
        cat_pred = np.sqrt(cat_var)

        # Get actual rounds at this event
        player_rounds = event_df[
            (event_df["event_id"] == eid) &
            (event_df["year"] == yr) &
            (event_df["player_name"] == pname)
        ]

        trailing_mean = stats["total_mean"]

        for _, rnd in player_rounds.iterrows():
            sg = rnd["sg_total_adj"]
            if pd.isna(sg):
                continue

            actual_dev = abs(sg - trailing_mean)

            rows.append({
                "event_id": eid,
                "player_name": pname,
                "year": yr,
                "round_num": rnd["round_num"],
                "sg_total_adj": sg,
                "trailing_mean": trailing_mean,
                "actual_dev": actual_dev,
                "actual_dev_sq": actual_dev ** 2,
                "simple_pred": simple_pred,
                "cat_pred": cat_pred,
                "pred_delta": cat_pred - simple_pred,  # positive = cat model thinks MORE volatile
            })

    return pd.DataFrame(rows)


def ols_r_squared(x, y):
    """Simple OLS R-squared."""
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_yy = np.sum((y - y_mean) ** 2)
    if ss_xx == 0 or ss_yy == 0:
        return np.nan, np.nan, np.nan
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_sq = (ss_xy ** 2) / (ss_xx * ss_yy)
    return r_sq, slope, intercept


def run_test():
    conn = sqlite3.connect(DG_HISTORICAL_DB)

    print("=" * 80)
    print("PLAYER-LEVEL PREDICTIVE TEST: Category-Weighted vs Simple Variance")
    print("Leave-one-year-out course multipliers")
    print("=" * 80)

    # Step 1: Find events
    events_df = find_qualifying_events(conn)
    event_ids = events_df["event_id"].tolist()
    print(f"\n{len(events_df)} qualifying events")

    # Step 2: Load data
    print("Loading data...")
    event_df, starts, rounds_df = load_all_data(conn, event_ids)
    conn.close()
    print(f"  {len(event_df)} event rounds, {len(rounds_df)} total rounds")

    # Step 3: Trailing stats
    print("Computing trailing stats...")
    trailing = compute_trailing_stats(starts, rounds_df)
    print(f"  {len(trailing)} qualified player-event-years")

    # Step 4: Leave-one-out course multipliers
    print("Computing LOO course multipliers...")
    course_mults = compute_course_multipliers_loo(event_df, trailing, event_ids)
    print(f"  {len(course_mults)} event-year multiplier sets")

    # Step 5: Build test dataset
    print("Building test dataset...")
    test_df = build_test_dataset(event_df, trailing, course_mults)
    print(f"  {len(test_df)} player-rounds in test set")
    print(f"  {test_df['event_id'].nunique()} events, {test_df['player_name'].nunique()} players")

    # ==================================================================
    # TEST A: Regression — which prediction better explains actual deviation?
    # ==================================================================
    print("\n" + "=" * 80)
    print("TEST A: REGRESSION (actual |deviation| ~ prediction)")
    print("Higher R-sq = better predictor of realized variance")
    print("=" * 80)

    x_simple = test_df["simple_pred"].values
    x_cat = test_df["cat_pred"].values
    y_abs = test_df["actual_dev"].values
    y_sq = test_df["actual_dev_sq"].values

    # Absolute deviation
    r2_simple_abs, slope_s, _ = ols_r_squared(x_simple, y_abs)
    r2_cat_abs, slope_c, _ = ols_r_squared(x_cat, y_abs)

    print(f"\n  Predicting |deviation| (N={len(y_abs)}):")
    print(f"    Simple std:        R-sq = {r2_simple_abs:.6f}  (slope = {slope_s:.4f})")
    print(f"    Category-weighted: R-sq = {r2_cat_abs:.6f}  (slope = {slope_c:.4f})")
    print(f"    Improvement:       {(r2_cat_abs - r2_simple_abs):.6f} ({(r2_cat_abs/r2_simple_abs - 1)*100:+.2f}%)")

    # Squared deviation
    r2_simple_sq, slope_s2, _ = ols_r_squared(x_simple ** 2, y_sq)
    r2_cat_sq, slope_c2, _ = ols_r_squared(x_cat ** 2, y_sq)

    print(f"\n  Predicting deviation^2 (N={len(y_sq)}):")
    print(f"    Simple var:        R-sq = {r2_simple_sq:.6f}  (slope = {slope_s2:.4f})")
    print(f"    Category-weighted: R-sq = {r2_cat_sq:.6f}  (slope = {slope_c2:.4f})")
    print(f"    Improvement:       {(r2_cat_sq - r2_simple_sq):.6f} ({(r2_cat_sq/r2_simple_sq - 1)*100:+.2f}%)")

    # Incremental test: does cat_pred add value BEYOND simple_pred?
    # Multiple regression: y ~ simple + cat
    X = np.column_stack([x_simple, x_cat, np.ones(len(x_simple))])
    try:
        betas = np.linalg.lstsq(X, y_abs, rcond=None)[0]
        y_hat = X @ betas
        ss_res = np.sum((y_abs - y_hat) ** 2)
        ss_tot = np.sum((y_abs - np.mean(y_abs)) ** 2)
        r2_both = 1 - ss_res / ss_tot
        print(f"\n  Multiple regression |dev| ~ simple + cat:")
        print(f"    R-sq = {r2_both:.6f}  (simple coef = {betas[0]:.4f}, cat coef = {betas[1]:.4f})")
        print(f"    Incremental over simple: {(r2_both - r2_simple_abs):.6f}")
        print(f"    Incremental over cat:    {(r2_both - r2_cat_abs):.6f}")
    except Exception as e:
        print(f"    Multiple regression failed: {e}")

    # ==================================================================
    # TEST B: Quintile analysis — calibration check
    # ==================================================================
    print("\n" + "=" * 80)
    print("TEST B: QUINTILE ANALYSIS")
    print("Sort by each prediction, check if higher predicted = higher actual")
    print("=" * 80)

    for pred_col, label in [("simple_pred", "Simple Std"), ("cat_pred", "Category-Weighted")]:
        test_df["quintile"] = pd.qcut(test_df[pred_col], 5, labels=[1, 2, 3, 4, 5])
        quintile_stats = test_df.groupby("quintile").agg(
            pred_mean=(pred_col, "mean"),
            actual_dev_mean=("actual_dev", "mean"),
            actual_dev_sq_mean=("actual_dev_sq", "mean"),
            n=("actual_dev", "count"),
        )
        print(f"\n  {label}:")
        print(f"  {'Q':>3} | {'Pred Std':>8} | {'Actual |dev|':>12} | {'Actual dev^2':>12} | {'N':>6}")
        print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}")
        for q, row in quintile_stats.iterrows():
            print(f"  {q:>3} | {row['pred_mean']:>8.4f} | {row['actual_dev_mean']:>12.4f} | "
                  f"{row['actual_dev_sq_mean']:>12.4f} | {row['n']:>6.0f}")

        # Monotonicity: rank correlation between quintile and actual dev
        from scipy import stats as sp_stats
        rho, p_val = sp_stats.spearmanr(quintile_stats.index.astype(int), quintile_stats["actual_dev_mean"])
        print(f"  Rank corr (quintile vs actual): rho={rho:.4f}, p={p_val:.4f}")

    # ==================================================================
    # TEST C: Where the two predictions DISAGREE — who's right?
    # ==================================================================
    print("\n" + "=" * 80)
    print("TEST C: DISAGREEMENT ANALYSIS")
    print("Players where cat-weighted and simple predictions diverge most")
    print("Which prediction is closer to reality?")
    print("=" * 80)

    # Quintile by pred_delta (cat_pred - simple_pred)
    test_df["delta_quintile"] = pd.qcut(test_df["pred_delta"], 5, labels=[1, 2, 3, 4, 5])

    delta_stats = test_df.groupby("delta_quintile").agg(
        delta_mean=("pred_delta", "mean"),
        simple_mean=("simple_pred", "mean"),
        cat_mean=("cat_pred", "mean"),
        actual_dev_mean=("actual_dev", "mean"),
        n=("actual_dev", "count"),
    )

    print(f"\n  Quintiles by (cat_pred - simple_pred):")
    print(f"  Q1 = cat model predicts LESS volatile than simple")
    print(f"  Q5 = cat model predicts MORE volatile than simple")
    print(f"\n  {'Q':>3} | {'Delta':>7} | {'Simple':>7} | {'Cat':>7} | {'Actual |dev|':>12} | {'N':>5}")
    print(f"  {'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*12}-+-{'-'*5}")
    for q, row in delta_stats.iterrows():
        print(f"  {q:>3} | {row['delta_mean']:>+7.4f} | {row['simple_mean']:>7.4f} | "
              f"{row['cat_mean']:>7.4f} | {row['actual_dev_mean']:>12.4f} | {row['n']:>5.0f}")

    # Key comparison: in Q1 (cat says less volatile), is actual closer to cat or simple?
    # In Q5 (cat says more volatile), is actual closer to cat or simple?
    print(f"\n  Interpretation:")
    q1 = delta_stats.loc[1]
    q5 = delta_stats.loc[5]
    print(f"    Q1: Cat predicts {q1['cat_mean']:.3f}, Simple predicts {q1['simple_mean']:.3f}, "
          f"Actual = {q1['actual_dev_mean']:.3f}")
    print(f"    Q5: Cat predicts {q5['cat_mean']:.3f}, Simple predicts {q5['simple_mean']:.3f}, "
          f"Actual = {q5['actual_dev_mean']:.3f}")
    q1_to_q5_actual = q5["actual_dev_mean"] - q1["actual_dev_mean"]
    q1_to_q5_simple = q5["simple_mean"] - q1["simple_mean"]
    q1_to_q5_cat = q5["cat_mean"] - q1["cat_mean"]
    print(f"\n    Q5 - Q1 spread:")
    print(f"      Actual |dev|:  {q1_to_q5_actual:+.4f}")
    print(f"      Simple pred:   {q1_to_q5_simple:+.4f}")
    print(f"      Cat pred:      {q1_to_q5_cat:+.4f}")
    if abs(q1_to_q5_actual - q1_to_q5_cat) < abs(q1_to_q5_actual - q1_to_q5_simple):
        print(f"    --> Category model tracks actual spread better")
    else:
        print(f"    --> Simple model tracks actual spread better (or no difference)")

    # ==================================================================
    # TEST D: Per-event breakdown — is category model better at some events?
    # ==================================================================
    print("\n" + "=" * 80)
    print("TEST D: PER-EVENT R-SQUARED COMPARISON")
    print("Events where category model outperforms simple model")
    print("=" * 80)

    event_r2 = []
    for eid in test_df["event_id"].unique():
        edf = test_df[test_df["event_id"] == eid]
        if len(edf) < 30:
            continue
        ename = events_df[events_df["event_id"] == eid]["event_name"].iloc[0]

        r2_s, _, _ = ols_r_squared(edf["simple_pred"].values, edf["actual_dev"].values)
        r2_c, _, _ = ols_r_squared(edf["cat_pred"].values, edf["actual_dev"].values)

        event_r2.append({
            "event_id": eid,
            "event_name": ename,
            "r2_simple": r2_s,
            "r2_cat": r2_c,
            "r2_diff": r2_c - r2_s,
            "n_rounds": len(edf),
        })

    er2_df = pd.DataFrame(event_r2).sort_values("r2_diff", ascending=False)

    print(f"\n  {'Event':<40} | {'R2 Simple':>9} | {'R2 Cat':>7} | {'Diff':>7} | {'N':>5}")
    print(f"  {'-'*40}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")
    for _, row in er2_df.iterrows():
        marker = " <--" if abs(row["r2_diff"]) > 0.005 else ""
        print(f"  {row['event_name'][:40]:<40} | {row['r2_simple']:>9.5f} | {row['r2_cat']:>7.5f} | "
              f"{row['r2_diff']:>+7.5f} | {row['n_rounds']:>5.0f}{marker}")

    n_better = (er2_df["r2_diff"] > 0).sum()
    n_worse = (er2_df["r2_diff"] < 0).sum()
    print(f"\n  Cat model better at {n_better}/{len(er2_df)} events, worse at {n_worse}")
    print(f"  Mean R2 diff: {er2_df['r2_diff'].mean():+.6f}")
    print(f"  Median R2 diff: {er2_df['r2_diff'].median():+.6f}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "sg_category_predictive_results.csv")
    test_df.to_csv(output_path, index=False)
    print(f"\n  Full test data saved to {output_path}")


if __name__ == "__main__":
    run_test()

"""
Independent reproduction of compute_sg_category_variance_analysis.
Does NOT import scoring_baseline. Recomputes category_mults & category_skew
from the DB for event 32 / min_year 2019 / tour 'pga'.
"""
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

DG_HISTORICAL_DB = "C:/Users/mckin/OneDrive/dg_historical.db"

# ---- This-week inputs (mirror task spec) ----
event_id_list = [32]
min_year = 2019
tour_override = "pga"

CAT_COLS = ["sg_ott_adj", "sg_app_adj", "sg_arg_adj", "sg_putt_adj"]
CAT_NAMES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]

current_year = datetime.now().year  # 2026 per system clock; we override below if needed


def compute_recency_weights(years):
    """w(year) = 1 + 0.025*(year - min_year), normalized to sum 1."""
    if not years:
        return {}
    sorted_years = sorted(set(years))
    min_yr = sorted_years[0]
    raw = {y: 1.0 + 0.025 * (y - min_yr) for y in sorted_years}
    total = sum(raw.values())
    return {y: w / total for y, w in raw.items()}


def main():
    cy = current_year
    print(f"datetime.now().year = {cy}")
    conn = sqlite3.connect(DG_HISTORICAL_DB)

    # --- Step 1b: tour-wide baseline skew, last 5 yrs raw adjusted-SG ---
    baseline_skew_query = f"""
        SELECT {', '.join(CAT_COLS)}
        FROM player_rounds
        WHERE tour = 'pga' AND year >= ?
          AND sg_total_adj IS NOT NULL
    """
    baseline_raw = pd.read_sql_query(baseline_skew_query, conn, params=[cy - 5])
    baseline_skew = {}
    for c, name in zip(CAT_COLS, CAT_NAMES):
        vals = baseline_raw[c].dropna()
        baseline_skew[name] = round(float(sp_stats.skew(vals)), 2)
    print(f"baseline window: year >= {cy-5}; baseline rows={len(baseline_raw)}")
    print("baseline_skew:", baseline_skew)

    # --- Step 2: player-year combos at this course ---
    placeholders = ",".join("?" * len(event_id_list))
    player_year_query = f"""
        SELECT DISTINCT player_name, year, MIN(round_date) AS event_start
        FROM player_rounds
        WHERE event_id IN ({placeholders}) AND year >= ? AND tour = ?
        GROUP BY player_name, year
    """
    params = list(event_id_list) + [min_year, tour_override]
    player_years = pd.read_sql_query(player_year_query, conn, params=params)
    player_years["event_start"] = pd.to_datetime(player_years["event_start"])
    print(f"player-year combos at course: {len(player_years)}")
    if player_years.empty:
        conn.close()
        print("EMPTY")
        return

    # --- Step 3: all rounds for trailing std ---
    unique_players = player_years["player_name"].unique().tolist()
    cat_cols_str = ", ".join(CAT_COLS)
    all_rounds = []
    batch_size = 500
    for i in range(0, len(unique_players), batch_size):
        batch = unique_players[i:i + batch_size]
        ph = ",".join("?" * len(batch))
        batch_query = f"""
            SELECT player_name, round_date, {cat_cols_str}
            FROM player_rounds
            WHERE player_name IN ({ph}) AND sg_total_adj IS NOT NULL
            ORDER BY player_name, round_date
        """
        all_rounds.append(pd.read_sql_query(batch_query, conn, params=batch))
    rounds_df = pd.concat(all_rounds, ignore_index=True)
    rounds_df["round_date"] = pd.to_datetime(rounds_df["round_date"])

    # --- Step 4: event SG categories at this course ---
    event_sg_query = f"""
        SELECT player_name, year, {cat_cols_str}
        FROM player_rounds
        WHERE event_id IN ({placeholders}) AND year >= ? AND tour = ?
          AND sg_total_adj IS NOT NULL
    """
    event_sg = pd.read_sql_query(
        event_sg_query, conn, params=list(event_id_list) + [min_year, tour_override]
    )
    conn.close()

    # --- Step 5: trailing 50-round std per cat per player-year ---
    trailing_cat_stds = {}
    for _, py in player_years.iterrows():
        pname = py["player_name"]
        yr = py["year"]
        estart = py["event_start"]
        player_data = rounds_df[
            (rounds_df["player_name"] == pname) &
            (rounds_df["round_date"] < estart)
        ].sort_values("round_date")
        if len(player_data) < 50:
            continue
        last_50 = player_data.tail(50)
        cat_stds = {}
        for c in CAT_COLS:
            valid = last_50[c].dropna()
            cat_stds[c] = float(valid.std()) if len(valid) >= 30 else None
        trailing_cat_stds[(pname, yr)] = cat_stds
    print(f"player-years with 50+ trailing rounds: {len(trailing_cat_stds)}")

    # --- Step 6: per-year results ---
    years_in_data = sorted(player_years["year"].unique())
    result_rows = []
    for yr in years_in_data:
        qual_keys = [(p, y) for (p, y) in trailing_cat_stds.keys() if y == yr]
        qual_players = [k[0] for k in qual_keys]
        n_qual = len(qual_players)
        if n_qual < 10:
            print(f"  year {yr}: only {n_qual} qualified players -> DROPPED")
            continue
        yr_event = event_sg[
            (event_sg["year"] == yr) &
            (event_sg["player_name"].isin(qual_players))
        ]
        if yr_event.empty:
            print(f"  year {yr}: yr_event empty -> DROPPED")
            continue
        row = {"year": int(yr), "qual_players": n_qual}
        for c, name in zip(CAT_COLS, CAT_NAMES):
            valid_event = yr_event[c].dropna()
            actual_std = float(valid_event.std()) if len(valid_event) >= 10 else None
            field_stds = [trailing_cat_stds[k][c] for k in qual_keys
                          if trailing_cat_stds[k][c] is not None]
            field_exp_std = float(np.mean(field_stds)) if field_stds else None
            row[f"{name}_actual"] = round(actual_std, 3) if actual_std else None
            row[f"{name}_field_exp"] = round(field_exp_std, 3) if field_exp_std else None
            if actual_std and field_exp_std and field_exp_std > 0:
                row[f"{name}_mult"] = round(actual_std / field_exp_std, 2)
            else:
                row[f"{name}_mult"] = None
            if len(valid_event) >= 20:
                row[f"{name}_skew"] = round(float(sp_stats.skew(valid_event)), 2)
            else:
                row[f"{name}_skew"] = None
        result_rows.append(row)

    if not result_rows:
        print("No years with enough qualified players")
        return

    # --- Step 7: recency-weighted averages ---
    row_years = [r["year"] for r in result_rows]
    weights = compute_recency_weights(row_years)
    sum_w = sum(weights.values())

    weighted = {}
    category_mults = {}
    category_skew = {}
    for name in CAT_NAMES:
        w_actual = w_field = w_total = 0.0
        w_skew = w_skew_total = 0.0
        for r in result_rows:
            w = weights[r["year"]]
            if r[f"{name}_actual"] is not None and r[f"{name}_field_exp"] is not None:
                w_actual += r[f"{name}_actual"] * w
                w_field += r[f"{name}_field_exp"] * w
                w_total += w
            if r.get(f"{name}_skew") is not None:
                w_skew += r[f"{name}_skew"] * w
                w_skew_total += w
        if w_total > 0:
            w_actual /= w_total
            w_field /= w_total
            w_actual *= sum_w
            w_field *= sum_w
        weighted[f"{name}_actual"] = round(w_actual, 3)
        weighted[f"{name}_field_exp"] = round(w_field, 3)
        category_mults[name] = round(w_actual / w_field, 2) if w_field > 0 else 1.0
        if w_skew_total > 0:
            w_skew /= w_skew_total
            w_skew *= sum_w
        category_skew[name] = round(w_skew, 2)

    # ---- Reporting ----
    print("\n=== PER-YEAR TABLE ===")
    hdr = f"{'year':>5} {'qual':>4}"
    for name in CAT_NAMES:
        s = name.replace("sg_", "").upper()
        hdr += f" | {s}_act {s}_exp {s}_mult {s}_skew"
    print(hdr)
    for r in result_rows:
        line = f"{r['year']:>5} {r['qual_players']:>4}"
        for name in CAT_NAMES:
            line += (f" | {str(r[f'{name}_actual']):>7} {str(r[f'{name}_field_exp']):>7}"
                     f" {str(r[f'{name}_mult']):>8} {str(r.get(f'{name}_skew')):>8}")
        print(line)

    print("\nweights (year->w):", {y: round(w, 4) for y, w in weights.items()})
    print("\nweighted actual/field:")
    for name in CAT_NAMES:
        print(f"  {name}: act={weighted[f'{name}_actual']} "
              f"field={weighted[f'{name}_field_exp']}")

    print("\n=== FINAL category_mults ===", category_mults)
    print("=== FINAL category_skew  ===", category_skew)
    print("n_years_used =", len(result_rows))

    # emit machine-readable
    import json
    print("JSON_OUT", json.dumps({
        "category_mults": category_mults,
        "category_skew": category_skew,
        "baseline_skew": baseline_skew,
        "n_years_used": len(result_rows),
        "per_year": result_rows,
        "weighted": weighted,
        "weights": {int(y): w for y, w in weights.items()},
        "current_year": cy,
    }))


if __name__ == "__main__":
    main()

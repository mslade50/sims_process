"""Test pre-event shot-process profiles against future category SG.

The comparison baseline is each player's event-safe trailing category SG:
the current event is excluded wholesale, avoiding the common R2/R3 leakage
mistake. Discovery is walk-forward 2022-2025; 2026 is confirmation-only.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from shot_feature_predictive_study import (
    CONFIRM_YEAR,
    DISCOVERY_YEARS,
    RIDGE_ALPHAS,
    walk_forward_score,
)


DEFAULT_SIM_PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
CATEGORIES = ("ott", "app", "arg", "putt")

GROUP_SPECS = {
    "app_process": (
        "app",
        [
            "app_prox_core_trail16",
            "opportunities_gained_15ft_trail16",
            "app_dir_rms_along_trail16",
            "app_dir_rms_cross_trail16",
            "app_dir_mean_cross_trail16",
            "pin_high_pct_trail16",
            "on_line_pct_3yd_trail16",
            "dialed_pct_5yd_trail16",
        ],
    ),
    "putting_process": (
        "putt",
        [
            "lag_putt_leave_control_trail36",
            "short_putt_make_skill_trail100",
            "burned_edge_count_trail8",
            "speed_ideal_pct_trail24",
            "speed_short_pct_trail24",
            "speed_mean_signed_leave_trail24",
            "speed_sd_signed_leave_trail24",
        ],
    ),
    "arg_process": (
        "arg",
        [
            "arg_leave_vs_conversion_trail24",
            "arg_conv_luck_trail8",
            "short_side_miss_rate_trail24",
            "recovery_shot_count_trail8",
        ],
    ),
    "ott_process": (
        "ott",
        [
            "drive_dist_baseline_trail20",
        ],
    ),
}


def _norm(value: object) -> str:
    return str(value).lower().strip()


def load_feature_snapshot(sim_prep: Path) -> pd.DataFrame:
    bridge = sim_prep / "shot_bridge"
    frames = []
    for filename in ("shot_feature_library.parquet", "shot_feature_library2.parquet"):
        frame = pd.read_parquet(bridge / filename)
        frame = frame[frame["round_num"].eq(1)].copy()
        frame["_name"] = frame["player_name"].map(_norm)
        keys = ["dg_id", "event_id", "year"]
        wanted = [
            column for column in frame.columns
            if "_trail" in column
        ]
        frames.append(frame[keys + wanted].drop_duplicates(keys))
    keys = ["dg_id", "event_id", "year"]
    return frames[0].merge(frames[1], on=keys, how="outer", validate="one_to_one")


def load_rounds(sim_prep: Path) -> pd.DataFrame:
    db = sim_prep / "work" / "dg_historical.db"
    columns = ", ".join(f"sg_{category}_adj" for category in CATEGORIES)
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as connection:
        rounds = pd.read_sql_query(
            "SELECT dg_id, player_name, event_id, year, round_num, round_date, "
            f"{columns} FROM player_rounds WHERE tour='pga' AND year>=2020",
            connection,
        )
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce")
    rounds["_name"] = rounds["player_name"].map(_norm)
    return rounds.dropna(subset=["dg_id", "round_date"])


def add_event_safe_history(rounds: pd.DataFrame) -> pd.DataFrame:
    event_keys = ["dg_id", "event_id", "year"]
    event_rows = []
    for keys, event in rounds.groupby(event_keys, sort=False):
        first_two = event[event["round_num"].isin([1, 2])]
        if first_two["round_num"].nunique() != 2:
            continue
        row = dict(zip(event_keys, keys))
        row["event_date"] = event["round_date"].min()
        row["round_values"] = {
            category: event[f"sg_{category}_adj"].dropna().tolist()
            for category in CATEGORIES
        }
        for category in CATEGORIES:
            row[f"target_{category}"] = first_two[f"sg_{category}_adj"].mean()
        event_rows.append(row)
    events = pd.DataFrame(event_rows).sort_values(["dg_id", "event_date"])

    output = []
    for _, player in events.groupby("dg_id", sort=False):
        histories = {category: deque(maxlen=50) for category in CATEGORIES}
        for event in player.itertuples(index=False):
            row = event._asdict()
            for category in CATEGORIES:
                values = np.asarray(histories[category], dtype=float)
                row[f"prior_{category}_mean50"] = (
                    float(values.mean()) if len(values) >= 8 else np.nan
                )
                row[f"prior_{category}_sd50"] = (
                    float(values.std(ddof=1)) if len(values) >= 12 else np.nan
                )
                histories[category].extend(row["round_values"][category])
            output.append(row)
    return pd.DataFrame(output).drop(columns="round_values")


def evaluate(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, (category, requested) in GROUP_SPECS.items():
        features = [feature for feature in requested if feature in panel.columns]
        target = f"target_{category}"
        base = [f"prior_{category}_mean50", f"prior_{category}_sd50"]
        best = None
        for alpha in RIDGE_ALPHAS:
            result = walk_forward_score(
                panel,
                features,
                target,
                alpha,
                DISCOVERY_YEARS,
                base_columns=base,
            )
            candidate = (result["delta_mse"], alpha, result)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _, alpha, discovery = best
        confirm = walk_forward_score(
            panel,
            features,
            target,
            alpha,
            (CONFIRM_YEAR,),
            base_columns=base,
        )
        rows.append({
            "group": group_name,
            "category": category,
            "features": "|".join(features),
            "alpha": alpha,
            **{f"discovery_{key}": value for key, value in discovery.items()},
            **{f"confirm_{key}": value for key, value in confirm.items()},
        })
    return pd.DataFrame(rows).sort_values("discovery_delta_mse", ascending=False)


def evaluate_univariate(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, (category, requested) in GROUP_SPECS.items():
        target = f"target_{category}"
        base = [f"prior_{category}_mean50", f"prior_{category}_sd50"]
        for feature in requested:
            if feature not in panel.columns:
                continue
            discovery = walk_forward_score(
                panel, [feature], target, 10.0, DISCOVERY_YEARS,
                base_columns=base,
            )
            confirm = walk_forward_score(
                panel, [feature], target, 10.0, (CONFIRM_YEAR,),
                base_columns=base,
            )
            rows.append({
                "group": group_name,
                "category": category,
                "feature": feature,
                **{f"discovery_{key}": value for key, value in discovery.items()},
                **{f"confirm_{key}": value for key, value in confirm.items()},
            })
    return pd.DataFrame(rows).sort_values("discovery_delta_mse", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim-prep", type=Path, default=DEFAULT_SIM_PREP)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    features = load_feature_snapshot(args.sim_prep)
    history = add_event_safe_history(load_rounds(args.sim_prep))
    panel = history.merge(
        features,
        on=["dg_id", "event_id", "year"],
        how="inner",
        validate="one_to_one",
    )
    print(
        f"panel: {len(panel):,} player-events, years "
        f"{panel['year'].min()}-{panel['year'].max()}"
    )
    results = evaluate(panel)
    print(results[[
        "group", "category", "alpha", "discovery_delta_mse_pct",
        "discovery_event_t", "confirm_delta_mse_pct", "confirm_event_t",
    ]].to_string(index=False))
    univariate = evaluate_univariate(panel)
    print("\nTop univariate category features:")
    print(univariate.head(20)[[
        "group", "category", "feature", "discovery_delta_mse_pct",
        "discovery_event_t", "confirm_delta_mse_pct", "confirm_event_t",
    ]].to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        univariate.to_csv(
            args.output.with_name(args.output.stem + "_univariate.csv"),
            index=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

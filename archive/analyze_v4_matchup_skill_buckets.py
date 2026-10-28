"""Reconstruct v4 baseline skills and analyze best-of-three matchup bets.

This is a read-only supplement to the frozen v4 closing-market backtest.  It
replays only the deterministic event-field construction; it does not rerun any
Monte Carlo simulations or rescore any wagers.  Matchup bets are bucketed on
the bet-on player's baseline pre-event skill, common to every model arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_shot_mean_closing_markets import (
    DEFAULT_CROSSWALK,
    DEFAULT_DB,
    DEFAULT_FACTOR_PANEL,
    DEFAULT_ODDS_CACHE,
    DEFAULT_PREDS,
    HistoricalOddsCacheReader,
    build_event_field,
    canonical_player_id,
    load_name_replacements,
    load_player_histories,
    load_r1_predictions,
    load_source_crosswalk,
    normalize_prediction_rosters,
    read_event_metadata,
)
from market_backtest_scoring import event_cluster_bootstrap


REPO = Path(__file__).resolve().parents[1]
V4 = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v4"
)
OUTPUT = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v4_skill_buckets"
)

BUCKET_ORDER = ("high", "mid_high", "mid_low", "low")
BUCKET_LABELS = {
    "high": "> 1.0",
    "mid_high": "(0.5, 1.0]",
    "mid_low": "(-0.5, 0.5]",
    "low": "<= -0.5",
}


def skill_bucket(values: pd.Series) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="raise")
    labels = np.select(
        [numeric > 1.0, numeric > 0.5, numeric > -0.5],
        ["high", "mid_high", "mid_low"],
        default="low",
    )
    return pd.Categorical(labels, categories=BUCKET_ORDER, ordered=True)


def reconstruct_skill_map(event_keys: list[str]) -> pd.DataFrame:
    replacements = load_name_replacements()
    reader = HistoricalOddsCacheReader(DEFAULT_ODDS_CACHE)
    rosters = normalize_prediction_rosters(reader.predictions(), replacements)
    rosters = rosters[rosters["event_key"].isin(event_keys)].copy()
    metadata = read_event_metadata(DEFAULT_DB, event_keys)
    if set(metadata) != set(event_keys):
        missing = sorted(set(event_keys) - set(metadata))
        raise ValueError(f"Missing event metadata: {missing}")

    union_ids = sorted(rosters["dg_id"].dropna().astype(int).unique())
    _, model_histories = load_player_histories(
        DEFAULT_DB,
        union_ids,
        max(metadata[event].start for event in event_keys) + pd.Timedelta(days=1),
    )
    r1_predictions = load_r1_predictions(DEFAULT_PREDS, replacements)
    source_crosswalk = load_source_crosswalk(DEFAULT_CROSSWALK)
    factor_panel = pd.read_parquet(DEFAULT_FACTOR_PANEL)

    pieces: list[pd.DataFrame] = []
    for event_key in event_keys:
        event_roster = rosters[rosters["event_key"].eq(event_key)]
        event_factor = factor_panel[factor_panel["event_key"].eq(event_key)]
        field, _ = build_event_field(
            event_roster,
            metadata[event_key],
            r1_predictions,
            model_histories,
            source_crosswalk,
            event_factor,
        )
        keep = field[
            [
                "dg_id",
                "player_name",
                "skill_est",
                "skill_source",
                "factor_available",
                "factor_delta_sg_per_round",
            ]
        ].copy()
        keep.insert(0, "event_start", metadata[event_key].start)
        keep.insert(0, "event_key", event_key)
        keep["selection_id"] = keep["dg_id"].map(canonical_player_id)
        keep = keep.rename(
            columns={
                "skill_est": "baseline_skill_sg_per_round",
                "factor_delta_sg_per_round": "walk_forward_delta_sg_per_round",
            }
        )
        max_abs_delta = float(keep["walk_forward_delta_sg_per_round"].abs().max())
        cap_scale = min(1.0, 0.15 / max_abs_delta) if max_abs_delta > 0 else 1.0
        keep["walk_forward_cap015_delta_sg_per_round"] = (
            keep["walk_forward_delta_sg_per_round"] * cap_scale
        )
        keep["skill_bucket"] = skill_bucket(keep["baseline_skill_sg_per_round"])
        keep["skill_bucket_range"] = keep["skill_bucket"].astype(str).map(BUCKET_LABELS)
        pieces.append(keep)

    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["event_key", "selection_id"]).any():
        raise ValueError("Duplicate event/player keys in reconstructed skill map")
    return result.sort_values(
        ["event_start", "event_key", "baseline_skill_sg_per_round"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_against_v4(skill_map: pd.DataFrame) -> dict[str, object]:
    audit = pd.read_parquet(V4 / "event_audit.parquet")
    reconstructed = (
        skill_map.groupby("event_key", observed=True)
        .agg(
            roster_players=("selection_id", "size"),
            r1_prediction_players=(
                "skill_source", lambda s: int(s.eq("r1_prediction").sum())
            ),
            dated_model_fallback_players=(
                "skill_source", lambda s: int(s.eq("dated_model_fallback").sum())
            ),
            field_floor_fallback_players=(
                "skill_source", lambda s: int(s.eq("field_floor_fallback").sum())
            ),
            skill_min=("baseline_skill_sg_per_round", "min"),
            skill_max=("baseline_skill_sg_per_round", "max"),
        )
        .reset_index()
    )
    columns = [
        "roster_players",
        "r1_prediction_players",
        "dated_model_fallback_players",
        "field_floor_fallback_players",
    ]
    audit_columns = {
        column: f"field.{column}"
        for column in [*columns, "skill_min", "skill_max"]
    }
    audit_subset = audit[["event_key", *audit_columns.values()]].rename(
        columns={value: key for key, value in audit_columns.items()}
    )
    merged = reconstructed.merge(
        audit_subset,
        on="event_key",
        suffixes=("_reconstructed", "_v4"),
        validate="one_to_one",
    )
    count_mismatch = sum(
        int(
            (
                merged[f"{column}_reconstructed"]
                != merged[f"{column}_v4"]
            ).sum()
        )
        for column in columns
    )
    min_error = float(
        np.max(np.abs(merged["skill_min_reconstructed"] - merged["skill_min_v4"]))
    )
    max_error = float(
        np.max(np.abs(merged["skill_max_reconstructed"] - merged["skill_max_v4"]))
    )
    if count_mismatch or min_error > 1e-12 or max_error > 1e-12:
        raise ValueError(
            "Reconstructed fields do not match v4 event audit: "
            f"count_mismatch={count_mismatch}, min_error={min_error}, max_error={max_error}"
        )
    return {
        "event_count": int(len(merged)),
        "player_event_count": int(len(skill_map)),
        "count_mismatches": int(count_mismatch),
        "max_abs_skill_min_error": min_error,
        "max_abs_skill_max_error": max_error,
    }


def add_zero_event_rows(
    frame: pd.DataFrame, event_universe: pd.DataFrame
) -> pd.DataFrame:
    arms = ("baseline", "challenger")
    existing = set(zip(frame["event_key"], frame["arm"]))
    rows = []
    for event_key, event_start in event_universe[
        ["event_key", "event_start"]
    ].itertuples(index=False, name=None):
        for arm in arms:
            if (event_key, arm) not in existing:
                rows.append(
                    {
                        "event_key": event_key,
                        "event_start": event_start,
                        "arm": arm,
                        "stake": 0.0,
                        "pnl": 0.0,
                    }
                )
    if rows:
        return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return frame


def analyze_bets(
    skill_map: pd.DataFrame, event_keys: list[str], draws: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bets = pd.read_parquet(V4 / "bets.parquet")
    bets = bets[
        bets["line_policy_name"].eq("best_of_sharp")
        & bets["market"].eq("tournament_matchup")
    ].copy()
    join = skill_map[
        [
            "event_key",
            "selection_id",
            "baseline_skill_sg_per_round",
            "skill_bucket",
            "skill_bucket_range",
        ]
    ].copy()
    bets = bets.merge(
        join,
        on=["event_key", "selection_id"],
        how="left",
        validate="many_to_one",
    )
    if bets["baseline_skill_sg_per_round"].isna().any():
        missing = bets.loc[
            bets["baseline_skill_sg_per_round"].isna(),
            ["event_key", "selection_id", "selection_name"],
        ].drop_duplicates()
        raise ValueError(f"Unmapped matchup selections:\n{missing.head(20)}")
    if set(bets["book"].unique()) - {"betonline", "betcris", "pinnacle"}:
        raise ValueError("Best-of-sharp slice contains a book outside the three-book set")

    event_universe = skill_map[["event_key", "event_start"]].drop_duplicates()
    if set(event_universe["event_key"]) != set(event_keys):
        raise ValueError("Event universe differs from v4 run config")

    performance_rows: list[dict[str, object]] = []
    bootstrap_pieces: list[pd.DataFrame] = []
    for challenger_name in ("walk_forward_uncapped", "walk_forward_cap015"):
        for sizing_name in ("flat_1u", "production_style"):
            comparison = bets[
                bets["challenger_name"].eq(challenger_name)
                & bets["sizing_name"].eq(sizing_name)
            ].copy()
            for bucket in BUCKET_ORDER:
                bucket_bets = comparison[comparison["skill_bucket"].eq(bucket)].copy()
                by_arm: dict[str, dict[str, float | int]] = {}
                for arm in ("baseline", "challenger"):
                    arm_bets = bucket_bets[bucket_bets["arm"].eq(arm)]
                    stake = float(arm_bets["stake"].sum())
                    pnl = float(arm_bets["pnl"].sum())
                    by_arm[arm] = {
                        "bet_count": int(len(arm_bets)),
                        "event_count_with_bet": int(arm_bets["event_key"].nunique()),
                        "stake": stake,
                        "pnl": pnl,
                        "roi": pnl / stake if stake else np.nan,
                    }
                    performance_rows.append(
                        {
                            "challenger_name": challenger_name,
                            "sizing_name": sizing_name,
                            "skill_bucket": bucket,
                            "skill_bucket_range": BUCKET_LABELS[bucket],
                            "arm": "baseline" if arm == "baseline" else challenger_name,
                            **by_arm[arm],
                        }
                    )
                performance_rows.append(
                    {
                        "challenger_name": challenger_name,
                        "sizing_name": sizing_name,
                        "skill_bucket": bucket,
                        "skill_bucket_range": BUCKET_LABELS[bucket],
                        "arm": f"{challenger_name}_minus_baseline",
                        "bet_count": int(
                            by_arm["challenger"]["bet_count"]
                            - by_arm["baseline"]["bet_count"]
                        ),
                        "event_count_with_bet": np.nan,
                        "stake": float(
                            by_arm["challenger"]["stake"] - by_arm["baseline"]["stake"]
                        ),
                        "pnl": float(
                            by_arm["challenger"]["pnl"] - by_arm["baseline"]["pnl"]
                        ),
                        "roi": float(
                            by_arm["challenger"]["roi"] - by_arm["baseline"]["roi"]
                        ),
                    }
                )

                boot_input = add_zero_event_rows(
                    bucket_bets[
                        ["event_key", "event_start", "arm", "stake", "pnl"]
                    ].copy(),
                    event_universe,
                )
                boot = event_cluster_bootstrap(
                    boot_input,
                    event_col="event_key",
                    event_order_col="event_start",
                    draws=draws,
                    seed=20260822,
                )
                boot["challenger_name"] = challenger_name
                boot["sizing_name"] = sizing_name
                boot["skill_bucket"] = bucket
                boot["skill_bucket_range"] = BUCKET_LABELS[bucket]
                boot["arm"] = boot["arm"].replace(
                    {
                        "challenger": challenger_name,
                        "challenger_minus_baseline": f"{challenger_name}_minus_baseline",
                    }
                )
                bootstrap_pieces.append(boot)

    performance = pd.DataFrame(performance_rows)
    bootstrap = pd.concat(bootstrap_pieces, ignore_index=True)
    return bets, performance, bootstrap


def write_report(
    validation: dict[str, object],
    bets: pd.DataFrame,
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    lines = [
        "# Best-of-three matchup performance by baseline skill",
        "",
        "This supplement filters the frozen v4 wager export to tournament matchups",
        "whose selected closing price is the best available across BetOnline, Betcris,",
        "and Pinnacle. The skill bucket is always the bet-on golfer's reconstructed",
        "baseline pre-event skill, so it is fixed and common across model arms.",
        "",
        "The four thresholds are the production R1 skill regimes: high >1.0, mid-high",
        "(0.5, 1.0], mid-low (-0.5, 0.5], and low <=-0.5 SG/round.",
        "",
        f"Exact field validation: {validation['event_count']} events and",
        f"{validation['player_event_count']:,} player-events; zero source-count",
        "mismatches and exact event skill extrema versus the v4 event audit.",
        "",
        "## Challenger minus baseline P&L",
        "",
        "| Challenger | Sizing | Skill | Baseline bets/stake | Challenger bets/stake | Baseline P&L/ROI | Challenger P&L/ROI | Delta | 95% event CI |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    deltas = performance[performance["arm"].str.endswith("_minus_baseline")]
    for row in deltas.itertuples(index=False):
        base = performance[
            performance["challenger_name"].eq(row.challenger_name)
            & performance["sizing_name"].eq(row.sizing_name)
            & performance["skill_bucket"].eq(row.skill_bucket)
            & performance["arm"].eq("baseline")
        ].iloc[0]
        challenger = performance[
            performance["challenger_name"].eq(row.challenger_name)
            & performance["sizing_name"].eq(row.sizing_name)
            & performance["skill_bucket"].eq(row.skill_bucket)
            & performance["arm"].eq(row.challenger_name)
        ].iloc[0]
        ci = bootstrap[
            bootstrap["challenger_name"].eq(row.challenger_name)
            & bootstrap["sizing_name"].eq(row.sizing_name)
            & bootstrap["skill_bucket"].eq(row.skill_bucket)
            & bootstrap["arm"].eq(row.arm)
            & bootstrap["metric"].eq("pnl")
        ].iloc[0]
        lines.append(
            f"| {row.challenger_name} | {row.sizing_name} | {row.skill_bucket_range} "
            f"| {int(base.bet_count):,} / {base.stake:.2f}u "
            f"| {int(challenger.bet_count):,} / {challenger.stake:.2f}u "
            f"| {base.pnl:+.2f}u / {base.roi:+.2%} "
            f"| {challenger.pnl:+.2f}u / {challenger.roi:+.2%} "
            f"| {row.pnl:+.2f}u "
            f"| [{ci.ci_low:+.2f}, {ci.ci_high:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "All confidence intervals resample whole events and retain all 105 eligible",
            "events, including zero-bet events within a bucket.",
            "",
            "## Selected-book counts",
            "",
        ]
    )
    book_counts = (
        bets.groupby("book", observed=True).size().sort_values(ascending=False)
    )
    for book, count in book_counts.items():
        lines.append(f"- {book}: {int(count):,} wager rows across all arm/sizing comparisons")
    lines.append("")
    (OUTPUT / "MATCHUP_SKILL_BUCKETS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = json.loads((V4 / "run_config.json").read_text(encoding="utf-8"))
    event_keys = list(config["event_keys"])
    skill_map = reconstruct_skill_map(event_keys)
    validation = validate_against_v4(skill_map)
    bets, performance, bootstrap = analyze_bets(skill_map, event_keys)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    skill_map.to_parquet(OUTPUT / "baseline_skill_map.parquet", index=False)
    performance.to_csv(OUTPUT / "matchup_skill_bucket_performance.csv", index=False)
    bootstrap.to_csv(OUTPUT / "matchup_skill_bucket_bootstrap.csv", index=False)
    write_report(validation, bets, performance, bootstrap)
    audit = {
        "contract": "v4_best_of_three_matchup_skill_bucket_supplement_v1",
        "source_v4": str(V4),
        "line_policy": "best_of_sharp",
        "allowed_books": ["betonline", "betcris", "pinnacle"],
        "bucket_variable": "bet_on_golfer_baseline_pre_event_skill_sg_per_round",
        "bucket_edges": BUCKET_LABELS,
        "bootstrap_draws": 10_000,
        "bootstrap_cluster_universe": "all_105_v4_eligible_events_including_zero_bet_events",
        "validation": validation,
        "mapped_wager_rows": int(len(bets)),
        "missing_skill_rows": int(bets["baseline_skill_sg_per_round"].isna().sum()),
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

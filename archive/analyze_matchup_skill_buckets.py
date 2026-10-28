"""Analyze best-price tournament matchups by the bet-on golfer's baseline skill.

This is a post-run audit of the corrected shot-mean closing-market backtest.  It
reconstructs the exact pre-event ``skill_est`` used to seed each event field,
joins that common baseline skill to settled matchup bets, and reports results
for both absolute SG/round bands and within-field quintiles.  Buckets are fixed
from baseline skill and therefore do not move between model arms.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_shot_mean_closing_markets as bt
from market_backtest_scoring import event_cluster_bootstrap


DEFAULT_BACKTEST = (
    bt.REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v4"
)
ABSOLUTE_EDGES = [-np.inf, 0.0, 0.5, 1.0, 1.5, np.inf]
ABSOLUTE_LABELS = ["<0", "0 to <0.5", "0.5 to <1.0", "1.0 to <1.5", ">=1.5"]
QUINTILE_LABELS = ["top 20%", "20-40%", "40-60%", "60-80%", "bottom 20%"]
GAP_EDGES = [-np.inf, -0.5, -0.15, 0.15, 0.5, np.inf]
GAP_LABELS = ["<-0.5 weaker", "-0.5 to <-0.15", "similar +/-0.15", "+0.15 to <+0.5", ">=+0.5 stronger"]


def stable_seed(*values: object) -> int:
    payload = "|".join(map(str, values)).encode("utf-8")
    return 20260822 + int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def reconstruct_skill_map(backtest_dir: Path) -> pd.DataFrame:
    manifest = pd.read_json(backtest_dir / "manifest.json", typ="series")
    event_keys = [str(value) for value in manifest["event_keys"]]
    replacements = bt.load_name_replacements()

    factor_panel = pd.read_parquet(bt.DEFAULT_FACTOR_PANEL)
    factor_panel = factor_panel[factor_panel["event_key"].isin(event_keys)].copy()
    reader = bt.HistoricalOddsCacheReader(bt.DEFAULT_ODDS_CACHE)
    rosters = bt.normalize_prediction_rosters(reader.predictions(), replacements)
    rosters = rosters[rosters["event_key"].isin(event_keys)].copy()
    metadata = bt.read_event_metadata(bt.DEFAULT_DB, event_keys)
    if set(metadata) != set(event_keys):
        missing = sorted(set(event_keys) - set(metadata))
        raise ValueError(f"Missing event metadata for {missing}")

    union_ids = sorted(rosters["dg_id"].dropna().astype(int).unique())
    _, model_histories = bt.load_player_histories(
        bt.DEFAULT_DB,
        union_ids,
        max(metadata[event].start for event in event_keys) + pd.Timedelta(days=1),
    )
    r1_predictions = bt.load_r1_predictions(bt.DEFAULT_PREDS, replacements)
    source_crosswalk = bt.load_source_crosswalk(bt.DEFAULT_CROSSWALK)

    pieces: list[pd.DataFrame] = []
    for event_key in event_keys:
        field, _ = bt.build_event_field(
            rosters[rosters["event_key"].eq(event_key)],
            metadata[event_key],
            r1_predictions,
            model_histories,
            source_crosswalk,
            factor_panel[factor_panel["event_key"].eq(event_key)],
        )
        field = field.copy()
        field["event_id"] = event_key
        field["event_start"] = metadata[event_key].start
        field["event_name"] = metadata[event_key].event_name
        field["selection_id"] = field["dg_id"].map(bt.canonical_player_id)
        field["projected_skill"] = pd.to_numeric(field["skill_est"], errors="raise")
        field["absolute_skill_bucket"] = pd.cut(
            field["projected_skill"],
            bins=ABSOLUTE_EDGES,
            labels=ABSOLUTE_LABELS,
            right=False,
            ordered=True,
        )
        rank = field["projected_skill"].rank(method="first", ascending=False)
        percentile = (rank - 1.0) / len(field)
        field["field_skill_quintile"] = pd.cut(
            percentile,
            bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
            labels=QUINTILE_LABELS,
            right=False,
            ordered=True,
        )
        pieces.append(
            field[
                [
                    "event_id",
                    "event_start",
                    "event_name",
                    "selection_id",
                    "player_name",
                    "projected_skill",
                    "skill_source",
                    "absolute_skill_bucket",
                    "field_skill_quintile",
                ]
            ]
        )
    skill_map = pd.concat(pieces, ignore_index=True)
    if skill_map.duplicated(["event_id", "selection_id"]).any():
        raise ValueError("Duplicate event-player keys in reconstructed skill map")
    return skill_map


def add_skill_columns(bets: pd.DataFrame, skill_map: pd.DataFrame) -> pd.DataFrame:
    selection = skill_map.rename(
        columns={
            "player_name": "skill_player_name",
            "projected_skill": "selection_projected_skill",
            "skill_source": "selection_skill_source",
        }
    )
    keep = [
        "event_id",
        "selection_id",
        "selection_projected_skill",
        "selection_skill_source",
        "absolute_skill_bucket",
        "field_skill_quintile",
    ]
    result = bets.merge(selection[keep], on=["event_id", "selection_id"], how="left", validate="many_to_one")
    opponent = skill_map[["event_id", "selection_id", "projected_skill"]].rename(
        columns={"selection_id": "opponent_id", "projected_skill": "opponent_projected_skill"}
    )
    result = result.merge(opponent, on=["event_id", "opponent_id"], how="left", validate="many_to_one")
    if result[["selection_projected_skill", "opponent_projected_skill"]].isna().any().any():
        missing = result[result["selection_projected_skill"].isna() | result["opponent_projected_skill"].isna()]
        raise ValueError(f"Matchup bets missing reconstructed skills: {len(missing)}")
    result["projected_skill_gap"] = result["selection_projected_skill"] - result["opponent_projected_skill"]
    result["skill_gap_bucket"] = pd.cut(
        result["projected_skill_gap"],
        bins=GAP_EDGES,
        labels=GAP_LABELS,
        right=False,
        ordered=True,
    )
    return result


def add_zero_event_rows(group: pd.DataFrame, event_universe: pd.DataFrame) -> pd.DataFrame:
    present = set(zip(group["event_id"].astype(str), group["arm"].astype(str)))
    rows = [
        {"event_id": event_id, "event_start": event_start, "arm": arm, "stake": 0.0, "pnl": 0.0}
        for event_id, event_start in event_universe.itertuples(index=False, name=None)
        for arm in ("baseline", "challenger")
        if (str(event_id), arm) not in present
    ]
    return pd.concat([group, pd.DataFrame(rows)], ignore_index=True, sort=False) if rows else group


def summarize_buckets(
    bets: pd.DataFrame,
    skill_map: pd.DataFrame,
    bucket_columns: list[str],
    draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_universe = skill_map[["event_id", "event_start"]].drop_duplicates("event_id")
    performance_rows: list[dict[str, object]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    for challenger_name in sorted(bets["challenger_name"].unique()):
        for sizing_name in sorted(bets["sizing_name"].unique()):
            scenario = bets[
                bets["challenger_name"].eq(challenger_name)
                & bets["sizing_name"].eq(sizing_name)
            ]
            for bucket_column in bucket_columns:
                observed = scenario[bucket_column].dropna()
                if isinstance(observed.dtype, pd.CategoricalDtype):
                    values = [value for value in observed.cat.categories if (observed == value).any()]
                else:
                    values = observed.drop_duplicates().tolist()
                for bucket in values:
                    group = scenario[scenario[bucket_column].eq(bucket)].copy()
                    for arm in ("baseline", "challenger"):
                        arm_bets = group[group["arm"].eq(arm)]
                        stake = float(arm_bets["stake"].sum())
                        pnl = float(arm_bets["pnl"].sum())
                        performance_rows.append(
                            {
                                "challenger_name": challenger_name,
                                "sizing_name": sizing_name,
                                "bucket_type": bucket_column,
                                "bucket": str(bucket),
                                "arm": arm,
                                "bet_count": int(len(arm_bets)),
                                "bet_event_count": int(arm_bets["event_id"].nunique()),
                                "stake": stake,
                                "pnl": pnl,
                                "roi": pnl / stake if stake > 0 else np.nan,
                            }
                        )
                    boot = event_cluster_bootstrap(
                        add_zero_event_rows(group, event_universe),
                        event_col="event_id",
                        event_order_col="event_start",
                        draws=draws,
                        seed=stable_seed(challenger_name, sizing_name, bucket_column, bucket),
                    )
                    boot["challenger_name"] = challenger_name
                    boot["sizing_name"] = sizing_name
                    boot["bucket_type"] = bucket_column
                    boot["bucket"] = str(bucket)
                    bootstrap_rows.append(boot)
    performance = pd.DataFrame(performance_rows)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)
    return performance, bootstrap


def audit_best_price_books(event_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reader = bt.HistoricalOddsCacheReader(bt.DEFAULT_ODDS_CACHE)
    raw = bt.normalize_matchup_lines(reader.matchups())
    allowed = raw[raw["book"].isin(bt.MATCHUP_BOOKS) & raw["event_id"].isin(event_ids)].copy()
    availability = (
        allowed.groupby(bt.KEY_COLS, observed=True)
        .agg(available_book_count=("book", "nunique"), available_books=("book", lambda values: ",".join(sorted(set(values)))))
        .reset_index()
    )
    chosen = bt.select_closing_lines(
        allowed,
        bt.LinePolicy(mode="best_of_sharp", books=bt.MATCHUP_BOOKS),
        bt.KEY_COLS,
    ).merge(availability, on=bt.KEY_COLS, how="left", validate="one_to_one")
    summary = (
        chosen.groupby(["available_book_count", "book"], observed=True)
        .size()
        .rename("logical_line_count")
        .reset_index()
    )
    return chosen, summary


def comparison_table(
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    challenger_name: str,
    sizing_name: str,
    bucket_type: str,
) -> pd.DataFrame:
    perf = performance[
        performance["challenger_name"].eq(challenger_name)
        & performance["sizing_name"].eq(sizing_name)
        & performance["bucket_type"].eq(bucket_type)
    ]
    pivot = perf.pivot(index="bucket", columns="arm", values=["bet_count", "stake", "pnl", "roi"])
    output = pd.DataFrame(index=pivot.index)
    for metric in ("bet_count", "stake", "pnl", "roi"):
        output[f"baseline_{metric}"] = pivot[(metric, "baseline")]
        output[f"challenger_{metric}"] = pivot[(metric, "challenger")]
    delta = bootstrap[
        bootstrap["challenger_name"].eq(challenger_name)
        & bootstrap["sizing_name"].eq(sizing_name)
        & bootstrap["bucket_type"].eq(bucket_type)
        & bootstrap["arm"].eq("challenger_minus_baseline")
        & bootstrap["metric"].eq("pnl")
    ].set_index("bucket")
    output["delta_pnl"] = delta["estimate"]
    output["delta_ci_low"] = delta["ci_low"]
    output["delta_ci_high"] = delta["ci_high"]
    output["delta_probability_positive"] = delta["probability_above_zero"]
    output = output.reset_index()
    bucket_orders = {
        "absolute_skill_bucket": ABSOLUTE_LABELS,
        "field_skill_quintile": QUINTILE_LABELS,
        "skill_gap_bucket": GAP_LABELS,
    }
    order = {label: index for index, label in enumerate(bucket_orders[bucket_type])}
    output["_bucket_order"] = output["bucket"].map(order)
    output = output.sort_values("_bucket_order", kind="mergesort").drop(columns="_bucket_order")
    for column in ("baseline_bet_count", "challenger_bet_count"):
        output[column] = output[column].astype(int)
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backtest-dir", type=Path, default=DEFAULT_BACKTEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    backtest_dir = args.backtest_dir.resolve()
    output_dir = (args.output_dir or (backtest_dir / "matchup_skill_bucket_audit")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_map = reconstruct_skill_map(backtest_dir)
    bets = pd.read_parquet(backtest_dir / "bets.parquet")
    bets = bets[
        bets["market_group"].eq("matchup")
        & bets["line_policy_name"].eq("best_of_sharp")
    ].copy()
    bets = add_skill_columns(bets, skill_map)
    performance, bootstrap = summarize_buckets(
        bets,
        skill_map,
        ["absolute_skill_bucket", "field_skill_quintile", "skill_gap_bucket"],
        args.bootstrap,
    )
    chosen_lines, book_summary = audit_best_price_books(set(skill_map["event_id"]))

    skill_map.to_parquet(output_dir / "event_player_projected_skill.parquet", index=False)
    bets.to_parquet(output_dir / "best_price_matchup_bets_with_skill.parquet", index=False)
    performance.to_csv(output_dir / "matchup_skill_bucket_performance.csv", index=False)
    bootstrap.to_csv(output_dir / "matchup_skill_bucket_bootstrap.csv", index=False)
    chosen_lines.to_parquet(output_dir / "best_price_matchup_line_audit.parquet", index=False)
    book_summary.to_csv(output_dir / "best_price_book_summary.csv", index=False)

    lines = [
        "# Best-Price Matchup Performance by Projected Skill",
        "",
        "Best price is the highest American closing price among Betcris, Pinnacle, and BetOnline for the same event/selection/opponent. If only one of those books offered the line, that quote remains eligible; identical prices resolve by Betcris, then Pinnacle, then BetOnline, and the latest duplicate close at that book wins.",
        "",
        f"- Events: {skill_map['event_id'].nunique()}",
        f"- Event-player skills: {len(skill_map):,}",
        f"- Best-price logical matchup sides: {len(chosen_lines):,}",
        f"- Settled matchup rows analyzed across scenarios: {len(bets):,}",
        f"- Missing skill joins: {int(bets['selection_projected_skill'].isna().sum())}",
        f"- Bootstrap: {args.bootstrap:,} whole-event draws per bucket",
        "",
        "## Production-style, capped factor: absolute projected skill",
        "",
        markdown_table(
            comparison_table(performance, bootstrap, "walk_forward_cap015", "production_style", "absolute_skill_bucket")
        ),
        "",
        "## Production-style, uncapped factor: absolute projected skill",
        "",
        markdown_table(
            comparison_table(performance, bootstrap, "walk_forward_uncapped", "production_style", "absolute_skill_bucket")
        ),
        "",
        "## Production-style, capped factor: within-field skill quintile",
        "",
        markdown_table(
            comparison_table(performance, bootstrap, "walk_forward_cap015", "production_style", "field_skill_quintile")
        ),
        "",
        "## Production-style, capped factor: projected skill gap to opponent",
        "",
        markdown_table(
            comparison_table(performance, bootstrap, "walk_forward_cap015", "production_style", "skill_gap_bucket")
        ),
        "",
        "## Interpretation",
        "",
        "Buckets use the bet-on golfer's common baseline pre-event skill, not a challenger-adjusted value. They are descriptive subgroup results: the EV gate selects different wagers in each arm, bets within an event are correlated, and examining several buckets creates multiple-comparison risk. Field quintiles control for changing tournament strength; absolute bands preserve SG/round meaning.",
        "",
    ]
    (output_dir / "MATCHUP_SKILL_BUCKET_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote matchup skill-bucket audit to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

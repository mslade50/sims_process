"""Audit 72-hole matchup performance by theoretical Kelly-edge bucket.

This research-only analysis reuses the frozen v4 simulation probabilities and
the checksummed DataGolf odds cache.  It compares the best available quote
among Betcris, Pinnacle, and BetOnline at two snapshots: DataGolf's archived
open and archived close.

Production calls expected return per unit risked the "Kelly edge":

    p(win) * profit_multiple - p(loss)

For markets without pushes this is ``p * decimal_odds - 1``.  The analysis
lowers the matchup EV gate from 3% to zero only so the requested 0-3% bucket
can be observed; those rows are explicitly counterfactual and were not part of
the sealed v4 betting result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
for module_path in (REPO, ARCHIVE):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import backtest_shot_mean_closing_markets as bt  # noqa: E402
from historical_odds_cache import HistoricalOddsCacheReader  # noqa: E402
from market_backtest_scoring import (  # noqa: E402
    GateConfig,
    LinePolicy,
    SizingConfig,
    event_cluster_bootstrap,
    select_arm_bets,
    select_closing_lines,
    settle_bets,
)


V4 = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v4"
)
OUT = V4 / "matchup_kelly_edge_open_close_audit"
CACHE = REPO / "permanent_data" / "research" / "dg_closing_cache_20230822_20260822"
KEYS = ["event_id", "market", "selection_id", "opponent_id"]
BOOKS = ("betcris", "pinnacle", "betonline")
VARIANTS = ("walk_forward_cap015", "walk_forward_uncapped")
BUCKETS = ("0-3%", "3-5%", "5-10%", ">=10%")


def valid_american(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & ((numeric <= -100.0) | (numeric >= 100.0))


def normalize_matchup_snapshot(
    rows: list[dict[str, object]], snapshot: str
) -> pd.DataFrame:
    """Expand cached two-sided 72-hole lines at one archived price snapshot."""

    if snapshot not in {"open", "close"}:
        raise ValueError(f"Unsupported snapshot: {snapshot}")
    raw = bt.add_event_key(pd.DataFrame(rows))
    bet_type = raw.get("bet_type", "").astype(str).str.casefold()
    raw = raw[bet_type.str.contains("tournament|72", regex=True, na=False)].copy()
    expanded: list[dict[str, object]] = []
    for row in raw.to_dict(orient="records"):
        for side, other in (("p1", "p2"), ("p2", "p1")):
            expanded.append(
                {
                    "event_id": row["event_key"],
                    "event_key": row["event_key"],
                    "year": row["year"],
                    "dg_event_id": row["dg_event_id"],
                    "event_name": row.get("event_name"),
                    "market": "tournament_matchup",
                    "selection_id": bt.canonical_player_id(row.get(f"{side}_dg_id")),
                    "selection_name": row.get(f"{side}_player_name"),
                    "opponent_id": bt.canonical_player_id(row.get(f"{other}_dg_id")),
                    "opponent_name": row.get(f"{other}_player_name"),
                    "book": bt.normalize_text(row.get("book")),
                    # Scoring helpers use the historical name close_odds.  At
                    # the open snapshot this column intentionally holds open.
                    "close_odds": row.get(f"{side}_{snapshot}"),
                    "close_time": row.get(f"{snapshot}_time"),
                    "tie_rule": row.get("tie_rule"),
                    "outcome": row.get(
                        f"{side}_outcome_text", row.get(f"{side}_outcome")
                    ),
                    "outcome_numeric": row.get(f"{side}_outcome"),
                    "record_id": row.get("record_id"),
                    "price_snapshot": snapshot,
                }
            )
    frame = pd.DataFrame(expanded)
    valid = (
        frame["selection_id"].ne("")
        & frame["opponent_id"].ne("")
        & valid_american(frame["close_odds"])
    )
    frame = frame.loc[valid].copy()
    frame["close_odds"] = pd.to_numeric(frame["close_odds"], errors="raise")
    frame["outcome_numeric"] = pd.to_numeric(
        frame["outcome_numeric"], errors="coerce"
    )
    return frame


def select_best_lines(
    raw_rows: list[dict[str, object]],
    snapshot: str,
    event_keys: set[str],
    event_starts: pd.DataFrame,
) -> pd.DataFrame:
    normalized = normalize_matchup_snapshot(raw_rows, snapshot)
    normalized = normalized[normalized["event_key"].isin(event_keys)].copy()
    # The cache has exactly one row per logical side/book, so opening-time
    # duplicate ordering is immaterial.  This assertion prevents a future
    # cache shape from silently changing that contract.
    if normalized.duplicated(KEYS + ["book"]).any():
        raise AssertionError(f"{snapshot} cache has duplicate side/book rows")
    selected = select_closing_lines(
        normalized,
        LinePolicy(mode="best_of_sharp", books=BOOKS),
        KEYS,
    )
    selected = selected.merge(
        event_starts,
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    if selected["event_start"].isna().any():
        raise AssertionError(f"{snapshot} selected lines are missing event dates")
    selected["price_snapshot"] = snapshot
    return selected


def production_matchup_bets(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
    arm_label: str,
) -> pd.DataFrame:
    """Select every nonnegative-EV line, then apply matchup unit convention."""

    selected = select_arm_bets(
        lines,
        probabilities,
        arm_label,
        GateConfig(min_ev=0.0, allowed_markets=("tournament_matchup",)),
        SizingConfig(method="flat", flat_stake=1.0),
        KEYS,
    )
    settlement_columns = [
        "settlement_status",
        "pnl",
        "return_amount",
        "paid_fraction",
        "is_graded",
    ]
    selected = selected.drop(
        columns=[column for column in settlement_columns if column in selected],
        errors="ignore",
    )
    selected["stake"] = np.where(
        selected["close_odds"] < 0,
        selected["close_odds"].abs() / 100.0,
        1.0,
    )
    selected = settle_bets(selected)
    selected["kelly_edge"] = selected["ev_per_unit"].astype(float)
    selected["full_kelly_fraction"] = np.maximum(
        (
            selected["pricing_probability"]
            * np.where(
                selected["close_odds"] > 0,
                selected["close_odds"] / 100.0,
                100.0 / selected["close_odds"].abs(),
            )
            - (1.0 - selected["pricing_probability"])
        )
        / np.where(
            selected["close_odds"] > 0,
            selected["close_odds"] / 100.0,
            100.0 / selected["close_odds"].abs(),
        ),
        0.0,
    )
    selected["edge_bucket"] = pd.cut(
        selected["kelly_edge"],
        bins=[-1e-12, 0.03, 0.05, 0.10, np.inf],
        labels=BUCKETS,
        right=False,
        ordered=True,
    )
    if selected["edge_bucket"].isna().any():
        bad = selected.loc[selected["edge_bucket"].isna(), "kelly_edge"]
        raise AssertionError(f"Unbucketed Kelly edges: {bad.describe().to_dict()}")
    selected["sealed_gate_eligible"] = selected["kelly_edge"] >= 0.03
    return selected


def summarize(bets: pd.DataFrame) -> pd.DataFrame:
    summary = (
        bets.groupby(
            ["price_snapshot", "challenger_name", "arm", "edge_bucket"],
            observed=True,
            sort=False,
        )
        .agg(
            bet_count=("pnl", "size"),
            event_count=("event_id", "nunique"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
            mean_theoretical_edge=("kelly_edge", "mean"),
            median_theoretical_edge=("kelly_edge", "median"),
            win_count=("settlement_status", lambda values: int((values == "win").sum())),
            loss_count=("settlement_status", lambda values: int((values == "loss").sum())),
            void_count=("settlement_status", lambda values: int((values == "void").sum())),
        )
        .reset_index()
    )
    summary["roi"] = summary["pnl"] / summary["stake"]
    summary["calibration_gap"] = summary["roi"] - summary["mean_theoretical_edge"]
    summary["edge_bucket"] = summary["edge_bucket"].astype(str)
    return summary


def summarize_gates(bets: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for gate_name, minimum_edge in (("positive_edge", 0.0), ("sealed_3pct", 0.03)):
        selected = bets[bets["kelly_edge"] >= minimum_edge]
        summary = (
            selected.groupby(
                ["price_snapshot", "challenger_name", "arm"],
                observed=True,
                sort=False,
            )
            .agg(
                bet_count=("pnl", "size"),
                event_count=("event_id", "nunique"),
                stake=("stake", "sum"),
                pnl=("pnl", "sum"),
            )
            .reset_index()
        )
        summary["roi"] = summary["pnl"] / summary["stake"]
        summary["gate_name"] = gate_name
        summary["minimum_edge"] = minimum_edge
        outputs.append(summary)
    return pd.concat(outputs, ignore_index=True, sort=False)


def add_zero_clusters(
    bets: pd.DataFrame, event_universe: pd.DataFrame
) -> pd.DataFrame:
    present = set(zip(bets["event_id"].astype(str), bets["arm"].astype(str)))
    rows: list[dict[str, object]] = []
    for event_id, event_start in event_universe.itertuples(index=False, name=None):
        for arm in ("baseline", "challenger"):
            if (str(event_id), arm) not in present:
                rows.append(
                    {
                        "event_id": event_id,
                        "event_start": event_start,
                        "arm": arm,
                        "stake": 0.0,
                        "pnl": 0.0,
                    }
                )
    if rows:
        return pd.concat([bets, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return bets


def bootstrap_buckets(
    bets: pd.DataFrame, event_universe: pd.DataFrame, draws: int = 10_000
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    group_cols = ["price_snapshot", "challenger_name", "edge_bucket"]
    for index, (key, group) in enumerate(
        bets.groupby(group_cols, observed=True, sort=False)
    ):
        snapshot, challenger_name, bucket = key
        bootstrap_input = add_zero_clusters(group, event_universe)
        result = event_cluster_bootstrap(
            bootstrap_input,
            event_col="event_id",
            event_order_col="event_start",
            draws=draws,
            seed=20260822 + index,
        )
        result["price_snapshot"] = snapshot
        result["challenger_name"] = challenger_name
        result["edge_bucket"] = str(bucket)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True, sort=False)


def bootstrap_gates(
    bets: pd.DataFrame, event_universe: pd.DataFrame, draws: int = 10_000
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    index = 0
    for gate_name, minimum_edge in (("positive_edge", 0.0), ("sealed_3pct", 0.03)):
        selected = bets[bets["kelly_edge"] >= minimum_edge]
        for (snapshot, challenger_name), group in selected.groupby(
            ["price_snapshot", "challenger_name"], observed=True, sort=False
        ):
            bootstrap_input = add_zero_clusters(group, event_universe)
            result = event_cluster_bootstrap(
                bootstrap_input,
                event_col="event_id",
                event_order_col="event_start",
                draws=draws,
                seed=20260822 + index,
            )
            result["price_snapshot"] = snapshot
            result["challenger_name"] = challenger_name
            result["gate_name"] = gate_name
            result["minimum_edge"] = minimum_edge
            outputs.append(result)
            index += 1
    return pd.concat(outputs, ignore_index=True, sort=False)


def line_coverage(
    lines_by_snapshot: dict[str, pd.DataFrame], probability_keys: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot, lines in lines_by_snapshot.items():
        joined = lines.merge(
            probability_keys.assign(has_probability=True),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        rows.append(
            {
                "price_snapshot": snapshot,
                "logical_sides": len(lines),
                "events": lines["event_id"].nunique(),
                "sides_with_frozen_probability": int(
                    joined["has_probability"].fillna(False).sum()
                ),
                "missing_probability_sides": int(joined["has_probability"].isna().sum()),
                "selected_betcris": int(lines["book"].eq("betcris").sum()),
                "selected_betonline": int(lines["book"].eq("betonline").sum()),
                "selected_pinnacle": int(lines["book"].eq("pinnacle").sum()),
            }
        )
    return pd.DataFrame(rows)


def format_cell(row: pd.Series) -> str:
    return (
        f"{int(row['bet_count']):,} / {row['pnl']:+.2f}u / "
        f"{100.0 * row['roi']:+.2f}%"
    )


def markdown_table(summary: pd.DataFrame, snapshot: str) -> str:
    data = summary[summary["price_snapshot"].eq(snapshot)].copy()
    lines = [
        "| Kelly edge | Prod baseline | Capped adjustment | Uncapped adjustment |",
        "|---|---:|---:|---:|",
    ]
    arm_map = {
        ("walk_forward_cap015", "baseline"): "baseline",
        ("walk_forward_cap015", "challenger"): "capped",
        ("walk_forward_uncapped", "challenger"): "uncapped",
    }
    for bucket in BUCKETS:
        cells: dict[str, str] = {}
        for (variant, arm), label in arm_map.items():
            match = data[
                data["challenger_name"].eq(variant)
                & data["arm"].eq(arm)
                & data["edge_bucket"].eq(bucket)
            ]
            cells[label] = "--" if match.empty else format_cell(match.iloc[0])
        lines.append(
            f"| {bucket} | {cells['baseline']} | {cells['capped']} | {cells['uncapped']} |"
        )
    return "\n".join(lines)


def coverage_markdown(coverage: pd.DataFrame) -> str:
    columns = list(coverage.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in coverage.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gate_summary: pd.DataFrame,
    gate_bootstrap: pd.DataFrame,
    coverage: pd.DataFrame,
    close_reproduction_error: float,
    same_best_book: int,
) -> str:
    delta = bootstrap[
        bootstrap["arm"].eq("challenger_minus_baseline")
        & bootstrap["metric"].eq("pnl")
        & bootstrap["challenger_name"].eq("walk_forward_cap015")
    ].copy()
    delta_lines = []
    for snapshot in ("close", "open"):
        for bucket in BUCKETS:
            row = delta[
                delta["price_snapshot"].eq(snapshot)
                & delta["edge_bucket"].eq(bucket)
            ]
            if row.empty:
                continue
            value = row.iloc[0]
            delta_lines.append(
                f"- {snapshot} {bucket}: {value['estimate']:+.2f}u; 95% CI "
                f"[{value['ci_low']:+.2f}, {value['ci_high']:+.2f}]"
            )
    overall_lines = [
        "| Snapshot | Model | Bets | Stake | P&L | ROI | Delta vs prod (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    sealed = gate_summary[gate_summary["gate_name"].eq("sealed_3pct")]
    sealed_boot = gate_bootstrap[
        gate_bootstrap["gate_name"].eq("sealed_3pct")
        & gate_bootstrap["arm"].eq("challenger_minus_baseline")
        & gate_bootstrap["metric"].eq("pnl")
    ]
    model_rows = (
        ("baseline", "walk_forward_cap015", "baseline", "Prod baseline"),
        ("capped", "walk_forward_cap015", "challenger", "Capped adjustment"),
        ("uncapped", "walk_forward_uncapped", "challenger", "Uncapped adjustment"),
    )
    for snapshot in ("close", "open"):
        for model_key, variant, arm, label in model_rows:
            row = sealed[
                sealed["price_snapshot"].eq(snapshot)
                & sealed["challenger_name"].eq(variant)
                & sealed["arm"].eq(arm)
            ].iloc[0]
            if model_key == "baseline":
                delta_text = "--"
            else:
                delta = sealed_boot[
                    sealed_boot["price_snapshot"].eq(snapshot)
                    & sealed_boot["challenger_name"].eq(variant)
                ].iloc[0]
                delta_text = (
                    f"{delta['estimate']:+.2f}u "
                    f"[{delta['ci_low']:+.2f}, {delta['ci_high']:+.2f}]"
                )
            overall_lines.append(
                f"| {snapshot} | {label} | {int(row['bet_count']):,} | "
                f"{row['stake']:.2f}u | {row['pnl']:+.2f}u | "
                f"{100.0 * row['roi']:+.2f}% | {delta_text} |"
            )
    return f"""# Matchup Kelly-Edge Buckets at Open and Close

This audit covers only pre-tournament 72-hole matchups. It uses the frozen v4
probabilities and the best archived price among Betcris, Pinnacle, and
BetOnline separately at DataGolf's open and close.

"Kelly edge" follows production terminology: expected return per unit risked,
including matchup push probability. The 0-3% rows are a counterfactual lowering
of the sealed 3% matchup gate; they were not bets in the declared v4 result.

- Events: 105
- Bootstrap: 10,000 whole-event draws per bucket
- Close reproduction maximum odds error: {close_reproduction_error:.12g}
- Same book supplied best open and best close: {same_best_book:,} logical sides
- Line coverage:\n{coverage_markdown(coverage)}

Each cell below is `bets / realized P&L / realized ROI` with production-style
matchup stakes (risk to win 1u at negative odds, otherwise risk 1u).

## Sealed 3% gate overall

{chr(10).join(overall_lines)}

## Closing prices

{markdown_table(summary, 'close')}

## Opening prices

{markdown_table(summary, 'open')}

## Capped-minus-production bucket deltas

{chr(10).join(delta_lines)}

## Interpretation constraints

- Bucket membership is arm-specific because each model supplies its own edge.
  The incremental rows therefore compare model portfolios, not identical bets.
- Best-of-open and best-of-close use archived snapshots. They do not establish
  simultaneous availability, limits, latency, or fillability across books.
- Several bucket cuts were inspected, so subgroup results are descriptive even
  with whole-event bootstrap intervals.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    probabilities = pd.read_parquet(V4 / "probabilities.parquet")
    probabilities = probabilities[
        probabilities["market"].eq("tournament_matchup")
    ].copy()
    event_keys = set(probabilities["event_id"].astype(str))
    event_starts = (
        probabilities[["event_id", "event_start"]]
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )
    event_universe = event_starts.sort_values("event_start").reset_index(drop=True)
    baseline_probabilities = probabilities[probabilities["arm"].eq("baseline")]
    probability_keys = baseline_probabilities[KEYS].drop_duplicates()

    reader = HistoricalOddsCacheReader(CACHE)
    raw_rows = reader.matchups()
    lines_by_snapshot = {
        snapshot: select_best_lines(
            raw_rows, snapshot, event_keys, event_starts
        )
        for snapshot in ("close", "open")
    }

    prior_close = pd.read_parquet(
        V4 / "matchup_skill_bucket_audit" / "best_price_matchup_line_audit.parquet"
    )
    close_check = lines_by_snapshot["close"].merge(
        prior_close[KEYS + ["book", "close_odds"]].rename(
            columns={"book": "prior_book", "close_odds": "prior_odds"}
        ),
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not close_check["_merge"].eq("both").all():
        raise AssertionError("Closing line menu does not reproduce the sealed audit")
    if not close_check["book"].eq(close_check["prior_book"]).all():
        raise AssertionError("Closing selected books do not reproduce the sealed audit")
    close_reproduction_error = float(
        (close_check["close_odds"] - close_check["prior_odds"]).abs().max()
    )
    if close_reproduction_error != 0.0:
        raise AssertionError("Closing selected prices do not reproduce the sealed audit")

    open_close = lines_by_snapshot["open"][KEYS + ["book"]].merge(
        lines_by_snapshot["close"][KEYS + ["book"]],
        on=KEYS,
        how="inner",
        suffixes=("_open", "_close"),
        validate="one_to_one",
    )
    same_best_book = int(
        open_close["book_open"].eq(open_close["book_close"]).sum()
    )

    bet_outputs: list[pd.DataFrame] = []
    for snapshot, lines in lines_by_snapshot.items():
        for variant in VARIANTS:
            challenger_probabilities = probabilities[
                probabilities["arm"].eq(variant)
            ]
            baseline = production_matchup_bets(
                lines, baseline_probabilities, "baseline"
            )
            challenger = production_matchup_bets(
                lines, challenger_probabilities, "challenger"
            )
            for frame in (baseline, challenger):
                frame["price_snapshot"] = snapshot
                frame["challenger_name"] = variant
            bet_outputs.extend([baseline, challenger])

    bets = pd.concat(bet_outputs, ignore_index=True, sort=False)
    summary = summarize(bets)
    bootstrap = bootstrap_buckets(bets, event_universe)
    gate_summary = summarize_gates(bets)
    gate_bootstrap = bootstrap_gates(bets, event_universe)
    coverage = line_coverage(lines_by_snapshot, probability_keys)

    bets.to_parquet(OUT / "matchup_positive_edge_bets.parquet", index=False)
    summary.to_csv(OUT / "kelly_edge_bucket_performance.csv", index=False)
    bootstrap.to_csv(OUT / "kelly_edge_bucket_bootstrap.csv", index=False)
    gate_summary.to_csv(OUT / "kelly_edge_gate_performance.csv", index=False)
    gate_bootstrap.to_csv(OUT / "kelly_edge_gate_bootstrap.csv", index=False)
    coverage.to_csv(OUT / "line_coverage.csv", index=False)
    report = build_report(
        summary,
        bootstrap,
        gate_summary,
        gate_bootstrap,
        coverage,
        close_reproduction_error,
        same_best_book,
    )
    (OUT / "MATCHUP_KELLY_EDGE_OPEN_CLOSE.md").write_text(
        report, encoding="utf-8"
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit best-of-three opening-price matchup performance for the v4 backtest.

The expensive simulations are not rerun.  This script reuses the frozen v4
pre-event probabilities, rebuilds the 72-hole matchup board from the verified
DataGolf cache, shops Betcris/Pinnacle/BetOnline independently at OPEN and at
CLOSE, and settles every non-negative-model-EV side.  The 0-3% bucket is thus
an explicit counterfactual below the production backtest's 3% matchup gate;
the remaining buckets agree with that gate.

"Kelly edge" here means expected profit per unit risked (``ev_per_unit``), not
the full-Kelly bankroll fraction and not raw probability-point edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
for module_path in (str(REPO), str(ARCHIVE)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import backtest_shot_mean_closing_markets as bt  # noqa: E402
from market_backtest_scoring import (  # noqa: E402
    GateConfig,
    LinePolicy,
    SizingConfig,
    select_arm_bets,
    select_closing_lines,
    settle_bets,
)


DEFAULT_BACKTEST = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v4"
)
DEFAULT_OUTPUT = DEFAULT_BACKTEST / "matchup_open_price_audit"
BOOKS = ("betcris", "pinnacle", "betonline")
KEYS = ["event_id", "market", "selection_id", "opponent_id"]
BUCKET_ORDER = ("0-<3%", "3-<5%", "5-10%", ">10%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backtest-dir", type=Path, default=DEFAULT_BACKTEST)
    parser.add_argument("--odds-cache", type=Path, default=bt.DEFAULT_ODDS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_matchup_prices(rows: list[dict[str, object]]) -> pd.DataFrame:
    raw = bt.add_event_key(pd.DataFrame(rows))
    if raw.empty:
        return pd.DataFrame()
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
                    "open_odds": row.get(f"{side}_open"),
                    "close_odds": row.get(f"{side}_close"),
                    "open_time": row.get("open_time"),
                    "close_time": row.get("close_time"),
                    "tie_rule": row.get("tie_rule"),
                    "outcome": row.get(
                        f"{side}_outcome_text", row.get(f"{side}_outcome")
                    ),
                    "outcome_numeric": row.get(f"{side}_outcome"),
                    "record_id": row.get("record_id"),
                }
            )
    frame = pd.DataFrame(expanded)
    valid_identity = frame["selection_id"].ne("") & frame["opponent_id"].ne("")
    frame = frame.loc[valid_identity & frame["book"].isin(BOOKS)].copy()
    for column in ("open_odds", "close_odds"):
        if not bt._valid_american(frame[column]).all():
            bad = int((~bt._valid_american(frame[column])).sum())
            raise ValueError(f"{column} has {bad} missing/invalid tournament sides")
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame["outcome_numeric"] = pd.to_numeric(frame["outcome_numeric"], errors="coerce")
    return frame


def select_best_prices(
    raw: pd.DataFrame, event_ids: set[str], price_type: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = raw[raw["event_id"].isin(event_ids)].copy()
    availability = (
        eligible.groupby(KEYS, observed=True)
        .agg(
            available_book_count=("book", "nunique"),
            available_books=("book", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
    )
    if eligible.duplicated(KEYS + ["book"]).any():
        raise ValueError(
            "Duplicate logical side/book rows require a separately audited timestamp policy"
        )

    if price_type == "close":
        selected = select_closing_lines(
            eligible,
            LinePolicy(mode="best_of_sharp", books=BOOKS),
            KEYS,
        )
        selected["source_price"] = selected["close_odds"]
        selected["source_time"] = selected["close_time"]
    elif price_type == "open":
        rank = {book: offset for offset, book in enumerate(BOOKS)}
        selected = eligible.copy()
        selected["_book_rank"] = selected["book"].map(rank).astype(int)
        # Price first, then the same deterministic equal-price book priority as
        # the closing-price sensitivity.  There are zero side/book duplicates
        # in this cache, so no ambiguous "first opener" collapse is required.
        selected = selected.sort_values(
            KEYS + ["open_odds", "_book_rank"],
            ascending=[True] * len(KEYS) + [False, True],
            kind="mergesort",
        ).drop_duplicates(KEYS, keep="first")
        selected = selected.drop(columns="_book_rank").reset_index(drop=True)
        selected["archived_close_odds"] = selected["close_odds"]
        selected["close_odds"] = selected["open_odds"]
        selected["source_price"] = selected["open_odds"]
        selected["source_time"] = selected["open_time"]
        selected["line_policy"] = "best_of_sharp_open"
    else:
        raise ValueError(price_type)

    selected["price_type"] = price_type
    selected = selected.merge(availability, on=KEYS, how="left", validate="one_to_one")
    if selected.duplicated(KEYS).any():
        raise ValueError(f"{price_type} selection did not produce one line per logical side")
    return selected, availability


def edge_bucket(value: float) -> str | None:
    if not np.isfinite(value) or value < 0.0:
        return None
    if value < 0.03:
        return "0-<3%"
    if value < 0.05:
        return "3-<5%"
    if value <= 0.10:
        return "5-10%"
    return ">10%"


def select_positive_ev_bets(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
    model_arm: str,
    comparison_arm: str,
) -> pd.DataFrame:
    model = probabilities[probabilities["arm"].eq(model_arm)].copy()
    selected = select_arm_bets(
        lines,
        model,
        comparison_arm,
        GateConfig(
            min_ev=0.0,
            allowed_markets=("tournament_matchup",),
        ),
        SizingConfig(method="flat", flat_stake=1.0),
        KEYS,
    )
    selected["model_arm"] = model_arm
    selected["comparison_arm"] = comparison_arm
    selected["kelly_edge_bucket"] = pd.Categorical(
        selected["ev_per_unit"].map(edge_bucket),
        categories=BUCKET_ORDER,
        ordered=True,
    )
    if selected["kelly_edge_bucket"].isna().any():
        raise ValueError("Selected non-negative EV bet did not receive a bucket")
    return selected


def production_style(frame: pd.DataFrame) -> pd.DataFrame:
    dropped = [
        "settlement_status",
        "pnl",
        "return_amount",
        "paid_fraction",
        "is_graded",
        "stake",
    ]
    result = frame.drop(columns=[column for column in dropped if column in frame]).copy()
    result["stake"] = np.where(
        result["close_odds"] < 0,
        result["close_odds"].abs() / 100.0,
        1.0,
    )
    return settle_bets(result)


def roi_bootstrap(
    frame: pd.DataFrame,
    event_ids: list[str],
    draws: int,
    seed: int,
) -> tuple[float, float]:
    clustered = (
        frame.groupby("event_id", observed=True)[["stake", "pnl"]]
        .sum()
        .reindex(event_ids, fill_value=0.0)
    )
    values = clustered[["stake", "pnl"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rois = np.empty(draws, dtype=float)
    for start in range(0, draws, 1_000):
        count = min(1_000, draws - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        totals = values[indices].sum(axis=1)
        rois[start : start + count] = np.divide(
            totals[:, 1],
            totals[:, 0],
            out=np.full(count, np.nan),
            where=totals[:, 0] > 0,
        )
    return tuple(np.nanquantile(rois, [0.025, 0.975]))


def summarize(
    all_bets: pd.DataFrame,
    event_ids: list[str],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouping = ["price_type", "challenger_name", "comparison_arm", "kelly_edge_bucket"]
    for group_key, raw_group in all_bets.groupby(grouping, observed=True, sort=False):
        price_type, challenger, comparison_arm, bucket = group_key
        for sizing_name, group in (
            ("flat_1u", raw_group),
            ("production_style", production_style(raw_group)),
        ):
            stake = float(group["stake"].sum())
            pnl = float(group["pnl"].sum())
            expected_pnl = float((group["stake"] * group["ev_per_unit"]).sum())
            token = f"{price_type}|{challenger}|{comparison_arm}|{bucket}|{sizing_name}"
            token_seed = seed + int.from_bytes(
                hashlib.sha256(token.encode("utf-8")).digest()[:4], "big"
            )
            roi_low, roi_high = roi_bootstrap(group, event_ids, draws, token_seed)
            rows.append(
                {
                    "price_type": price_type,
                    "challenger_name": challenger,
                    "arm": comparison_arm,
                    "kelly_edge_bucket": str(bucket),
                    "sizing_name": sizing_name,
                    "bet_count": int(len(group)),
                    "bet_event_count": int(group["event_id"].nunique()),
                    "wins": int(group["settlement_status"].eq("win").sum()),
                    "losses": int(group["settlement_status"].isin(["loss", "tie_loss"]).sum()),
                    "pushes": int(group["settlement_status"].eq("push").sum()),
                    "voids": int(group["settlement_status"].eq("void").sum()),
                    "ungraded": int(group["settlement_status"].eq("ungraded").sum()),
                    "stake": stake,
                    "pnl": pnl,
                    "roi": pnl / stake if stake else np.nan,
                    "roi_ci_low": roi_low,
                    "roi_ci_high": roi_high,
                    "mean_kelly_edge": float(group["ev_per_unit"].mean()),
                    "stake_weighted_expected_roi": expected_pnl / stake if stake else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate_gate_summary(all_bets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, raw_group in all_bets.groupby(
        ["price_type", "challenger_name", "comparison_arm"],
        observed=True,
        sort=False,
    ):
        price_type, challenger, comparison_arm = key
        for opportunity_set, eligible in (
            ("all_positive_ev", raw_group),
            ("sealed_gate_ge_3pct", raw_group[raw_group["ev_per_unit"].ge(0.03)]),
        ):
            for sizing_name, group in (
                ("flat_1u", eligible),
                ("production_style", production_style(eligible)),
            ):
                stake = float(group["stake"].sum())
                pnl = float(group["pnl"].sum())
                rows.append(
                    {
                        "price_type": price_type,
                        "challenger_name": challenger,
                        "arm": comparison_arm,
                        "opportunity_set": opportunity_set,
                        "sizing_name": sizing_name,
                        "bet_count": int(len(group)),
                        "event_count": int(group["event_id"].nunique()),
                        "stake": stake,
                        "pnl": pnl,
                        "roi": pnl / stake if stake else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _pct(value: object) -> str:
    return f"{float(value):+.2%}"


def _units(value: object) -> str:
    return f"{float(value):+.2f}u"


def build_report(
    output_dir: Path,
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    line_summary: pd.DataFrame,
    audit: dict[str, object],
) -> None:
    production = summary[summary["sizing_name"].eq("production_style")].copy()
    # Baseline repeats exactly for each challenger scenario; retain one copy.
    production = production[
        production["arm"].eq("challenger")
        | production["challenger_name"].eq("walk_forward_cap015")
    ].copy()
    production["model"] = np.where(
        production["arm"].eq("baseline"),
        "baseline",
        production["challenger_name"].map(
            {
                "walk_forward_cap015": "capped",
                "walk_forward_uncapped": "uncapped",
            }
        ),
    )
    aggregate_production = aggregate[aggregate["sizing_name"].eq("production_style")].copy()
    aggregate_production = aggregate_production[
        aggregate_production["arm"].eq("challenger")
        | aggregate_production["challenger_name"].eq("walk_forward_cap015")
    ].copy()
    aggregate_production["model"] = np.where(
        aggregate_production["arm"].eq("baseline"),
        "baseline",
        aggregate_production["challenger_name"].map(
            {
                "walk_forward_cap015": "capped",
                "walk_forward_uncapped": "uncapped",
            }
        ),
    )

    lines = [
        "# Matchup Kelly-edge audit: best open versus best close",
        "",
        "This post-run audit reuses the frozen v4 tournament probabilities and shops only Betcris, Pinnacle, and BetOnline. Kelly edge means expected profit per unit risked (`ev_per_unit`), not probability-point edge or full-Kelly bankroll fraction.",
        "",
        "The production-style matchup convention is used below: risk enough to win 1u at negative American odds and risk 1u at positive odds.",
        "",
        "## Results by theoretical edge",
        "",
    ]
    for price_type in ("close", "open"):
        lines.extend(
            [
                f"### Best {price_type}",
                "",
                "| Model | Edge | Bets | Stake | P&L | ROI | Event-bootstrap 95% ROI interval | Model expected ROI |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        table = production[production["price_type"].eq(price_type)].copy()
        table["_model_order"] = table["model"].map(
            {"baseline": 0, "capped": 1, "uncapped": 2}
        )
        table["_bucket_order"] = table["kelly_edge_bucket"].map(
            {value: offset for offset, value in enumerate(BUCKET_ORDER)}
        )
        table = table.sort_values(["_model_order", "_bucket_order"], kind="mergesort")
        for row in table.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.kelly_edge_bucket} | {int(row.bet_count):,} | "
                f"{row.stake:,.2f}u | {_units(row.pnl)} | {_pct(row.roi)} | "
                f"[{_pct(row.roi_ci_low)}, {_pct(row.roi_ci_high)}] | "
                f"{_pct(row.stake_weighted_expected_roi)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Sealed 3% gate versus all positive EV",
            "",
            "The original v4 matchup test sealed a 3% minimum EV gate. The `0-<3%` rows above are therefore an explicitly labeled diagnostic extension; they were not members of the original wager set.",
            "",
            "| Price | Model | Opportunity set | Bets | Stake | P&L | ROI |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    aggregate_production["_price_order"] = aggregate_production["price_type"].map(
        {"close": 0, "open": 1}
    )
    aggregate_production["_model_order"] = aggregate_production["model"].map(
        {"baseline": 0, "capped": 1, "uncapped": 2}
    )
    aggregate_production["_set_order"] = aggregate_production["opportunity_set"].map(
        {"sealed_gate_ge_3pct": 0, "all_positive_ev": 1}
    )
    aggregate_production = aggregate_production.sort_values(
        ["_price_order", "_model_order", "_set_order"], kind="mergesort"
    )
    for row in aggregate_production.itertuples(index=False):
        label = (
            "sealed >=3%" if row.opportunity_set == "sealed_gate_ge_3pct" else "all >=0%"
        )
        lines.append(
            f"| {row.price_type} | {row.model} | {label} | {int(row.bet_count):,} | "
            f"{row.stake:,.2f}u | {_units(row.pnl)} | {_pct(row.roi)} |"
        )

    book_totals = (
        line_summary.groupby(["price_type", "book"], observed=True)["logical_side_count"]
        .sum()
        .reset_index()
        .sort_values(["price_type", "logical_side_count"], ascending=[True, False])
    )
    lines.extend(
        [
            "",
            "## Price-board integrity",
            "",
            f"- Events: {int(audit['events'])}",
            f"- Logical directional matchup sides: {int(audit['logical_sides_by_price_type']['open']):,} at open and close",
            f"- Missing opening timestamps: {int(audit['timing_summary']['open_timestamp_missing'])}",
            f"- Duplicate logical side/book rows: {int(audit['duplicate_logical_side_book_rows'])}",
            f"- Best open price versus best close on the same side: open better {int(audit['open_close_price_movement']['open_better'])} ({float(audit['open_close_price_movement']['open_better_fraction']):.1%}), same {int(audit['open_close_price_movement']['same'])}, close better {int(audit['open_close_price_movement']['close_better'])}",
            f"- Mean implied-price move from open to close: {float(audit['open_close_price_movement']['mean_implied_probability_close_minus_open']):+.3%}",
            "- Best-price book counts:",
            "",
        ]
    )
    for row in book_totals.itertuples(index=False):
        lines.append(
            f"  - {row.price_type}: {row.book} {int(row.logical_side_count):,}"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Opening prices are each book's archived opener, not a synchronized screen. The three books opened at different timestamps, so `best open` does not prove those quotes coexisted.",
            "- The frozen v4 probability is the reconstructed final pre-event model state. The archive does not contain an as-of snapshot proving that exact model was available when each book opened. Open-price results are therefore a price-timing sensitivity, not a fully executable point-in-time opening-line backtest.",
            "- The reconstructed production coefficients were fit retrospectively on the full historical sample, as documented by the parent v4 audit. Absolute ROI should not be interpreted as a clean point-in-time out-of-sample claim.",
            "- The raw cache has no duplicate side/book rows, so there was no hidden latest-versus-earliest dedup decision. Equal prices use deterministic priority Betcris, then Pinnacle, then BetOnline.",
            "- Full row-level and flat-1u results are retained in the supporting artifacts.",
            "",
        ]
    )
    (output_dir / "MATCHUP_OPEN_PRICE_KELLY_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def validate_sealed_close_against_v4(
    all_bets: pd.DataFrame, backtest_dir: Path
) -> dict[str, object]:
    archived = pd.read_parquet(backtest_dir / "bets.parquet")
    archived = archived[
        archived["line_policy_name"].eq("best_of_sharp")
        & archived["sizing_name"].eq("production_style")
        & archived["market"].eq("tournament_matchup")
    ].copy()
    rows: list[dict[str, object]] = []
    compare_columns = KEYS + ["book", "close_odds", "stake", "pnl", "settlement_status"]
    for challenger in ("walk_forward_uncapped", "walk_forward_cap015"):
        for arm in ("baseline", "challenger"):
            rebuilt = all_bets[
                all_bets["price_type"].eq("close")
                & all_bets["challenger_name"].eq(challenger)
                & all_bets["comparison_arm"].eq(arm)
                & all_bets["ev_per_unit"].ge(0.03)
            ].copy()
            rebuilt = production_style(rebuilt)
            reference = archived[
                archived["challenger_name"].eq(challenger)
                & archived["arm"].eq(arm)
            ].copy()
            left = rebuilt[compare_columns].sort_values(KEYS, kind="mergesort").reset_index(drop=True)
            right = reference[compare_columns].sort_values(KEYS, kind="mergesort").reset_index(drop=True)
            if left[KEYS].to_dict("records") != right[KEYS].to_dict("records"):
                raise ValueError(f"Sealed close key mismatch for {challenger}/{arm}")
            for column in ("book", "settlement_status"):
                if not left[column].equals(right[column]):
                    raise ValueError(
                        f"Sealed close {column} mismatch for {challenger}/{arm}"
                    )
            numeric_error = float(
                np.nanmax(
                    np.abs(
                        left[["close_odds", "stake", "pnl"]].to_numpy(dtype=float)
                        - right[["close_odds", "stake", "pnl"]].to_numpy(dtype=float)
                    )
                )
            )
            if numeric_error > 1e-12:
                raise ValueError(
                    f"Sealed close numeric mismatch for {challenger}/{arm}: {numeric_error}"
                )
            rows.append(
                {
                    "challenger_name": challenger,
                    "arm": arm,
                    "bet_count": len(left),
                    "max_numeric_error": numeric_error,
                }
            )
    return {
        "passed": True,
        "scenarios": rows,
        "comparison": "keys, book, close odds, stake, pnl, settlement status",
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    manifest_path = args.backtest_dir / "manifest.json"
    probabilities_path = args.backtest_dir / "probabilities.parquet"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_ids = [str(value) for value in run_manifest["event_keys"]]
    event_id_set = set(event_ids)

    probabilities = pd.read_parquet(probabilities_path)
    probabilities = probabilities[
        probabilities["event_id"].isin(event_id_set)
        & probabilities["market"].eq("tournament_matchup")
    ].copy()
    expected_arms = {"baseline", "walk_forward_uncapped", "walk_forward_cap015"}
    if set(probabilities["arm"]) != expected_arms:
        raise ValueError(f"Unexpected probability arms: {sorted(set(probabilities['arm']))}")
    if probabilities.duplicated(KEYS + ["arm"]).any():
        raise ValueError("Frozen probabilities contain duplicate matchup keys")

    reader = bt.HistoricalOddsCacheReader(args.odds_cache)
    raw = normalize_matchup_prices(reader.matchups())
    duplicate_count = int(raw[raw["event_id"].isin(event_id_set)].duplicated(KEYS + ["book"]).sum())

    selected_price_sets: list[pd.DataFrame] = []
    availability_rows: list[pd.DataFrame] = []
    for price_type in ("close", "open"):
        chosen, availability = select_best_prices(raw, event_id_set, price_type)
        selected_price_sets.append(chosen)
        availability = availability.copy()
        availability["price_type"] = price_type
        availability_rows.append(availability)
    selected_lines = pd.concat(selected_price_sets, ignore_index=True, sort=False)
    availability = pd.concat(availability_rows, ignore_index=True, sort=False)

    event_dates = (
        probabilities[["event_id", "event_start"]]
        .drop_duplicates("event_id")
    )
    selected_lines = selected_lines.merge(
        event_dates,
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    if selected_lines["event_start"].isna().any():
        raise ValueError("Selected price row is missing event_start")

    bet_pieces: list[pd.DataFrame] = []
    for price_type, price_lines in selected_lines.groupby("price_type", sort=False):
        for challenger in ("walk_forward_uncapped", "walk_forward_cap015"):
            baseline = select_positive_ev_bets(
                price_lines,
                probabilities,
                "baseline",
                "baseline",
            )
            challenger_bets = select_positive_ev_bets(
                price_lines,
                probabilities,
                challenger,
                "challenger",
            )
            for frame in (baseline, challenger_bets):
                frame["challenger_name"] = challenger
                frame["price_type"] = price_type
            bet_pieces.extend([baseline, challenger_bets])
    all_bets = pd.concat(bet_pieces, ignore_index=True, sort=False)
    summary = summarize(all_bets, event_ids, args.bootstrap, args.seed)
    aggregate = aggregate_gate_summary(all_bets)
    sealed_close_validation = validate_sealed_close_against_v4(
        all_bets, args.backtest_dir
    )

    line_summary = (
        selected_lines.groupby(["price_type", "available_book_count", "book"], observed=True)
        .size()
        .rename("logical_side_count")
        .reset_index()
    )
    price_movement = selected_lines.pivot(
        index=KEYS,
        columns="price_type",
        values=["source_price", "book"],
    ).reset_index()
    price_movement.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in price_movement.columns
    ]
    open_price = pd.to_numeric(price_movement["source_price_open"], errors="raise")
    close_price = pd.to_numeric(price_movement["source_price_close"], errors="raise")
    open_implied = np.where(
        open_price < 0,
        -open_price / (-open_price + 100.0),
        100.0 / (open_price + 100.0),
    )
    close_implied = np.where(
        close_price < 0,
        -close_price / (-close_price + 100.0),
        100.0 / (close_price + 100.0),
    )
    movement_audit = {
        "open_better": int((open_price > close_price).sum()),
        "same": int((open_price == close_price).sum()),
        "close_better": int((open_price < close_price).sum()),
        "open_better_fraction": float((open_price > close_price).mean()),
        "same_selected_book": int(
            (price_movement["book_open"] == price_movement["book_close"]).sum()
        ),
        "mean_implied_probability_open": float(np.mean(open_implied)),
        "mean_implied_probability_close": float(np.mean(close_implied)),
        "mean_implied_probability_close_minus_open": float(
            np.mean(close_implied - open_implied)
        ),
    }

    open_lines = selected_lines[selected_lines["price_type"].eq("open")].copy()
    open_dates = pd.to_datetime(open_lines["open_time"], errors="coerce").dt.normalize()
    event_dates_normalized = pd.to_datetime(open_lines["event_start"], errors="coerce").dt.normalize()
    timing_summary = {
        "open_timestamp_missing": int(open_dates.isna().sum()),
        "open_calendar_before_event": int((open_dates < event_dates_normalized).sum()),
        "open_calendar_on_event_date": int((open_dates == event_dates_normalized).sum()),
        "open_calendar_after_event_date": int((open_dates > event_dates_normalized).sum()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "matchup_kelly_edge_price_summary.csv", index=False)
    aggregate.to_csv(args.output_dir / "matchup_gate_aggregate.csv", index=False)
    all_bets.to_parquet(args.output_dir / "matchup_positive_ev_bets.parquet", index=False)
    selected_lines.to_parquet(args.output_dir / "best_price_matchup_lines.parquet", index=False)
    availability.to_parquet(args.output_dir / "matchup_book_availability.parquet", index=False)
    line_summary.to_csv(args.output_dir / "best_price_book_summary.csv", index=False)
    price_movement.to_csv(args.output_dir / "open_close_best_price_comparison.csv", index=False)

    audit = {
        "contract": "v4_frozen_probability_best_of_three_open_close_matchup_edge_audit_v1",
        "backtest_manifest_sha256": sha256_file(manifest_path),
        "probabilities_sha256": sha256_file(probabilities_path),
        "odds_cache_manifest_sha256": sha256_file(args.odds_cache / "manifest.json"),
        "books": list(BOOKS),
        "events": len(event_ids),
        "raw_tournament_sides_in_selected_events": int(len(raw[raw["event_id"].isin(event_id_set)])),
        "duplicate_logical_side_book_rows": duplicate_count,
        "logical_sides_by_price_type": selected_lines.groupby("price_type").size().astype(int).to_dict(),
        "positive_ev_bet_rows_including_scenarios": int(len(all_bets)),
        "kelly_edge_definition": "expected profit per unit risked (ev_per_unit)",
        "bucket_policy": {
            "0-<3%": "0 <= EV < 0.03; below the original 3% gate",
            "3-<5%": "0.03 <= EV < 0.05",
            "5-10%": "0.05 <= EV <= 0.10",
            ">10%": "EV > 0.10",
        },
        "open_price_policy": "highest archived book-opening American odds across Betcris/Pinnacle/BetOnline; equal-price tie priority in that order",
        "open_price_execution_caveat": "book open timestamps need not be simultaneous and archived pre-event model inputs have no at-open snapshot timestamp",
        "open_close_price_movement": movement_audit,
        "sealed_close_validation": sealed_close_validation,
        "timing_summary": timing_summary,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    build_report(args.output_dir, summary, aggregate, line_summary, audit)
    print(summary.to_string(index=False))
    print(aggregate.to_string(index=False))
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

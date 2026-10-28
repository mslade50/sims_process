"""Research-only audit of ARG/PUTT concentration in matchup fair probabilities.

The analysis deliberately separates three questions that are easy to conflate:

1. Are players with strong absolute ARG/PUTT histories miscalibrated?
2. Conditional on total category skill, is a short-game-heavy profile
   miscalibrated relative to an opponent?
3. For R2-R4, are scalar live updates that follow short-game-heavy prior-round
   performance miscalibrated?

The primary universe is every offered matchup in the audited 2024-2026 score
authority, collapsed to one physical pair per event/market. Same-book prices
are converted to no-vig probabilities and averaged within pair. All inference
uses whole-event resampling, so three books and repeated pairs do not masquerade
as independent events.

The script is research-only. It does not touch production inputs, dashboards,
bet ledgers, or cloud resources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "permanent_data" / "research"
BASE_SKILL = RESEARCH / "base_skill_rebuild_20200101_20260822"
DEFAULT_SCORE_AUTHORITY = (
    BASE_SKILL
    / "all_round_joint_market_eval_2024_2026_score_authority_v5_authorized_20260823"
    / "scored_evaluation_2024_2026.parquet"
)
DEFAULT_ROUND_PANEL = BASE_SKILL / "player_round_panel.parquet"
DEFAULT_ROUND_STATE = (
    BASE_SKILL
    / "all_round_frozen_raw_predictions_score_authority_v4_20260823"
    / "winner_composed_round_predictions_2024_2026.parquet"
)
DEFAULT_COUNTERFACTUAL = (
    RESEARCH
    / "category_profile_full_board_2026_plus_bmw_50k_final_v6_sealed_walkforward_20260825"
)
DEFAULT_OUTPUT = RESEARCH / "arg_putt_matchup_calibration_2024_2026_20260825_v1"

CATEGORIES = ("ott", "app", "arg", "putt")
ROUND_MARKETS = ("r1", "r2", "r3", "r4")
ALL_MARKETS = ("72h", *ROUND_MARKETS)
PROFILE_SPAN = 50
MIN_CATEGORY_OBS = 12
EPSILON = 1e-9

PROFILE_FEATURES = (
    "profile_short_diff",
    "short_absolute_level",
    "putt_profile_diff",
    "arg_profile_diff",
    "putt_absolute_level",
    "arg_absolute_level",
)
LIVE_FEATURES = (
    "live_short_balance_diff",
    "live_putt_diff",
    "live_arg_diff",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score-authority", type=Path, default=DEFAULT_SCORE_AUTHORITY)
    parser.add_argument("--round-panel", type=Path, default=DEFAULT_ROUND_PANEL)
    parser.add_argument("--round-state", type=Path, default=DEFAULT_ROUND_STATE)
    parser.add_argument("--counterfactual-dir", type=Path, default=DEFAULT_COUNTERFACTUAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def american_implied(values: pd.Series) -> np.ndarray:
    odds = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(odds), np.nan, dtype=float)
    negative = odds < 0.0
    output[negative] = -odds[negative] / (-odds[negative] + 100.0)
    output[~negative] = 100.0 / (odds[~negative] + 100.0)
    return output


def american_profit(values: pd.Series) -> np.ndarray:
    odds = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(odds), np.nan, dtype=float)
    negative = odds < 0.0
    output[negative] = 100.0 / -odds[negative]
    output[~negative] = odds[~negative] / 100.0
    return output


def conditional_probability(win: pd.Series, lose: pd.Series) -> np.ndarray:
    left = pd.to_numeric(win, errors="coerce").to_numpy(float)
    right = pd.to_numeric(lose, errors="coerce").to_numpy(float)
    denominator = left + right
    return np.divide(
        left,
        denominator,
        out=np.full(len(left), np.nan, dtype=float),
        where=denominator > 0.0,
    )


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(values, float), EPSILON, 1.0 - EPSILON)
    return np.log(array / (1.0 - array))


def event_start_from_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str).str.rsplit("|", n=1).str[-1], errors="raise")


def weighted_ema(values: np.ndarray, span: int = PROFILE_SPAN) -> float:
    array = np.clip(np.asarray(values, float), -8.0, 8.0)
    if not len(array):
        return np.nan
    alpha = 2.0 / (float(span) + 1.0)
    weights = (1.0 - alpha) ** np.arange(len(array) - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    return float(weights @ array)


def zscore_by_event(frame: pd.DataFrame, column: str) -> pd.Series:
    group = frame.groupby("event_key", sort=False)[column]
    centered = frame[column] - group.transform("mean")
    scale = group.transform("std").replace(0.0, np.nan)
    return centered / scale


def load_baseline_quotes(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[frame["candidate"].eq("baseline_current_prod")].copy()
    if frame.empty:
        raise ValueError("No baseline_current_prod rows in score authority")
    if frame["quote_id"].duplicated().any():
        raise ValueError("Baseline score authority contains duplicate quote_id values")
    if not set(frame["market"].dropna().unique()).issubset(set(ALL_MARKETS)):
        raise ValueError("Unexpected market in score authority")
    frame["p1_id"] = frame["p1_id"].astype(str)
    frame["p2_id"] = frame["p2_id"].astype(str)
    frame["event_start"] = event_start_from_key(frame["event_key"])
    frame["year"] = frame["event_start"].dt.year.astype(int)
    frame["market_round"] = frame["market"].map(
        {"72h": 0, "r1": 1, "r2": 2, "r3": 3, "r4": 4}
    ).astype(int)
    frame["model_p1"] = conditional_probability(
        frame["p1_probability"], frame["p2_probability"]
    )
    for point in ("open", "close"):
        p1 = american_implied(frame[f"p1_{point}"])
        p2 = american_implied(frame[f"p2_{point}"])
        frame[f"market_p1_{point}"] = p1 / (p1 + p2)
    return frame


def _category_history_lookup(
    round_panel: pd.DataFrame, player_ids: set[int]
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    columns = {
        category: f"dg_sg_{category}_field_adjusted" for category in CATEGORIES
    }
    required = ["dg_id", "round_date", *columns.values()]
    history = round_panel.loc[
        round_panel["dg_id"].isin(player_ids), required
    ].copy()
    history["round_date"] = pd.to_datetime(history["round_date"], errors="coerce")
    history = history.dropna(subset=["round_date", "dg_id"])
    history["dg_id"] = pd.to_numeric(history["dg_id"], errors="raise").astype(int)
    history = history.sort_values(["dg_id", "round_date"], kind="mergesort")
    output: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for player_id, group in history.groupby("dg_id", sort=False):
        player: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for category, column in columns.items():
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna()
            player[category] = (
                group.loc[valid, "round_date"].to_numpy(dtype="datetime64[ns]"),
                values.loc[valid].to_numpy(float),
            )
        output[int(player_id)] = player
    return output


def build_pre_event_profiles(
    quotes: pd.DataFrame, round_panel: pd.DataFrame
) -> pd.DataFrame:
    left = quotes[["event_key", "event_start", "p1_id"]].rename(
        columns={"p1_id": "dg_id_key"}
    )
    right = quotes[["event_key", "event_start", "p2_id"]].rename(
        columns={"p2_id": "dg_id_key"}
    )
    requested = (
        pd.concat([left, right], ignore_index=True)
        .drop_duplicates(["event_key", "dg_id_key"])
        .sort_values(["event_start", "event_key", "dg_id_key"], kind="mergesort")
    )
    requested["dg_id"] = pd.to_numeric(requested["dg_id_key"], errors="raise").astype(int)
    lookup = _category_history_lookup(round_panel, set(requested["dg_id"]))
    rows: list[dict[str, object]] = []
    for event_key, event_start, dg_id_key, dg_id in requested.itertuples(index=False):
        player = lookup.get(int(dg_id), {})
        row: dict[str, object] = {
            "event_key": event_key,
            "event_start": event_start,
            "dg_id_key": str(dg_id_key),
        }
        for category in CATEGORIES:
            dates, values = player.get(
                category,
                (np.array([], dtype="datetime64[ns]"), np.array([], dtype=float)),
            )
            stop = int(
                np.searchsorted(dates, np.datetime64(pd.Timestamp(event_start)), side="left")
            )
            prior = values[:stop]
            row[f"sg_{category}"] = weighted_ema(prior)
            row[f"sg_{category}_n"] = int(len(prior))
        rows.append(row)
    profile = pd.DataFrame(rows)
    count_columns = [f"sg_{category}_n" for category in CATEGORIES]
    value_columns = [f"sg_{category}" for category in CATEGORIES]
    profile["profile_eligible"] = (
        profile[count_columns].min(axis=1).ge(MIN_CATEGORY_OBS)
        & profile[value_columns].notna().all(axis=1)
    )
    profile["category_center"] = profile[value_columns].mean(axis=1)
    for category in CATEGORIES:
        profile[f"{category}_dev"] = (
            profile[f"sg_{category}"] - profile["category_center"]
        )
    profile["short_level"] = profile["sg_arg"] + profile["sg_putt"]
    profile["short_dev"] = profile["arg_dev"] + profile["putt_dev"]
    profile["category_total"] = profile[value_columns].sum(axis=1)
    for column in (
        *[f"sg_{category}" for category in CATEGORIES],
        *[f"{category}_dev" for category in CATEGORIES],
        "short_level",
        "short_dev",
        "category_total",
    ):
        profile[f"{column}_z"] = zscore_by_event(profile, column)
    return profile.sort_values(["event_start", "event_key", "dg_id_key"]).reset_index(drop=True)


def build_live_prior_profiles(
    quotes: pd.DataFrame, round_panel: pd.DataFrame
) -> pd.DataFrame:
    events = set(quotes["event_key"])
    # These are the exact realized category inputs used by the frozen legacy
    # live path. Field-adjusted variants are useful robustness features, but
    # would not faithfully describe the historical baseline being scored.
    category_columns = {category: f"dg_sg_{category}_raw" for category in CATEGORIES}
    columns = ["event_key", "dg_id", "round_num", *category_columns.values()]
    rounds = round_panel.loc[
        round_panel["event_key"].isin(events)
        & round_panel["round_num"].between(1, 3),
        columns,
    ].copy()
    rounds["dg_id_key"] = pd.to_numeric(rounds["dg_id"], errors="raise").astype(int).astype(str)
    if rounds.duplicated(["event_key", "dg_id_key", "round_num"]).any():
        raise ValueError("Round panel is not unique by event/player/round")
    rows: list[dict[str, object]] = []
    for (event_key, dg_id_key), group in rounds.groupby(
        ["event_key", "dg_id_key"], sort=False
    ):
        group = group.sort_values("round_num")
        for target_round in (2, 3, 4):
            prior = group[group["round_num"].lt(target_round)]
            row: dict[str, object] = {
                "event_key": event_key,
                "dg_id_key": dg_id_key,
                "round_num": target_round,
                "prior_rounds": int(prior["round_num"].nunique()),
            }
            for category, column in category_columns.items():
                row[f"live_{category}"] = pd.to_numeric(
                    prior[column], errors="coerce"
                ).mean()
                row[f"live_{category}_count"] = int(
                    pd.to_numeric(prior[column], errors="coerce").notna().sum()
                )
            rows.append(row)
    live = pd.DataFrame(rows)
    expected = live["round_num"] - 1
    count_columns = [f"live_{category}_count" for category in CATEGORIES]
    live["live_complete"] = live[count_columns].eq(expected, axis=0).all(axis=1)
    live["live_short"] = live["live_arg"] + live["live_putt"]
    live["live_long"] = live["live_ott"] + live["live_app"]
    live["live_total"] = live["live_short"] + live["live_long"]
    live["live_short_balance"] = live["live_short"] - live["live_long"]
    return live.sort_values(["event_key", "round_num", "dg_id_key"]).reset_index(drop=True)


def collapse_physical_pairs(quotes: pd.DataFrame) -> pd.DataFrame:
    keys = ["event_key", "market", "pair_key"]
    constants = (
        "p1_id",
        "p2_id",
        "model_p1",
        "settlement_p1_outcome",
        "p1_prediction",
        "p2_prediction",
        "push_probability",
        "event_start",
        "year",
        "market_round",
    )
    for column in constants:
        if (quotes.groupby(keys, sort=False)[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"{column} is not constant within physical pair")
    pair = quotes.groupby(keys, as_index=False, sort=False).agg(
        p1_id=("p1_id", "first"),
        p2_id=("p2_id", "first"),
        model_p1=("model_p1", "first"),
        outcome_p1=("settlement_p1_outcome", "first"),
        p1_prediction=("p1_prediction", "first"),
        p2_prediction=("p2_prediction", "first"),
        push_probability=("push_probability", "first"),
        event_start=("event_start", "first"),
        year=("year", "first"),
        round_num=("market_round", "first"),
        market_p1_open=("market_p1_open", "mean"),
        market_p1_close=("market_p1_close", "mean"),
        books=("book", "nunique"),
        quotes=("quote_id", "size"),
    )
    return pair


def attach_side_frame(
    pair: pd.DataFrame,
    source: pd.DataFrame,
    side: int,
    source_id: str = "dg_id_key",
    extra_keys: Iterable[str] = (),
) -> pd.DataFrame:
    side_id = f"p{side}_id"
    keys = ["event_key", *extra_keys]
    source_columns = [column for column in source.columns if column not in keys]
    renamed = source.rename(
        columns={
            source_id: side_id,
            **{
                column: f"p{side}_{column}"
                for column in source_columns
                if column != source_id
            },
        }
    )
    join_keys = [*keys, side_id]
    keep = [*join_keys, *[column for column in renamed if column.startswith(f"p{side}_") and column != side_id]]
    return pair.merge(
        renamed[keep],
        on=join_keys,
        how="left",
        validate="many_to_one",
    )


def attach_profiles_and_states(
    pair: pd.DataFrame,
    profile: pd.DataFrame,
    live: pd.DataFrame,
    round_state_path: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    profile = profile.drop(columns=["event_start"], errors="ignore")
    output = pair.copy()
    for side in (1, 2):
        output = attach_side_frame(output, profile, side)
    live_markets = output[output["round_num"].between(2, 4)].copy()
    other = output[~output["round_num"].between(2, 4)].copy()
    for side in (1, 2):
        live_markets = attach_side_frame(
            live_markets, live, side, extra_keys=("round_num",)
        )
    output = pd.concat([other, live_markets], ignore_index=True, sort=False)

    state = pd.read_parquet(round_state_path)
    state = state[
        [
            "event_key",
            "dg_id_key",
            "round_num",
            "current_control_prediction",
            "baseline_r1_prediction",
            "legacy_stage_adjustment",
        ]
    ].copy()
    state["dg_id_key"] = state["dg_id_key"].astype(str)
    if state.duplicated(["event_key", "dg_id_key", "round_num"]).any():
        raise ValueError("Frozen round state is not unique by event/player/round")
    round_rows = output[output["round_num"].between(1, 4)].copy()
    nonround = output[~output["round_num"].between(1, 4)].copy()
    for side in (1, 2):
        round_rows = attach_side_frame(
            round_rows, state, side, extra_keys=("round_num",)
        )
    output = pd.concat([nonround, round_rows], ignore_index=True, sort=False)
    round_mask = output["round_num"].between(1, 4)
    errors: list[float] = []
    for side in (1, 2):
        error = (
            output.loc[round_mask, f"p{side}_current_control_prediction"]
            - output.loc[round_mask, f"p{side}_prediction"]
        ).abs()
        errors.extend(error.dropna().tolist())
    audit = {
        "round_state_prediction_max_abs_error": float(max(errors)) if errors else np.nan,
        "profile_both_sides_coverage": float(
            output[["p1_profile_eligible", "p2_profile_eligible"]]
            .eq(True)
            .all(axis=1)
            .mean()
        ),
        "live_both_sides_coverage_r2_r4": float(
            output.loc[
                output["round_num"].between(2, 4),
                ["p1_live_complete", "p2_live_complete"],
            ]
            .eq(True)
            .all(axis=1)
            .mean()
        ),
    }
    if audit["round_state_prediction_max_abs_error"] > 1e-12:
        raise ValueError("Scored baseline does not match frozen scalar round state")
    return output.sort_values(["event_start", "market", "pair_key"]).reset_index(drop=True), audit


def orient_pairs(pair: pd.DataFrame, price_point: str) -> pd.DataFrame:
    output = pair.copy()
    market_p1 = output[f"market_p1_{price_point}"].to_numpy(float)
    edge = logit(output["model_p1"]) - logit(market_p1)
    sign = np.where(edge >= 0.0, 1.0, -1.0)
    output["price_point"] = price_point
    output["edge_side"] = np.where(sign > 0.0, "p1", "p2")
    output["orientation_sign"] = sign
    output["edge_logit"] = np.abs(edge)
    output["model_probability"] = np.where(
        sign > 0.0, output["model_p1"], 1.0 - output["model_p1"]
    )
    output["market_probability"] = np.where(sign > 0.0, market_p1, 1.0 - market_p1)
    output["outcome"] = np.where(
        sign > 0.0, output["outcome_p1"], 1.0 - output["outcome_p1"]
    )
    output["model_premium"] = output["model_probability"] - output["market_probability"]

    pair_features = {
        "profile_short_diff": "short_dev_z",
        "putt_profile_diff": "putt_dev_z",
        "arg_profile_diff": "arg_dev_z",
        "ott_profile_diff": "ott_dev_z",
        "app_profile_diff": "app_dev_z",
        "category_total_diff": "category_total_z",
    }
    level_features = {
        "short_absolute_level": "short_level_z",
        "putt_absolute_level": "sg_putt_z",
        "arg_absolute_level": "sg_arg_z",
        "ott_absolute_level": "sg_ott_z",
        "app_absolute_level": "sg_app_z",
    }
    for target, source in pair_features.items():
        output[target] = sign * (output[f"p1_{source}"] - output[f"p2_{source}"])
    for target, source in level_features.items():
        output[target] = np.where(sign > 0.0, output[f"p1_{source}"], output[f"p2_{source}"])

    if "p1_live_short_balance" in output:
        output["live_short_balance_diff"] = sign * (
            output["p1_live_short_balance"] - output["p2_live_short_balance"]
        )
        output["live_putt_diff"] = sign * (
            output["p1_live_putt"] - output["p2_live_putt"]
        )
        output["live_arg_diff"] = sign * (
            output["p1_live_arg"] - output["p2_live_arg"]
        )
        output["live_total_diff"] = sign * (
            output["p1_live_total"] - output["p2_live_total"]
        )
    if "p1_legacy_stage_adjustment" in output:
        output["stage_adjustment_diff"] = sign * (
            output["p1_legacy_stage_adjustment"]
            - output["p2_legacy_stage_adjustment"]
        )

    resolved = output["outcome"].isin([0.0, 1.0])
    output["resolved"] = resolved
    y = output["outcome"].to_numpy(float)
    model = np.clip(output["model_probability"].to_numpy(float), EPSILON, 1.0 - EPSILON)
    market = np.clip(output["market_probability"].to_numpy(float), EPSILON, 1.0 - EPSILON)
    output["calibration_gap"] = np.where(resolved, y - model, np.nan)
    output["market_calibration_gap"] = np.where(resolved, y - market, np.nan)
    model_ll = -(y * np.log(model) + (1.0 - y) * np.log(1.0 - model))
    market_ll = -(y * np.log(market) + (1.0 - y) * np.log(1.0 - market))
    output["excess_log_loss"] = np.where(resolved, model_ll - market_ll, np.nan)
    output["excess_brier"] = np.where(
        resolved, (y - model) ** 2 - (y - market) ** 2, np.nan
    )
    return output


def event_equal_summary(group: pd.DataFrame, feature: str) -> pd.Series:
    event = group.groupby("event_key", sort=False).agg(
        pairs=("outcome", "size"),
        feature=(feature, "mean"),
        observed=("outcome", "mean"),
        model=("model_probability", "mean"),
        market=("market_probability", "mean"),
        gap=("calibration_gap", "mean"),
        market_gap=("market_calibration_gap", "mean"),
        excess_ll=("excess_log_loss", "mean"),
        excess_brier=("excess_brier", "mean"),
        premium=("model_premium", "mean"),
    )
    return pd.Series(
        {
            "pairs": int(len(group)),
            "events": int(len(event)),
            "feature_mean": event["feature"].mean(),
            "observed": event["observed"].mean(),
            "model": event["model"].mean(),
            "market_probability": event["market"].mean(),
            "calibration_gap": event["gap"].mean(),
            "market_calibration_gap": event["market_gap"].mean(),
            "excess_log_loss": event["excess_ll"].mean(),
            "excess_brier": event["excess_brier"].mean(),
            "model_premium": event["premium"].mean(),
        }
    )


def period_subsets(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all_2024_2026", frame
    yield "replication_2024_2025", frame[frame["year"].isin([2024, 2025])]
    yield "hypothesis_2026", frame[frame["year"].eq(2026)]


def build_quintile_summary(oriented: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_list = (*PROFILE_FEATURES, *LIVE_FEATURES)
    for price_point in ("close", "open"):
        priced = oriented[oriented["price_point"].eq(price_point)]
        for period, period_frame in period_subsets(priced):
            markets: list[tuple[str, pd.DataFrame]] = [
                (market, period_frame[period_frame["market"].eq(market)])
                for market in ALL_MARKETS
            ]
            markets.append(("rounds_pooled", period_frame[period_frame["market"].isin(ROUND_MARKETS)]))
            for market, market_frame in markets:
                for feature in feature_list:
                    resolved = market_frame[
                        market_frame["resolved"] & market_frame[feature].notna()
                    ].copy()
                    if len(resolved) < 50 or resolved[feature].nunique() < 5:
                        continue
                    resolved["quintile"] = pd.qcut(
                        resolved[feature], 5, labels=False, duplicates="drop"
                    )
                    summary = (
                        resolved.groupby("quintile", sort=True)
                        .apply(event_equal_summary, feature=feature, include_groups=False)
                        .reset_index()
                    )
                    summary.insert(0, "feature", feature)
                    summary.insert(0, "market", market)
                    summary.insert(0, "period", period)
                    summary.insert(0, "price_point", price_point)
                    rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_overall_calibration(oriented: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for price_point in ("close", "open"):
        priced = oriented[oriented["price_point"].eq(price_point)]
        for period, period_frame in period_subsets(priced):
            markets: list[tuple[str, pd.DataFrame]] = [
                (market, period_frame[period_frame["market"].eq(market)])
                for market in ALL_MARKETS
            ]
            markets.append(
                ("rounds_pooled", period_frame[period_frame["market"].isin(ROUND_MARKETS)])
            )
            for market, market_frame in markets:
                resolved = market_frame[market_frame["resolved"]]
                if resolved.empty:
                    continue
                event = resolved.groupby("event_key", sort=False).agg(
                    pairs=("outcome", "size"),
                    observed=("outcome", "mean"),
                    model_probability=("model_probability", "mean"),
                    market_probability=("market_probability", "mean"),
                    calibration_gap=("calibration_gap", "mean"),
                    market_calibration_gap=("market_calibration_gap", "mean"),
                    excess_log_loss=("excess_log_loss", "mean"),
                    excess_brier=("excess_brier", "mean"),
                )
                means = event.drop(columns="pairs").mean().to_dict()
                rows.append(
                    {
                        "price_point": price_point,
                        "period": period,
                        "market": market,
                        "pairs": int(len(resolved)),
                        "events": int(len(event)),
                        **means,
                    }
                )
    return pd.DataFrame(rows)


def _event_cross_products(
    frame: pd.DataFrame,
    outcome: str,
    feature: str,
    add_live_controls: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    columns = ["event_key", "market", outcome, feature, "edge_logit", "market_probability"]
    if add_live_controls:
        columns.extend(["live_total_diff", "stage_adjustment_diff"])
    data = frame[columns].dropna().copy()
    if data.empty:
        raise ValueError("No complete rows for regression")
    event_counts = data.groupby("event_key")["event_key"].transform("size").to_numpy(float)
    weights = 1.0 / event_counts

    def weighted_standardize(values: pd.Series) -> np.ndarray:
        array = values.to_numpy(float)
        mean = float(np.average(array, weights=weights))
        variance = float(np.average((array - mean) ** 2, weights=weights))
        scale = math.sqrt(max(variance, 1e-12))
        return (array - mean) / scale

    design: dict[str, np.ndarray] = {
        "intercept": np.ones(len(data), dtype=float),
        "feature_sd": weighted_standardize(data[feature]),
        "edge_logit_sd": weighted_standardize(data["edge_logit"]),
        "market_logit": logit(data["market_probability"]),
    }
    if data["market"].nunique() > 1:
        for market in ("r2", "r3", "r4"):
            design[f"market_{market}"] = data["market"].eq(market).to_numpy(float)
    if add_live_controls:
        design["live_total_diff_sd"] = weighted_standardize(data["live_total_diff"])
        design["stage_adjustment_diff_sd"] = weighted_standardize(
            data["stage_adjustment_diff"]
        )
    names = list(design)
    matrix = np.column_stack([design[name] for name in names])
    y = data[outcome].to_numpy(float)
    event_names = data["event_key"].drop_duplicates().to_numpy(str)
    a, b = [], []
    for event in event_names:
        mask = data["event_key"].eq(event).to_numpy()
        x_event = matrix[mask]
        w_event = weights[mask]
        y_event = y[mask]
        a.append(x_event.T @ (w_event[:, None] * x_event))
        b.append(x_event.T @ (w_event * y_event))
    return np.asarray(a), np.asarray(b), names, event_names, y


def event_bootstrap_regression(
    frame: pd.DataFrame,
    outcome: str,
    feature: str,
    draws: int,
    seed: int,
    add_live_controls: bool = False,
) -> dict[str, object]:
    a, b, names, events, y = _event_cross_products(
        frame, outcome, feature, add_live_controls
    )
    feature_index = names.index("feature_sd")
    estimate = float(np.linalg.lstsq(a.sum(axis=0), b.sum(axis=0), rcond=None)[0][feature_index])
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    n_events = len(events)
    probabilities = np.full(n_events, 1.0 / n_events, dtype=float)
    batch_size = 1_000
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        counts = rng.multinomial(n_events, probabilities, size=stop - start).astype(float)
        batch_a = np.einsum("be,eij->bij", counts, a, optimize=True)
        batch_b = np.einsum("be,ej->bj", counts, b, optimize=True)
        try:
            coefficients = np.linalg.solve(batch_a, batch_b)
        except np.linalg.LinAlgError:
            coefficients = np.vstack(
                [
                    np.linalg.lstsq(matrix, vector, rcond=None)[0]
                    for matrix, vector in zip(batch_a, batch_b)
                ]
            )
        boot[start:stop] = coefficients[:, feature_index]
    low, high = np.quantile(boot, [0.025, 0.975])
    p_value = min(1.0, 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0))))
    complete = frame[["event_key", outcome, feature]].dropna()
    return {
        "outcome": outcome,
        "feature": feature,
        "estimate_per_sd": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_p": p_value,
        "bootstrap_draws": int(draws),
        "events": int(complete["event_key"].nunique()),
        "pairs": int(len(complete)),
        "controls": "+".join(names[2:]),
    }


def build_regression_summary(
    oriented: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for price_index, price_point in enumerate(("close", "open")):
        priced = oriented[oriented["price_point"].eq(price_point)]
        for period_index, (period, period_frame) in enumerate(period_subsets(priced)):
            markets: list[tuple[str, pd.DataFrame]] = [
                (market, period_frame[period_frame["market"].eq(market)])
                for market in ALL_MARKETS
            ]
            markets.append(("rounds_pooled", period_frame[period_frame["market"].isin(ROUND_MARKETS)]))
            for market_index, (market, market_frame) in enumerate(markets):
                features = list(PROFILE_FEATURES)
                if market in ("r2", "r3", "r4", "rounds_pooled"):
                    features.extend(LIVE_FEATURES)
                for feature_index, feature in enumerate(features):
                    resolved = market_frame[market_frame["resolved"] & market_frame[feature].notna()]
                    if len(resolved) < 100 or resolved["event_key"].nunique() < 10:
                        continue
                    live_controls = feature in LIVE_FEATURES
                    for outcome_index, outcome in enumerate(("calibration_gap", "excess_log_loss")):
                        row_seed = (
                            seed
                            + price_index * 100_000
                            + period_index * 10_000
                            + market_index * 1_000
                            + feature_index * 10
                            + outcome_index
                        )
                        result = event_bootstrap_regression(
                            resolved,
                            outcome,
                            feature,
                            draws,
                            row_seed,
                            add_live_controls=live_controls,
                        )
                        result.update(
                            {
                                "price_point": price_point,
                                "period": period,
                                "market": market,
                            }
                        )
                        rows.append(result)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["holm_p_within_market_outcome"] = summary.groupby(
        ["price_point", "period", "market", "outcome"], sort=False
    )["bootstrap_p"].transform(holm_adjust)
    return summary


def holm_adjust(values: pd.Series) -> pd.Series:
    probabilities = pd.to_numeric(values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(probabilities), np.nan, dtype=float)
    valid = np.isfinite(probabilities)
    if not valid.any():
        return pd.Series(adjusted, index=values.index)
    valid_positions = np.flatnonzero(valid)
    order = valid_positions[np.argsort(probabilities[valid])]
    count = len(order)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (count - rank) * probabilities[position])
        running = max(running, candidate)
        adjusted[position] = running
    return pd.Series(adjusted, index=values.index)


def settle_discovered_bets(quotes: pd.DataFrame, price_point: str, min_ev: float = 0.03) -> pd.DataFrame:
    frame = quotes.copy()
    p1_profit = american_profit(frame[f"p1_{price_point}"])
    p2_profit = american_profit(frame[f"p2_{price_point}"])
    p1_prob = frame["p1_probability"].to_numpy(float)
    p2_prob = frame["p2_probability"].to_numpy(float)
    ev1 = p1_prob * p1_profit - p2_prob
    ev2 = p2_prob * p2_profit - p1_prob
    choose_p1 = ev1 >= ev2
    frame["selected_side"] = np.where(choose_p1, "p1", "p2")
    frame["selection_id"] = np.where(choose_p1, frame["p1_id"], frame["p2_id"])
    frame["opponent_id"] = np.where(choose_p1, frame["p2_id"], frame["p1_id"])
    frame["ev_per_unit"] = np.where(choose_p1, ev1, ev2)
    frame["odds"] = np.where(
        choose_p1, frame[f"p1_{price_point}"], frame[f"p2_{price_point}"]
    )
    frame["profit_if_win"] = np.where(choose_p1, p1_profit, p2_profit)
    frame["selection_outcome"] = np.where(
        choose_p1,
        frame["settlement_p1_outcome"],
        1.0 - frame["settlement_p1_outcome"],
    )
    frame = frame[frame["ev_per_unit"].ge(min_ev)].copy()
    frame = (
        frame.sort_values(
            ["event_key", "market", "pair_key", "ev_per_unit", "book"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["event_key", "market", "pair_key"], keep="first")
        .reset_index(drop=True)
    )
    frame["pnl"] = np.select(
        [frame["selection_outcome"].eq(1.0), frame["selection_outcome"].eq(0.0)],
        [frame["profit_if_win"], -1.0],
        default=0.0,
    )
    frame["price_point"] = price_point
    return frame


def attach_bet_profiles(bets: pd.DataFrame, profile: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_key",
        "dg_id_key",
        "short_dev_z",
        "putt_dev_z",
        "arg_dev_z",
        "short_level_z",
        "sg_putt_z",
        "sg_arg_z",
    ]
    selection = profile[columns].rename(
        columns={
            "dg_id_key": "selection_id",
            **{column: f"selection_{column}" for column in columns[2:]},
        }
    )
    opponent = profile[columns].rename(
        columns={
            "dg_id_key": "opponent_id",
            **{column: f"opponent_{column}" for column in columns[2:]},
        }
    )
    output = bets.merge(selection, on=["event_key", "selection_id"], how="left", validate="many_to_one")
    output = output.merge(opponent, on=["event_key", "opponent_id"], how="left", validate="many_to_one")
    output["profile_short_diff"] = output["selection_short_dev_z"] - output["opponent_short_dev_z"]
    output["putt_profile_diff"] = output["selection_putt_dev_z"] - output["opponent_putt_dev_z"]
    output["arg_profile_diff"] = output["selection_arg_dev_z"] - output["opponent_arg_dev_z"]
    output["short_absolute_level"] = output["selection_short_level_z"]
    output["putt_absolute_level"] = output["selection_sg_putt_z"]
    output["arg_absolute_level"] = output["selection_sg_arg_z"]
    round_bets = output[output["market"].isin(["r2", "r3", "r4"])].copy()
    other = output[~output["market"].isin(["r2", "r3", "r4"])].copy()
    if len(round_bets):
        round_bets["round_num"] = round_bets["market"].str[1].astype(int)
        live_columns = [
            "event_key",
            "dg_id_key",
            "round_num",
            "live_short_balance",
            "live_putt",
            "live_arg",
            "live_total",
        ]
        selection_live = live[live_columns].rename(
            columns={
                "dg_id_key": "selection_id",
                "live_short_balance": "selection_live_short_balance",
                "live_putt": "selection_live_putt",
                "live_arg": "selection_live_arg",
                "live_total": "selection_live_total",
            }
        )
        opponent_live = live[live_columns].rename(
            columns={
                "dg_id_key": "opponent_id",
                "live_short_balance": "opponent_live_short_balance",
                "live_putt": "opponent_live_putt",
                "live_arg": "opponent_live_arg",
                "live_total": "opponent_live_total",
            }
        )
        round_bets = round_bets.merge(
            selection_live,
            on=["event_key", "round_num", "selection_id"],
            how="left",
            validate="many_to_one",
        ).merge(
            opponent_live,
            on=["event_key", "round_num", "opponent_id"],
            how="left",
            validate="many_to_one",
        )
        round_bets["live_short_balance_diff"] = (
            round_bets["selection_live_short_balance"]
            - round_bets["opponent_live_short_balance"]
        )
        round_bets["live_putt_diff"] = (
            round_bets["selection_live_putt"] - round_bets["opponent_live_putt"]
        )
        round_bets["live_arg_diff"] = (
            round_bets["selection_live_arg"] - round_bets["opponent_live_arg"]
        )
    return pd.concat([other, round_bets], ignore_index=True, sort=False)


def build_bet_summary(bets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for price_point in ("close", "open"):
        priced = bets[bets["price_point"].eq(price_point)]
        for period, period_frame in period_subsets(priced):
            markets: list[tuple[str, pd.DataFrame]] = [
                (market, period_frame[period_frame["market"].eq(market)])
                for market in ALL_MARKETS
            ]
            markets.append(("rounds_pooled", period_frame[period_frame["market"].isin(ROUND_MARKETS)]))
            for market, market_frame in markets:
                for feature in (*PROFILE_FEATURES, *LIVE_FEATURES):
                    complete = market_frame[market_frame[feature].notna()].copy()
                    if len(complete) < 25 or complete[feature].nunique() < 5:
                        continue
                    complete["quintile"] = pd.qcut(
                        complete[feature], 5, labels=False, duplicates="drop"
                    )
                    for quintile, group in complete.groupby("quintile", sort=True):
                        rows.append(
                            {
                                "price_point": price_point,
                                "period": period,
                                "market": market,
                                "feature": feature,
                                "quintile": int(quintile),
                                "bets": int(len(group)),
                                "events": int(group["event_key"].nunique()),
                                "mean_feature": float(group[feature].mean()),
                                "mean_ev": float(group["ev_per_unit"].mean()),
                                "pnl": float(group["pnl"].sum()),
                                "roi": float(group["pnl"].mean()),
                            }
                        )
    return pd.DataFrame(rows)


def bootstrap_event_mean(values: pd.Series, draws: int, seed: int) -> dict[str, float]:
    array = values.dropna().to_numpy(float)
    if not len(array):
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    p_value = min(1.0, 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0))))
    return {
        "estimate": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p": p_value,
    }


def counterfactual_score_audit(
    directory: Path, draws: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probabilities = pd.read_parquet(directory / "probabilities.parquet")
    lines = pd.read_parquet(directory / "best_of_3_lines.parquet")
    probabilities = probabilities[
        probabilities["board"].eq("72h")
        & probabilities["market"].eq("tournament_matchup")
    ].copy()
    lines = lines[
        lines["board"].eq("72h")
        & lines["market"].eq("tournament_matchup")
        & lines["price_point"].eq("close")
    ].copy()
    for frame in (probabilities, lines):
        frame["selection_id"] = frame["selection_id"].astype(str)
        frame["opponent_id"] = frame["opponent_id"].astype(str)
        frame["pair_key"] = [
            "|".join(sorted((left, right), key=int))
            for left, right in frame[["selection_id", "opponent_id"]].itertuples(index=False, name=None)
        ]
    canonical_lines = lines[
        lines["selection_id"].astype(int) < lines["opponent_id"].astype(int)
    ][["event_key", "pair_key", "selection_id", "opponent_id", "outcome_numeric"]]
    canonical_lines = canonical_lines.drop_duplicates(["event_key", "pair_key"])
    canonical_probability = probabilities[
        probabilities["selection_id"].astype(int)
        < probabilities["opponent_id"].astype(int)
    ]
    pivot = canonical_probability.pivot(
        index=["event_key", "pair_key", "selection_id", "opponent_id"],
        columns="arm",
        values=["probability", "push_probability"],
    ).reset_index()
    pivot.columns = [
        "_".join((str(left), str(right))).rstrip("_") for left, right in pivot.columns
    ]
    panel = pivot.merge(
        canonical_lines,
        on=["event_key", "pair_key", "selection_id", "opponent_id"],
        validate="one_to_one",
    )
    panel = panel[panel["outcome_numeric"].isin([0.0, 1.0])].copy()
    for arm in ("baseline", "targeted_arg_putt"):
        panel[f"conditional_{arm}"] = panel[f"probability_{arm}"] / (
            1.0 - panel[f"push_probability_{arm}"]
        )
        probability = np.clip(panel[f"conditional_{arm}"], EPSILON, 1.0 - EPSILON)
        outcome = panel["outcome_numeric"]
        panel[f"log_loss_{arm}"] = -(
            outcome * np.log(probability) + (1.0 - outcome) * np.log(1.0 - probability)
        )
        panel[f"brier_{arm}"] = (outcome - probability) ** 2
    panel["targeted_minus_baseline_log_loss"] = (
        panel["log_loss_targeted_arg_putt"] - panel["log_loss_baseline"]
    )
    panel["targeted_minus_baseline_brier"] = (
        panel["brier_targeted_arg_putt"] - panel["brier_baseline"]
    )
    panel["targeted_minus_baseline_probability_pp"] = 100.0 * (
        panel["conditional_targeted_arg_putt"] - panel["conditional_baseline"]
    )
    event = panel.groupby("event_key", as_index=False).agg(
        pairs=("outcome_numeric", "size"),
        log_loss_delta=("targeted_minus_baseline_log_loss", "mean"),
        brier_delta=("targeted_minus_baseline_brier", "mean"),
        mean_probability_delta_pp=("targeted_minus_baseline_probability_pp", "mean"),
        mean_abs_probability_delta_pp=(
            "targeted_minus_baseline_probability_pp",
            lambda values: values.abs().mean(),
        ),
    )
    rows = []
    for index, metric in enumerate(("log_loss_delta", "brier_delta")):
        result = bootstrap_event_mean(event[metric], draws, seed + index)
        rows.append(
            {
                "metric": metric,
                **result,
                "events": int(len(event)),
                "pairs": int(len(panel)),
                "events_improved": int((event[metric] < 0.0).sum()),
                "mean_abs_probability_delta_pp": float(
                    panel["targeted_minus_baseline_probability_pp"].abs().mean()
                ),
                "max_abs_probability_delta_pp": float(
                    panel["targeted_minus_baseline_probability_pp"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows), event


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.select_dtypes(include=["float", "float32", "float64"]):
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def build_report(
    coverage: pd.DataFrame,
    overall: pd.DataFrame,
    quintiles: pd.DataFrame,
    regressions: pd.DataFrame,
    bet_summary: pd.DataFrame,
    counterfactual: pd.DataFrame,
    audit: dict[str, object],
) -> str:
    primary_reg = regressions[
        regressions["price_point"].eq("close")
        & regressions["period"].isin(["all_2024_2026", "replication_2024_2025", "hypothesis_2026"])
        & regressions["market"].isin(["72h", "rounds_pooled"])
        & regressions["feature"].isin(
            ["profile_short_diff", "short_absolute_level", "live_short_balance_diff"]
        )
    ][
        [
            "period",
            "market",
            "feature",
            "outcome",
            "estimate_per_sd",
            "ci_low",
            "ci_high",
            "bootstrap_p",
            "holm_p_within_market_outcome",
            "events",
            "pairs",
        ]
    ]
    extreme = quintiles[
        quintiles["price_point"].eq("close")
        & quintiles["period"].eq("all_2024_2026")
        & quintiles["market"].isin(["72h", "rounds_pooled"])
        & quintiles["feature"].isin(
            ["profile_short_diff", "short_absolute_level", "live_short_balance_diff"]
        )
        & quintiles["quintile"].isin([0, 4])
    ][
        [
            "market",
            "feature",
            "quintile",
            "pairs",
            "events",
            "feature_mean",
            "observed",
            "model",
            "market_probability",
            "calibration_gap",
            "excess_log_loss",
        ]
    ]
    bet_extreme = bet_summary[
        bet_summary["price_point"].eq("close")
        & bet_summary["period"].eq("all_2024_2026")
        & bet_summary["market"].isin(["72h", "rounds_pooled"])
        & bet_summary["feature"].isin(
            ["profile_short_diff", "short_absolute_level", "live_short_balance_diff"]
        )
        & bet_summary["quintile"].isin([0, 4])
    ]
    overall_primary = overall[
        overall["price_point"].eq("close")
        & overall["period"].eq("all_2024_2026")
    ]
    lines = [
        "# ARG/PUTT matchup-fair calibration audit",
        "",
        "## What this can identify",
        "",
        "- `short_absolute_level`: whether the model-edge side is simply a good historical ARG/PUTT player.",
        "- `profile_short_diff`: whether that side is more short-game-heavy than the opponent after subtracting each player's four-category mean.",
        "- `live_short_balance_diff`: for R2-R4 only, whether completed prior rounds were more ARG/PUTT-heavy than OTT/APP-heavy, controlling total live SG and the exact scalar stage update.",
        "  The frozen baseline used raw round categories: R2 used R1 PUTT (ARG is a negative-control exposure there), while R3/R4 used prior-round PUTT and ARG averages.",
        "- Negative calibration-gap slopes and positive excess-log-loss slopes support overvaluation. Intervals are whole-event bootstrap intervals.",
        "",
        "## Coverage and integrity",
        "",
        markdown_table(coverage),
        "",
        f"Frozen round-state maximum identity error: {audit['round_state_prediction_max_abs_error']:.3g}.",
        "Pre-event profiles use only rounds dated before the event. Live profiles use only rounds numbered before the priced round.",
        "Same-book quotes are averaged within physical pair before inference.",
        "",
        "## Overall model-edge calibration",
        "",
        "Each pair is oriented toward the side our model likes more than the same-book no-vig market. This is the background calibration problem against which the category slices should be judged.",
        "",
        markdown_table(overall_primary, digits=6),
        "",
        "## Primary continuous tests",
        "",
        markdown_table(primary_reg, digits=6),
        "",
        "## Extreme-quintile calibration",
        "",
        markdown_table(extreme, digits=5),
        "",
        "## Direct 2026 targeted counterfactual",
        "",
        "The targeted arm holds every player's total category mean fixed. It is therefore a profile/distribution test, not a test of shrinking the player's total-SG median.",
        "Negative score deltas favor the targeted arm.",
        "",
        markdown_table(counterfactual, digits=7),
        "",
        "## Independently discovered 3% EV bets (descriptive)",
        "",
        markdown_table(bet_extreme, digits=5),
        "",
        "## Interpretation limits",
        "",
        "- The hypothesis was generated after examining 2026 diagnostics. The 2026 slice is exploratory; 2024-2025 is the cleaner temporal replication.",
        "- The round kernel erases zero-sum pre-event category-profile changes, so R1-R4 profile cohorts diagnose calibration association, not causal profile weight.",
        "- Frozen R2-R4 outputs retain the exact scalar stage adjustment but not its row-level ARG/PUTT component decomposition. The leakage-safe prior-round category test can detect a miss associated with short-game-heavy live state, but cannot uniquely assign causality to one coefficient.",
        "- Betting ROI is tertiary because a 3% threshold turns very small fair changes into discontinuous bet discovery.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    inputs = [
        args.score_authority,
        args.round_panel,
        args.round_state,
        args.counterfactual_dir / "probabilities.parquet",
        args.counterfactual_dir / "best_of_3_lines.parquet",
    ]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(map(str, missing)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen score authority and round panel ...", flush=True)
    quotes = load_baseline_quotes(args.score_authority)
    round_panel = pd.read_parquet(args.round_panel)
    print("Building strict-prior EMA50 category profiles ...", flush=True)
    profile = build_pre_event_profiles(quotes, round_panel)
    print("Building leakage-safe prior-round category states ...", flush=True)
    live = build_live_prior_profiles(quotes, round_panel)
    pair = collapse_physical_pairs(quotes)
    pair, state_audit = attach_profiles_and_states(
        pair, profile, live, args.round_state
    )
    oriented = pd.concat(
        [orient_pairs(pair, "close"), orient_pairs(pair, "open")],
        ignore_index=True,
        sort=False,
    )

    coverage_rows = []
    for market, group in pair.groupby("market", sort=False):
        coverage_rows.append(
            {
                "market": market,
                "pairs": int(len(group)),
                "events": int(group["event_key"].nunique()),
                "resolved_pairs": int(group["outcome_p1"].isin([0.0, 1.0]).sum()),
                "profile_both_sides": int(
                    group[["p1_profile_eligible", "p2_profile_eligible"]]
                    .eq(True)
                    .all(axis=1)
                    .sum()
                ),
                "live_both_sides": int(
                    group[["p1_live_complete", "p2_live_complete"]]
                    .eq(True)
                    .all(axis=1)
                    .sum()
                )
                if market in ("r2", "r3", "r4")
                else 0,
                "mean_books": float(group["books"].mean()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    print("Computing cohort summaries ...", flush=True)
    overall = build_overall_calibration(oriented)
    quintiles = build_quintile_summary(oriented)
    print(f"Running {args.bootstrap:,}-draw whole-event regressions ...", flush=True)
    regressions = build_regression_summary(oriented, args.bootstrap, args.seed)

    print("Replaying independent 3% EV discovery ...", flush=True)
    bets = pd.concat(
        [
            settle_discovered_bets(quotes, "close"),
            settle_discovered_bets(quotes, "open"),
        ],
        ignore_index=True,
        sort=False,
    )
    bets = attach_bet_profiles(bets, profile, live)
    bet_summary = build_bet_summary(bets)
    print("Scoring the direct 2026 targeted counterfactual ...", flush=True)
    counterfactual, counterfactual_events = counterfactual_score_audit(
        args.counterfactual_dir, args.bootstrap, args.seed + 900_000
    )

    profile.to_parquet(args.output_dir / "pre_event_category_profiles.parquet", index=False)
    live.to_parquet(args.output_dir / "prior_round_category_states.parquet", index=False)
    pair.to_parquet(args.output_dir / "physical_pair_panel.parquet", index=False)
    oriented.to_parquet(args.output_dir / "oriented_pair_panel.parquet", index=False)
    bets.to_parquet(args.output_dir / "discovered_bets.parquet", index=False)
    coverage.to_csv(args.output_dir / "coverage.csv", index=False)
    overall.to_csv(args.output_dir / "overall_calibration.csv", index=False)
    quintiles.to_csv(args.output_dir / "quintile_calibration.csv", index=False)
    regressions.to_csv(args.output_dir / "event_bootstrap_regressions.csv", index=False)
    bet_summary.to_csv(args.output_dir / "bet_quintiles.csv", index=False)
    counterfactual.to_csv(args.output_dir / "counterfactual_score_summary.csv", index=False)
    counterfactual_events.to_csv(
        args.output_dir / "counterfactual_event_influence.csv", index=False
    )

    audit: dict[str, object] = {
        **state_audit,
        "quotes": int(len(quotes)),
        "physical_pairs": int(len(pair)),
        "events": int(pair["event_key"].nunique()),
        "profile_rows": int(len(profile)),
        "live_state_rows": int(len(live)),
        "discovered_bets": int(len(bets)),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "inputs": {str(path.resolve()): sha256_file(path) for path in inputs},
        "source_sha256": sha256_file(Path(__file__)),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_report(
        coverage, overall, quintiles, regressions, bet_summary, counterfactual, audit
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Audit complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

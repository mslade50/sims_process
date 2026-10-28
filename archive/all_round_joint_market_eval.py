"""Research-only joint 72-hole and R1-R4 matchup evaluation.

This module is intentionally independent of the older frozen-handoff and hash
chains.  It consumes already-built, point-in-time player predictions and
normalized matchup quotes; it never fetches data and never writes a production
input.

Prediction schema
-----------------
``base_predictions`` is long-form with one row per
``candidate, event_key, player_id`` and these required columns::

    candidate, event_key, player_id, prediction

``dg_id_key`` may be used instead of ``player_id`` and ``mapped_event_key``
instead of ``event_key``.  The caller is responsible for supplying 2023 OOF
predictions and predictions frozen before each 2024-26 event.

``round_predictions`` has the same columns plus ``round_num`` (1-4), with one
prediction available immediately before the market being priced.  It is
mandatory in both authoritative CLI phases so a run cannot silently bypass
the cascade/live update.  Only the explicitly non-authoritative synthetic-test
helper may pass ``None``, in which case it carries the pre-event prediction
into all four round markets.  Candidates must have the same name in the base
and round panels.

Quote schema
------------
Both quote frames require one normalized same-book matchup per row with::

    event_key, event_start_date, book, p1_id, p2_id,
    p1_close, p2_close, p1_outcome, p2_outcome, tie_rule

``mapped_event_key`` is accepted for ``event_key``.  Round quotes additionally
require ``round_num`` (or an unambiguous ``round_market``).  Outcomes are
physical player-one results: 1=win, 0=loss and .5=tie.  If a ties-lose archive
stores a physical tie as (0, 0), it is reconstructed as a tie.  Supplying
``p1_round_score``/``p2_round_score`` or ``derived_score_outcome`` is preferred
because it distinguishes a physical tie from an administrative void.

The evaluator uses a global exact-common pair menu for every candidate within
each sample/market.  April-June 2023 chronological OOF pairs fit
candidate/market Gaussian sigma and tie-width maps; those maps are frozen for
July-December architecture selection.  Final maps use all April-December OOF
pairs and are frozen for 2024-26.  Selection uses the contract's equal-market
70% flat-risk one-unit P&L rank plus 30% three-way-log-loss rank across 72h and
R1-R4, with the baseline dominance exclusion applied before ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

try:
    from archive.build_all_round_selected_handoff import validate_prediction_authority
except ModuleNotFoundError:  # direct ``python archive/...py`` execution
    from build_all_round_selected_handoff import validate_prediction_authority
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive.market_backtest_scoring import (
    american_decimal_odds,
    settle_bets,
    ties_lose,
    valid_american_odds_mask,
)


MARKETS = ("72h", "r1", "r2", "r3", "r4")
BOOKS = ("betonline", "betcris", "pinnacle")
BOOK_PRIORITY = {book: index for index, book in enumerate(BOOKS)}
EPS = 1e-9
MIN_MAPPER_PAIRS = 8
MIN_MAPPER_EVENTS = 2
EXPECTED_CONTRACT_SHA256 = "4091a0192a959640a9dc88ff6b348759cf3052ed7aed3a116b5989019111374c"
DEFAULT_CONTRACT_PATH = REPO / "ALL_ROUND_BASE_SKILL_REBUILD_CONTRACT_2026_08.md"
MAPPER_TRAIN_START = pd.Timestamp("2023-04-01")
MAPPER_TRAIN_END = pd.Timestamp("2023-06-30")
ARCHITECTURE_SELECTION_START = pd.Timestamp("2023-07-01")
ARCHITECTURE_SELECTION_END = pd.Timestamp("2023-12-31")
LOCKED_EVALUATION_START = pd.Timestamp("2024-01-01")
LOCKED_EVALUATION_END = pd.Timestamp("2026-12-31")
LOCKED_BOOTSTRAP_DRAWS = 10_000
LOCKED_BOOTSTRAP_SEED = 20260823
PRIMARY_SELECTION_MODE = "primary_locked_rule"
PRIMARY_SELECTION_OUTCOME = "eligible_challenger_selected"
DIAGNOSTIC_SELECTION_MODE = "diagnostic_fallback_after_failed_gate"
DIAGNOSTIC_SELECTION_OUTCOME = "no_challenger_survived"
DIAGNOSTIC_FALLBACK_RULE = (
    "only_after_no_nonbaseline_challenger_survives_the_baseline_dominance_gate; "
    "choose_the_lowest_existing_70_30_rank_nonbaseline_without_recomputing_ranks"
)
NO_DIAGNOSTIC_FALLBACK_RULE = "not_applicable"
SELECTION_STATUS_FIELDS = (
    "primary_selection_outcome",
    "promotion_eligible",
    "selection_mode",
    "diagnostic_fallback_rule",
)

# Deliberately narrow quote projection.  Selection must not materialize later
# outcome labels or unrelated cached columns merely because they share a wide
# historical Parquet with the 2023 board.
QUOTE_READ_COLUMNS = frozenset(
    {
        "event_key",
        "mapped_event_key",
        "event_start_date",
        "mapped_event_start_date",
        "market",
        "round_market",
        "round_num",
        "book",
        "bookmaker",
        "p1_id",
        "p2_id",
        "p1_dg_id",
        "p2_dg_id",
        "p1_close",
        "p2_close",
        "p1_close_odds",
        "p2_close_odds",
        "p1_open",
        "p2_open",
        "p1_outcome",
        "p2_outcome",
        "p1_outcome_text",
        "p2_outcome_text",
        "tie_rule",
        "p1_round_score",
        "p2_round_score",
        "physical_p1_outcome",
        "derived_score_outcome",
        "p1_derived_score_outcome",
        "void",
        "no_score_void",
        "other_void_with_scores",
    }
)
PREDICTION_READ_COLUMNS = frozenset(
    {
        "candidate",
        "model",
        "event_key",
        "mapped_event_key",
        "event_start_date",
        "player_id",
        "dg_id_key",
        "prediction",
        "round_num",
        "round_market",
        "market",
    }
)


@dataclass
class JointMarketEvaluation:
    """In-memory outputs from :func:`evaluate_joint_markets`."""

    selected_candidate: str
    primary_selection_outcome: str
    promotion_eligible: bool
    selection_mode: str
    diagnostic_fallback_rule: str
    selection_summary: pd.DataFrame
    selection_joint_summary: pd.DataFrame
    evaluation_summary: pd.DataFrame
    mapper_crossfit: pd.DataFrame
    mapper_final: pd.DataFrame
    common_menu_audit: pd.DataFrame
    scored_selection: pd.DataFrame
    scored_evaluation: pd.DataFrame
    selection_bets: pd.DataFrame
    selection_betting_summary: pd.DataFrame
    bets: pd.DataFrame
    betting_summary: pd.DataFrame
    open_bets: pd.DataFrame
    open_betting_summary: pd.DataFrame
    edge_bucket_summary: pd.DataFrame
    betting_breakdowns: pd.DataFrame
    bootstrap: pd.DataFrame


@dataclass
class JointCandidateSelection:
    """Frozen 2023 OOF selection-phase artifacts."""

    selected_candidate: str
    primary_selection_outcome: str
    promotion_eligible: bool
    selection_mode: str
    diagnostic_fallback_rule: str
    selection_summary: pd.DataFrame
    selection_joint_summary: pd.DataFrame
    mapper_crossfit: pd.DataFrame
    common_menu_audit: pd.DataFrame
    scored_selection: pd.DataFrame
    selection_bets: pd.DataFrame
    selection_betting_summary: pd.DataFrame


@dataclass
class FrozenJointEvaluation:
    """2024-26 results for exactly the frozen selected challenger and control."""

    selected_candidate: str
    primary_selection_outcome: str
    promotion_eligible: bool
    selection_mode: str
    diagnostic_fallback_rule: str
    evaluation_summary: pd.DataFrame
    mapper_final: pd.DataFrame
    common_menu_audit: pd.DataFrame
    scored_evaluation: pd.DataFrame
    bets: pd.DataFrame
    betting_summary: pd.DataFrame
    open_bets: pd.DataFrame
    open_betting_summary: pd.DataFrame
    edge_bucket_summary: pd.DataFrame
    betting_breakdowns: pd.DataFrame
    bootstrap: pd.DataFrame


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _selection_status(
    *, diagnostic_fallback: bool
) -> dict[str, str | bool]:
    if diagnostic_fallback:
        return {
            "primary_selection_outcome": DIAGNOSTIC_SELECTION_OUTCOME,
            "promotion_eligible": False,
            "selection_mode": DIAGNOSTIC_SELECTION_MODE,
            "diagnostic_fallback_rule": DIAGNOSTIC_FALLBACK_RULE,
        }
    return {
        "primary_selection_outcome": PRIMARY_SELECTION_OUTCOME,
        "promotion_eligible": True,
        "selection_mode": PRIMARY_SELECTION_MODE,
        "diagnostic_fallback_rule": NO_DIAGNOSTIC_FALLBACK_RULE,
    }


def _validate_selection_status(status: Mapping[str, Any]) -> dict[str, str | bool]:
    missing = sorted(set(SELECTION_STATUS_FIELDS) - set(status))
    if missing:
        raise ValueError(f"selection status is missing required fields: {missing}")
    normalized: dict[str, str | bool] = {
        "primary_selection_outcome": str(status["primary_selection_outcome"]),
        "promotion_eligible": bool(status["promotion_eligible"]),
        "selection_mode": str(status["selection_mode"]),
        "diagnostic_fallback_rule": str(status["diagnostic_fallback_rule"]),
    }
    if normalized["selection_mode"] == PRIMARY_SELECTION_MODE:
        expected = _selection_status(diagnostic_fallback=False)
    elif normalized["selection_mode"] == DIAGNOSTIC_SELECTION_MODE:
        expected = _selection_status(diagnostic_fallback=True)
    else:
        raise ValueError("selection status has an unknown selection_mode")
    if normalized != expected:
        raise ValueError("selection status fields are internally inconsistent")
    return normalized


def _annotate_selection_status(
    frame: pd.DataFrame, status: Mapping[str, Any]
) -> pd.DataFrame:
    normalized = _validate_selection_status(status)
    output = frame.copy()
    for key, value in normalized.items():
        output[key] = value
    return output


def _status_from_selection(selection: JointCandidateSelection) -> dict[str, str | bool]:
    return _validate_selection_status(
        {field: getattr(selection, field) for field in SELECTION_STATUS_FIELDS}
    )


def _stable_id(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0].lstrip("+")
    return text.casefold()


def _book(value: object) -> str:
    text = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    aliases = {
        "betonlineag": "betonline",
        "cris": "betcris",
        "bookmaker": "betcris",
        "pinnaclesports": "pinnacle",
    }
    return aliases.get(text, text)


def _outcome_number(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().casefold()
    words = {
        "w": 1.0,
        "win": 1.0,
        "won": 1.0,
        "l": 0.0,
        "loss": 0.0,
        "lost": 0.0,
        "t": 0.5,
        "tie": 0.5,
        "tied": 0.5,
        "push": 0.5,
        "draw": 0.5,
    }
    if text in words:
        return words[text]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if numeric in (0.0, 0.5, 1.0) else np.nan


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _physical_p1_outcome(
    row: pd.Series, *, allow_numeric_void_tie: bool = True
) -> float:
    """Return the physical p1 win/tie/loss, not necessarily bet settlement."""

    # These two flags explicitly distinguish an administrative/no-score void
    # from DataGolf's normal settlement text for a tied two-way matchup.
    for void_column in ("no_score_void", "other_void_with_scores"):
        if void_column in row.index and pd.notna(row[void_column]) and _as_bool(row[void_column]):
            return np.nan

    if "p1_round_score" in row.index and "p2_round_score" in row.index:
        p1_score = pd.to_numeric(pd.Series([row["p1_round_score"]]), errors="coerce").iloc[0]
        p2_score = pd.to_numeric(pd.Series([row["p2_round_score"]]), errors="coerce").iloc[0]
        if np.isfinite(p1_score) and np.isfinite(p2_score):
            return 1.0 if p1_score < p2_score else (0.0 if p1_score > p2_score else 0.5)

    for column in (
        "physical_p1_outcome",
        "derived_score_outcome",
        "p1_derived_score_outcome",
    ):
        if column in row.index:
            value = _outcome_number(row[column])
            if np.isfinite(value):
                return value

    p1 = _outcome_number(row.get("p1_outcome"))
    p2 = _outcome_number(row.get("p2_outcome"))
    # DataGolf encodes a physical round-matchup tie as numeric 0.5/0.5 while
    # its settlement text says ``void`` because the wager is refunded.  That
    # convention is verified against the score panel for R1-R4.  The 72-hole
    # archive also uses 0.5/0.5 for administrative voids, so tournament rows
    # may take this branch only when the caller supplied score authority (the
    # explicit physical/derived fields above) or the text is not a void.
    if np.isfinite(p1) and np.isfinite(p2):
        if (
            np.isclose(p1, 0.5)
            and np.isclose(p2, 0.5)
            and allow_numeric_void_tie
        ):
            return 0.5
        if np.isclose(p1, 0.0) and np.isclose(p2, 0.0) and ties_lose(row.get("tie_rule", "")):
            return 0.5

    if "void" in row.index and pd.notna(row["void"]) and _as_bool(row["void"]):
        return np.nan
    for text_column in ("outcome", "p1_outcome_text", "result", "settlement"):
        if text_column in row.index and pd.notna(row[text_column]):
            if str(row[text_column]).strip().casefold() in {
                "void",
                "voided",
                "refund",
                "refunded",
                "cancelled",
                "canceled",
                "no action",
            }:
                return np.nan

    if np.isfinite(p1) and np.isfinite(p2):
        if np.isclose(p1 + p2, 1.0):
            return p1
        return np.nan
    return p1 if np.isfinite(p1) else np.nan


def _settlement_p1_outcome(row: pd.Series) -> float:
    """Return the archived wager settlement, separate from physical result."""

    p1 = _outcome_number(row.get("p1_outcome"))
    p2 = _outcome_number(row.get("p2_outcome"))
    if np.isfinite(p1) and np.isfinite(p2):
        if np.isclose(p1, 0.5) and np.isclose(p2, 0.5):
            return 0.5
        if np.isclose(p1, 0.0) and np.isclose(p2, 0.0) and ties_lose(row.get("tie_rule", "")):
            return 0.5
        if np.isclose(p1 + p2, 1.0):
            return p1
        return np.nan
    return p1 if np.isfinite(p1) else np.nan


def _round_market(frame: pd.DataFrame) -> pd.Series:
    if "round_num" in frame:
        number = pd.to_numeric(frame["round_num"], errors="coerce")
    else:
        source = frame.get("round_market", frame.get("market", pd.Series("", index=frame.index)))
        number = pd.to_numeric(
            source.astype(str).str.casefold().str.extract(r"(?:round|r)[ _-]?([1-4])")[0],
            errors="coerce",
        )
    invalid = ~number.isin([1, 2, 3, 4])
    if invalid.any():
        raise ValueError("round quotes/predictions contain an invalid or missing round_num")
    return "r" + number.astype(int).astype(str)


def _normalize_predictions(frame: pd.DataFrame, *, rounds: bool) -> pd.DataFrame:
    output = frame.copy()
    renames: dict[str, str] = {}
    if "event_key" not in output and "mapped_event_key" in output:
        renames["mapped_event_key"] = "event_key"
    if "player_id" not in output and "dg_id_key" in output:
        renames["dg_id_key"] = "player_id"
    if "candidate" not in output and "model" in output:
        renames["model"] = "candidate"
    output = output.rename(columns=renames)
    _require(output, {"candidate", "event_key", "player_id", "prediction"}, "predictions")
    output["candidate"] = output["candidate"].astype(str)
    output["event_key"] = output["event_key"].map(_stable_id)
    output["player_id"] = output["player_id"].map(_stable_id)
    output["prediction"] = pd.to_numeric(output["prediction"], errors="coerce")
    if output["event_key"].eq("").any() or output["player_id"].eq("").any():
        raise ValueError("predictions contain blank event/player identities")
    output = output.loc[np.isfinite(output["prediction"])].copy()
    if rounds:
        output["market"] = _round_market(output)
    else:
        output["market"] = "72h"
    keys = ["candidate", "market", "event_key", "player_id"]
    if output.duplicated(keys).any():
        raise ValueError(f"predictions are ambiguous on {keys}")
    return output[keys + ["prediction"]].reset_index(drop=True)


def _normalize_quotes(frame: pd.DataFrame, *, rounds: bool) -> pd.DataFrame:
    output = frame.copy()
    if not rounds and "market" in output:
        source_market = (
            output["market"]
            .astype(str)
            .str.casefold()
            .str.replace(r"[^a-z0-9]", "", regex=True)
        )
        tournament_mask = source_market.isin(
            {"72h", "72hole", "72holematchup", "72holematchups", "tournamentmatchup"}
        )
        output = output.loc[tournament_mask].copy()
        if output.empty:
            raise ValueError("tournament quote input contains no 72h matchup rows")
    aliases = {
        "mapped_event_key": "event_key",
        "mapped_event_start_date": "event_start_date",
        "bookmaker": "book",
        "p1_dg_id": "p1_id",
        "p2_dg_id": "p2_id",
        "p1_close_odds": "p1_close",
        "p2_close_odds": "p2_close",
    }
    output = output.rename(
        columns={source: target for source, target in aliases.items() if target not in output}
    )
    required = {
        "event_key",
        "event_start_date",
        "book",
        "p1_id",
        "p2_id",
        "p1_close",
        "p2_close",
        "p1_outcome",
        "p2_outcome",
    }
    _require(output, required, "normalized quotes")
    if "tie_rule" not in output:
        output["tie_rule"] = "ties push"
    output["market"] = _round_market(output) if rounds else "72h"
    output["event_key"] = output["event_key"].map(_stable_id)
    output["book"] = output["book"].map(_book)
    output["p1_id"] = output["p1_id"].map(_stable_id)
    output["p2_id"] = output["p2_id"].map(_stable_id)
    output["event_start_date"] = pd.to_datetime(
        output["event_start_date"], errors="coerce"
    ).dt.normalize()
    if output["event_start_date"].isna().any():
        raise ValueError("normalized quotes contain invalid event_start_date values")
    if output["event_key"].eq("").any() or output["p1_id"].eq("").any() or output["p2_id"].eq("").any():
        raise ValueError("normalized quotes contain blank event/player identities")
    if output["p1_id"].eq(output["p2_id"]).any():
        raise ValueError("normalized quotes contain a self-matchup")
    output = output.loc[output["book"].isin(BOOKS)].copy()
    output["p1_close"] = pd.to_numeric(output["p1_close"], errors="coerce")
    output["p2_close"] = pd.to_numeric(output["p2_close"], errors="coerce")
    valid = valid_american_odds_mask(output["p1_close"]) & valid_american_odds_mask(
        output["p2_close"]
    )
    output = output.loc[valid].copy()
    if output.empty:
        raise ValueError("normalized quotes have no valid two-sided close at the three books")

    physical = output.apply(
        lambda row: _physical_p1_outcome(
            row, allow_numeric_void_tie=bool(rounds)
        ),
        axis=1,
    ).astype(float)
    settlement = output.apply(_settlement_p1_outcome, axis=1).astype(float)
    flip = output["p1_id"] > output["p2_id"]
    paired_columns = [
        column for column in output.columns if column.startswith("p1_") and "p2_" + column[3:] in output
    ]
    for p1_column in paired_columns:
        p2_column = "p2_" + p1_column[3:]
        left = output[p1_column].copy()
        right = output[p2_column].copy()
        output.loc[flip, p1_column] = right.loc[flip]
        output.loc[flip, p2_column] = left.loc[flip]
    output["physical_p1_outcome"] = np.where(
        flip & np.isfinite(physical), 1.0 - physical, physical
    )
    output["settlement_p1_outcome"] = np.where(
        flip & np.isfinite(settlement), 1.0 - settlement, settlement
    )
    output["pair_key"] = output["p1_id"] + "|" + output["p2_id"]
    output["source_sides_swapped_to_canonical"] = flip.to_numpy(bool)
    output["quote_id"] = (
        output["event_key"]
        + "|"
        + output["market"]
        + "|"
        + output["book"]
        + "|"
        + output["pair_key"]
    )
    logical_book_keys = ["event_key", "market", "book", "pair_key"]
    if output.duplicated(logical_book_keys).any():
        raise ValueError(
            "normalized quotes contain duplicate event/market/book/pair rows"
        )
    inconsistent = (
        output.loc[output["physical_p1_outcome"].notna()]
        .groupby(["event_key", "market", "pair_key"], observed=True)["physical_p1_outcome"]
        .nunique()
        .gt(1)
    )
    if inconsistent.any():
        raise ValueError("physical matchup outcomes disagree across books")
    return output.reset_index(drop=True)


def _date_sample(
    dates: pd.Series,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> pd.Series:
    selection = dates.between(selection_start, selection_end)
    evaluation = dates.between(eval_start, eval_end)
    return pd.Series(
        np.select([selection, evaluation], ["selection", "evaluation"], default=""),
        index=dates.index,
        dtype="object",
    )


def _prefilter_raw_quote_frame(
    frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    label: str,
) -> pd.DataFrame:
    """Apply the label boundary before quote normalization or outcome parsing."""

    date_column = next(
        (column for column in ("event_start_date", "mapped_event_start_date") if column in frame),
        None,
    )
    if date_column is None:
        raise ValueError(f"{label} lacks an event-date column for pre-normalization filtering")
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    keep = dates.between(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    return frame.loc[keep].copy()


def build_exact_common_panel(
    base_predictions: pd.DataFrame,
    round_predictions: pd.DataFrame | None,
    tournament_quotes: pd.DataFrame,
    round_quotes: pd.DataFrame,
    *,
    baseline: str,
    selection_start: str | pd.Timestamp,
    selection_end: str | pd.Timestamp,
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp,
    selection_candidates: Sequence[str] | None = None,
    evaluation_candidates: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach predictions and enforce one global pair menu per sample/market."""

    selection_start_ts = pd.Timestamp(selection_start).normalize()
    selection_end_ts = pd.Timestamp(selection_end).normalize()
    eval_start_ts = pd.Timestamp(eval_start).normalize()
    eval_end_ts = pd.Timestamp(eval_end).normalize()
    if selection_start_ts > selection_end_ts or eval_start_ts > eval_end_ts:
        raise ValueError("sample date ranges are inverted")
    if selection_end_ts >= eval_start_ts:
        raise ValueError("selection and evaluation ranges must not overlap")

    base = _normalize_predictions(base_predictions, rounds=False)
    if round_predictions is None:
        round_parts = [base.assign(market=f"r{round_num}") for round_num in range(1, 5)]
        rounds = pd.concat(round_parts, ignore_index=True)
    else:
        rounds = _normalize_predictions(round_predictions, rounds=True)
    predictions = pd.concat([base, rounds], ignore_index=True)

    tournament = _normalize_quotes(tournament_quotes, rounds=False)
    round_frame = _normalize_quotes(round_quotes, rounds=True)
    quotes = pd.concat([tournament, round_frame], ignore_index=True, sort=False)
    quotes["sample"] = _date_sample(
        quotes["event_start_date"],
        selection_start_ts,
        selection_end_ts,
        eval_start_ts,
        eval_end_ts,
    )
    quotes = quotes.loc[quotes["sample"].ne("")].copy()
    if quotes.empty:
        raise ValueError("no quotes fall inside the requested selection/evaluation dates")

    market_candidates = {
        market: set(group["candidate"].astype(str))
        for market, group in predictions.groupby("market", observed=True)
    }
    missing_market = sorted(set(MARKETS) - set(market_candidates))
    if missing_market:
        raise ValueError(f"prediction panel is missing markets: {missing_market}")
    all_joint_candidates = set.intersection(
        *(market_candidates[market] for market in MARKETS)
    )
    if baseline not in all_joint_candidates:
        raise ValueError(f"baseline {baseline!r} is not available in all five markets")
    if not all_joint_candidates:
        raise ValueError("no candidate is available in all five markets")
    selection_universe = (
        set(map(str, selection_candidates))
        if selection_candidates is not None
        else set(all_joint_candidates)
    )
    evaluation_universe = (
        set(map(str, evaluation_candidates))
        if evaluation_candidates is not None
        else set(selection_universe)
    )
    for label, universe in (
        ("selection", selection_universe),
        ("evaluation", evaluation_universe),
    ):
        missing = universe - all_joint_candidates
        if missing:
            raise ValueError(f"{label} candidate universe is unavailable: {sorted(missing)}")
        if baseline not in universe:
            raise ValueError(f"baseline must be present in the {label} candidate universe")
    active_candidates = selection_universe | evaluation_universe
    predictions = predictions.loc[
        predictions["candidate"].isin(active_candidates)
    ].copy()

    left = predictions.rename(
        columns={"player_id": "p1_id", "prediction": "p1_prediction"}
    )
    right = predictions.rename(
        columns={"player_id": "p2_id", "prediction": "p2_prediction"}
    )
    paired = quotes.merge(
        left,
        on=["market", "event_key", "p1_id"],
        how="inner",
        validate="many_to_many",
    )
    paired = paired.merge(
        right,
        on=["candidate", "market", "event_key", "p2_id"],
        how="inner",
        validate="many_to_one",
    )
    paired["skill_difference"] = paired["p1_prediction"] - paired["p2_prediction"]
    paired = paired.loc[np.isfinite(paired["skill_difference"])].copy()

    audit_rows: list[dict[str, Any]] = []
    common_parts: list[pd.DataFrame] = []
    opportunity_columns = ["event_key", "pair_key"]
    for (sample, market), quote_group in quotes.groupby(["sample", "market"], observed=True):
        candidate_universe = (
            selection_universe if sample == "selection" else evaluation_universe
        )
        quote_opportunities = set(map(tuple, quote_group[opportunity_columns].drop_duplicates().to_numpy()))
        candidate_sets: dict[str, set[tuple[str, str]]] = {}
        paired_group = paired.loc[
            paired["sample"].eq(sample) & paired["market"].eq(market)
        ]
        for candidate in sorted(candidate_universe):
            candidate_rows = paired_group.loc[paired_group["candidate"].eq(candidate)]
            candidate_sets[candidate] = set(
                map(tuple, candidate_rows[opportunity_columns].drop_duplicates().to_numpy())
            )
        common = set.intersection(
            *(candidate_sets[candidate] for candidate in sorted(candidate_universe))
        )
        if not common:
            raise ValueError(f"exact common menu is empty for {sample}/{market}")
        common_index = pd.MultiIndex.from_tuples(common, names=opportunity_columns)
        row_index = pd.MultiIndex.from_frame(paired_group[opportunity_columns])
        keep = row_index.isin(common_index)
        common_parts.append(paired_group.loc[keep].copy())
        for candidate in sorted(candidate_universe):
            available = candidate_sets[candidate]
            audit_rows.append(
                {
                    "sample": sample,
                    "market": market,
                    "candidate": candidate,
                    "quoted_opportunities": int(len(quote_opportunities)),
                    "candidate_available_opportunities": int(len(available)),
                    "exact_common_opportunities": int(len(common)),
                    "candidate_dropped_from_common": int(len(available - common)),
                    "quoted_dropped_from_common": int(len(quote_opportunities - common)),
                }
            )

    common_panel = pd.concat(common_parts, ignore_index=True)
    opportunity_counts = (
        common_panel[["sample", "market", "candidate", "event_key", "pair_key"]]
        .drop_duplicates()
        .groupby(["sample", "market", "candidate"], observed=True)
        .size()
        .unstack("candidate")
    )
    if opportunity_counts.nunique(axis=1).gt(1).any():
        raise AssertionError("candidate opportunity counts differ after exact-menu restriction")
    return common_panel.reset_index(drop=True), pd.DataFrame(audit_rows)


def _logical_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-book copies to one canonical logical pair per candidate."""

    if frame.empty:
        return frame.copy()
    keys = ["candidate", "sample", "event_key", "market", "pair_key"]
    invariants = [
        "skill_difference",
        "physical_p1_outcome",
        *[column for column in ("p1_probability", "p2_probability", "push_probability") if column in frame],
    ]
    for column in invariants:
        inconsistent = frame.groupby(keys, observed=True)[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise ValueError(f"logical matchup reconstruction disagrees across books for {column}")
    output = frame.assign(_book_rank=frame["book"].map(BOOK_PRIORITY).fillna(99))
    return (
        output.sort_values(keys + ["_book_rank"], kind="mergesort")
        .drop_duplicates(keys, keep="first")
        .drop(columns="_book_rank")
        .reset_index(drop=True)
    )


def gaussian_matchup_probabilities(
    difference: np.ndarray | pd.Series,
    sigma: float,
    tie_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric latent-normal win/loss/tie probabilities."""

    values = np.asarray(difference, dtype=float)
    p1 = norm.cdf((values - tie_threshold) / sigma)
    p2 = norm.cdf((-values - tie_threshold) / sigma)
    push = np.clip(1.0 - p1 - p2, 0.0, 1.0)
    total = p1 + p2 + push
    return p1 / total, p2 / total, push / total


def _event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_key", observed=True)["pair_key"].transform("size")
    return 1.0 / counts.to_numpy(float)


def fit_gaussian_mapper(
    logical_pairs: pd.DataFrame,
    *,
    min_pairs: int = MIN_MAPPER_PAIRS,
    min_events: int = MIN_MAPPER_EVENTS,
) -> tuple[float, float, float, int, int]:
    """Fit sigma and physical-tie width with event-equal three-way log loss."""

    graded = logical_pairs.loc[
        logical_pairs["physical_p1_outcome"].isin([0.0, 0.5, 1.0])
    ].copy()
    pair_count = int(len(graded))
    event_count = int(graded["event_key"].nunique())
    if pair_count < min_pairs or event_count < min_events:
        raise ValueError(
            f"mapper needs at least {min_pairs} pairs/{min_events} events; "
            f"received {pair_count}/{event_count}"
        )
    difference = graded["skill_difference"].to_numpy(float)
    outcome = graded["physical_p1_outcome"].to_numpy(float)
    weights = _event_equal_weights(graded)

    def objective(parameters: np.ndarray) -> float:
        sigma = math.exp(float(parameters[0]))
        tie_threshold = math.exp(float(parameters[1]))
        p1, p2, push = gaussian_matchup_probabilities(difference, sigma, tie_threshold)
        chosen = np.where(outcome == 1.0, p1, np.where(outcome == 0.0, p2, push))
        loss = -np.log(np.clip(chosen, EPS, 1.0))
        return float(np.sum(weights * loss) / np.sum(weights))

    scale = float(np.nanstd(difference))
    initial_sigma = max(scale, 0.5)
    initial_tie = max(0.02 * initial_sigma, 0.01)
    fit = minimize(
        objective,
        np.log(np.array([initial_sigma, initial_tie])),
        method="L-BFGS-B",
        bounds=[(math.log(0.01), math.log(100.0)), (math.log(1e-6), math.log(10.0))],
    )
    if not np.isfinite(fit.fun):
        raise ValueError("Gaussian mapper optimization produced a nonfinite objective")
    sigma, tie_threshold = np.exp(fit.x)
    return float(sigma), float(tie_threshold), float(fit.fun), pair_count, event_count


def fit_final_probability_mappers(selection_panel: pd.DataFrame) -> pd.DataFrame:
    """Fit frozen candidate/market mappers on every 2023 selection event."""

    logical = _logical_pairs(selection_panel)
    rows: list[dict[str, Any]] = []
    for (candidate, market), group in logical.groupby(["candidate", "market"], observed=True):
        sigma, tie, loss, pairs, events = fit_gaussian_mapper(group)
        rows.append(
            {
                "candidate": candidate,
                "market": market,
                "sigma": sigma,
                "tie_threshold": tie,
                "selection_event_equal_three_way_log_loss": loss,
                "selection_pairs": pairs,
                "selection_events": events,
                "fit_sample": "selection_only",
            }
        )
    return pd.DataFrame(rows)


def fit_chronological_selection_mappers(
    selection_panel: pd.DataFrame,
    *,
    mapper_start: str | pd.Timestamp = MAPPER_TRAIN_START,
    mapper_end: str | pd.Timestamp = MAPPER_TRAIN_END,
    architecture_start: str | pd.Timestamp = ARCHITECTURE_SELECTION_START,
    architecture_end: str | pd.Timestamp = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on April-June OOF pairs and score July-December without refitting."""

    mapper_start_ts = pd.Timestamp(mapper_start).normalize()
    mapper_end_ts = pd.Timestamp(mapper_end).normalize()
    architecture_start_ts = pd.Timestamp(architecture_start).normalize()
    architecture_end_ts = pd.Timestamp(architecture_end).normalize()
    if not (
        mapper_start_ts <= mapper_end_ts < architecture_start_ts <= architecture_end_ts
    ):
        raise ValueError("chronological mapper/architecture windows overlap or are inverted")
    calibration = selection_panel.loc[
        selection_panel["event_start_date"].between(mapper_start_ts, mapper_end_ts)
    ].copy()
    architecture = selection_panel.loc[
        selection_panel["event_start_date"].between(
            architecture_start_ts, architecture_end_ts
        )
    ].copy()
    if calibration.empty or architecture.empty:
        raise ValueError("April-June mapper and July-December architecture samples are required")
    mappers = fit_final_probability_mappers(calibration).rename(
        columns={
            "selection_event_equal_three_way_log_loss": "mapper_training_event_equal_three_way_log_loss",
            "selection_pairs": "mapper_training_pairs",
            "selection_events": "mapper_training_events",
        }
    )
    mappers["fit_sample"] = "apr_jun_2023_chronological_oof"
    mappers["mapper_training_start"] = mapper_start_ts
    mappers["mapper_training_end"] = mapper_end_ts
    mappers["architecture_scoring_start"] = architecture_start_ts
    mappers["architecture_scoring_end"] = architecture_end_ts
    scored = _score_with_mappers(architecture, mappers)
    scored["probability_mapper_scope"] = "frozen_apr_jun_2023_oof"
    return scored, mappers


def _score_with_mappers(
    panel: pd.DataFrame,
    mappers: pd.DataFrame,
    *,
    fold_column: str | None = None,
) -> pd.DataFrame:
    keys = ["candidate", "market"] + ([fold_column] if fold_column else [])
    mapper_columns = keys + ["sigma", "tie_threshold"]
    if mappers.duplicated(keys).any():
        raise ValueError(f"probability mappers are ambiguous on {keys}")
    output = panel.merge(
        mappers[mapper_columns], on=keys, how="left", validate="many_to_one"
    )
    if output[["sigma", "tie_threshold"]].isna().any().any():
        raise ValueError("one or more pair rows lack a probability mapper")
    p1, p2, push = gaussian_matchup_probabilities(
        output["skill_difference"].to_numpy(float),
        output["sigma"].to_numpy(float),
        output["tie_threshold"].to_numpy(float),
    )
    output["p1_probability"] = p1
    output["p2_probability"] = p2
    output["push_probability"] = push
    simplex = output[["p1_probability", "p2_probability", "push_probability"]].sum(axis=1)
    if not np.allclose(simplex, 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("Gaussian matchup probabilities do not sum to one")
    return output


def _weighted_calibration(
    event_keys: pd.Series,
    outcomes: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float, float, bool]:
    if len(outcomes) < 10 or len(np.unique(outcomes)) < 2:
        return np.nan, np.nan, np.nan, False
    probability = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    outcome = np.asarray(outcomes, dtype=float)
    counts = event_keys.groupby(event_keys, observed=True).transform("size").to_numpy(float)
    weights = 1.0 / counts
    x = logit(probability)

    def objective(parameters: np.ndarray) -> float:
        fitted = np.clip(expit(parameters[0] + parameters[1] * x), EPS, 1.0 - EPS)
        loss = -(outcome * np.log(fitted) + (1.0 - outcome) * np.log(1.0 - fitted))
        return float(np.sum(weights * loss) / np.sum(weights))

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    intercept, slope = fit.x if fit.success else (np.nan, np.nan)
    try:
        bins = pd.qcut(pd.Series(probability), q=10, duplicates="drop")
        calibration = pd.DataFrame(
            {"bin": bins, "predicted": probability, "actual": outcome, "weight": weights}
        )
        grouped = calibration.groupby("bin", observed=True).apply(
            lambda item: pd.Series(
                {
                    "predicted": np.average(item["predicted"], weights=item["weight"]),
                    "actual": np.average(item["actual"], weights=item["weight"]),
                    "weight": item["weight"].sum(),
                }
            ),
            include_groups=False,
        )
        ece = float(
            np.average(np.abs(grouped["predicted"] - grouped["actual"]), weights=grouped["weight"])
        )
    except (ValueError, ZeroDivisionError):
        ece = np.nan
    return float(intercept), float(slope), ece, bool(fit.success)


def summarize_probabilities(scored: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Event-equal three-way and resolved-binary proper scores/calibration."""

    logical = _logical_pairs(scored)
    rows: list[dict[str, Any]] = []
    for (candidate, market), group in logical.groupby(["candidate", "market"], observed=True):
        graded = group.loc[group["physical_p1_outcome"].isin([0.0, 0.5, 1.0])].copy()
        if graded.empty:
            continue
        outcome = graded["physical_p1_outcome"].to_numpy(float)
        p1 = np.clip(graded["p1_probability"].to_numpy(float), EPS, 1.0)
        p2 = np.clip(graded["p2_probability"].to_numpy(float), EPS, 1.0)
        push = np.clip(graded["push_probability"].to_numpy(float), EPS, 1.0)
        chosen = np.where(outcome == 1.0, p1, np.where(outcome == 0.0, p2, push))
        graded["three_way_log_loss"] = -np.log(chosen)
        graded["three_way_brier"] = (
            np.square(p1 - (outcome == 1.0))
            + np.square(p2 - (outcome == 0.0))
            + np.square(push - (outcome == 0.5))
        )
        event_three = graded.groupby("event_key", observed=True)[
            ["three_way_log_loss", "three_way_brier"]
        ].mean()
        resolved = graded.loc[~np.isclose(graded["physical_p1_outcome"], 0.5)].copy()
        binary = {
            "resolved_pairs": 0,
            "event_equal_log_loss": np.nan,
            "event_equal_brier": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "ece_decile": np.nan,
            "calibration_fit_success": False,
        }
        if not resolved.empty:
            y = resolved["physical_p1_outcome"].to_numpy(float)
            conditional = (
                resolved["p1_probability"]
                / (resolved["p1_probability"] + resolved["p2_probability"])
            ).clip(EPS, 1.0 - EPS).to_numpy(float)
            resolved["log_loss"] = -(y * np.log(conditional) + (1.0 - y) * np.log(1.0 - conditional))
            resolved["brier"] = np.square(conditional - y)
            event_binary = resolved.groupby("event_key", observed=True)[["log_loss", "brier"]].mean()
            intercept, slope, ece, fit_success = _weighted_calibration(
                resolved["event_key"], y, conditional
            )
            binary = {
                "resolved_pairs": int(len(resolved)),
                "event_equal_log_loss": float(event_binary["log_loss"].mean()),
                "event_equal_brier": float(event_binary["brier"].mean()),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece_decile": ece,
                "calibration_fit_success": fit_success,
            }
        rows.append(
            {
                "sample": sample,
                "candidate": candidate,
                "market": market,
                "events": int(graded["event_key"].nunique()),
                "graded_pairs": int(len(graded)),
                "physical_ties": int(np.isclose(outcome, 0.5).sum()),
                "observed_tie_rate": float(np.isclose(outcome, 0.5).mean()),
                "mean_predicted_tie_rate": float(graded["push_probability"].mean()),
                "event_equal_three_way_log_loss": float(event_three["three_way_log_loss"].mean()),
                "event_equal_three_way_brier": float(event_three["three_way_brier"].mean()),
                "mean_sigma": float(graded["sigma"].mean()),
                "mean_tie_threshold": float(graded["tie_threshold"].mean()),
                **binary,
            }
        )
    return pd.DataFrame(rows)


def joint_selection_summary(
    selection_summary: pd.DataFrame,
    baseline: str,
    selection_bets: pd.DataFrame,
    *,
    diagnostic_fallback_on_no_eligible: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Apply the frozen market-first 70/30 candidate selection rule.

    Each of 72h and R1-R4 receives equal weight.  Within a market, flat-risk
    one-unit P&L at the 3% EV gate is the sole betting-return rank; the five
    P&L ranks are averaged and receive 70% weight.  The corresponding OOF
    three-way-log-loss ranks receive 30%.  ROI is reported but never enters
    selection.  A challenger is ineligible only when both its aggregate P&L
    *and* equal-market log loss are worse than the baseline.  The baseline is
    retained in all ranks and dominance comparisons, but is a comparator rather
    than a selectable challenger.
    """

    complete = selection_summary.loc[selection_summary["market"].isin(MARKETS)].copy()
    counts = complete.groupby("candidate", observed=True)["market"].nunique()
    eligible = set(counts.loc[counts.eq(len(MARKETS))].index)
    if baseline not in eligible:
        raise ValueError("baseline lacks a complete five-market selection summary")
    complete = complete.loc[complete["candidate"].isin(eligible)]
    candidate_names = sorted(eligible)
    market_grid = pd.MultiIndex.from_product(
        [candidate_names, MARKETS], names=["candidate", "market"]
    ).to_frame(index=False)
    if selection_bets.empty:
        flat_market = market_grid.assign(bets=0, stake=0.0, pnl=0.0, roi=0.0)
    else:
        flat = selection_bets.loc[selection_bets["sizing"].eq("flat_1u")].copy()
        aggregated = flat.groupby(["candidate", "round_market"], observed=True).agg(
            bets=("pnl", "size"), stake=("stake", "sum"), pnl=("pnl", "sum")
        ).reset_index().rename(columns={"round_market": "market"})
        flat_market = market_grid.merge(
            aggregated, on=["candidate", "market"], how="left", validate="one_to_one"
        )
        flat_market[["bets", "stake", "pnl"]] = flat_market[
            ["bets", "stake", "pnl"]
        ].fillna(0.0)
        flat_market["roi"] = np.where(
            flat_market["stake"].gt(0.0),
            flat_market["pnl"] / flat_market["stake"],
            0.0,
        )
    flat_market["market_betting_rank"] = flat_market.groupby("market", observed=True)["pnl"].rank(
        ascending=False, method="average", pct=True
    )
    complete["market_log_loss_rank"] = complete.groupby("market", observed=True)[
        "event_equal_three_way_log_loss"
    ].rank(ascending=True, method="average", pct=True)

    joint = complete.groupby("candidate", observed=True).agg(
        markets=("market", "nunique"),
        selection_events=("events", "sum"),
        selection_pairs=("graded_pairs", "sum"),
        mean_market_event_equal_three_way_log_loss=("event_equal_three_way_log_loss", "mean"),
        mean_market_event_equal_three_way_brier=("event_equal_three_way_brier", "mean"),
        mean_market_event_equal_log_loss=("event_equal_log_loss", "mean"),
        mean_market_event_equal_brier=("event_equal_brier", "mean"),
        mean_market_log_loss_rank=("market_log_loss_rank", "mean"),
    ).reset_index()
    betting_joint = flat_market.groupby("candidate", observed=True).agg(
        aggregate_selection_bets=("bets", "sum"),
        aggregate_selection_stake=("stake", "sum"),
        aggregate_selection_pnl=("pnl", "sum"),
        mean_market_selection_roi=("roi", "mean"),
        mean_market_betting_rank=("market_betting_rank", "mean"),
    ).reset_index()
    joint = joint.merge(betting_joint, on="candidate", how="left", validate="one_to_one")
    baseline_loss = float(
        joint.loc[
            joint["candidate"].eq(baseline),
            "mean_market_event_equal_three_way_log_loss",
        ].iloc[0]
    )
    joint["three_way_log_loss_delta_vs_baseline"] = (
        joint["mean_market_event_equal_three_way_log_loss"] - baseline_loss
    )
    baseline_pnl = float(
        joint.loc[joint["candidate"].eq(baseline), "aggregate_selection_pnl"].iloc[0]
    )
    joint["pnl_delta_vs_baseline"] = joint["aggregate_selection_pnl"] - baseline_pnl
    joint["worse_than_baseline_pnl"] = joint["aggregate_selection_pnl"].lt(baseline_pnl)
    joint["worse_than_baseline_log_loss"] = joint[
        "mean_market_event_equal_three_way_log_loss"
    ].gt(baseline_loss)
    joint["eligible_under_baseline_dominance_rule"] = ~(
        joint["worse_than_baseline_pnl"] & joint["worse_than_baseline_log_loss"]
    )
    joint["eligible_for_challenger_selection"] = (
        joint["eligible_under_baseline_dominance_rule"]
        & ~joint["candidate"].eq(baseline)
    )
    joint["market_first_rank_score"] = (
        0.70 * joint["mean_market_betting_rank"]
        + 0.30 * joint["mean_market_log_loss_rank"]
    )
    eligible_joint = joint.loc[joint["eligible_for_challenger_selection"]].copy()
    used_diagnostic_fallback = False
    if eligible_joint.empty:
        if not diagnostic_fallback_on_no_eligible:
            raise ValueError(
                "no non-baseline challenger survives the baseline dominance gate"
            )
        eligible_joint = joint.loc[~joint["candidate"].eq(baseline)].copy()
        if eligible_joint.empty:
            raise ValueError("diagnostic fallback has no non-baseline challenger to evaluate")
        used_diagnostic_fallback = True
    best_loss = eligible_joint["market_first_rank_score"].min()
    tied = sorted(
        eligible_joint.loc[
            eligible_joint["market_first_rank_score"].le(best_loss + 1e-12),
            "candidate",
        ]
    )
    selected = tied[0]
    status = _selection_status(diagnostic_fallback=used_diagnostic_fallback)
    for key, value in status.items():
        joint[key] = value
    joint["selected"] = joint["candidate"].eq(selected)
    joint["qualified_winner"] = joint["selected"] & bool(status["promotion_eligible"])
    joint["diagnostic_fallback_selected"] = (
        joint["selected"] & used_diagnostic_fallback
    )
    joint = joint.sort_values(
        ["eligible_for_challenger_selection", "market_first_rank_score", "candidate"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return joint, selected


def select_and_settle_best_close(
    scored: pd.DataFrame,
    *,
    min_ev: float = 0.03,
    sizings: Sequence[str] = ("flat_1u", "risk_to_win_1u"),
    price_point: str = "close",
) -> pd.DataFrame:
    """Take one direction at the best executable close across the three books."""

    if price_point not in {"open", "close"}:
        raise ValueError("price_point must be 'open' or 'close'")
    p1_price = f"p1_{price_point}"
    p2_price = f"p2_{price_point}"
    if p1_price not in scored or p2_price not in scored:
        return pd.DataFrame()
    valid_price = valid_american_odds_mask(
        pd.to_numeric(scored[p1_price], errors="coerce")
    ) & valid_american_odds_mask(pd.to_numeric(scored[p2_price], errors="coerce"))
    settlement_column = (
        "settlement_p1_outcome"
        if "settlement_p1_outcome" in scored
        else "physical_p1_outcome"
    )
    graded = scored.loc[
        scored[settlement_column].isin([0.0, 0.5, 1.0])
        & scored["book"].isin(BOOKS)
        & valid_price
    ].copy()
    if graded.empty:
        return pd.DataFrame()
    side_parts: list[pd.DataFrame] = []
    for side, other in (("p1", "p2"), ("p2", "p1")):
        part = graded.copy()
        part["side"] = side
        part["selection_id"] = part[f"{side}_id"]
        part["opponent_id"] = part[f"{other}_id"]
        part["close_odds"] = part[f"{side}_{price_point}"].astype(float)
        part["probability"] = part[f"{side}_probability"].astype(float)
        other_probability = part[f"{other}_probability"].astype(float)
        tie_loss = part["tie_rule"].map(ties_lose)
        part["effective_push_probability"] = np.where(
            tie_loss, 0.0, part["push_probability"]
        )
        part["loss_probability"] = other_probability + np.where(
            tie_loss, part["push_probability"], 0.0
        )
        decimal = np.asarray(american_decimal_odds(part["close_odds"]), dtype=float)
        part["ev_per_unit"] = part["probability"] * (decimal - 1.0) - part["loss_probability"]
        part["pricing_probability"] = part["probability"] / np.clip(
            part["probability"] + part["loss_probability"], EPS, 1.0
        )
        outcome = part[settlement_column].to_numpy(float)
        side_outcome = np.where(
            np.isclose(outcome, 0.5),
            0.5,
            np.where(
                (side == "p1") & np.isclose(outcome, 1.0)
                | (side == "p2") & np.isclose(outcome, 0.0),
                1.0,
                0.0,
            ),
        )
        part["outcome_numeric"] = side_outcome
        part["is_tie"] = np.isclose(outcome, 0.5)
        part["selected_skill"] = part.get(
            f"{side}_prediction", pd.Series(np.nan, index=part.index)
        )
        part["opponent_skill"] = part.get(
            f"{other}_prediction", pd.Series(np.nan, index=part.index)
        )
        side_parts.append(part)
    lines = pd.concat(side_parts, ignore_index=True)
    lines["_book_rank"] = lines["book"].map(BOOK_PRIORITY).fillna(99)
    side_keys = ["candidate", "event_key", "market", "pair_key", "side"]
    # EV is the correct cross-format comparison when one book pushes ties and
    # another grades them as losses; raw price breaks remaining ties.
    best_side = (
        lines.sort_values(
            side_keys + ["ev_per_unit", "close_odds", "_book_rank"],
            ascending=[True] * len(side_keys) + [False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(side_keys, keep="first")
        .drop(columns="_book_rank")
    )
    direction_keys = ["candidate", "event_key", "market", "pair_key"]
    chosen = (
        best_side.sort_values(
            direction_keys + ["ev_per_unit", "close_odds", "side"],
            ascending=[True] * len(direction_keys) + [False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(direction_keys, keep="first")
    )
    chosen = chosen.loc[chosen["ev_per_unit"].ge(float(min_ev))].copy()
    if chosen.empty:
        return pd.DataFrame()
    settled_parts: list[pd.DataFrame] = []
    for sizing in sizings:
        piece = chosen.copy()
        if sizing == "flat_1u":
            piece["stake"] = 1.0
        elif sizing == "risk_to_win_1u":
            odds = pd.to_numeric(piece["close_odds"], errors="coerce")
            piece["stake"] = np.where(
                odds.lt(0), odds.abs() / 100.0, 100.0 / odds
            )
        elif sizing == "quarter_kelly_100u_cap2u":
            decimal = np.asarray(american_decimal_odds(piece["close_odds"]), dtype=float)
            resolved_mass = piece["probability"] + piece["loss_probability"]
            full_kelly = piece["ev_per_unit"] / np.clip(
                (decimal - 1.0) * resolved_mass, EPS, None
            )
            piece["stake"] = 100.0 * np.clip(0.25 * full_kelly, 0.0, 0.02)
        else:
            raise ValueError(f"unknown sizing: {sizing}")
        piece["sizing"] = sizing
        piece["round_market"] = piece["market"]
        piece["market"] = "tournament_matchup"
        piece["event_id"] = piece["event_key"]
        piece["price_point"] = price_point
        piece["line_policy"] = f"best_of_betonline_betcris_pinnacle_{price_point}"
        settled_parts.append(settle_bets(piece))
    return pd.concat(settled_parts, ignore_index=True, sort=False)


def summarize_bets(bets: pd.DataFrame, sample: str = "evaluation") -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    slices: list[tuple[str, pd.DataFrame]] = [
        (market, bets.loc[bets["round_market"].eq(market)]) for market in MARKETS
    ]
    slices.append(("pooled", bets))
    for market, market_frame in slices:
        for (candidate, sizing), group in market_frame.groupby(
            ["candidate", "sizing"], observed=True
        ):
            stake = float(group["stake"].sum())
            pnl = float(group["pnl"].sum())
            rows.append(
                {
                    "sample": sample,
                    "candidate": candidate,
                    "market": market,
                    "sizing": sizing,
                    "events": int(group["event_key"].nunique()),
                    "bets": int(len(group)),
                    "stake": stake,
                    "pnl": pnl,
                    "roi": pnl / stake if stake else np.nan,
                    "mean_ev_per_unit": float(group["ev_per_unit"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_edge_buckets(
    scored: pd.DataFrame,
    *,
    price_point: str,
    sample: str = "evaluation",
) -> pd.DataFrame:
    """Settle the frozen 0-3/3-5/5-10/>10% theoretical-EV buckets."""

    opportunities = select_and_settle_best_close(
        scored,
        min_ev=0.0,
        sizings=("flat_1u",),
        price_point=price_point,
    )
    if opportunities.empty:
        return pd.DataFrame()
    opportunities["edge_bucket"] = pd.cut(
        opportunities["ev_per_unit"],
        bins=[0.0, 0.03, 0.05, 0.10, np.inf],
        labels=["0-3%", "3-5%", "5-10%", ">10%"],
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for keys, group in opportunities.groupby(
        ["candidate", "round_market", "edge_bucket"], observed=True
    ):
        candidate, market, bucket = keys
        stake = float(group["stake"].sum())
        pnl = float(group["pnl"].sum())
        rows.append(
            {
                "sample": sample,
                "price_point": price_point,
                "candidate": candidate,
                "market": market,
                "edge_bucket": str(bucket),
                "events": int(group["event_key"].nunique()),
                "bets": int(len(group)),
                "stake": stake,
                "pnl": pnl,
                "roi": pnl / stake if stake else np.nan,
                "mean_ev_per_unit": float(group["ev_per_unit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_bet_breakdowns(
    bets: pd.DataFrame,
    *,
    sample: str = "evaluation",
) -> pd.DataFrame:
    """Long-form year/book/skill/event/golfer betting diagnostics."""

    if bets.empty:
        return pd.DataFrame()
    work = bets.copy()
    work["year"] = pd.to_datetime(work["event_start_date"], errors="coerce").dt.year
    work["skill_bucket"] = pd.cut(
        pd.to_numeric(work.get("selected_skill"), errors="coerce"),
        bins=[-np.inf, -0.5, 0.0, 0.5, np.inf],
        labels=["<-0.5", "-0.5-0", "0-0.5", ">=0.5"],
        right=False,
    ).astype(str)
    dimensions = {
        "year": "year",
        "selected_book": "book",
        "skill_bucket": "skill_bucket",
        "event_week": "event_key",
        "golfer": "selection_id",
    }
    rows: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        for keys, group in work.groupby(
            ["price_point", "candidate", "round_market", "sizing", column],
            observed=True,
            dropna=False,
        ):
            price_point, candidate, market, sizing, value = keys
            stake = float(group["stake"].sum())
            pnl = float(group["pnl"].sum())
            rows.append(
                {
                    "sample": sample,
                    "price_point": price_point,
                    "candidate": candidate,
                    "market": market,
                    "sizing": sizing,
                    "dimension": dimension,
                    "value": str(value),
                    "events": int(group["event_key"].nunique()),
                    "bets": int(len(group)),
                    "stake": stake,
                    "pnl": pnl,
                    "roi": pnl / stake if stake else np.nan,
                    "mean_ev_per_unit": float(group["ev_per_unit"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_interval(
    values: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    *,
    aggregate: str,
) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan, np.nan
    indices = rng.integers(0, len(values), size=(int(draws), len(values)))
    sampled = values[indices]
    statistic = sampled.mean(axis=1) if aggregate == "mean" else sampled.sum(axis=1)
    estimate = float(values.mean() if aggregate == "mean" else values.sum())
    return (
        estimate,
        float(np.quantile(statistic, 0.025)),
        float(np.quantile(statistic, 0.975)),
        float(np.mean(statistic < 0.0) if aggregate == "mean" else np.mean(statistic > 0.0)),
    )


def paired_event_bootstrap(
    scored: pd.DataFrame,
    bets: pd.DataFrame,
    *,
    baseline: str,
    draws: int,
    seed: int = 20260823,
) -> pd.DataFrame:
    """Paired whole-event intervals for proper-score and P&L deltas."""

    if draws <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    logical = _logical_pairs(scored)
    graded = logical.loc[logical["physical_p1_outcome"].isin([0.0, 0.5, 1.0])].copy()
    outcome = graded["physical_p1_outcome"].to_numpy(float)
    chosen = np.where(
        outcome == 1.0,
        graded["p1_probability"],
        np.where(outcome == 0.0, graded["p2_probability"], graded["push_probability"]),
    )
    graded["three_way_log_loss"] = -np.log(np.clip(chosen, EPS, 1.0))
    rows: list[dict[str, Any]] = []
    candidates = sorted(set(graded["candidate"]) - {baseline})
    for market in MARKETS:
        market_frame = graded.loc[graded["market"].eq(market)]
        baseline_frame = market_frame.loc[
            market_frame["candidate"].eq(baseline),
            ["event_key", "pair_key", "physical_p1_outcome", "three_way_log_loss"],
        ].rename(columns={"three_way_log_loss": "baseline_value"})
        for candidate in candidates:
            challenger = market_frame.loc[
                market_frame["candidate"].eq(candidate),
                ["event_key", "pair_key", "physical_p1_outcome", "three_way_log_loss"],
            ].rename(
                columns={
                    "physical_p1_outcome": "candidate_outcome",
                    "three_way_log_loss": "candidate_value",
                }
            )
            paired = baseline_frame.merge(
                challenger,
                on=["event_key", "pair_key"],
                how="inner",
                validate="one_to_one",
            )
            if paired.empty:
                continue
            if not np.allclose(paired["physical_p1_outcome"], paired["candidate_outcome"]):
                raise AssertionError("paired bootstrap outcomes disagree")
            event_delta = (
                paired.assign(delta=paired["candidate_value"] - paired["baseline_value"])
                .groupby("event_key", observed=True)["delta"]
                .mean()
            )
            estimate, low, high, better = _bootstrap_interval(
                event_delta.to_numpy(float), draws, rng, aggregate="mean"
            )
            rows.append(
                {
                    "metric": "event_equal_three_way_log_loss_delta",
                    "market": market,
                    "candidate": candidate,
                    "baseline": baseline,
                    "sizing": "none",
                    "events": int(len(event_delta)),
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "probability_better": better,
                    "draws": int(draws),
                }
            )

    if not bets.empty:
        all_events_by_market = {
            market: sorted(scored.loc[scored["market"].eq(market), "event_key"].unique())
            for market in MARKETS
        }
        all_events_by_market["pooled"] = sorted(scored["event_key"].unique())
        for sizing in sorted(bets["sizing"].unique()):
            sizing_bets = bets.loc[bets["sizing"].eq(sizing)]
            for market in (*MARKETS, "pooled"):
                market_bets = (
                    sizing_bets
                    if market == "pooled"
                    else sizing_bets.loc[sizing_bets["round_market"].eq(market)]
                )
                event_index = all_events_by_market[market]
                event_pnl = (
                    market_bets.groupby(["event_key", "candidate"], observed=True)["pnl"]
                    .sum()
                    .unstack("candidate")
                    .reindex(event_index, fill_value=0.0)
                    .fillna(0.0)
                )
                if baseline not in event_pnl:
                    event_pnl[baseline] = 0.0
                for candidate in candidates:
                    if candidate not in event_pnl:
                        event_pnl[candidate] = 0.0
                    delta = event_pnl[candidate].to_numpy(float) - event_pnl[baseline].to_numpy(float)
                    estimate, low, high, better = _bootstrap_interval(
                        delta, draws, rng, aggregate="sum"
                    )
                    rows.append(
                        {
                            "metric": "pnl_delta",
                            "market": market,
                            "candidate": candidate,
                            "baseline": baseline,
                            "sizing": sizing,
                            "events": int(len(delta)),
                            "estimate": estimate,
                            "ci_low": low,
                            "ci_high": high,
                            "probability_better": better,
                            "draws": int(draws),
                        }
                    )
    return pd.DataFrame(rows)


def _write_outputs(result: JointMarketEvaluation, output_dir: Path, config: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_outputs = {
        "selection_summary.csv": result.selection_summary,
        "selection_joint_summary.csv": result.selection_joint_summary,
        "evaluation_summary.csv": result.evaluation_summary,
        "probability_mappers_apr_jun.csv": result.mapper_crossfit,
        "probability_mappers_final_2023.csv": result.mapper_final,
        "exact_common_menu_audit.csv": result.common_menu_audit,
        "selection_betting_summary.csv": result.selection_betting_summary,
        "betting_summary.csv": result.betting_summary,
        "opening_price_betting_summary.csv": result.open_betting_summary,
        "edge_bucket_summary.csv": result.edge_bucket_summary,
        "betting_breakdowns.csv": result.betting_breakdowns,
        "paired_event_bootstrap.csv": result.bootstrap,
    }
    for name, frame in csv_outputs.items():
        frame.to_csv(output_dir / name, index=False)
    result.scored_selection.to_parquet(output_dir / "scored_selection_oof.parquet", index=False)
    result.scored_evaluation.to_parquet(output_dir / "scored_evaluation_2024_2026.parquet", index=False)
    result.selection_bets.to_parquet(output_dir / "selection_best_close_bets.parquet", index=False)
    result.bets.to_parquet(output_dir / "best_close_bets.parquet", index=False)
    result.open_bets.to_parquet(output_dir / "best_open_bets.parquet", index=False)
    payload = {
        **dict(config),
        "selected_candidate": result.selected_candidate,
        **{field: getattr(result, field) for field in SELECTION_STATUS_FIELDS},
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_candidate_catalog(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.casefold() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        rows = value.get("candidates", value) if isinstance(value, dict) else value
        catalog = pd.DataFrame(rows)
    else:
        catalog = _read_frame(source)
    _require(
        catalog,
        {"candidate", "course_fit_lambda", "live_update_lambda"},
        "candidate catalog",
    )
    catalog = catalog.copy()
    catalog["candidate"] = catalog["candidate"].astype(str)
    if catalog.duplicated("candidate").any():
        raise ValueError("candidate catalog has duplicate candidate rows")
    for column in ("course_fit_lambda", "live_update_lambda"):
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
        if catalog[column].isna().any() or ~catalog[column].isin([0.0, 0.5, 1.0]).all():
            raise ValueError(f"candidate catalog {column} must be one of 0, 0.5, 1")
    return catalog


def _selected_spec(catalog: pd.DataFrame, candidate: str) -> dict[str, Any]:
    selected = catalog.loc[catalog["candidate"].eq(candidate)]
    if len(selected) != 1:
        raise ValueError(f"candidate catalog has no unique row for {candidate!r}")
    row = selected.iloc[0]
    return {
        "candidate": candidate,
        "course_fit_lambda": float(row["course_fit_lambda"]),
        "live_update_lambda": float(row["live_update_lambda"]),
    }


def write_selection_freeze(
    output_dir: str | Path,
    selection: JointCandidateSelection,
    *,
    baseline: str,
    min_ev: float,
    candidate_catalog_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> Path:
    """Write a self-hashed selection authority binding every frozen choice."""

    if not math.isclose(float(min_ev), 0.03, abs_tol=1e-12):
        raise ValueError("the authoritative selection freeze requires the contract 3% EV gate")
    contract = Path(contract_path).resolve()
    contract_hash = _sha256_file(contract)
    if contract_hash.casefold() != EXPECTED_CONTRACT_SHA256:
        raise ValueError(
            f"all-round contract hash changed: {contract_hash}; expected {EXPECTED_CONTRACT_SHA256}"
        )
    catalog_path = Path(candidate_catalog_path).resolve()
    catalog = _read_candidate_catalog(catalog_path)
    scored_candidates = set(selection.selection_joint_summary["candidate"].astype(str))
    missing = scored_candidates - set(catalog["candidate"])
    if missing:
        raise ValueError(f"candidate catalog omits scored candidates: {sorted(missing)}")
    if baseline not in scored_candidates:
        raise ValueError("selection result does not contain the declared baseline")
    if selection.selected_candidate == baseline:
        raise ValueError("the baseline is a comparator and cannot be frozen as the challenger")
    status = _status_from_selection(selection)
    joint = selection.selection_joint_summary
    for field, expected in status.items():
        if field not in joint or not joint[field].eq(expected).all():
            raise ValueError(f"selection summary does not consistently bind {field}")
    eligible_challenger_count = int(
        joint["eligible_for_challenger_selection"].astype(bool).sum()
    )
    selected_row = joint.loc[joint["candidate"].eq(selection.selected_candidate)]
    if len(selected_row) != 1:
        raise ValueError("selection summary lacks one unique selected challenger row")
    selected_survived_dominance = bool(
        selected_row.iloc[0]["eligible_for_challenger_selection"]
    )
    if status["selection_mode"] == DIAGNOSTIC_SELECTION_MODE:
        if eligible_challenger_count != 0 or selected_survived_dominance:
            raise ValueError("diagnostic fallback is permitted only after no challenger survives")
    elif eligible_challenger_count < 1 or not selected_survived_dominance:
        raise ValueError("primary selection must freeze a dominance-eligible challenger")
    selected_spec = _selected_spec(catalog, selection.selected_candidate)
    frozen_spec = {
        **selected_spec,
        **status,
        "eligible_challenger_count": eligible_challenger_count,
        "selected_survived_dominance": selected_survived_dominance,
        "baseline": baseline,
        "books": list(BOOKS),
        "primary_price_point": "close",
        "primary_sizing": "flat_1u",
        "minimum_ev": 0.03,
        "probability_mapper_training_start": "2023-04-01",
        "probability_mapper_training_end": "2023-06-30",
        "architecture_selection_start": "2023-07-01",
        "architecture_selection_end": "2023-12-31",
        "final_mapper_training_start": "2023-04-01",
        "final_mapper_training_end": "2023-12-31",
        "evaluation_start": "2024-01-01",
        "evaluation_end": "2026-12-31",
        "bootstrap_draws": LOCKED_BOOTSTRAP_DRAWS,
        "bootstrap_seed": LOCKED_BOOTSTRAP_SEED,
        "candidate_selection_rule": (
            "equal-market 70% flat-risk 1u P&L rank at 3% EV + "
            "30% three-way-log-loss rank"
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 3,
        "research_only": True,
        "contract_path": str(contract),
        "contract_sha256": contract_hash,
        "candidate_catalog_path": str(catalog_path),
        "candidate_catalog_sha256": _sha256_file(catalog_path),
        "baseline": baseline,
        "selected_candidate": selection.selected_candidate,
        **status,
        "eligible_challenger_count": eligible_challenger_count,
        "selected_survived_dominance": selected_survived_dominance,
        "selected_spec": selected_spec,
        "frozen_spec": frozen_spec,
        "frozen_spec_sha256": _canonical_payload_hash(frozen_spec),
    }
    payload["freeze_payload_sha256"] = _canonical_payload_hash(payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    freeze_path = destination / "selection_freeze.json"
    freeze_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (destination / "selection_freeze.sha256").write_text(
        _sha256_file(freeze_path) + "  selection_freeze.json\n", encoding="utf-8"
    )
    return freeze_path


def validate_selection_freeze(
    freeze_path: str | Path,
    *,
    baseline: str,
    min_ev: float,
    selection_start: str | pd.Timestamp,
    selection_end: str | pd.Timestamp,
    candidate_catalog_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    eval_start: str | pd.Timestamp = LOCKED_EVALUATION_START,
    eval_end: str | pd.Timestamp = LOCKED_EVALUATION_END,
    bootstrap_draws: int = LOCKED_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = LOCKED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate the freeze and all bound authorities before test labels load."""

    path = Path(freeze_path).resolve()
    sidecar = path.with_suffix(".sha256")
    if not sidecar.exists():
        raise ValueError("selection freeze lacks its SHA-256 sidecar")
    expected_file_hash = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual_file_hash = _sha256_file(path)
    if actual_file_hash.casefold() != expected_file_hash.casefold():
        raise ValueError("selection freeze file hash disagrees with its sidecar")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 3:
        raise ValueError("unsupported selection freeze schema")
    recorded_payload_hash = str(payload.get("freeze_payload_sha256", ""))
    hash_input = dict(payload)
    hash_input.pop("freeze_payload_sha256", None)
    if _canonical_payload_hash(hash_input) != recorded_payload_hash:
        raise ValueError("selection freeze payload self-hash is invalid")

    contract = Path(contract_path).resolve()
    contract_hash = _sha256_file(contract)
    if contract_hash.casefold() != EXPECTED_CONTRACT_SHA256:
        raise ValueError("current all-round contract does not match the frozen authority")
    if str(payload.get("contract_sha256", "")).casefold() != contract_hash.casefold():
        raise ValueError("selection freeze contract hash mismatch")
    catalog_path = Path(candidate_catalog_path).resolve()
    _read_candidate_catalog(catalog_path)
    if str(payload.get("candidate_catalog_sha256", "")).casefold() != _sha256_file(
        catalog_path
    ).casefold():
        raise ValueError("selection freeze candidate-catalog hash mismatch")
    if str(payload.get("baseline")) != baseline:
        raise ValueError("selection freeze baseline mismatch")
    if not math.isclose(float(min_ev), 0.03, abs_tol=1e-12):
        raise ValueError("evaluation gate must remain at the frozen 3%")
    frozen_spec = payload.get("frozen_spec", {})
    if _canonical_payload_hash(frozen_spec) != str(payload.get("frozen_spec_sha256", "")):
        raise ValueError("selection freeze frozen-spec hash mismatch")
    if not math.isclose(float(frozen_spec.get("minimum_ev", np.nan)), 0.03, abs_tol=1e-12):
        raise ValueError("selection freeze does not bind the 3% gate")
    if str(frozen_spec.get("baseline")) != baseline:
        raise ValueError("selection freeze frozen-spec baseline mismatch")
    status = _validate_selection_status(payload)
    for field, expected in status.items():
        if frozen_spec.get(field) != expected:
            raise ValueError(f"selection freeze frozen-spec mismatch for {field}")
    eligible_challenger_count = int(payload.get("eligible_challenger_count", -1))
    selected_survived_dominance = bool(payload.get("selected_survived_dominance"))
    if int(frozen_spec.get("eligible_challenger_count", -1)) != eligible_challenger_count:
        raise ValueError("selection freeze eligible-challenger count mismatch")
    if bool(frozen_spec.get("selected_survived_dominance")) != selected_survived_dominance:
        raise ValueError("selection freeze selected dominance status mismatch")
    if status["selection_mode"] == DIAGNOSTIC_SELECTION_MODE:
        if eligible_challenger_count != 0 or selected_survived_dominance:
            raise ValueError("selection freeze diagnostic fallback was not gate-triggered")
    elif eligible_challenger_count < 1 or not selected_survived_dominance:
        raise ValueError("selection freeze primary challenger did not survive dominance")
    expected_periods = {
        "probability_mapper_training_start": "2023-04-01",
        "probability_mapper_training_end": "2023-06-30",
        "architecture_selection_start": "2023-07-01",
        "architecture_selection_end": "2023-12-31",
        "final_mapper_training_start": "2023-04-01",
        "final_mapper_training_end": "2023-12-31",
        "evaluation_start": "2024-01-01",
        "evaluation_end": "2026-12-31",
    }
    for key, expected in expected_periods.items():
        if str(frozen_spec.get(key)) != expected:
            raise ValueError(f"selection freeze period mismatch for {key}")
    if pd.Timestamp(selection_start).normalize() != MAPPER_TRAIN_START:
        raise ValueError("evaluation selection_start differs from the frozen boundary")
    if pd.Timestamp(selection_end).normalize() != ARCHITECTURE_SELECTION_END:
        raise ValueError("evaluation selection_end differs from the frozen boundary")
    if pd.Timestamp(eval_start).normalize() != LOCKED_EVALUATION_START:
        raise ValueError("evaluation eval_start differs from the frozen boundary")
    if pd.Timestamp(eval_end).normalize() != LOCKED_EVALUATION_END:
        raise ValueError("evaluation eval_end differs from the frozen boundary")
    if int(bootstrap_draws) != LOCKED_BOOTSTRAP_DRAWS:
        raise ValueError("evaluation bootstrap_draws differs from the frozen 10,000")
    if int(bootstrap_seed) != LOCKED_BOOTSTRAP_SEED:
        raise ValueError("evaluation bootstrap_seed differs from the frozen deterministic seed")
    if int(frozen_spec.get("bootstrap_draws", -1)) != LOCKED_BOOTSTRAP_DRAWS:
        raise ValueError("selection freeze bootstrap draw count mismatch")
    if int(frozen_spec.get("bootstrap_seed", -1)) != LOCKED_BOOTSTRAP_SEED:
        raise ValueError("selection freeze bootstrap seed mismatch")
    if tuple(frozen_spec.get("books", ())) != BOOKS:
        raise ValueError("selection freeze bookmaker menu mismatch")
    if frozen_spec.get("primary_sizing") != "flat_1u":
        raise ValueError("selection freeze primary sizing mismatch")
    selected = str(payload.get("selected_candidate"))
    if selected == baseline:
        raise ValueError("selection freeze improperly names the baseline as challenger")
    expected_selected_spec = _selected_spec(_read_candidate_catalog(catalog_path), selected)
    if payload.get("selected_spec") != expected_selected_spec:
        raise ValueError("selection freeze selected lambda specification mismatch")
    for key, expected in expected_selected_spec.items():
        if frozen_spec.get(key) != expected:
            raise ValueError(f"selection freeze frozen-spec mismatch for {key}")
    return payload


def validate_evaluation_prediction_authority(
    authority_path: str | Path,
    *,
    base_predictions_path: str | Path,
    round_predictions_path: str | Path,
    selection_freeze_path: str | Path,
    frozen_payload: Mapping[str, Any],
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Bind the sealed prediction panels to the already-validated freeze.

    This performs only authority/file hashing.  It must complete before either
    combined prediction Parquet is loaded and before any 2024-26 quote label is
    materialized.
    """

    authority = validate_prediction_authority(
        authority_path,
        combined_base_predictions_path=base_predictions_path,
        combined_round_predictions_path=round_predictions_path,
    )
    required = {
        "schema_version",
        "research_only",
        "production_files_written",
        "phase",
        "contract_sha256",
        "training_catalog_sha256",
        "selection_freeze_sha256",
        "selected_candidate",
        "baseline",
        "downstream_lambda",
        "live_update_lambda",
        "promotion_eligible",
        "diagnostic_fallback",
        "selection_mode",
        "candidate_closure",
        "selection_years",
        "evaluation_years",
        "evaluation_outcome_quote_or_label_columns_read",
        "outputs",
    }
    missing = sorted(required - set(authority))
    if missing:
        raise ValueError(f"prediction authority is missing required fields: {missing}")
    if int(authority["schema_version"]) != 1:
        raise ValueError("unsupported prediction authority schema")
    if authority["phase"] != "sealed_joint_market_prediction_panels":
        raise ValueError("prediction authority has the wrong phase")

    contract_hash = _sha256_file(contract_path)
    if contract_hash.casefold() != EXPECTED_CONTRACT_SHA256:
        raise ValueError("current contract differs from the locked all-round contract")
    if str(authority["contract_sha256"]).casefold() != contract_hash.casefold():
        raise ValueError("prediction authority contract hash mismatch")
    if str(frozen_payload.get("contract_sha256", "")).casefold() != contract_hash.casefold():
        raise ValueError("selection freeze and prediction authority contract mismatch")
    if authority["selection_years"] != [2023]:
        raise ValueError("prediction authority selection years are not exactly [2023]")
    if authority["evaluation_years"] != [2024, 2025, 2026]:
        raise ValueError("prediction authority evaluation years are not exactly 2024-2026")
    if authority["evaluation_outcome_quote_or_label_columns_read"] != []:
        raise ValueError("prediction authority reports reading evaluation labels or quotes")

    freeze_hash = _sha256_file(selection_freeze_path)
    if str(authority["selection_freeze_sha256"]).casefold() != freeze_hash.casefold():
        raise ValueError("prediction authority selection-freeze hash mismatch")
    selected = str(frozen_payload.get("selected_candidate", ""))
    baseline = str(frozen_payload.get("baseline", ""))
    if str(authority["selected_candidate"]) != selected:
        raise ValueError("prediction authority selected candidate disagrees with freeze")
    if str(authority["baseline"]) != baseline:
        raise ValueError("prediction authority baseline disagrees with freeze")
    if authority["candidate_closure"] != sorted([baseline, selected]):
        raise ValueError("prediction authority candidate closure is not baseline plus selected")

    status = _validate_selection_status(frozen_payload)
    if str(authority["selection_mode"]) != status["selection_mode"]:
        raise ValueError("prediction authority selection mode disagrees with freeze")
    for key in ("promotion_eligible", "diagnostic_fallback"):
        if not isinstance(authority[key], bool):
            raise ValueError(f"prediction authority {key} is not boolean")
    if authority["promotion_eligible"] is not status["promotion_eligible"]:
        raise ValueError("prediction authority promotion eligibility disagrees with freeze")
    expected_fallback = status["selection_mode"] == DIAGNOSTIC_SELECTION_MODE
    if authority["diagnostic_fallback"] is not expected_fallback:
        raise ValueError("prediction authority diagnostic-fallback status disagrees with freeze")

    selected_spec = frozen_payload.get("selected_spec", {})
    expected_downstream = float(selected_spec.get("course_fit_lambda", np.nan))
    expected_live = float(selected_spec.get("live_update_lambda", np.nan))
    if not math.isclose(
        float(authority["downstream_lambda"]), expected_downstream, abs_tol=1e-12
    ):
        raise ValueError("prediction authority downstream lambda disagrees with freeze")
    if not math.isclose(
        float(authority["live_update_lambda"]), expected_live, abs_tol=1e-12
    ):
        raise ValueError("prediction authority live lambda disagrees with freeze")

    outputs = authority["outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "combined_base_predictions",
        "combined_round_predictions",
    }:
        raise ValueError("prediction authority output menu is not the exact combined pair")
    supplied_paths = {
        "combined_base_predictions": Path(base_predictions_path).resolve(),
        "combined_round_predictions": Path(round_predictions_path).resolve(),
    }
    for name, supplied in supplied_paths.items():
        record = outputs[name]
        if not isinstance(record, Mapping) or not {"path", "sha256", "rows"}.issubset(record):
            raise ValueError(f"prediction authority {name} record is incomplete")
        if Path(str(record["path"])).resolve() != supplied:
            raise ValueError(f"prediction authority {name} path differs from CLI")
        if str(record["sha256"]).casefold() != _sha256_file(supplied).casefold():
            raise ValueError(f"prediction authority {name} hash mismatch")
        if int(record["rows"]) <= 0:
            raise ValueError(f"prediction authority {name} has no sealed rows")
    return authority


def select_joint_candidate_2023(
    base_predictions: pd.DataFrame,
    round_predictions: pd.DataFrame | None,
    tournament_quotes: pd.DataFrame,
    round_quotes: pd.DataFrame,
    output_dir: str | Path | None,
    baseline: str,
    selection_start: str | pd.Timestamp = "2023-04-01",
    selection_end: str | pd.Timestamp = "2023-12-31",
    folds: int = 5,
    *,
    min_ev: float = 0.03,
    diagnostic_fallback_on_no_eligible: bool = False,
) -> JointCandidateSelection:
    """Fit maps on Apr-Jun OOF pairs and select on frozen Jul-Dec scores."""

    requested_start = pd.Timestamp(selection_start).normalize()
    if requested_start != MAPPER_TRAIN_START:
        raise ValueError("selection_start must equal the frozen 2023-04-01 boundary")
    contract_selection_start = MAPPER_TRAIN_START
    tournament_selection_quotes = _prefilter_raw_quote_frame(
        tournament_quotes,
        contract_selection_start,
        selection_end,
        label="tournament selection quotes",
    )
    round_selection_quotes = _prefilter_raw_quote_frame(
        round_quotes,
        contract_selection_start,
        selection_end,
        label="round selection quotes",
    )

    # A deliberately empty far-future evaluation window lets the common-panel
    # validator serve the selection-only phase without inspecting 2024+ rows.
    panel, audit = build_exact_common_panel(
        base_predictions,
        round_predictions,
        tournament_selection_quotes,
        round_selection_quotes,
        baseline=baseline,
        selection_start=contract_selection_start,
        selection_end=selection_end,
        eval_start="2100-01-01",
        eval_end="2100-01-02",
    )
    selection_panel = panel.loc[panel["sample"].eq("selection")].copy()
    if selection_panel.empty:
        raise ValueError("the 2023 selection sample is empty")
    # ``folds`` is retained in the public signature for compatibility with the
    # superseded event-fold draft.  The frozen contract is chronological and
    # deliberately does not use it.
    if folds < 2:
        raise ValueError("folds compatibility argument must be at least 2")
    scored, crossfit = fit_chronological_selection_mappers(
        selection_panel, architecture_end=selection_end
    )
    summary = summarize_probabilities(scored, "selection_oof")
    selection_bets = select_and_settle_best_close(
        scored,
        min_ev=min_ev,
        sizings=("flat_1u", "risk_to_win_1u", "quarter_kelly_100u_cap2u"),
    )
    betting_summary = summarize_bets(selection_bets, sample="selection_oof")
    joint, selected = joint_selection_summary(
        summary,
        baseline,
        selection_bets,
        diagnostic_fallback_on_no_eligible=diagnostic_fallback_on_no_eligible,
    )
    status = _validate_selection_status(joint.iloc[0].to_dict())
    summary = _annotate_selection_status(summary, status)
    crossfit = _annotate_selection_status(crossfit, status)
    audit = _annotate_selection_status(
        audit.assign(phase="selection_2023_apr_dec"), status
    )
    scored = _annotate_selection_status(scored, status)
    selection_bets = _annotate_selection_status(selection_bets, status)
    betting_summary = _annotate_selection_status(betting_summary, status)
    result = JointCandidateSelection(
        selected_candidate=selected,
        **status,
        selection_summary=summary,
        selection_joint_summary=joint,
        mapper_crossfit=crossfit,
        common_menu_audit=audit,
        scored_selection=scored,
        selection_bets=selection_bets,
        selection_betting_summary=betting_summary,
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        result.selection_summary.to_csv(destination / "selection_summary.csv", index=False)
        result.selection_joint_summary.to_csv(
            destination / "selection_joint_summary.csv", index=False
        )
        result.mapper_crossfit.to_csv(
            destination / "probability_mappers_apr_jun.csv", index=False
        )
        result.common_menu_audit.to_csv(destination / "exact_common_menu_audit.csv", index=False)
        result.selection_betting_summary.to_csv(
            destination / "selection_betting_summary.csv", index=False
        )
        result.scored_selection.to_parquet(
            destination / "scored_selection_oof.parquet", index=False
        )
        result.selection_bets.to_parquet(
            destination / "selection_best_close_bets.parquet", index=False
        )
        (destination / "selection_status.json").write_text(
            json.dumps({"selected_candidate": selected, **status}, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def evaluate_frozen_joint_candidate(
    base_predictions: pd.DataFrame,
    round_predictions: pd.DataFrame | None,
    tournament_quotes: pd.DataFrame,
    round_quotes: pd.DataFrame,
    output_dir: str | Path | None,
    baseline: str,
    selected_candidate: str,
    selection_start: str | pd.Timestamp = "2023-04-01",
    selection_end: str | pd.Timestamp = "2023-12-31",
    eval_start: str | pd.Timestamp = "2024-01-01",
    eval_end: str | pd.Timestamp = "2026-12-31",
    bootstrap_draws: int = 10_000,
    *,
    bootstrap_seed: int = 20260823,
    min_ev: float = 0.03,
    selection_status: Mapping[str, Any] | None = None,
) -> FrozenJointEvaluation:
    """Score only the frozen selected challenger/control in 2024-26."""

    status = _validate_selection_status(
        _selection_status(diagnostic_fallback=False)
        if selection_status is None
        else selection_status
    )
    requested_start = pd.Timestamp(selection_start).normalize()
    if requested_start != MAPPER_TRAIN_START:
        raise ValueError("selection_start must equal the frozen 2023-04-01 boundary")
    contract_selection_start = MAPPER_TRAIN_START
    frozen_candidates = sorted({baseline, selected_candidate})
    panel, audit = build_exact_common_panel(
        base_predictions,
        round_predictions,
        tournament_quotes,
        round_quotes,
        baseline=baseline,
        selection_start=contract_selection_start,
        selection_end=selection_end,
        eval_start=eval_start,
        eval_end=eval_end,
        selection_candidates=frozen_candidates,
        evaluation_candidates=frozen_candidates,
    )
    selection_panel = panel.loc[panel["sample"].eq("selection")].copy()
    evaluation_panel = panel.loc[panel["sample"].eq("evaluation")].copy()
    if selection_panel.empty or evaluation_panel.empty:
        raise ValueError("frozen evaluation requires nonempty 2023 and 2024-26 panels")
    mapper_final = fit_final_probability_mappers(selection_panel)
    scored = _score_with_mappers(evaluation_panel, mapper_final)
    summary = summarize_probabilities(scored, "evaluation_frozen")
    summary["selected"] = summary["candidate"].eq(selected_candidate)
    summary["qualified_winner"] = summary["selected"] & bool(
        status["promotion_eligible"]
    )
    evaluation_sizings = (
        "flat_1u",
        "risk_to_win_1u",
        "quarter_kelly_100u_cap2u",
    )
    bets = select_and_settle_best_close(
        scored, min_ev=min_ev, sizings=evaluation_sizings, price_point="close"
    )
    betting_summary = summarize_bets(bets)
    open_bets = select_and_settle_best_close(
        scored, min_ev=min_ev, sizings=evaluation_sizings, price_point="open"
    )
    open_betting_summary = summarize_bets(open_bets)
    if not open_betting_summary.empty:
        open_betting_summary["price_point"] = "open"
    if not betting_summary.empty:
        betting_summary["price_point"] = "close"
    edge_bucket_summary = pd.concat(
        [
            summarize_edge_buckets(scored, price_point="close"),
            summarize_edge_buckets(scored, price_point="open"),
        ],
        ignore_index=True,
    )
    betting_breakdowns = summarize_bet_breakdowns(
        pd.concat([bets, open_bets], ignore_index=True, sort=False)
    )
    bootstrap = paired_event_bootstrap(
        scored,
        bets,
        baseline=baseline,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    summary = _annotate_selection_status(summary, status)
    mapper_final = _annotate_selection_status(mapper_final, status)
    audit = _annotate_selection_status(
        audit.assign(phase="frozen_selected_challenger_control"), status
    )
    scored = _annotate_selection_status(scored, status)
    bets = _annotate_selection_status(bets, status)
    betting_summary = _annotate_selection_status(betting_summary, status)
    open_bets = _annotate_selection_status(open_bets, status)
    open_betting_summary = _annotate_selection_status(open_betting_summary, status)
    edge_bucket_summary = _annotate_selection_status(edge_bucket_summary, status)
    betting_breakdowns = _annotate_selection_status(betting_breakdowns, status)
    bootstrap = _annotate_selection_status(bootstrap, status)
    result = FrozenJointEvaluation(
        selected_candidate=selected_candidate,
        **status,
        evaluation_summary=summary,
        mapper_final=mapper_final,
        common_menu_audit=audit,
        scored_evaluation=scored,
        bets=bets,
        betting_summary=betting_summary,
        open_bets=open_bets,
        open_betting_summary=open_betting_summary,
        edge_bucket_summary=edge_bucket_summary,
        betting_breakdowns=betting_breakdowns,
        bootstrap=bootstrap,
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        result.evaluation_summary.to_csv(destination / "evaluation_summary.csv", index=False)
        result.mapper_final.to_csv(
            destination / "probability_mappers_final_2023.csv", index=False
        )
        result.common_menu_audit.to_csv(destination / "frozen_common_menu_audit.csv", index=False)
        result.betting_summary.to_csv(destination / "betting_summary.csv", index=False)
        result.open_betting_summary.to_csv(
            destination / "opening_price_betting_summary.csv", index=False
        )
        result.edge_bucket_summary.to_csv(destination / "edge_bucket_summary.csv", index=False)
        result.betting_breakdowns.to_csv(destination / "betting_breakdowns.csv", index=False)
        result.bootstrap.to_csv(destination / "paired_event_bootstrap.csv", index=False)
        result.scored_evaluation.to_parquet(
            destination / "scored_evaluation_2024_2026.parquet", index=False
        )
        result.bets.to_parquet(destination / "best_close_bets.parquet", index=False)
        result.open_bets.to_parquet(destination / "best_open_bets.parquet", index=False)
        (destination / "selection_status.json").write_text(
            json.dumps(
                {"selected_candidate": selected_candidate, **status}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    return result


def evaluate_joint_markets(
    base_predictions: pd.DataFrame,
    round_predictions: pd.DataFrame | None,
    tournament_quotes: pd.DataFrame,
    round_quotes: pd.DataFrame,
    output_dir: str | Path | None,
    baseline: str,
    selection_start: str | pd.Timestamp = "2023-04-01",
    selection_end: str | pd.Timestamp = "2023-12-31",
    eval_start: str | pd.Timestamp = "2024-01-01",
    eval_end: str | pd.Timestamp = "2026-12-31",
    folds: int = 5,
    bootstrap_draws: int = 10_000,
    *,
    bootstrap_seed: int = 20260823,
    min_ev: float = 0.03,
    allow_unfrozen_test_only: bool = False,
    diagnostic_fallback_on_no_eligible: bool = False,
) -> JointMarketEvaluation:
    """Compose both phases for synthetic tests, never for the locked evaluation.

    This helper necessarily observes the evaluation panel in the same process
    that selects the challenger, without an intervening on-disk hash authority.
    The real 2024-26 run must use the phase-separated CLI.  Callers must opt in
    explicitly so this convenience path cannot be mistaken for authoritative.
    """

    if not allow_unfrozen_test_only:
        raise ValueError(
            "evaluate_joint_markets is non-authoritative and test-only; use the "
            "phase-separated CLI, or pass allow_unfrozen_test_only=True for a "
            "synthetic test"
        )

    selection = select_joint_candidate_2023(
        base_predictions,
        round_predictions,
        tournament_quotes,
        round_quotes,
        None,
        baseline,
        selection_start,
        selection_end,
        folds,
        min_ev=min_ev,
        diagnostic_fallback_on_no_eligible=diagnostic_fallback_on_no_eligible,
    )
    frozen = evaluate_frozen_joint_candidate(
        base_predictions,
        round_predictions,
        tournament_quotes,
        round_quotes,
        None,
        baseline,
        selection.selected_candidate,
        selection_start,
        selection_end,
        eval_start,
        eval_end,
        bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        min_ev=min_ev,
        selection_status=_status_from_selection(selection),
    )
    result = JointMarketEvaluation(
        selected_candidate=selection.selected_candidate,
        primary_selection_outcome=selection.primary_selection_outcome,
        promotion_eligible=selection.promotion_eligible,
        selection_mode=selection.selection_mode,
        diagnostic_fallback_rule=selection.diagnostic_fallback_rule,
        selection_summary=selection.selection_summary,
        selection_joint_summary=selection.selection_joint_summary,
        evaluation_summary=frozen.evaluation_summary,
        mapper_crossfit=selection.mapper_crossfit,
        mapper_final=frozen.mapper_final,
        common_menu_audit=pd.concat(
            [selection.common_menu_audit, frozen.common_menu_audit], ignore_index=True
        ),
        scored_selection=selection.scored_selection,
        scored_evaluation=frozen.scored_evaluation,
        selection_bets=selection.selection_bets,
        selection_betting_summary=selection.selection_betting_summary,
        bets=frozen.bets,
        betting_summary=frozen.betting_summary,
        open_bets=frozen.open_bets,
        open_betting_summary=frozen.open_betting_summary,
        edge_bucket_summary=frozen.edge_bucket_summary,
        betting_breakdowns=frozen.betting_breakdowns,
        bootstrap=frozen.bootstrap,
    )
    if output_dir is not None:
        _write_outputs(
            result,
            Path(output_dir),
            {
                "research_only": True,
                "production_files_written": False,
                "baseline": baseline,
                **_status_from_selection(selection),
                "selection_start": str(MAPPER_TRAIN_START.date()),
                "selection_end": str(pd.Timestamp(selection_end).date()),
                "evaluation_start": str(pd.Timestamp(eval_start).date()),
                "evaluation_end": str(pd.Timestamp(eval_end).date()),
                "probability_mapper_training": "Apr-Jun 2023 chronological OOF",
                "architecture_selection": "Jul-Dec 2023 frozen mapper",
                "final_mapper_training": "Apr-Dec 2023 chronological OOF",
                "bootstrap_draws": int(bootstrap_draws),
                "bootstrap_seed": int(bootstrap_seed),
                "minimum_ev": float(min_ev),
                "books": list(BOOKS),
                "candidate_selection_metric": (
                    "70% equal-market flat-risk 1u P&L rank + 30% equal-market "
                    "three-way log-loss rank"
                ),
            },
        )
    return result


def _read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.casefold() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.casefold() in {".csv", ".txt"}:
        return pd.read_csv(source)
    raise ValueError(f"unsupported tabular input: {source}")


def _parquet_filter_value(value: pd.Timestamp, arrow_type: object) -> object:
    """Match the scalar type used by an Arrow date/timestamp/string field."""

    import pyarrow.types as arrow_types

    if arrow_types.is_string(arrow_type) or arrow_types.is_large_string(arrow_type):
        return str(value.date())
    if arrow_types.is_date(arrow_type):
        return value.date()
    return value.to_pydatetime()


def _read_quote_frame_window(
    path: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    selection_phase: bool,
) -> pd.DataFrame:
    """Predicate-push a quote-label window before outcome columns materialize."""

    source = Path(path)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError("quote read window is inverted")
    if source.suffix.casefold() not in {".parquet", ".pq"}:
        if selection_phase:
            raise ValueError(
                "selection quote inputs must be Parquet so the 2023 label boundary "
                "can be predicate-pushed before materialization"
            )
        raw = _read_frame(source)
        return _prefilter_raw_quote_frame(
            raw, start_ts, end_ts, label=f"evaluation quotes {source}"
        )

    import pyarrow.dataset as arrow_dataset

    schema = arrow_dataset.dataset(source, format="parquet").schema
    schema_names = list(schema.names)
    date_column = next(
        (
            column
            for column in ("event_start_date", "mapped_event_start_date")
            if column in schema_names
        ),
        None,
    )
    if date_column is None:
        raise ValueError(f"quote parquet {source} lacks an event-date filter column")
    field_type = schema.field(date_column).type
    filters = [
        (date_column, ">=", _parquet_filter_value(start_ts, field_type)),
        (date_column, "<=", _parquet_filter_value(end_ts, field_type)),
    ]
    columns = [column for column in schema_names if column in QUOTE_READ_COLUMNS]
    required_projection = {
        date_column,
        "book",
        "p1_outcome",
        "p2_outcome",
    }
    missing = sorted(required_projection - set(columns))
    if missing:
        raise ValueError(f"quote parquet projection lacks required columns: {missing}")
    if not {"market", "round_market", "round_num"}.intersection(columns):
        raise ValueError("quote parquet projection lacks market/round identity")
    frame = pd.read_parquet(source, columns=columns, filters=filters)
    # Defense in depth: predicate behavior is asserted before normalization.
    filtered_dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    outside = ~filtered_dates.between(start_ts, end_ts)
    if outside.any():
        raise AssertionError(
            f"Parquet predicate returned {int(outside.sum())} quote labels outside "
            f"{start_ts.date()}..{end_ts.date()}"
        )
    return frame


def _read_prediction_frame_window(
    path: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Project only pre-event prediction fields; push dates when available."""

    source = Path(path)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if source.suffix.casefold() in {".parquet", ".pq"}:
        import pyarrow.dataset as arrow_dataset

        schema = arrow_dataset.dataset(source, format="parquet").schema
        names = list(schema.names)
        columns = [column for column in names if column in PREDICTION_READ_COLUMNS]
        filters = None
        if "event_start_date" in names:
            arrow_type = schema.field("event_start_date").type
            filters = [
                ("event_start_date", ">=", _parquet_filter_value(start_ts, arrow_type)),
                ("event_start_date", "<=", _parquet_filter_value(end_ts, arrow_type)),
            ]
        return pd.read_parquet(source, columns=columns, filters=filters)
    if source.suffix.casefold() in {".csv", ".txt"}:
        header = pd.read_csv(source, nrows=0)
        columns = [column for column in header.columns if column in PREDICTION_READ_COLUMNS]
        frame = pd.read_csv(source, usecols=columns)
        if "event_start_date" in frame:
            dates = pd.to_datetime(frame["event_start_date"], errors="coerce")
            frame = frame.loc[dates.between(start_ts, end_ts)].copy()
        return frame
    raise ValueError(f"unsupported prediction input: {source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "evaluate"), required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--round-predictions", required=True)
    parser.add_argument("--tournament-quotes", required=True)
    parser.add_argument("--round-quotes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--selected-candidate")
    parser.add_argument("--selection-freeze")
    parser.add_argument("--prediction-authority")
    parser.add_argument("--candidate-catalog")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--selection-start", default="2023-04-01")
    parser.add_argument("--selection-end", default="2023-12-31")
    parser.add_argument("--eval-start", default="2024-01-01")
    parser.add_argument("--eval-end", default="2026-12-31")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--min-ev", type=float, default=0.03)
    parser.add_argument(
        "--diagnostic-fallback-on-no-eligible",
        action="store_true",
        help=(
            "after a failed primary dominance gate only, freeze the best-ranked "
            "non-baseline challenger as promotion-ineligible diagnostic"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.phase == "select":
        if args.prediction_authority:
            raise ValueError("--prediction-authority is evaluate-phase only")
        if not args.candidate_catalog:
            raise ValueError("select phase requires --candidate-catalog with frozen lambdas")
        # Validate non-label authorities before opening prediction or quote data.
        if _sha256_file(args.contract).casefold() != EXPECTED_CONTRACT_SHA256:
            raise ValueError("all-round contract hash differs from the frozen authority")
        _read_candidate_catalog(args.candidate_catalog)
        if not math.isclose(float(args.min_ev), 0.03, abs_tol=1e-12):
            raise ValueError("select phase is frozen to the 3% EV gate")
        if pd.Timestamp(args.selection_start).normalize() != MAPPER_TRAIN_START:
            raise ValueError("select phase must start at the frozen 2023-04-01 boundary")
        if pd.Timestamp(args.selection_end).normalize() != ARCHITECTURE_SELECTION_END:
            raise ValueError("select phase must end at the frozen 2023-12-31 boundary")
        if pd.Timestamp(args.eval_start).normalize() != LOCKED_EVALUATION_START:
            raise ValueError("select phase must bind eval_start to 2024-01-01")
        if pd.Timestamp(args.eval_end).normalize() != LOCKED_EVALUATION_END:
            raise ValueError("select phase must bind eval_end to 2026-12-31")
        if int(args.bootstrap_draws) != LOCKED_BOOTSTRAP_DRAWS:
            raise ValueError("select phase must bind bootstrap_draws to 10,000")
        if int(args.bootstrap_seed) != LOCKED_BOOTSTRAP_SEED:
            raise ValueError("select phase must bind the deterministic bootstrap seed")
        base = _read_prediction_frame_window(
            args.base_predictions, start=MAPPER_TRAIN_START, end=args.selection_end
        )
        rounds = _read_prediction_frame_window(
            args.round_predictions, start=MAPPER_TRAIN_START, end=args.selection_end
        )
        tournament = _read_quote_frame_window(
            args.tournament_quotes,
            start=MAPPER_TRAIN_START,
            end=args.selection_end,
            selection_phase=True,
        )
        round_quotes = _read_quote_frame_window(
            args.round_quotes,
            start=MAPPER_TRAIN_START,
            end=args.selection_end,
            selection_phase=True,
        )
        result = select_joint_candidate_2023(
            base,
            rounds,
            tournament,
            round_quotes,
            args.output_dir,
            args.baseline,
            args.selection_start,
            args.selection_end,
            args.folds,
            min_ev=args.min_ev,
            diagnostic_fallback_on_no_eligible=(
                args.diagnostic_fallback_on_no_eligible
            ),
        )
        freeze_path = write_selection_freeze(
            args.output_dir,
            result,
            baseline=args.baseline,
            min_ev=args.min_ev,
            candidate_catalog_path=args.candidate_catalog,
            contract_path=args.contract,
        )
        if result.selection_mode == DIAGNOSTIC_SELECTION_MODE:
            print(
                "Primary selection outcome: no challenger survived; "
                f"diagnostic fallback {result.selected_candidate} frozen as "
                f"promotion-ineligible at {freeze_path.resolve()}"
            )
        else:
            print(
                f"Selected eligible challenger {result.selected_candidate}; "
                f"freeze written to {freeze_path.resolve()}"
            )
        return

    if args.diagnostic_fallback_on_no_eligible:
        raise ValueError(
            "--diagnostic-fallback-on-no-eligible is select-phase only; "
            "evaluation status comes exclusively from the freeze"
        )
    if not args.selection_freeze:
        raise ValueError("evaluate phase requires the written --selection-freeze artifact")
    if not args.candidate_catalog:
        raise ValueError("evaluate phase requires the frozen --candidate-catalog")
    # These are intentionally the first phase actions.  Neither predictions
    # nor 2024-26 quote labels are materialized until both frozen authorities
    # pass.  Freeze validation runs first so malformed locked CLI boundaries
    # fail on their own authority before a prediction-authority path is needed.
    frozen_payload = validate_selection_freeze(
        args.selection_freeze,
        baseline=args.baseline,
        min_ev=args.min_ev,
        selection_start=args.selection_start,
        selection_end=args.selection_end,
        candidate_catalog_path=args.candidate_catalog,
        contract_path=args.contract,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    if not args.prediction_authority:
        raise ValueError("evaluate phase requires the sealed --prediction-authority")
    validate_evaluation_prediction_authority(
        args.prediction_authority,
        base_predictions_path=args.base_predictions,
        round_predictions_path=args.round_predictions,
        selection_freeze_path=args.selection_freeze,
        frozen_payload=frozen_payload,
        contract_path=args.contract,
    )
    selected = str(frozen_payload["selected_candidate"])
    if args.selected_candidate is not None and args.selected_candidate != selected:
        raise ValueError("--selected-candidate disagrees with --selection-freeze")
    base = _read_prediction_frame_window(
        args.base_predictions, start=MAPPER_TRAIN_START, end=args.eval_end
    )
    rounds = _read_prediction_frame_window(
        args.round_predictions, start=MAPPER_TRAIN_START, end=args.eval_end
    )
    tournament = _read_quote_frame_window(
        args.tournament_quotes,
        start=MAPPER_TRAIN_START,
        end=args.eval_end,
        selection_phase=False,
    )
    round_quotes = _read_quote_frame_window(
        args.round_quotes,
        start=MAPPER_TRAIN_START,
        end=args.eval_end,
        selection_phase=False,
    )
    result = evaluate_frozen_joint_candidate(
        base,
        rounds,
        tournament,
        round_quotes,
        args.output_dir,
        args.baseline,
        selected,
        args.selection_start,
        args.selection_end,
        args.eval_start,
        args.eval_end,
        args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
        min_ev=args.min_ev,
        selection_status={field: frozen_payload[field] for field in SELECTION_STATUS_FIELDS},
    )
    print(
        f"Evaluated frozen {result.selected_candidate} versus {args.baseline}; "
        f"wrote research outputs to {Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()

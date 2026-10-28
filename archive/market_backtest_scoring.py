"""Reusable, offline scoring utilities for historical golf-market backtests.

The module deliberately does not fetch odds, results, or model inputs.  It
expects one normalized closing-line row per offered selection, plus one
probability row per model arm and selection.  Keeping IO outside this module
makes the bet policy identical for baseline and challenger models and makes the
settlement rules independently testable.

Required closing-line columns are ``event_id``, ``market``, ``selection_id``,
``book``, and ``close_odds``.  Matchup rows should also carry ``opponent_id``
and ``tie_rule``.  A matchup result may be supplied in ``outcome`` or
``outcome_numeric`` (win=1, loss=0, tie=0.5 or the string ``tie``).  Outright
rows use DataGolf's ``bet_outcome_numeric``: 1 is a full win, 0 a loss, and a
fraction is the paid dead-heat share.

Probability inputs require the same key columns and ``probability``.  For an
outright, probability is the expected paid finish share (the dead-heat-aware
probability produced by the simulator).  For a matchup it is the *unconditional*
win probability.  An optional ``push_probability`` makes push-market EV and
Kelly calculations exact; tie-loss markets ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MATCHUP_MARKETS = frozenset({"tournament_matchup"})
OUTRIGHT_MARKETS = frozenset({"win", "top_5", "top_10", "top_20"})

_MARKET_ALIASES = {
    "tournament_matchup": "tournament_matchup",
    "tournament_matchups": "tournament_matchup",
    "72_hole_matchup": "tournament_matchup",
    "72_hole_matchups": "tournament_matchup",
    "72hole_matchup": "tournament_matchup",
    "72hole_matchups": "tournament_matchup",
    "72_hole_h2h": "tournament_matchup",
    "72hole_h2h": "tournament_matchup",
    "win": "win",
    "winner": "win",
    "outright": "win",
    "top_5": "top_5",
    "top5": "top_5",
    "top_10": "top_10",
    "top10": "top_10",
    "top_20": "top_20",
    "top20": "top_20",
}

_WIN_WORDS = frozenset({"w", "win", "won", "winner", "1", "1.0"})
_LOSS_WORDS = frozenset({"l", "loss", "lost", "lose", "0", "0.0"})
_TIE_WORDS = frozenset({"t", "tie", "tied", "push", "draw", "0.5"})
_VOID_WORDS = frozenset(
    {"void", "voided", "refund", "refunded", "cancelled", "canceled", "no action"}
)


@dataclass(frozen=True)
class LinePolicy:
    """How one executable close is chosen for each selection.

    ``fixed_book`` uses the first available book in ``books``.  ``best_of_sharp``
    takes the highest American price among those books.  A duplicate quote at
    the selected book resolves to the latest ``close_time``.
    """

    mode: str = "fixed_book"
    books: tuple[str, ...] = ("betcris", "pinnacle")

    def __post_init__(self) -> None:
        if self.mode not in {"fixed_book", "best_of_sharp"}:
            raise ValueError("LinePolicy.mode must be 'fixed_book' or 'best_of_sharp'")
        if not self.books:
            raise ValueError("LinePolicy.books must contain at least one book")


@dataclass(frozen=True)
class GateConfig:
    """Predeclared bet gates, applied unchanged to every model arm."""

    min_ev: float = 0.0
    min_probability_edge: float = 0.0
    min_probability: float = 0.0
    max_probability: float = 1.0
    min_american_odds: float | None = None
    max_american_odds: float | None = None
    allowed_markets: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_probability <= self.max_probability <= 1.0:
            raise ValueError("Probability gates must satisfy 0 <= min <= max <= 1")


@dataclass(frozen=True)
class SizingConfig:
    """Flat-unit or capped fractional-Kelly staking.

    Kelly stakes use a fixed nominal ``bankroll_units`` at bet placement.  This
    avoids letting one model's earlier realized luck change later stake sizes.
    A sequential bankroll simulation can be layered on the selected bets later.
    """

    method: str = "flat"
    flat_stake: float = 1.0
    bankroll_units: float = 100.0
    kelly_fraction: float = 0.25
    max_bankroll_fraction: float = 0.02
    min_stake: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in {"flat", "quarter_kelly"}:
            raise ValueError("SizingConfig.method must be 'flat' or 'quarter_kelly'")
        if self.flat_stake <= 0 or self.bankroll_units <= 0:
            raise ValueError("flat_stake and bankroll_units must be positive")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not 0 < self.max_bankroll_fraction <= 1:
            raise ValueError("max_bankroll_fraction must be in (0, 1]")
        if self.min_stake < 0:
            raise ValueError("min_stake cannot be negative")


@dataclass
class TwoArmBacktestResult:
    """Outputs from :func:`run_two_arm_backtest`."""

    lines: pd.DataFrame
    bets: pd.DataFrame
    attribution: pd.DataFrame
    attribution_summary: pd.DataFrame
    performance: pd.DataFrame
    bootstrap: pd.DataFrame


def canonical_market(value: object) -> str:
    """Return the canonical market name used by this module."""

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return _MARKET_ALIASES.get(text, text)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_american_odds(values: pd.Series, label: str = "close_odds") -> pd.Series:
    odds = pd.to_numeric(values, errors="coerce")
    invalid = ~valid_american_odds_mask(odds)
    if invalid.any():
        examples = values.loc[invalid].head(5).tolist()
        raise ValueError(f"{label} contains invalid American odds: {examples}")
    return odds.astype(float)


def valid_american_odds_mask(
    odds: float | Sequence[float] | pd.Series,
) -> np.ndarray:
    """Return a non-raising mask for finite, conventionally valid prices."""

    values = np.asarray(odds, dtype=float)
    return np.isfinite(values) & ((values <= -100.0) | (values >= 100.0))


def american_profit_multiple(odds: float | Sequence[float] | pd.Series) -> float | np.ndarray:
    """Profit (excluding returned stake) earned by a one-unit winning wager."""

    values = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values > -100) & (values < 100)):
        raise ValueError("American odds must be finite and <= -100 or >= +100")
    result = np.where(values > 0, values / 100.0, 100.0 / np.abs(values))
    return float(result) if values.ndim == 0 else result


def american_decimal_odds(odds: float | Sequence[float] | pd.Series) -> float | np.ndarray:
    """Convert American prices to decimal odds, including returned stake."""

    return 1.0 + american_profit_multiple(odds)


def american_implied_probability(
    odds: float | Sequence[float] | pd.Series,
) -> float | np.ndarray:
    """Raw (not de-vigged) probability implied by an American price."""

    decimal = american_decimal_odds(odds)
    return 1.0 / decimal


def infer_key_columns(frame: pd.DataFrame) -> list[str]:
    """Infer the normalized selection identity columns."""

    if "bet_key" in frame.columns:
        return ["bet_key"]
    event_identity = "event_key" if "event_key" in frame.columns else "event_id"
    required = [event_identity, "market", "selection_id"]
    _require_columns(frame, required, "frame")
    if "opponent_id" in frame.columns:
        required.append("opponent_id")
    return required


def _normalize_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "market" in result.columns:
        result["market"] = result["market"].map(canonical_market)
    if "book" in result.columns:
        result["book"] = result["book"].astype(str).str.strip().str.lower()
    # DataGolf event_id values repeat from season to season.  Never let three
    # years of the same tournament collapse into one betting opportunity or
    # bootstrap cluster merely because the caller did not prebuild event_key.
    if "event_key" not in result.columns and "event_id" in result.columns:
        if "season" in result.columns:
            season_column = "season"
        elif "year" in result.columns:
            season_column = "year"
        else:
            season_column = None
        if season_column is None:
            result["event_key"] = result["event_id"].map(_stable_identity_text)
        else:
            result["event_key"] = (
                result[season_column].map(_stable_identity_text)
                + ":"
                + result["event_id"].map(_stable_identity_text)
            )
    return result


def _stable_identity_text(value: object) -> str:
    """Make integer-like IDs stable across int/float CSV inference."""

    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def select_closing_lines(
    close_rows: pd.DataFrame,
    policy: LinePolicy = LinePolicy(),
    key_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Choose exactly one closing line per normalized betting selection."""

    _require_columns(
        close_rows,
        ["event_id", "market", "selection_id", "book", "close_odds"],
        "close_rows",
    )
    rows = _normalize_identity(close_rows)
    rows["close_odds"] = _validate_american_odds(rows["close_odds"])
    keys = list(key_cols) if key_cols is not None else infer_key_columns(rows)
    _require_columns(rows, keys, "close_rows")

    normalized_books = tuple(str(book).strip().lower() for book in policy.books)
    rank = {book: index for index, book in enumerate(normalized_books)}
    rows = rows.loc[rows["book"].isin(rank)].copy()
    if rows.empty:
        return rows.assign(line_policy=pd.Series(dtype="object"))

    rows["_book_rank"] = rows["book"].map(rank).astype(int)
    if "close_time" in rows.columns:
        rows["_close_time_sort"] = pd.to_datetime(rows["close_time"], errors="coerce", utc=True)
    else:
        rows["_close_time_sort"] = pd.NaT

    # na_position='last' means a timestamped quote wins over an undated duplicate.
    if policy.mode == "fixed_book":
        sort_cols = keys + ["_book_rank", "_close_time_sort"]
        ascending = [True] * len(keys) + [True, False]
    else:
        sort_cols = keys + ["close_odds", "_book_rank", "_close_time_sort"]
        ascending = [True] * len(keys) + [False, True, False]

    rows = rows.sort_values(
        sort_cols,
        ascending=ascending,
        kind="mergesort",
        na_position="last",
    )
    chosen = rows.drop_duplicates(keys, keep="first").copy()
    chosen["line_policy"] = policy.mode
    return chosen.drop(columns=["_book_rank", "_close_time_sort"]).reset_index(drop=True)


def ties_lose(value: object) -> bool:
    """Canonical predicate for matchup formats in which a tie is a loss.

    DataGolf uses both prose (``ties lose``/``three way``) and the word
    ``separate`` for two separately priced sides where a tie is not a push.
    Pricing and settlement must call this same predicate.
    """

    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return (
        "separate" in text
        or "ties lose" in text
        or "tie loses" in text
        or "three way" in text
        or "3 way" in text
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _result_text(row: pd.Series) -> str:
    for column in ("outcome", "result", "settlement"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column]).strip().lower()
    return ""


def _numeric_result(row: pd.Series, outright: bool) -> float | None:
    columns = (
        ("bet_outcome_numeric", "outcome_numeric")
        if outright
        else ("outcome_numeric", "bet_outcome_numeric")
    )
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            try:
                return float(row[column])
            except (TypeError, ValueError):
                return None
    return None


def _is_void(row: pd.Series, result_text: str) -> bool:
    if "void" in row.index and pd.notna(row["void"]) and _as_bool(row["void"]):
        return True
    return result_text in _VOID_WORDS


def _settle_one(row: pd.Series) -> tuple[str, float, float, float]:
    market = canonical_market(row["market"])
    stake = float(row["stake"])
    decimal = float(american_decimal_odds(float(row["close_odds"])))
    text = _result_text(row)

    if _is_void(row, text):
        return "void", 0.0, stake, np.nan

    if market in MATCHUP_MARKETS:
        numeric = _numeric_result(row, outright=False)
        is_tie = (
            ("is_tie" in row.index and pd.notna(row["is_tie"]) and _as_bool(row["is_tie"]))
            or text in _TIE_WORDS
            or (numeric is not None and np.isclose(numeric, 0.5))
        )
        if is_tie:
            if ties_lose(row.get("tie_rule", "")):
                return "tie_loss", -stake, 0.0, 0.0
            return "push", 0.0, stake, np.nan

        if text in _WIN_WORDS or (numeric is not None and np.isclose(numeric, 1.0)):
            profit = stake * (decimal - 1.0)
            return "win", profit, stake * decimal, 1.0
        if text in _LOSS_WORDS or (numeric is not None and np.isclose(numeric, 0.0)):
            return "loss", -stake, 0.0, 0.0
        return "ungraded", np.nan, np.nan, np.nan

    if market in OUTRIGHT_MARKETS:
        fraction = _numeric_result(row, outright=True)
        if fraction is None:
            if text in _WIN_WORDS:
                fraction = 1.0
            elif text in _LOSS_WORDS:
                fraction = 0.0
        if fraction is None or not np.isfinite(fraction):
            return "ungraded", np.nan, np.nan, np.nan
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(
                f"bet_outcome_numeric must be in [0, 1], got {fraction} "
                f"for event {row.get('event_id')}"
            )
        returned = stake * fraction * decimal
        pnl = returned - stake
        if np.isclose(fraction, 0.0):
            status = "loss"
        elif np.isclose(fraction, 1.0):
            status = "win"
        else:
            status = "dead_heat"
        return status, pnl, returned, fraction

    raise ValueError(f"Unsupported market for settlement: {market!r}")


def settle_bets(bets: pd.DataFrame) -> pd.DataFrame:
    """Settle normalized selected bets at their raw American closing prices.

    Dead-heat return is ``stake * paid_fraction * decimal_odds``.  This is the
    standard bookmaker rule: the unpaid fraction of stake loses rather than
    being refunded.
    """

    _require_columns(bets, ["event_id", "market", "close_odds", "stake"], "bets")
    result = _normalize_identity(bets)
    result["close_odds"] = _validate_american_odds(result["close_odds"])
    if (pd.to_numeric(result["stake"], errors="coerce") <= 0).any():
        raise ValueError("Every selected bet must have a positive stake")
    result["stake"] = pd.to_numeric(result["stake"], errors="raise").astype(float)

    if result.empty:
        result["settlement_status"] = pd.Series(dtype="object")
        result["pnl"] = pd.Series(dtype=float)
        result["return_amount"] = pd.Series(dtype=float)
        result["paid_fraction"] = pd.Series(dtype=float)
        result["is_graded"] = pd.Series(dtype=bool)
        return result

    settled = result.apply(_settle_one, axis=1, result_type="expand")
    settled.columns = ["settlement_status", "pnl", "return_amount", "paid_fraction"]
    for column in settled.columns:
        result[column] = settled[column]
    result["is_graded"] = result["settlement_status"] != "ungraded"
    return result


def _expected_value_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    market = result["market"].map(canonical_market)
    p_win = result["probability"].astype(float)
    push = result.get("push_probability", pd.Series(0.0, index=result.index)).fillna(0.0).astype(float)
    if ((p_win < 0) | (p_win > 1) | (push < 0) | (push > 1)).any():
        raise ValueError("probability and push_probability must be in [0, 1]")
    if ((p_win + push) > 1.0 + 1e-10).any():
        raise ValueError("win probability plus push probability cannot exceed 1")

    decimal = np.asarray(american_decimal_odds(result["close_odds"]), dtype=float)
    profit_multiple = decimal - 1.0
    is_matchup = market.isin(MATCHUP_MARKETS).to_numpy()
    tie_is_loss = result.get("tie_rule", pd.Series("", index=result.index)).map(ties_lose).to_numpy()
    effective_push = np.where(is_matchup & ~tie_is_loss, push.to_numpy(), 0.0)
    loss_probability = 1.0 - p_win.to_numpy() - effective_push
    ev = p_win.to_numpy() * profit_multiple - loss_probability

    resolved_probability = 1.0 - effective_push
    pricing_probability = np.divide(
        p_win.to_numpy(),
        resolved_probability,
        out=np.zeros(len(result), dtype=float),
        where=resolved_probability > 0,
    )
    implied = np.asarray(american_implied_probability(result["close_odds"]), dtype=float)
    result["push_probability_used"] = effective_push
    result["pricing_probability"] = pricing_probability
    result["implied_probability"] = implied
    result["probability_edge"] = pricing_probability - implied
    result["ev_per_unit"] = ev
    return result


def _size_selected_bets(frame: pd.DataFrame, sizing: SizingConfig) -> pd.DataFrame:
    result = frame.copy()
    if sizing.method == "flat":
        result["full_kelly_fraction"] = np.nan
        result["stake"] = float(sizing.flat_stake)
        return result

    probability = result["pricing_probability"].to_numpy(dtype=float)
    profit_multiple = np.asarray(american_profit_multiple(result["close_odds"]), dtype=float)
    full_kelly = np.maximum(
        (probability * profit_multiple - (1.0 - probability)) / profit_multiple,
        0.0,
    )
    sized_fraction = np.minimum(
        sizing.kelly_fraction * full_kelly,
        sizing.max_bankroll_fraction,
    )
    stake = sizing.bankroll_units * sized_fraction
    result["full_kelly_fraction"] = full_kelly
    result["stake"] = stake
    positive_stake = result["stake"] > 0
    if sizing.min_stake > 0:
        positive_stake &= result["stake"] >= sizing.min_stake
    return result.loc[positive_stake].copy()


def select_arm_bets(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
    arm: str,
    gate: GateConfig = GateConfig(),
    sizing: SizingConfig = SizingConfig(),
    key_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join one model arm to fixed lines, apply gates, size, and settle bets."""

    normalized_lines = _normalize_identity(lines)
    normalized_probabilities = _normalize_identity(probabilities)
    keys = list(key_cols) if key_cols is not None else infer_key_columns(normalized_lines)
    _require_columns(normalized_lines, keys + ["close_odds"], "lines")
    _require_columns(normalized_probabilities, keys + ["probability"], "probabilities")
    if normalized_probabilities.duplicated(keys).any():
        raise ValueError(f"{arm} probabilities contain duplicate keys")

    probability_columns = keys + ["probability"]
    if "push_probability" in normalized_probabilities.columns:
        probability_columns.append("push_probability")
    joined = normalized_lines.merge(
        normalized_probabilities[probability_columns],
        how="inner",
        on=keys,
        validate="one_to_one",
    )
    if joined.empty:
        return settle_bets(joined.assign(stake=pd.Series(dtype=float), arm=arm))

    joined = _expected_value_inputs(joined)
    mask = (
        (joined["ev_per_unit"] >= gate.min_ev)
        & (joined["probability_edge"] >= gate.min_probability_edge)
        & (joined["pricing_probability"] >= gate.min_probability)
        & (joined["pricing_probability"] <= gate.max_probability)
    )
    if gate.min_american_odds is not None:
        mask &= joined["close_odds"] >= gate.min_american_odds
    if gate.max_american_odds is not None:
        mask &= joined["close_odds"] <= gate.max_american_odds
    if gate.allowed_markets is not None:
        allowed = {canonical_market(market) for market in gate.allowed_markets}
        mask &= joined["market"].isin(allowed)

    selected = _size_selected_bets(joined.loc[mask].copy(), sizing)
    selected["arm"] = str(arm)
    return settle_bets(selected.reset_index(drop=True))


def compare_arm_bets(
    baseline_bets: pd.DataFrame,
    challenger_bets: pd.DataFrame,
    key_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Label overlapping, newly added, and dropped challenger wagers."""

    reference = baseline_bets if not baseline_bets.empty else challenger_bets
    keys = list(key_cols) if key_cols is not None else infer_key_columns(reference)
    value_cols = ["probability", "ev_per_unit", "stake", "pnl"]

    def side(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        _require_columns(frame, keys, prefix)
        available = [column for column in value_cols if column in frame.columns]
        subset = frame[keys + available].copy()
        return subset.rename(columns={column: f"{prefix}_{column}" for column in available})

    baseline = side(baseline_bets, "baseline")
    challenger = side(challenger_bets, "challenger")
    comparison = baseline.merge(challenger, how="outer", on=keys, validate="one_to_one")
    comparison["baseline_selected"] = comparison.get("baseline_stake", np.nan).notna()
    comparison["challenger_selected"] = comparison.get("challenger_stake", np.nan).notna()
    comparison["attribution"] = np.select(
        [
            comparison["baseline_selected"] & comparison["challenger_selected"],
            ~comparison["baseline_selected"] & comparison["challenger_selected"],
            comparison["baseline_selected"] & ~comparison["challenger_selected"],
        ],
        ["overlap", "new", "dropped"],
        default="unknown",
    )
    for column in ("baseline_stake", "challenger_stake", "baseline_pnl", "challenger_pnl"):
        if column not in comparison:
            comparison[column] = 0.0
        comparison[column] = comparison[column].fillna(0.0)
    comparison["pnl_delta"] = comparison["challenger_pnl"] - comparison["baseline_pnl"]
    comparison["stake_delta"] = comparison["challenger_stake"] - comparison["baseline_stake"]
    return comparison


def summarize_attribution(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the overlap/new/dropped decomposition."""

    columns = [
        "attribution",
        "bet_count",
        "baseline_stake",
        "challenger_stake",
        "baseline_pnl",
        "challenger_pnl",
        "pnl_delta",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        comparison.groupby("attribution", sort=False, observed=True)
        .agg(
            bet_count=("attribution", "size"),
            baseline_stake=("baseline_stake", "sum"),
            challenger_stake=("challenger_stake", "sum"),
            baseline_pnl=("baseline_pnl", "sum"),
            challenger_pnl=("challenger_pnl", "sum"),
            pnl_delta=("pnl_delta", "sum"),
        )
        .reset_index()
    )
    return summary[columns]


def _ordered_event_totals(
    bets: pd.DataFrame,
    event_col: str | None,
    event_order_col: str | None,
) -> pd.DataFrame:
    event_col = _resolve_event_col(bets, event_col)
    valid = bets.loc[bets["pnl"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=[event_col, "pnl", "stake", "bet_count", "_event_order"])
    if event_order_col is not None and event_order_col in valid.columns:
        order = pd.to_datetime(valid[event_order_col], errors="coerce", utc=True)
    elif "close_time" in valid.columns:
        order = pd.to_datetime(valid["close_time"], errors="coerce", utc=True)
    else:
        order = pd.Series(np.arange(len(valid)), index=valid.index, dtype=float)
    valid["_row_order"] = order
    totals = (
        valid.groupby(event_col, sort=False, dropna=False)
        .agg(
            pnl=("pnl", "sum"),
            stake=("stake", "sum"),
            bet_count=("pnl", "size"),
            _event_order=("_row_order", "min"),
        )
        .reset_index()
        .sort_values("_event_order", kind="mergesort", na_position="last")
        .reset_index(drop=True)
    )
    return totals


def _resolve_event_col(frame: pd.DataFrame, event_col: str | None) -> str:
    if event_col is not None:
        _require_columns(frame, [event_col], "bets")
        return event_col
    if "event_key" in frame.columns:
        return "event_key"
    _require_columns(frame, ["event_id"], "bets")
    return "event_id"


def max_drawdown_from_pnl(pnl: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown of an ordered P&L stream, in units."""

    values = np.asarray(pnl, dtype=float)
    if values.size == 0:
        return 0.0
    equity = np.concatenate(([0.0], np.cumsum(values)))
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def performance_summary(
    bets: pd.DataFrame,
    event_col: str | None = None,
    event_order_col: str | None = None,
    starting_bankroll: float = 100.0,
) -> pd.DataFrame:
    """Summarize settled P&L, raw-odds ROI, and chronological drawdown by arm."""

    event_col = _resolve_event_col(bets, event_col)
    _require_columns(bets, ["arm", "stake", "pnl"], "bets")
    rows: list[dict[str, object]] = []
    for arm, frame in bets.groupby("arm", sort=False, observed=True):
        valid = frame.loc[frame["pnl"].notna()]
        events = _ordered_event_totals(valid, event_col, event_order_col)
        stake = float(valid["stake"].sum())
        pnl = float(valid["pnl"].sum())
        rows.append(
            {
                "arm": arm,
                "bet_count": int(len(valid)),
                "event_count": int(events[event_col].nunique()),
                "stake": stake,
                "pnl": pnl,
                "roi": pnl / stake if stake > 0 else np.nan,
                "max_drawdown": max_drawdown_from_pnl(events["pnl"]),
                "starting_bankroll": float(starting_bankroll),
                "ending_bankroll": float(starting_bankroll + pnl),
            }
        )
    columns = [
        "arm",
        "bet_count",
        "event_count",
        "stake",
        "pnl",
        "roi",
        "max_drawdown",
        "starting_bankroll",
        "ending_bankroll",
    ]
    return pd.DataFrame(rows, columns=columns)


def event_cluster_bootstrap(
    bets: pd.DataFrame,
    event_col: str | None = None,
    event_order_col: str | None = None,
    draws: int = 5_000,
    confidence: float = 0.95,
    seed: int = 20260822,
    comparison: tuple[str, str] | None = ("baseline", "challenger"),
) -> pd.DataFrame:
    """Bootstrap whole events and report P&L/ROI/drawdown intervals.

    Every draw samples event clusters, never individual bets.  The comparison
    rows are paired because the same sampled event indices feed both arms.
    Drawdown is computed over the sampled event-slot sequence.
    """

    if draws <= 0:
        raise ValueError("draws must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    event_col = _resolve_event_col(bets, event_col)
    _require_columns(bets, ["arm", "stake", "pnl"], "bets")
    valid = bets.loc[bets["pnl"].notna()].copy()
    output_columns = [
        "arm",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "probability_above_zero",
        "draws",
        "cluster_count",
    ]
    if valid.empty:
        return pd.DataFrame(columns=output_columns)

    if event_order_col is not None and event_order_col in valid.columns:
        valid["_event_order"] = pd.to_datetime(valid[event_order_col], errors="coerce", utc=True)
    elif "close_time" in valid.columns:
        valid["_event_order"] = pd.to_datetime(valid["close_time"], errors="coerce", utc=True)
    else:
        valid["_event_order"] = np.arange(len(valid), dtype=float)

    event_order = (
        valid.groupby(event_col, sort=False, dropna=False)["_event_order"]
        .min()
        .sort_values(kind="mergesort", na_position="last")
    )
    events = event_order.index.tolist()
    arms = valid["arm"].drop_duplicates().astype(str).tolist()
    aggregate = (
        valid.groupby([event_col, "arm"], sort=False, dropna=False)
        .agg(pnl=("pnl", "sum"), stake=("stake", "sum"))
        .reset_index()
    )
    pnl_pivot = aggregate.pivot(index=event_col, columns="arm", values="pnl").reindex(events).fillna(0.0)
    stake_pivot = aggregate.pivot(index=event_col, columns="arm", values="stake").reindex(events).fillna(0.0)
    pnl_matrix = pnl_pivot.reindex(columns=arms, fill_value=0.0).to_numpy(dtype=float)
    stake_matrix = stake_pivot.reindex(columns=arms, fill_value=0.0).to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(events), size=(draws, len(events)))
    sampled_pnl_path = pnl_matrix[sampled, :]
    sampled_pnl = sampled_pnl_path.sum(axis=1)
    sampled_stake = stake_matrix[sampled, :].sum(axis=1)
    sampled_roi = np.divide(
        sampled_pnl,
        sampled_stake,
        out=np.full_like(sampled_pnl, np.nan),
        where=sampled_stake > 0,
    )
    cumulative = np.cumsum(sampled_pnl_path, axis=1)
    cumulative_with_zero = np.concatenate(
        [np.zeros((draws, 1, len(arms))), cumulative], axis=1
    )
    peaks = np.maximum.accumulate(cumulative_with_zero, axis=1)
    sampled_drawdown = np.max(peaks - cumulative_with_zero, axis=1)

    alpha = (1.0 - confidence) / 2.0
    low_q, high_q = 100.0 * alpha, 100.0 * (1.0 - alpha)
    rows: list[dict[str, object]] = []

    def append_metric(arm: str, metric: str, estimate: float, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        if finite.size:
            ci_low, ci_high = np.percentile(finite, [low_q, high_q])
            probability = float(np.mean(finite > 0))
        else:
            ci_low = ci_high = probability = np.nan
        rows.append(
            {
                "arm": arm,
                "metric": metric,
                "estimate": float(estimate),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "probability_above_zero": probability,
                "draws": int(draws),
                "cluster_count": int(len(events)),
            }
        )

    actual_by_arm: dict[str, dict[str, float]] = {}
    for index, arm in enumerate(arms):
        actual_pnl = float(pnl_matrix[:, index].sum())
        actual_stake = float(stake_matrix[:, index].sum())
        actual_roi = actual_pnl / actual_stake if actual_stake > 0 else np.nan
        actual_drawdown = max_drawdown_from_pnl(pnl_matrix[:, index])
        actual_by_arm[arm] = {
            "pnl": actual_pnl,
            "roi": actual_roi,
            "max_drawdown": actual_drawdown,
        }
        append_metric(arm, "pnl", actual_pnl, sampled_pnl[:, index])
        append_metric(arm, "roi", actual_roi, sampled_roi[:, index])
        append_metric(
            arm,
            "max_drawdown",
            actual_drawdown,
            sampled_drawdown[:, index],
        )

    if comparison is not None and all(arm in arms for arm in comparison):
        base, challenger = comparison
        base_index, challenger_index = arms.index(base), arms.index(challenger)
        label = f"{challenger}_minus_{base}"
        for metric, values in (
            ("pnl", sampled_pnl[:, challenger_index] - sampled_pnl[:, base_index]),
            ("roi", sampled_roi[:, challenger_index] - sampled_roi[:, base_index]),
            (
                "max_drawdown",
                sampled_drawdown[:, challenger_index] - sampled_drawdown[:, base_index],
            ),
        ):
            estimate = actual_by_arm[challenger][metric] - actual_by_arm[base][metric]
            append_metric(label, metric, estimate, values)

    return pd.DataFrame(rows, columns=output_columns)


def run_two_arm_backtest(
    close_rows: pd.DataFrame,
    baseline_probabilities: pd.DataFrame,
    challenger_probabilities: pd.DataFrame,
    *,
    line_policy: LinePolicy = LinePolicy(),
    gate: GateConfig = GateConfig(),
    sizing: SizingConfig = SizingConfig(),
    key_cols: Sequence[str] | None = None,
    event_col: str | None = None,
    event_order_col: str | None = None,
    starting_bankroll: float = 100.0,
    bootstrap_draws: int = 5_000,
    bootstrap_seed: int = 20260822,
) -> TwoArmBacktestResult:
    """Run a paired, identically governed baseline/challenger money test."""

    lines = select_closing_lines(close_rows, line_policy, key_cols)
    keys = list(key_cols) if key_cols is not None else infer_key_columns(lines)
    baseline = select_arm_bets(lines, baseline_probabilities, "baseline", gate, sizing, keys)
    challenger = select_arm_bets(
        lines,
        challenger_probabilities,
        "challenger",
        gate,
        sizing,
        keys,
    )
    bets = pd.concat([baseline, challenger], ignore_index=True, sort=False)
    attribution = compare_arm_bets(baseline, challenger, keys)
    return TwoArmBacktestResult(
        lines=lines,
        bets=bets,
        attribution=attribution,
        attribution_summary=summarize_attribution(attribution),
        performance=performance_summary(
            bets,
            event_col=event_col,
            event_order_col=event_order_col,
            starting_bankroll=starting_bankroll,
        ),
        bootstrap=event_cluster_bootstrap(
            bets,
            event_col=event_col,
            event_order_col=event_order_col,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        ),
    )


def run_sizing_scenarios(
    close_rows: pd.DataFrame,
    baseline_probabilities: pd.DataFrame,
    challenger_probabilities: pd.DataFrame,
    sizing_scenarios: Mapping[str, SizingConfig],
    **kwargs: object,
) -> dict[str, TwoArmBacktestResult]:
    """Convenience wrapper for the required flat and capped-Kelly reports."""

    if not sizing_scenarios:
        raise ValueError("sizing_scenarios cannot be empty")
    return {
        name: run_two_arm_backtest(
            close_rows,
            baseline_probabilities,
            challenger_probabilities,
            sizing=sizing,
            **kwargs,
        )
        for name, sizing in sizing_scenarios.items()
    }

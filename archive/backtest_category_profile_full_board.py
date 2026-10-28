"""Research-only 2026 full-board replay of category-profile calibration arms.

This script does not mutate production inputs or betting ledgers.  It prices
the archived 2026 pre-event 72-hole matchup and finish-position boards with a
strict-prior, physical-course category-first tournament cascade, then compares
three category-mean arms under common Monte Carlo random numbers:

* ``baseline``: the unmodified point-in-time category profile;
* ``blanket_065``: shrink every eligible category deviation 35% toward the
  player's four-category mean, with one proportional 0.15 SG cap; and
* ``targeted_arg_putt``: shrink only ARG and PUTT deviations 35%, compensating
  the zero-sum movement equally into OTT and APP, with the same cap.

R1-R4 probabilities come from the separately audited scored-authority panel.
They are copied byte-for-byte to all three arms because the production
single-round kernel recenters category means to the fixed total skill.  Those
boards are therefore a structural-invariance/new-discovery audit, not evidence
that the category calibration was historically used by the live model.

The replay discovers each arm's bets independently at both archived open and
close, using the best price from Betcris, Pinnacle, and BetOnline, a 3% matchup
EV gate, and the production 5% finish-position EV gate.  It reports flat stakes
and production-style stakes.  Results are
exploratory: the targeted arm was proposed after inspecting 2026 diagnostics,
the current coefficients/kernel are retrospective, historical weather
forecasts were not archived, and best-of-three asynchronous prices are not a
claim of simultaneous executability.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sims_kernel


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
SIM_PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
for _path in (str(REPO), str(ARCHIVE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backtest_distribution_engines_2026 import (  # noqa: E402
    DB_CAT,
    build_category_params,
    load_correlation,
)
from backtest_shot_dispersion_recent import CAT_ORDER  # noqa: E402
from backtest_shot_mean_closing_markets import (  # noqa: E402
    COEFF_PATH,
    CORR_PATH,
    DEFAULT_CROSSWALK,
    DEFAULT_DB,
    DEFAULT_ODDS_CACHE,
    DEFAULT_PREDS,
    EXPORT,
    EventMeta,
    canonical_event_key,
    canonical_player_id,
    course_shape,
    cut_line_for_event,
    event_format_exclusion,
    finish_probabilities,
    load_coefficients,
    load_name_replacements,
    load_player_histories,
    load_r1_predictions,
    load_source_crosswalk,
    normalize_name,
    normalize_prediction_rosters,
    normalize_text,
    probabilities_for_lines,
    read_event_metadata,
    simulate_scores,
    sqlite_rows_for_players,
    stable_seed,
)
from build_bmw_shot_features_shadow import (  # noqa: E402
    build_dispersion_features,
    load_event_dates,
    load_shot_round_stats,
)
from historical_odds_cache import HistoricalOddsCacheReader  # noqa: E402
from market_backtest_scoring import (  # noqa: E402
    GateConfig,
    LinePolicy,
    SizingConfig,
    event_cluster_bootstrap,
    performance_summary,
    select_arm_bets,
    select_closing_lines,
    settle_bets,
    valid_american_odds_mask,
)


START = "2026-01-01"
END = "2026-08-22"
ARMS = ("baseline", "blanket_065", "targeted_arg_putt")
CHALLENGERS = ARMS[1:]
PRICE_POINTS = ("open", "close")
SIZING_SCENARIOS = ("flat_1u", "production_style")
BOOKS = ("betcris", "pinnacle", "betonline")
MIN_EV = 0.03
FINISH_MIN_EV = 0.05
SHRINK_FACTOR = 0.65
PROFILE_CAP = 0.15
MIN_PRIOR_ROUNDS = 20
MIN_CATEGORY_OBS = 12
MAX_AGE = 55.0
WEEK_LATENT_SD = 0.65
CHECKPOINT_VERSION = "category_profile_full_board_v2"
KEY_COLS = ["event_id", "board", "market", "selection_id", "opponent_id"]

DEFAULT_SCORE_AUTHORITY = (
    REPO
    / "permanent_data"
    / "research"
    / "base_skill_rebuild_20200101_20260822"
    / "all_round_joint_market_eval_2024_2026_score_authority_v5_authorized_20260823"
    / "scored_evaluation_2024_2026.parquet"
)
DEFAULT_OUTPUT = (
    REPO
    / "permanent_data"
    / "research"
    / "category_profile_full_board_2026_v1"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--simulations", type=int, default=50_000)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--min-ev", type=float, default=MIN_EV)
    parser.add_argument("--finish-min-ev", type=float, default=FINISH_MIN_EV)
    parser.add_argument("--week-latent-sd", type=float, default=WEEK_LATENT_SD)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--shot-export", type=Path, default=EXPORT)
    parser.add_argument("--odds-cache", type=Path, default=DEFAULT_ODDS_CACHE)
    parser.add_argument(
        "--supplemental-odds-cache",
        type=Path,
        action="append",
        default=[],
        help="Repeatable additional verified cache (for example the BMW completion cache).",
    )
    parser.add_argument("--score-authority", type=Path, default=DEFAULT_SCORE_AUTHORITY)
    parser.add_argument("--correlation", type=Path, default=CORR_PATH)
    parser.add_argument("--coefficients", type=Path, default=COEFF_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-key", action="append")
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--no-course-shape", action="store_true")
    parser.add_argument(
        "--no-shot-variance",
        action="store_true",
        help="Diagnostic only: disable the common production shot-dispersion overlay.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write board/eligibility coverage without running Monte Carlo.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the existing immutable run_config matches exactly.",
    )
    return parser.parse_args(argv)


def _json_default(value: object) -> object:
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def config_digest(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_output(output: Path, config: dict[str, object], resume: bool) -> str:
    digest = config_digest(config)
    config_path = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite nonempty research output {output}; use a new path "
                "or --resume for an identical run."
            )
        if not config_path.is_file():
            raise FileNotFoundError(f"Resume directory has no run_config.json: {output}")
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("config_digest") != digest:
            raise ValueError("Resume run_config does not match the requested configuration")
        return digest
    output.mkdir(parents=True, exist_ok=True)
    payload = {**config, "config_digest": digest}
    _atomic_text(config_path, json.dumps(payload, indent=2, default=_json_default) + "\n")
    return digest


def canonical_internal_event_key(value: object) -> str:
    text = str(value)
    if text.startswith("pga:R"):
        return text
    pieces = text.split("|")
    if len(pieces) >= 3 and pieces[0].casefold() == "pga":
        return canonical_event_key(pieces[1], pieces[2])
    raise ValueError(f"Unrecognized event key: {value!r}")


def _valid_price(value: object) -> bool:
    try:
        return bool(valid_american_odds_mask([float(value)])[0])
    except (TypeError, ValueError):
        return False


def normalize_cache_board(
    matchup_rows: Sequence[dict[str, Any]],
    outright_rows: Sequence[dict[str, Any]],
    price_point: str,
) -> pd.DataFrame:
    """Expand both sides and normalize an archived open or close board."""

    if price_point not in PRICE_POINTS:
        raise ValueError(f"Unknown price point: {price_point}")
    rows: list[dict[str, object]] = []
    matchup_price = f"{{side}}_{price_point}"
    quote_time = f"{price_point}_time"
    for raw in matchup_rows:
        bet_type = normalize_text(raw.get("bet_type"))
        if "72" not in bet_type and "tournament" not in bet_type:
            continue
        try:
            event_key = canonical_event_key(raw.get("season", raw.get("query_year")), raw["event_id"])
        except (KeyError, TypeError, ValueError):
            continue
        for side, other in (("p1", "p2"), ("p2", "p1")):
            odds = raw.get(matchup_price.format(side=side))
            selection_id = canonical_player_id(raw.get(f"{side}_dg_id"))
            opponent_id = canonical_player_id(raw.get(f"{other}_dg_id"))
            if not selection_id or not opponent_id or not _valid_price(odds):
                continue
            rows.append(
                {
                    "event_id": event_key,
                    "event_key": event_key,
                    "event_start": raw.get("event_completed"),
                    "board": "72h",
                    "round_num": 0,
                    "market": "tournament_matchup",
                    "selection_id": selection_id,
                    "selection_name": raw.get(f"{side}_player_name"),
                    "opponent_id": opponent_id,
                    "opponent_name": raw.get(f"{other}_player_name"),
                    "book": normalize_text(raw.get("book")),
                    "close_odds": float(odds),
                    "close_time": raw.get(quote_time),
                    "price_point": price_point,
                    "tie_rule": raw.get("tie_rule", ""),
                    "outcome": raw.get(f"{side}_outcome_text"),
                    "outcome_numeric": raw.get(f"{side}_outcome"),
                    "record_id": raw.get("record_id"),
                }
            )
    for raw in outright_rows:
        market = normalize_text(raw.get("market")).replace("-", "_").replace(" ", "_")
        market = {"top5": "top_5", "top10": "top_10", "top20": "top_20", "winner": "win"}.get(market, market)
        if market not in {"win", "top_5", "top_10", "top_20"}:
            continue
        odds = raw.get(f"{price_point}_odds")
        selection_id = canonical_player_id(raw.get("dg_id"))
        if not selection_id or not _valid_price(odds):
            continue
        try:
            event_key = canonical_event_key(raw.get("season", raw.get("query_year")), raw["event_id"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            {
                "event_id": event_key,
                "event_key": event_key,
                "event_start": raw.get("event_completed"),
                "board": "72h",
                "round_num": 0,
                "market": market,
                "selection_id": selection_id,
                "selection_name": raw.get("player_name"),
                "opponent_id": "",
                "opponent_name": "",
                "book": normalize_text(raw.get("book")),
                "close_odds": float(odds),
                "close_time": raw.get(quote_time),
                "price_point": price_point,
                "tie_rule": "",
                "outcome": raw.get("bet_outcome_text", raw.get("outcome")),
                "bet_outcome_numeric": raw.get("bet_outcome_numeric"),
                "record_id": raw.get("record_id"),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["event_start"] = pd.to_datetime(output["event_start"], errors="coerce")
    return output


def select_best_of_three(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return lines.copy()
    selected = select_closing_lines(
        lines,
        LinePolicy(mode="best_of_sharp", books=BOOKS),
        KEY_COLS,
    )
    selected["line_policy"] = "best_of_3"
    return selected


def _invert_outcome(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return np.nan
    if np.isclose(float(numeric), 0.5):
        return 0.5
    if np.isclose(float(numeric), 1.0):
        return 0.0
    if np.isclose(float(numeric), 0.0):
        return 1.0
    return np.nan


def normalize_round_authority(
    source: pd.DataFrame, price_point: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand the audited baseline R1-R4 authority into lines/probabilities."""

    if price_point not in PRICE_POINTS:
        raise ValueError(f"Unknown price point: {price_point}")
    data = source[
        source["candidate"].eq("baseline_current_prod")
        & source["market"].isin(("r1", "r2", "r3", "r4"))
    ].copy()
    event_dates = pd.to_datetime(data["event_start_date"], errors="coerce")
    data = data[event_dates.dt.year.eq(2026)].copy()
    price_col = f"{{side}}_{price_point}"
    rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    for raw in data.to_dict(orient="records"):
        event_key = canonical_internal_event_key(raw["event_key"])
        board = str(raw["market"]).lower()
        round_num = int(board[1:])
        p1_outcome = raw.get("settlement_p1_outcome")
        for side, other in (("p1", "p2"), ("p2", "p1")):
            odds = raw.get(price_col.format(side=side))
            selection_id = canonical_player_id(raw.get(f"{side}_id", raw.get(f"{side}_dg_id")))
            opponent_id = canonical_player_id(raw.get(f"{other}_id", raw.get(f"{other}_dg_id")))
            if not selection_id or not opponent_id or not _valid_price(odds):
                continue
            outcome = p1_outcome if side == "p1" else _invert_outcome(p1_outcome)
            line = {
                "event_id": event_key,
                "event_key": event_key,
                "event_start": pd.Timestamp(raw["event_start_date"]),
                "board": board,
                "round_num": round_num,
                "market": "tournament_matchup",
                "selection_id": selection_id,
                "selection_name": selection_id,
                "opponent_id": opponent_id,
                "opponent_name": opponent_id,
                "book": normalize_text(raw.get("book")),
                "close_odds": float(odds),
                "close_time": pd.NaT,
                "price_point": price_point,
                "tie_rule": raw.get("tie_rule", ""),
                "outcome_numeric": outcome,
                "record_id": raw.get("quote_id"),
                "authority_event_key": raw.get("event_key"),
            }
            rows.append(line)
            probability_rows.append(
                {
                    **{key: line[key] for key in KEY_COLS},
                    "event_key": event_key,
                    "event_start": line["event_start"],
                    "round_num": round_num,
                    "probability": float(raw[f"{side}_probability"]),
                    "push_probability": float(raw.get("push_probability", 0.0)),
                    "arm": "baseline",
                    "probability_source": "audited_scored_authority_invariant",
                }
            )
    lines = pd.DataFrame(rows)
    probabilities = pd.DataFrame(probability_rows)
    if probabilities.empty:
        return lines, probabilities
    probabilities = probabilities.drop_duplicates(KEY_COLS, keep="first")
    return lines, probabilities


def load_age_histories(
    db_path: Path, player_ids: Sequence[int], end: pd.Timestamp
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    ages = sqlite_rows_for_players(
        db_path,
        "player_rounds",
        ["dg_id", "round_date", "age"],
        player_ids,
        "tour='pga' AND round_date < ? AND age IS NOT NULL",
        [end.strftime("%Y-%m-%d")],
    )
    if ages.empty:
        return {}
    ages["round_date"] = pd.to_datetime(ages["round_date"], errors="coerce").dt.normalize()
    ages["age"] = pd.to_numeric(ages["age"], errors="coerce")
    ages = ages.dropna(subset=["round_date", "age"])
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for dg_id, group in ages.groupby("dg_id", sort=False):
        group = group.sort_values("round_date", kind="mergesort")
        output[int(dg_id)] = (
            group["round_date"].to_numpy(dtype="datetime64[ns]"),
            group["age"].to_numpy(float),
        )
    return output


def latest_prior_age(
    ages: dict[int, tuple[np.ndarray, np.ndarray]], dg_id: int, cutoff: pd.Timestamp
) -> float:
    history = ages.get(int(dg_id))
    if history is None:
        return np.nan
    dates, values = history
    index = int(np.searchsorted(dates, np.datetime64(cutoff), side="left") - 1)
    return float(values[index]) if index >= 0 else np.nan


def profile_eligibility(
    field: pd.DataFrame,
    histories: dict[int, pd.DataFrame],
    age_histories: dict[int, tuple[np.ndarray, np.ndarray]],
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dg_id in field["dg_id"].astype(int):
        history = histories.get(int(dg_id), pd.DataFrame())
        if not history.empty:
            history = history[history["round_date"].lt(cutoff)]
        counts = {
            category: int(history[source].notna().sum()) if source in history else 0
            for category, source in DB_CAT.items()
        }
        prior_rounds = int(len(history))
        age = latest_prior_age(age_histories, int(dg_id), cutoff)
        rounds_ok = prior_rounds >= MIN_PRIOR_ROUNDS
        categories_ok = all(count >= MIN_CATEGORY_OBS for count in counts.values())
        age_ok = bool(np.isfinite(age) and age < MAX_AGE)
        eligible = bool(rounds_ok and categories_ok and age_ok)
        reason = "eligible"
        if not rounds_ok:
            reason = "prior_rounds_lt20"
        elif not categories_ok:
            reason = "category_obs_lt12"
        elif not np.isfinite(age):
            reason = "missing_prior_age"
        elif not age_ok:
            reason = "age_ge55"
        rows.append(
            {
                "dg_id": int(dg_id),
                "prior_rounds": prior_rounds,
                "prior_age": age,
                **{f"{category}_obs": count for category, count in counts.items()},
                "eligible": eligible,
                "eligibility_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def calibrate_profiles(
    mu: np.ndarray,
    eligible: np.ndarray,
    arm: str,
    cap: float = PROFILE_CAP,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a zero-sum profile arm with one proportional per-player cap."""

    values = np.asarray(mu, dtype=float)
    mask = np.asarray(eligible, dtype=bool)
    if values.ndim != 2 or values.shape[1] != len(CAT_ORDER):
        raise ValueError(f"mu must have shape (n, {len(CAT_ORDER)})")
    if mask.shape != (len(values),):
        raise ValueError("eligible mask length does not match mu")
    if arm == "baseline":
        shifts = np.zeros_like(values)
    else:
        center = values.mean(axis=1)
        if arm == "blanket_065":
            shifts = (SHRINK_FACTOR - 1.0) * (values - center[:, None])
        elif arm == "targeted_arg_putt":
            index = {category: position for position, category in enumerate(CAT_ORDER)}
            shifts = np.zeros_like(values)
            arg = index["sg_arg"]
            putt = index["sg_putt"]
            ott = index["sg_ott"]
            app = index["sg_app"]
            shifts[:, arg] = -0.35 * (values[:, arg] - center)
            shifts[:, putt] = -0.35 * (values[:, putt] - center)
            compensation = -(shifts[:, arg] + shifts[:, putt]) / 2.0
            shifts[:, ott] = compensation
            shifts[:, app] = compensation
        else:
            raise ValueError(f"Unknown category-profile arm: {arm}")
        shifts[~mask] = 0.0
        maximum = np.max(np.abs(shifts), axis=1)
        raw_maximum = maximum.copy()
        scale = np.ones(len(values), dtype=float)
        positive = maximum > 0
        scale[positive] = np.minimum(1.0, float(cap) / maximum[positive])
        shifts *= scale[:, None]
    output = values + shifts
    total_error = np.max(np.abs(output.sum(axis=1) - values.sum(axis=1))) if len(values) else 0.0
    if total_error > 1e-12:
        raise AssertionError(f"Category calibration changed total skill by {total_error}")
    max_shift = np.max(np.abs(shifts), axis=1) if len(values) else np.array([], dtype=float)
    raw_cap_bound = (
        int(np.sum(raw_maximum > float(cap) + 1e-12)) if arm != "baseline" else 0
    )
    return np.ascontiguousarray(output), {
        "arm": arm,
        "eligible_players": int(mask.sum()),
        "changed_players": int(np.any(np.abs(shifts) > 1e-14, axis=1).sum()),
        "cap_bound_players": raw_cap_bound,
        "max_abs_shift": float(max_shift.max()) if len(max_shift) else 0.0,
        "mean_abs_shift": float(np.abs(shifts).mean()) if shifts.size else 0.0,
        "max_total_error": float(total_error),
    }


def build_event_field_core(
    roster: pd.DataFrame,
    meta: EventMeta,
    r1_predictions: pd.DataFrame,
    model_histories: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_crosswalk: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build an archived roster with strictly dated total-skill fallbacks."""

    from backtest_shot_mean_closing_markets import latest_model_skill

    field = roster[["dg_id", "player_name"]].drop_duplicates("dg_id").copy()
    exact = r1_predictions[
        r1_predictions["year"].eq(meta.year)
        & r1_predictions["event_id"].eq(meta.event_id)
    ][["player_name", "prediction"]]
    field = field.merge(exact, on="player_name", how="left", validate="many_to_one")
    field["skill_est"] = field["prediction"]
    field["skill_source"] = np.where(field["prediction"].notna(), "r1_prediction", "")
    missing = field["skill_est"].isna()
    field.loc[missing, "skill_est"] = [
        latest_model_skill(model_histories, int(dg_id), meta.start)
        for dg_id in field.loc[missing, "dg_id"]
    ]
    dated = missing & field["skill_est"].notna()
    field.loc[dated, "skill_source"] = "dated_model_fallback"
    still_missing = field["skill_est"].isna()
    finite = field.loc[~still_missing, "skill_est"]
    floor = float(finite.quantile(0.10) - 0.50) if len(finite) else -2.0
    field.loc[still_missing, "skill_est"] = floor
    field.loc[still_missing, "skill_source"] = "field_floor_fallback"
    if source_crosswalk is None:
        field["source_player_id"] = ""
    else:
        field = field.merge(
            source_crosswalk[["dg_id", "source_player_id"]],
            on="dg_id",
            how="left",
            validate="many_to_one",
        )
        field["source_player_id"] = field["source_player_id"].fillna("").astype(str)
    field = field.sort_values(["player_name", "dg_id"], kind="mergesort").reset_index(drop=True)
    return field, {
        "roster_players": int(len(field)),
        "r1_prediction_players": int(field["skill_source"].eq("r1_prediction").sum()),
        "dated_model_fallback_players": int(field["skill_source"].eq("dated_model_fallback").sum()),
        "field_floor_fallback_players": int(field["skill_source"].eq("field_floor_fallback").sum()),
        "shot_source_id_players": int(field["source_player_id"].ne("").sum()),
    }


def finish_event_exclusion(meta: EventMeta) -> str:
    inherited = event_format_exclusion(meta)
    if inherited:
        return inherited
    if "zurich classic" in normalize_text(meta.event_name):
        return "team_format_unmodeled"
    return ""


def event_cut_line(meta: EventMeta, field_size: int) -> tuple[int, str]:
    name = normalize_text(meta.event_name)
    if "open championship" in name:
        return min(70, field_size), "open_championship_top70"
    return cut_line_for_event(meta, field_size)


def _clone_params(params: dict[str, np.ndarray], mu: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mu": np.ascontiguousarray(mu),
        "std_current": np.ascontiguousarray(params["std_current"]),
        # Preserve the reconstructed production variance arm.  Category means
        # are the only candidate-specific input; dispersion stays common.
        "std_shot": np.ascontiguousarray(params["std_shot"]),
        "skew": np.ascontiguousarray(params["skew"]),
    }


def _event_checkpoint_name(event_key: str) -> str:
    return event_key.replace(":", "_").replace("|", "_")


def event_field_shot_history(
    shot_stats: pd.DataFrame,
    feature_ids: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only strict-prior shot rows for players in the current event field."""

    ids = set(feature_ids["source_player_id"].dropna().astype(str))
    if not ids or shot_stats.empty:
        return shot_stats.iloc[0:0].copy()
    source_ids = shot_stats["source_player_id"].astype(str)
    dates = pd.to_datetime(shot_stats["event_date"], errors="coerce")
    return shot_stats[source_ids.isin(ids) & dates.lt(cutoff)].copy()


def simulate_event_probabilities(
    event_key: str,
    meta: EventMeta,
    roster: pd.DataFrame,
    opportunity_lines: pd.DataFrame,
    histories: dict[int, pd.DataFrame],
    model_histories: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    age_histories: dict[int, tuple[np.ndarray, np.ndarray]],
    source_crosswalk: pd.DataFrame,
    shot_stats: pd.DataFrame,
    r1_predictions: pd.DataFrame,
    chol: np.ndarray,
    coefficients: dict[str, Any],
    db_path: Path,
    simulations: int,
    base_seed: int,
    week_latent_sd: float,
    no_course_shape: bool,
    no_shot_variance: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    field, field_audit = build_event_field_core(
        roster, meta, r1_predictions, model_histories, source_crosswalk
    )
    eligibility = profile_eligibility(field, histories, age_histories, meta.start)
    mults, course_skews, course_audit = course_shape(db_path, meta, no_course_shape)
    if no_shot_variance:
        features = pd.DataFrame(
            {"source_player_id": field["source_player_id"].drop_duplicates()}
        )
        feature_prior: dict[str, float] = {}
    else:
        feature_ids = (
            field.loc[field["source_player_id"].ne(""), ["source_player_id"]]
            .drop_duplicates("source_player_id")
        )
        features, feature_prior = build_dispersion_features(
            event_field_shot_history(shot_stats, feature_ids, meta.start),
            feature_ids,
        )
    params, category_audit = build_category_params(
        histories,
        field,
        meta.start,
        features,
        mults,
        course_skews,
    )
    cut_line, cut_rule = event_cut_line(meta, len(field))
    zero_delta = np.zeros(len(field), dtype=float)
    probability_pieces: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    seed = stable_seed(event_key, base_seed)
    for arm in ARMS:
        calibrated_mu, arm_audit = calibrate_profiles(
            params["mu"], eligibility["eligible"].to_numpy(bool), arm
        )
        scores = simulate_scores(
            _clone_params(params, calibrated_mu),
            field,
            chol,
            coefficients,
            zero_delta,
            simulations,
            seed,
            cut_line,
            week_latent_sd,
        )
        probabilities, market_audit = probabilities_for_lines(
            scores,
            field,
            opportunity_lines,
            event_key,
            meta.start,
        )
        probabilities["board"] = "72h"
        probabilities["round_num"] = 0
        probabilities["arm"] = arm
        probabilities["probability_source"] = "strict_prior_category_cascade"
        probability_pieces.append(probabilities)
        calibration_rows.append({"event_key": event_key, **arm_audit})
        del scores
        gc.collect()
    eligibility.insert(0, "event_key", event_key)
    event_audit: dict[str, object] = {
        "event_key": event_key,
        "event_name": meta.event_name,
        "event_start": meta.start,
        "status": "simulated",
        "simulations": int(simulations),
        "seed": int(seed),
        "cut_line": int(cut_line),
        "cut_rule": cut_rule,
        "physical_course_count": int(len(meta.course_counts)),
        "eligibility_eligible_players": int(eligibility["eligible"].sum()),
        **field_audit,
        **{f"category_{key}": value for key, value in category_audit.items()},
        "course_shape_status": course_audit.get("status"),
        "shot_variance_status": "disabled_diagnostic" if no_shot_variance else "production_overlay_common",
        "shot_feature_prior": feature_prior,
    }
    return (
        pd.concat(probability_pieces, ignore_index=True, sort=False),
        eligibility,
        {"event": event_audit, "calibration": calibration_rows},
    )


def invariant_round_probabilities(baseline: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for arm in ARMS:
        copy = baseline.copy()
        copy["arm"] = arm
        pieces.append(copy)
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()


def round_invariance_audit(probabilities: pd.DataFrame) -> pd.DataFrame:
    keys = KEY_COLS
    baseline = probabilities[probabilities["arm"].eq("baseline")][
        keys + ["probability", "push_probability"]
    ]
    rows: list[dict[str, object]] = []
    for arm in CHALLENGERS:
        candidate = probabilities[probabilities["arm"].eq(arm)][
            keys + ["probability", "push_probability"]
        ]
        joined = baseline.merge(candidate, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one")
        p_delta = joined["probability_candidate"] - joined["probability_baseline"]
        push_delta = joined["push_probability_candidate"] - joined["push_probability_baseline"]
        rows.append(
            {
                "arm": arm,
                "opportunities": int(len(joined)),
                "max_abs_probability_delta": float(p_delta.abs().max()) if len(joined) else np.nan,
                "max_abs_push_delta": float(push_delta.abs().max()) if len(joined) else np.nan,
                "structurally_invariant": bool(
                    len(joined) == len(baseline)
                    and np.array_equal(
                        joined["probability_baseline"].to_numpy(),
                        joined["probability_candidate"].to_numpy(),
                    )
                    and np.array_equal(
                        joined["push_probability_baseline"].to_numpy(),
                        joined["push_probability_candidate"].to_numpy(),
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def select_candidate_bets(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
    arm: str,
    sizing_name: str,
    matchup_min_ev: float,
    finish_min_ev: float,
) -> pd.DataFrame:
    probability = probabilities[probabilities["arm"].eq(arm)].copy()
    matchup_gate = GateConfig(
        min_ev=float(matchup_min_ev),
        allowed_markets=("tournament_matchup",),
    )
    finish_gate = GateConfig(
        min_ev=float(finish_min_ev),
        allowed_markets=("win", "top_5", "top_10", "top_20"),
    )
    matchup_lines = lines[lines["market"].eq("tournament_matchup")]
    finish_lines = lines[~lines["market"].eq("tournament_matchup")]
    if sizing_name == "flat_1u":
        sizing = SizingConfig(method="flat", flat_stake=1.0)
        matchups = select_arm_bets(
            matchup_lines, probability, arm, matchup_gate, sizing, KEY_COLS
        )
        matchups = dedupe_matchup_direction(matchups)
        finishes = select_arm_bets(
            finish_lines, probability, arm, finish_gate, sizing, KEY_COLS
        )
        selected = pd.concat([matchups, finishes], ignore_index=True, sort=False)
    elif sizing_name == "production_style":
        matchups = select_arm_bets(
            matchup_lines,
            probability,
            arm,
            matchup_gate,
            SizingConfig(method="flat", flat_stake=1.0),
            KEY_COLS,
        )
        matchups = dedupe_matchup_direction(matchups)
        if not matchups.empty:
            matchups = matchups.drop(
                columns=["settlement_status", "pnl", "return_amount", "paid_fraction", "is_graded"],
                errors="ignore",
            )
            matchups["stake"] = np.where(
                matchups["close_odds"] < 0,
                matchups["close_odds"].abs() / 100.0,
                1.0,
            )
            matchups = settle_bets(matchups)
        finishes = select_arm_bets(
            finish_lines,
            probability,
            arm,
            finish_gate,
            SizingConfig(
                method="quarter_kelly",
                bankroll_units=50.0,
                kelly_fraction=0.25,
                max_bankroll_fraction=1.0,
                min_stake=0.005,
            ),
            KEY_COLS,
        )
        selected = pd.concat([matchups, finishes], ignore_index=True, sort=False)
    else:
        raise ValueError(f"Unknown sizing scenario: {sizing_name}")
    selected["sizing_name"] = sizing_name
    return selected


def dedupe_matchup_direction(selected: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-EV direction for each event/board/unordered pair."""

    if selected.empty:
        return selected.copy()
    output = selected.copy()
    output["_pair_key"] = _pair_key(output)
    identity = ["event_id", "board", "market", "_pair_key"]
    output["_book_sort"] = output.get("book", "").astype(str)
    output["_selection_sort"] = output["selection_id"].astype(str)
    output = output.sort_values(
        [*identity, "ev_per_unit", "close_odds", "_book_sort", "_selection_sort"],
        ascending=[True] * len(identity) + [False, False, True, True],
        kind="mergesort",
    )
    output = output.drop_duplicates(identity, keep="first")
    return output.drop(columns=["_pair_key", "_book_sort", "_selection_sort"]).reset_index(drop=True)


def _pair_key(frame: pd.DataFrame) -> pd.Series:
    left = frame["selection_id"].astype(str)
    right = frame["opponent_id"].fillna("").astype(str)
    ordered = ["|".join(sorted((a, b))) if b else a for a, b in zip(left, right)]
    return pd.Series(ordered, index=frame.index)


def compare_discovery(
    baseline: pd.DataFrame, challenger: pd.DataFrame, challenger_name: str
) -> pd.DataFrame:
    """Compare independent portfolios, explicitly collapsing opposite-side flips."""

    identity = KEY_COLS
    pair_identity = ["event_id", "board", "market", "pair_key"]
    base = baseline.copy()
    candidate = challenger.copy()
    base["pair_key"] = _pair_key(base)
    candidate["pair_key"] = _pair_key(candidate)
    exact_base = {
        tuple(row[identity].astype(str)): row for _, row in base.iterrows()
    }
    exact_candidate = {
        tuple(row[identity].astype(str)): row for _, row in candidate.iterrows()
    }
    candidate_by_pair = {
        tuple(row[pair_identity].astype(str)): row for _, row in candidate.iterrows()
    }
    rows: list[dict[str, object]] = []
    consumed_candidate: set[tuple[str, ...]] = set()

    def value(row: pd.Series | None, column: str) -> float:
        if row is None or column not in row or pd.isna(row[column]):
            return 0.0
        return float(row[column])

    def append_record(
        attribution: str,
        base_row: pd.Series | None,
        candidate_row: pd.Series | None,
    ) -> None:
        representative = candidate_row if candidate_row is not None else base_row
        assert representative is not None
        record = representative.to_dict()
        record.update(
            {
                "attribution": attribution,
                "challenger": challenger_name,
                "baseline_selection_id": (
                    str(base_row["selection_id"]) if base_row is not None else ""
                ),
                "challenger_selection_id": (
                    str(candidate_row["selection_id"]) if candidate_row is not None else ""
                ),
                "baseline_probability": value(base_row, "probability"),
                "challenger_probability": value(candidate_row, "probability"),
                "baseline_ev_per_unit": value(base_row, "ev_per_unit"),
                "challenger_ev_per_unit": value(candidate_row, "ev_per_unit"),
                "baseline_stake": value(base_row, "stake"),
                "challenger_stake": value(candidate_row, "stake"),
                "baseline_pnl": value(base_row, "pnl"),
                "challenger_pnl": value(candidate_row, "pnl"),
            }
        )
        record["stake_delta"] = record["challenger_stake"] - record["baseline_stake"]
        record["pnl_delta"] = record["challenger_pnl"] - record["baseline_pnl"]
        rows.append(record)

    for exact, base_row in exact_base.items():
        candidate_row = exact_candidate.get(exact)
        if candidate_row is not None:
            append_record("overlap", base_row, candidate_row)
            consumed_candidate.add(exact)
            continue
        pair = tuple(base_row[pair_identity].astype(str))
        candidate_row = candidate_by_pair.get(pair)
        candidate_exact = (
            tuple(candidate_row[identity].astype(str))
            if candidate_row is not None
            else None
        )
        if (
            candidate_row is not None
            and str(candidate_row["selection_id"]) != str(base_row["selection_id"])
        ):
            append_record("side_flip", base_row, candidate_row)
            assert candidate_exact is not None
            consumed_candidate.add(candidate_exact)
        else:
            append_record("dropped", base_row, None)
    for exact, candidate_row in exact_candidate.items():
        if exact not in consumed_candidate and exact not in exact_base:
            append_record("new", None, candidate_row)
    output = pd.DataFrame(rows)
    return output


def summarize_performance(selected: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("all", selected),
        ("pre_event_all", selected[selected["board"].eq("72h")]),
        ("72h_matchup", selected[selected["board"].eq("72h") & selected["market"].eq("tournament_matchup")]),
        ("finish", selected[~selected["market"].eq("tournament_matchup")]),
    ]
    groups.extend(
        (board, selected[selected["board"].eq(board)])
        for board in ("r1", "r2", "r3", "r4")
    )
    groups.extend(
        (market, selected[selected["market"].eq(market)])
        for market in ("win", "top_5", "top_10", "top_20")
    )
    rows: list[pd.DataFrame] = []
    for label, frame in groups:
        if frame.empty:
            continue
        summary = performance_summary(
            frame,
            event_col="event_id",
            event_order_col="event_start",
            starting_bankroll=100.0,
        )
        summary["report_group"] = label
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def add_zero_event_rows(
    bets: pd.DataFrame, lines: pd.DataFrame, arms: Sequence[str]
) -> pd.DataFrame:
    universe = lines[["event_id", "event_start"]].drop_duplicates("event_id")
    present = set(zip(bets["event_id"].astype(str), bets["arm"].astype(str))) if not bets.empty else set()
    rows = []
    for event_id, event_start in universe.itertuples(index=False, name=None):
        for arm in arms:
            if (str(event_id), str(arm)) not in present:
                rows.append(
                    {
                        "event_id": event_id,
                        "event_start": event_start,
                        "arm": arm,
                        "stake": 0.0,
                        "pnl": 0.0,
                    }
                )
    if not rows:
        return bets
    return pd.concat([bets, pd.DataFrame(rows)], ignore_index=True, sort=False)


def bootstrap_summaries(
    selected: pd.DataFrame,
    lines: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    group_masks: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=selected.index)),
        ("pre_event_all", selected["board"].eq("72h")),
        ("72h_matchup", selected["board"].eq("72h") & selected["market"].eq("tournament_matchup")),
        ("finish", ~selected["market"].eq("tournament_matchup")),
    ]
    group_masks.extend((board, selected["board"].eq(board)) for board in ("r1", "r2", "r3", "r4"))
    group_masks.extend(
        (market, selected["market"].eq(market))
        for market in ("win", "top_5", "top_10", "top_20")
    )
    pieces: list[pd.DataFrame] = []
    for challenger in CHALLENGERS:
        pair_arms = ("baseline", challenger)
        for label, mask in group_masks:
            group = selected.loc[mask & selected["arm"].isin(pair_arms)].copy()
            if label == "all":
                line_group = lines
            elif label == "pre_event_all":
                line_group = lines[lines["board"].eq("72h")]
            elif label == "72h_matchup":
                line_group = lines[lines["board"].eq("72h") & lines["market"].eq("tournament_matchup")]
            elif label == "finish":
                line_group = lines[~lines["market"].eq("tournament_matchup")]
            elif label in {"win", "top_5", "top_10", "top_20"}:
                line_group = lines[lines["market"].eq(label)]
            else:
                line_group = lines[lines["board"].eq(label)]
            if line_group.empty:
                continue
            group = add_zero_event_rows(group, line_group, pair_arms)
            boot = event_cluster_bootstrap(
                group,
                event_col="event_id",
                event_order_col="event_start",
                draws=draws,
                seed=stable_seed(f"bootstrap|{challenger}|{label}", seed),
                comparison=pair_arms,
            )
            boot["challenger"] = challenger
            boot["report_group"] = label
            pieces.append(boot)
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()


def probability_shift_summary(probabilities: pd.DataFrame) -> pd.DataFrame:
    baseline = probabilities[probabilities["arm"].eq("baseline")][KEY_COLS + ["probability"]]
    rows: list[dict[str, object]] = []
    for arm in CHALLENGERS:
        candidate = probabilities[probabilities["arm"].eq(arm)][KEY_COLS + ["probability"]]
        joined = baseline.merge(candidate, on=KEY_COLS, suffixes=("_baseline", "_candidate"), validate="one_to_one")
        joined["delta"] = joined["probability_candidate"] - joined["probability_baseline"]
        for board, frame in joined.groupby("board", sort=False):
            rows.append(
                {
                    "arm": arm,
                    "board": board,
                    "opportunities": int(len(frame)),
                    "mean_delta_pp": float(frame["delta"].mean() * 100.0),
                    "mean_abs_delta_pp": float(frame["delta"].abs().mean() * 100.0),
                    "max_abs_delta_pp": float(frame["delta"].abs().max() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def board_coverage_summary(
    selected_lines: pd.DataFrame, probabilities: pd.DataFrame
) -> pd.DataFrame:
    if selected_lines.empty:
        return pd.DataFrame()
    lines = selected_lines.copy()
    lines["pair_key"] = _pair_key(lines)
    lines["event_pair_key"] = lines["event_id"].astype(str) + "|" + lines["pair_key"]
    coverage = (
        lines.groupby(["price_point", "board", "market"], observed=True, dropna=False)
        .agg(
            event_count=("event_id", "nunique"),
            directional_line_count=("selection_id", "size"),
            unordered_opportunity_count=("event_pair_key", "nunique"),
        )
        .reset_index()
    )
    baseline = probabilities[probabilities["arm"].eq("baseline")].copy()
    available = (
        baseline.groupby(["board", "market"], observed=True, dropna=False)
        .agg(
            probability_event_count=("event_id", "nunique"),
            probability_opportunity_count=("selection_id", "size"),
        )
        .reset_index()
    )
    return coverage.merge(available, on=["board", "market"], how="left")


def discovery_summary(attribution: pd.DataFrame) -> pd.DataFrame:
    if attribution.empty:
        return pd.DataFrame()
    return (
        attribution.groupby(
            [
                "price_point",
                "sizing_name",
                "challenger",
                "board",
                "market",
                "attribution",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            bet_count=("attribution", "size"),
            baseline_stake=("baseline_stake", "sum"),
            challenger_stake=("challenger_stake", "sum"),
            baseline_pnl=("baseline_pnl", "sum"),
            challenger_pnl=("challenger_pnl", "sum"),
            stake_delta=("stake_delta", "sum"),
            pnl_delta=("pnl_delta", "sum"),
        )
        .reset_index()
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small report table without pandas' optional tabulate dependency."""

    def render(value: object) -> str:
        if pd.isna(value):
            text = ""
        elif isinstance(value, (float, np.floating)):
            text = f"{float(value):.6g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    columns = [render(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def build_report(
    config: dict[str, object],
    event_audit: pd.DataFrame,
    shift: pd.DataFrame,
    performance: pd.DataFrame,
    discovery: pd.DataFrame,
    invariance: pd.DataFrame,
    board_coverage: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    simulated = event_audit[event_audit["status"].eq("simulated")] if not event_audit.empty else event_audit
    excluded = event_audit[~event_audit["status"].eq("simulated")] if not event_audit.empty else event_audit
    paired = bootstrap[
        bootstrap.get("sizing_name", pd.Series(index=bootstrap.index, dtype=object)).eq(
            "production_style"
        )
        & bootstrap.get("metric", pd.Series(index=bootstrap.index, dtype=object)).eq("pnl")
        & bootstrap.get("arm", pd.Series(index=bootstrap.index, dtype=object))
        .astype(str)
        .str.endswith("_minus_baseline")
        & bootstrap.get("report_group", pd.Series(index=bootstrap.index, dtype=object)).isin(
            ["all", "pre_event_all", "72h_matchup", "finish"]
        )
    ].copy()
    paired_columns = [
        "price_point",
        "report_group",
        "challenger",
        "estimate",
        "ci_low",
        "ci_high",
        "probability_above_zero",
        "cluster_count",
    ]
    paired = paired[[column for column in paired_columns if column in paired.columns]]
    lines = [
        "# 2026 category-profile full-board replay",
        "",
        f"- Events simulated: {len(simulated)}",
        f"- Events excluded/failed closed: {len(excluded)}",
        f"- Simulations per event/arm: {config['simulations']:,}",
        f"- Independent discovery gates: matchups EV >= {float(config['min_ev']):.1%}; "
        f"finish positions EV >= {float(config['finish_min_ev']):.1%}.",
        "- Prices: best archived Betcris/Pinnacle/BetOnline at open and close.",
        "- Pre-event markets: 72-hole matchups plus win/top-5/top-10/top-20.",
        "- Round markets: R1-R4 audited scored-authority board where frozen probabilities exist; structurally invariant.",
        "",
        "## Interpretation contract",
        "",
        "The 72-hole and finish boards are the substantive category-cascade test. "
        "R1-R4 are copied from the audited baseline authority for every arm because "
        "the production single-round kernel removes zero-sum profile movement when "
        "it recenters category means to fixed total skill. Any nonzero round delta is a bug.",
        "",
        "## Probability movement",
        "",
        markdown_table(shift) if not shift.empty else "No probability rows.",
        "",
        "## Round invariance",
        "",
        markdown_table(invariance) if not invariance.empty else "No round rows.",
        "",
        "## Board coverage",
        "",
        markdown_table(board_coverage) if not board_coverage.empty else "No board coverage rows.",
        "",
        "## Paired production P&L deltas",
        "",
        "Event-cluster 95% intervals; positive means the challenger beat baseline.",
        "",
        markdown_table(paired) if not paired.empty else "No paired bootstrap rows.",
        "",
        "## Portfolio performance",
        "",
        markdown_table(performance) if not performance.empty else "No selected bets.",
        "",
        "## Independent discovery by market",
        "",
        markdown_table(discovery) if not discovery.empty else "No attribution rows.",
        "",
        "## Exploratory and leakage caveats",
        "",
        "- The targeted ARG/PUTT arm was designed after inspecting 2026 diagnostics; this is exploratory, not a sealed out-of-sample selection.",
        "- Player category histories and ages are strictly before each event, and course shape is selected by physical course. Current prediction coefficients and the installed kernel are retrospective snapshots.",
        "- Historical weather forecast states were not archived, so every arm uses the same neutral weather input. This isolates category profiles but is not an exact weekly-production reconstruction.",
        "- Best-of-three archived prices can have asynchronous timestamps and are not proof that every quoted combination was simultaneously executable.",
        "- Round coverage is not 35 events for every day: R1 covers 34 events, R2 and R3 cover 31, and R4 covers 30. BMW has substantive pre-event/finish coverage but no frozen scored round authority; AmEx, Farmers, and Puerto Rico lack R2-R4 authority, and Valero lacks R4.",
        "- The audited R1-R4 authority is its frozen scored/evaluation board; its earlier exact-common-menu construction dropped some raw quotes. It should not be described as every raw DataGolf quote.",
        "- Unsupported Stableford, team, or staggered-start formats fail closed rather than being forced through a 72-hole stroke-play kernel.",
        "- ROI is descriptive and highly sensitive to selection, settlement, and a small number of events; use the event-cluster intervals and probability movement as primary evidence.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.simulations <= 0 or args.bootstrap <= 0:
        raise ValueError("--simulations and --bootstrap must be positive")
    if not 0.0 <= args.min_ev < 1.0 or not 0.0 <= args.finish_min_ev < 1.0:
        raise ValueError("--min-ev and --finish-min-ev must be in [0, 1)")
    cache_dirs = [args.odds_cache, *args.supplemental_odds_cache]
    shot_inputs = [
        args.shot_export / f"pga_{year}_shot_sg.parquet"
        for year in range(2020, pd.Timestamp(args.end).year + 1)
    ]
    inputs = [
        args.db,
        args.predictions,
        args.crosswalk,
        args.shot_export / "manifest.json",
        *shot_inputs,
        *(cache / "manifest.json" for cache in cache_dirs),
        args.score_authority,
        args.correlation,
        args.coefficients,
    ]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(map(str, missing)))
    config: dict[str, object] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "start": args.start,
        "end": args.end,
        "simulations": args.simulations,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "min_ev": args.min_ev,
        "finish_min_ev": args.finish_min_ev,
        "week_latent_sd": args.week_latent_sd,
        "arms": ARMS,
        "books": BOOKS,
        "price_points": PRICE_POINTS,
        "sizing_scenarios": SIZING_SCENARIOS,
        "eligibility": {
            "tour": "pga",
            "max_age_exclusive": MAX_AGE,
            "min_prior_rounds": MIN_PRIOR_ROUNDS,
            "min_non_null_each_category": MIN_CATEGORY_OBS,
        },
        "blanket_factor": SHRINK_FACTOR,
        "targeted_raw_factor": -0.35,
        "common_cap": PROFILE_CAP,
        "common_random_numbers": "identical_event_seed_all_arms",
        "weather": "neutral_common_all_arms_no_historical_forecast_archive",
        "course_shape_disabled": bool(args.no_course_shape),
        "shot_variance_disabled": bool(args.no_shot_variance),
        "preflight_only": bool(args.preflight_only),
        "event_keys": sorted(args.event_key or []),
        "event_limit": args.event_limit,
        "odds_caches": [str(path.resolve()) for path in cache_dirs],
        "inputs": [file_signature(path) for path in inputs],
        "source_files": [
            file_signature(path)
            for path in (
                Path(__file__),
                REPO / "sim_inputs.py",
                ARCHIVE / "backtest_distribution_engines_2026.py",
                ARCHIVE / "backtest_shot_mean_closing_markets.py",
                ARCHIVE / "backtest_shot_dispersion_recent.py",
                ARCHIVE / "build_bmw_shot_features_shadow.py",
                ARCHIVE / "historical_odds_cache.py",
                ARCHIVE / "market_backtest_scoring.py",
            )
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sims_kernel": sims_kernel.version(),
        },
        "research_only": True,
    }
    run_digest = prepare_output(args.output_dir, config, args.resume)
    readers = [HistoricalOddsCacheReader(path) for path in cache_dirs]
    replacements = load_name_replacements()
    rosters = normalize_prediction_rosters(
        [row for reader in readers for row in reader.predictions()], replacements
    )
    matchup_rows = [row for reader in readers for row in reader.matchups()]
    outright_rows = [row for reader in readers for row in reader.outrights()]
    cache_lines_by_price = {
        price: normalize_cache_board(matchup_rows, outright_rows, price)
        for price in PRICE_POINTS
    }
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    for price, frame in cache_lines_by_price.items():
        dates = pd.to_datetime(frame["event_start"], errors="coerce")
        cache_lines_by_price[price] = frame[
            dates.between(start, end + pd.Timedelta(days=30), inclusive="both")
        ].copy()
    cache_event_keys = set.intersection(
        *(set(frame["event_key"].unique()) for frame in cache_lines_by_price.values()),
        set(rosters.loc[rosters["year"].eq(2026), "event_key"].unique()),
    )
    metadata = read_event_metadata(args.db, cache_event_keys)
    event_keys = sorted(
        (
            key
            for key in cache_event_keys & set(metadata)
            if start <= metadata[key].start <= end
        ),
        key=lambda key: metadata[key].start,
    )
    if args.event_key:
        requested = {canonical_internal_event_key(key) for key in args.event_key}
        event_keys = [key for key in event_keys if key in requested]
    if args.event_limit is not None:
        event_keys = event_keys[: args.event_limit]
    if not event_keys:
        raise ValueError("No eligible 2026 events remain after filters")

    round_source = pd.read_parquet(args.score_authority)
    round_lines_by_price: dict[str, pd.DataFrame] = {}
    round_probability_pieces: list[pd.DataFrame] = []
    for price in PRICE_POINTS:
        raw_lines, raw_probability = normalize_round_authority(round_source, price)
        raw_lines = raw_lines[raw_lines["event_key"].isin(event_keys)]
        raw_probability = raw_probability[raw_probability["event_key"].isin(event_keys)]
        round_lines_by_price[price] = select_best_of_three(raw_lines)
        round_probability_pieces.append(raw_probability)
    round_baseline_probability = (
        pd.concat(round_probability_pieces, ignore_index=True, sort=False)
        .drop_duplicates(KEY_COLS, keep="first")
    )
    round_probabilities = invariant_round_probabilities(round_baseline_probability)
    invariance = round_invariance_audit(round_probabilities)
    _atomic_csv(invariance, args.output_dir / "round_invariance_audit.csv")

    all_player_ids = sorted(
        rosters[rosters["event_key"].isin(event_keys)]["dg_id"].astype(int).unique()
    )
    histories, model_histories = load_player_histories(
        args.db, all_player_ids, max(metadata[key].start for key in event_keys) + pd.Timedelta(days=1)
    )
    age_histories = load_age_histories(
        args.db, all_player_ids, max(metadata[key].start for key in event_keys) + pd.Timedelta(days=1)
    )
    r1_predictions = load_r1_predictions(args.predictions, replacements)
    source_crosswalk = load_source_crosswalk(args.crosswalk)
    source_ids = (
        source_crosswalk[source_crosswalk["dg_id"].isin(all_player_ids)]["source_player_id"]
        .dropna()
        .astype(str)
        .loc[lambda values: values.ne("")]
        .drop_duplicates()
        .tolist()
    )
    if args.no_shot_variance or args.preflight_only:
        shot_stats = pd.DataFrame(columns=["event_date"])
        valid_shots = 0
    else:
        print(
            f"Loading strict-prior shot dispersion for {len(source_ids)} source players ...",
            flush=True,
        )
        event_dates = load_event_dates(args.db)
        shot_stats, valid_shots = load_shot_round_stats(
            args.shot_export,
            source_ids,
            event_dates,
            max(metadata[key].start for key in event_keys) + pd.Timedelta(days=1),
        )
        print(
            f"Loaded {len(shot_stats):,} player/category/round rows from {valid_shots:,} shots.",
            flush=True,
        )
    _, chol = load_correlation(args.correlation)
    coefficients = load_coefficients(args.coefficients)

    probability_pieces: list[pd.DataFrame] = [round_probabilities]
    eligibility_pieces: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    opportunity_union = pd.concat(cache_lines_by_price.values(), ignore_index=True, sort=False)
    opportunity_union = opportunity_union.drop_duplicates(KEY_COLS)

    for number, event_key in enumerate(event_keys, start=1):
        meta = metadata[event_key]
        checkpoint = checkpoint_dir / f"{_event_checkpoint_name(event_key)}.parquet"
        eligibility_checkpoint = checkpoint_dir / f"{_event_checkpoint_name(event_key)}_eligibility.parquet"
        audit_checkpoint = checkpoint_dir / f"{_event_checkpoint_name(event_key)}_audit.json"
        exclusion = finish_event_exclusion(meta)
        if exclusion:
            event_rows.append(
                {
                    "event_key": event_key,
                    "event_name": meta.event_name,
                    "event_start": meta.start,
                    "status": "excluded",
                    "exclusion_reason": exclusion,
                }
            )
            print(f"[{number}/{len(event_keys)}] excluded {event_key}: {exclusion}")
            continue
        if args.resume and checkpoint.is_file() and eligibility_checkpoint.is_file() and audit_checkpoint.is_file():
            audit = json.loads(audit_checkpoint.read_text(encoding="utf-8"))
            checkpoint_meta = audit.get("checkpoint", {})
            if checkpoint_meta.get("event_key") != event_key:
                raise ValueError(f"Checkpoint event mismatch for {event_key}")
            if checkpoint_meta.get("config_digest") != run_digest:
                raise ValueError(f"Checkpoint config mismatch for {event_key}")
            for label, path in (
                ("probabilities", checkpoint),
                ("eligibility", eligibility_checkpoint),
            ):
                if checkpoint_meta.get(label) != file_signature(path):
                    raise ValueError(f"Checkpoint artifact mismatch for {event_key}: {label}")
            probabilities = pd.read_parquet(checkpoint)
            eligibility = pd.read_parquet(eligibility_checkpoint)
            probability_pieces.append(probabilities)
            eligibility_pieces.append(eligibility)
            event_rows.append(audit["event"])
            calibration_rows.extend(audit["calibration"])
            print(f"[{number}/{len(event_keys)}] resumed {event_key}")
            continue
        roster = rosters[rosters["event_key"].eq(event_key)]
        lines = opportunity_union[opportunity_union["event_key"].eq(event_key)]
        if args.preflight_only:
            field, field_audit = build_event_field_core(
                roster, meta, r1_predictions, model_histories, source_crosswalk
            )
            eligibility = profile_eligibility(field, histories, age_histories, meta.start)
            eligibility.insert(0, "event_key", event_key)
            eligibility_pieces.append(eligibility)
            event_rows.append(
                {
                    "event_key": event_key,
                    "event_name": meta.event_name,
                    "event_start": meta.start,
                    "status": "preflight",
                    "eligibility_eligible_players": int(eligibility["eligible"].sum()),
                    "opportunity_count": int(len(lines)),
                    **field_audit,
                }
            )
            print(f"[{number}/{len(event_keys)}] preflight {event_key}")
            continue
        started = time.perf_counter()
        probabilities, eligibility, audit = simulate_event_probabilities(
            event_key,
            meta,
            roster,
            lines,
            histories,
            model_histories,
            age_histories,
            source_crosswalk,
            shot_stats,
            r1_predictions,
            chol,
            coefficients,
            args.db,
            args.simulations,
            args.seed,
            args.week_latent_sd,
            args.no_course_shape,
            args.no_shot_variance,
        )
        audit["event"]["runtime_seconds"] = time.perf_counter() - started
        _atomic_parquet(probabilities, checkpoint)
        _atomic_parquet(eligibility, eligibility_checkpoint)
        audit["checkpoint"] = {
            "event_key": event_key,
            "config_digest": run_digest,
            "probabilities": file_signature(checkpoint),
            "eligibility": file_signature(eligibility_checkpoint),
        }
        _atomic_text(audit_checkpoint, json.dumps(audit, indent=2, default=_json_default) + "\n")
        probability_pieces.append(probabilities)
        eligibility_pieces.append(eligibility)
        event_rows.append(audit["event"])
        calibration_rows.extend(audit["calibration"])
        print(
            f"[{number}/{len(event_keys)}] simulated {event_key} in "
            f"{audit['event']['runtime_seconds']:.1f}s"
        )

    event_audit = pd.DataFrame(event_rows)
    eligibility_audit = pd.concat(eligibility_pieces, ignore_index=True, sort=False) if eligibility_pieces else pd.DataFrame()
    calibration_audit = pd.DataFrame(calibration_rows)
    _atomic_csv(event_audit, args.output_dir / "event_audit.csv")
    _atomic_parquet(eligibility_audit, args.output_dir / "eligibility_audit.parquet")
    _atomic_csv(calibration_audit, args.output_dir / "calibration_audit.csv")
    if args.preflight_only:
        print(f"Preflight complete: {args.output_dir}")
        return 0

    probabilities = pd.concat(probability_pieces, ignore_index=True, sort=False)
    probabilities = probabilities.drop_duplicates([*KEY_COLS, "arm"], keep="first")
    _atomic_parquet(probabilities, args.output_dir / "probabilities.parquet")
    shift = probability_shift_summary(probabilities)
    _atomic_csv(shift, args.output_dir / "probability_shift_summary.csv")

    selected_pieces: list[pd.DataFrame] = []
    attribution_pieces: list[pd.DataFrame] = []
    performance_pieces: list[pd.DataFrame] = []
    bootstrap_pieces: list[pd.DataFrame] = []
    all_selected_lines: list[pd.DataFrame] = []
    for price in PRICE_POINTS:
        preevent = select_best_of_three(
            cache_lines_by_price[price][cache_lines_by_price[price]["event_key"].isin(event_keys)]
        )
        preevent["event_start"] = preevent["event_key"].map(
            {key: metadata[key].start for key in event_keys}
        )
        lines = pd.concat([preevent, round_lines_by_price[price]], ignore_index=True, sort=False)
        lines["price_point"] = price
        all_selected_lines.append(lines)
        for sizing in SIZING_SCENARIOS:
            scenario_bets = []
            for arm in ARMS:
                bets = select_candidate_bets(
                    lines,
                    probabilities,
                    arm,
                    sizing,
                    args.min_ev,
                    args.finish_min_ev,
                )
                bets["price_point"] = price
                scenario_bets.append(bets)
                selected_pieces.append(bets)
            scenario = pd.concat(scenario_bets, ignore_index=True, sort=False)
            perf = summarize_performance(scenario)
            perf["price_point"] = price
            perf["sizing_name"] = sizing
            performance_pieces.append(perf)
            boot = bootstrap_summaries(scenario, lines, args.bootstrap, args.seed)
            boot["price_point"] = price
            boot["sizing_name"] = sizing
            bootstrap_pieces.append(boot)
            baseline = scenario[scenario["arm"].eq("baseline")]
            for challenger in CHALLENGERS:
                comparison = compare_discovery(
                    baseline,
                    scenario[scenario["arm"].eq(challenger)],
                    challenger,
                )
                attribution_pieces.append(comparison)

    selected = pd.concat(selected_pieces, ignore_index=True, sort=False) if selected_pieces else pd.DataFrame()
    attribution = pd.concat(attribution_pieces, ignore_index=True, sort=False) if attribution_pieces else pd.DataFrame()
    performance = pd.concat(performance_pieces, ignore_index=True, sort=False) if performance_pieces else pd.DataFrame()
    bootstrap = pd.concat(bootstrap_pieces, ignore_index=True, sort=False) if bootstrap_pieces else pd.DataFrame()
    selected_lines = pd.concat(all_selected_lines, ignore_index=True, sort=False)
    discovery = discovery_summary(attribution)
    board_coverage = board_coverage_summary(selected_lines, probabilities)
    _atomic_parquet(selected_lines, args.output_dir / "best_of_3_lines.parquet")
    _atomic_csv(board_coverage, args.output_dir / "board_coverage.csv")
    _atomic_parquet(selected, args.output_dir / "selected_bets.parquet")
    _atomic_parquet(attribution, args.output_dir / "discovery_attribution.parquet")
    _atomic_csv(discovery, args.output_dir / "discovery_summary.csv")
    _atomic_csv(performance, args.output_dir / "performance_summary.csv")
    _atomic_csv(bootstrap, args.output_dir / "event_cluster_bootstrap.csv")
    report = build_report(
        config,
        event_audit,
        shift,
        performance,
        discovery,
        invariance,
        board_coverage,
        bootstrap,
    )
    _atomic_text(args.output_dir / "report.md", report)
    print(f"Replay complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

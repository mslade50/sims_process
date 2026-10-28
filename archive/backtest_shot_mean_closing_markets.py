"""Three-year closing-market backtest for a shot-derived APP mean factor.

This is a retrospective research replay of today's own-model production core,
not a claim that the exact historical weekly production artifacts were
preserved.  Every event uses a DataGolf archived pre-tournament roster,
event-dated prediction rows, strictly earlier category and shot histories,
current category-first production mechanics, the current shot-variance overlay
in both arms, and common Monte Carlo random numbers.  The production prediction
coefficients themselves were fit retrospectively on the full historical sample;
absolute ROI is therefore a current-code sensitivity, not a point-in-time OOS
estimate.  The challenger shot coefficient is still refit strictly walk-forward.

The only difference between the baseline and the primary challenger is:

    challenger APP mean = baseline APP mean + walk-forward shot delta
    challenger total mean = baseline total mean + the same shot delta

Closing odds never enter the simulation or the walk-forward coefficient fit.
They are used only after probabilities have been frozen, for bet selection and
settlement.  R2-R4 live boards are intentionally out of scope because their
historical live model states were not archived.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
SIM_PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")
DEFAULT_DB = SIM_PREP / "work" / "dg_historical.db"
DEFAULT_PREDS = SIM_PREP / "work" / "sg_preds_resids.csv"
DEFAULT_CROSSWALK = SIM_PREP / "shot_bridge" / "crosswalk.parquet"
DEFAULT_FACTOR_PANEL = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_factor_panel_20230822_20260822.parquet"
)
DEFAULT_ODDS_CACHE = (
    REPO
    / "permanent_data"
    / "research"
    / "dg_closing_cache_20230822_20260822"
)
DEFAULT_OUTPUT = (
    REPO
    / "permanent_data"
    / "research"
    / "shot_mean_closing_backtest_20230822_20260822_v2"
)
CORR_PATH = REPO / "permanent_data" / "sg_cat_corr_tour_within_player_pearson.csv"
COEFF_PATH = REPO / "coeffs_cache.json"

for path in (str(REPO), str(ARCHIVE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backtest_distribution_engines_2026 import (  # noqa: E402
    build_category_params,
    load_correlation,
)
from backtest_shot_dispersion_recent import (  # noqa: E402
    BASELINE_SKEW,
    CAT_ORDER,
    reconstruct_course_shape,
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
    compare_arm_bets,
    event_cluster_bootstrap,
    performance_summary,
    select_arm_bets,
    select_closing_lines,
    settle_bets,
    summarize_attribution,
)


START = "2023-08-22"
END = "2026-08-22"
FIXED_BETA = -0.098195
CAP_ABS_SG = 0.15
MATCHUP_GATE = 0.03
OUTRIGHT_GATE = 0.05
MARKETS = ("tournament_matchup", "win", "top_5", "top_10", "top_20")
MATCHUP_BOOKS = ("betcris", "pinnacle", "betonline")
OUTRIGHT_BOOKS = ("betonline", "pinnacle", "betcris")
KEY_COLS = ["event_id", "market", "selection_id", "opponent_id"]
PRIMARY_ARM = "walk_forward_uncapped"
DEFAULT_ARMS = (PRIMARY_ARM, "walk_forward_cap015")
ALL_ARMS = (*DEFAULT_ARMS, "fixed_beta_sensitivity")


@dataclass(frozen=True)
class EventMeta:
    event_key: str
    year: int
    event_id: int
    event_name: str
    start: pd.Timestamp
    course_counts: tuple[tuple[int, int], ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--simulations", type=int, default=50_000)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--export", type=Path, default=EXPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--factor-panel", type=Path, default=DEFAULT_FACTOR_PANEL)
    parser.add_argument("--odds-cache", type=Path, default=DEFAULT_ODDS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help=f"Comma-separated challenger arms from: {', '.join(ALL_ARMS)}",
    )
    parser.add_argument(
        "--event-key",
        action="append",
        help="Optional canonical event key filter; repeat for multiple events.",
    )
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--no-course-shape", action="store_true")
    parser.add_argument("--no-shot-variance", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build and export the eligible-event/market audit without simulating.",
    )
    return parser.parse_args(argv)


def canonical_event_key(year: object, event_id: object) -> str:
    return f"pga:R{int(float(year)):04d}{int(float(event_id)):03d}"


def canonical_player_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return str(value).strip().lower()


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").casefold().strip().split())


def load_name_replacements(path: Path = REPO / "sim_inputs.py") -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "name_replacements"
            for target in node.targets
        ):
            continue
        values = ast.literal_eval(node.value)
        return {normalize_text(key): normalize_text(value) for key, value in values.items()}
    return {}


def load_sim_input_literal(name: str, path: Path = REPO / "sim_inputs.py") -> object:
    """Read one literal weekly setting without importing config side effects."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"No literal {name} setting found in {path}")


def normalize_name(value: object, replacements: dict[str, str]) -> str:
    key = normalize_text(value)
    return replacements.get(key, key)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(event_key: str, base_seed: int) -> int:
    value = hashlib.sha256(f"{base_seed}|{event_key}".encode("utf-8")).digest()
    return int.from_bytes(value[:4], "little") & 0x7FFFFFFF


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def normalize_event_audit_rows(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Build a stable audit frame from fresh and JSON-resumed checkpoints."""
    frame = pd.json_normalize(rows, sep=".")
    if "start" in frame.columns:
        frame["start"] = pd.to_datetime(frame["start"], errors="coerce")
    return frame


def _row_year(frame: pd.DataFrame) -> pd.Series:
    for column in ("season", "query_year", "year"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=frame.index)


def add_event_key(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    years = _row_year(output)
    event_ids = pd.to_numeric(output.get("event_id"), errors="coerce")
    valid = years.notna() & event_ids.notna()
    output = output.loc[valid].copy()
    years = years.loc[valid]
    event_ids = event_ids.loc[valid]
    output["event_key"] = [
        canonical_event_key(year, event_id)
        for year, event_id in zip(years, event_ids)
    ]
    output["year"] = years.astype(int).to_numpy()
    output["dg_event_id"] = event_ids.astype(int).to_numpy()
    return output


def normalize_prediction_rosters(
    rows: Sequence[dict[str, Any]], replacements: dict[str, str]
) -> pd.DataFrame:
    frame = add_event_key(pd.DataFrame(rows))
    if frame.empty:
        return pd.DataFrame(columns=["event_key", "year", "dg_event_id", "dg_id", "player_name"])
    if "model" in frame.columns and frame["model"].eq("baseline").any():
        frame = frame[frame["model"].eq("baseline")].copy()
    frame["dg_id"] = pd.to_numeric(frame.get("dg_id"), errors="coerce")
    frame["player_name"] = frame.get("player_name", "").map(
        lambda value: normalize_name(value, replacements)
    )
    frame = frame.dropna(subset=["dg_id"])
    frame["dg_id"] = frame["dg_id"].astype(int)
    frame = frame.sort_values(["event_key", "dg_id"], kind="mergesort")
    return frame.drop_duplicates(["event_key", "dg_id"], keep="first")


def _valid_american(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & ((numeric <= -100.0) | (numeric >= 100.0))


def normalize_matchup_lines(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    raw = add_event_key(pd.DataFrame(rows))
    if raw.empty:
        return pd.DataFrame()
    bet_type = raw.get("bet_type", "").astype(str).str.casefold()
    raw = raw[bet_type.str.contains("tournament|72", regex=True, na=False)].copy()
    expanded: list[dict[str, Any]] = []
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
                    "selection_id": canonical_player_id(row.get(f"{side}_dg_id")),
                    "selection_name": row.get(f"{side}_player_name"),
                    "opponent_id": canonical_player_id(row.get(f"{other}_dg_id")),
                    "opponent_name": row.get(f"{other}_player_name"),
                    "book": normalize_text(row.get("book")),
                    "close_odds": row.get(f"{side}_close"),
                    "close_time": row.get("close_time"),
                    "tie_rule": row.get("tie_rule"),
                    "outcome": row.get(f"{side}_outcome_text", row.get(f"{side}_outcome")),
                    "outcome_numeric": row.get(f"{side}_outcome"),
                    "record_id": row.get("record_id"),
                }
            )
    frame = pd.DataFrame(expanded)
    if frame.empty:
        return frame
    valid = (
        frame["selection_id"].ne("")
        & frame["opponent_id"].ne("")
        & _valid_american(frame["close_odds"])
    )
    frame = frame.loc[valid].copy()
    frame["close_odds"] = pd.to_numeric(frame["close_odds"], errors="raise")
    frame["outcome_numeric"] = pd.to_numeric(frame["outcome_numeric"], errors="coerce")
    return frame


def canonical_market(value: object) -> str:
    text = normalize_text(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "winner": "win",
        "top5": "top_5",
        "top10": "top_10",
        "top20": "top_20",
    }
    return aliases.get(text, text)


def normalize_outright_lines(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    raw = add_event_key(pd.DataFrame(rows))
    if raw.empty:
        return pd.DataFrame()
    raw["market"] = raw.get("market", "").map(canonical_market)
    raw = raw[raw["market"].isin(MARKETS[1:])].copy()
    frame = pd.DataFrame(
        {
            "event_id": raw["event_key"],
            "event_key": raw["event_key"],
            "year": raw["year"],
            "dg_event_id": raw["dg_event_id"],
            "event_name": raw.get("event_name"),
            "market": raw["market"],
            "selection_id": raw.get("dg_id").map(canonical_player_id),
            "selection_name": raw.get("player_name"),
            "opponent_id": "",
            "opponent_name": "",
            "book": raw.get("book", "").map(normalize_text),
            "close_odds": raw.get("close_odds"),
            "close_time": raw.get("close_time"),
            "tie_rule": "",
            "outcome": raw.get("outcome", raw.get("bet_outcome_text")),
            "bet_outcome_numeric": raw.get("bet_outcome_numeric"),
            "record_id": raw.get("record_id"),
        }
    )
    valid = frame["selection_id"].ne("") & _valid_american(frame["close_odds"])
    frame = frame.loc[valid].copy()
    frame["close_odds"] = pd.to_numeric(frame["close_odds"], errors="raise")
    frame["bet_outcome_numeric"] = pd.to_numeric(
        frame["bet_outcome_numeric"], errors="coerce"
    )
    return frame


def select_line_sets(lines: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matchup = lines[lines["market"].eq("tournament_matchup")]
    outright = lines[~lines["market"].eq("tournament_matchup")]
    outputs: dict[str, pd.DataFrame] = {}
    for mode in ("fixed_book", "best_of_sharp"):
        h2h = select_closing_lines(
            matchup,
            LinePolicy(mode=mode, books=MATCHUP_BOOKS),
            KEY_COLS,
        )
        finish = select_closing_lines(
            outright,
            LinePolicy(mode=mode, books=OUTRIGHT_BOOKS),
            KEY_COLS,
        )
        output = pd.concat([h2h, finish], ignore_index=True, sort=False)
        output["line_policy"] = mode
        outputs[mode] = output
    return outputs


def read_event_metadata(db_path: Path, event_keys: Iterable[str]) -> dict[str, EventMeta]:
    requested = set(event_keys)
    years = sorted({int(key.split("R", 1)[1][:4]) for key in requested})
    if not years:
        return {}
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        frame = pd.read_sql_query(
            "SELECT year, event_id, event_name, round_num, round_date, course_num "
            "FROM player_rounds WHERE tour='pga' AND year BETWEEN ? AND ?",
            connection,
            params=[min(years), max(years)],
        )
    frame["round_date"] = pd.to_datetime(frame["round_date"], errors="coerce").dt.normalize()
    frame["event_key"] = [
        canonical_event_key(year, event_id)
        for year, event_id in frame[["year", "event_id"]].itertuples(index=False, name=None)
    ]
    frame = frame[frame["event_key"].isin(requested)]
    output: dict[str, EventMeta] = {}
    for event_key, event in frame.groupby("event_key", sort=False):
        r1 = event[event["round_num"].eq(1)]
        counts = r1["course_num"].dropna().astype(int).value_counts()
        course_counts = tuple((int(course), int(count)) for course, count in counts.items())
        names = r1["event_name"].dropna()
        output[event_key] = EventMeta(
            event_key=event_key,
            year=int(r1["year"].iloc[0]),
            event_id=int(r1["event_id"].iloc[0]),
            event_name=str(names.mode().iloc[0] if len(names) else ""),
            start=pd.Timestamp(r1["round_date"].min()).normalize(),
            course_counts=course_counts,
        )
    return output


def sqlite_rows_for_players(
    db_path: Path,
    table: str,
    columns: Sequence[str],
    player_ids: Sequence[int],
    where: str,
    params: Sequence[object],
    batch_size: int = 700,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for offset in range(0, len(player_ids), batch_size):
            batch = [int(value) for value in player_ids[offset : offset + batch_size]]
            marks = ",".join("?" for _ in batch)
            query = (
                f"SELECT {', '.join(columns)} FROM {table} "
                f"WHERE dg_id IN ({marks}) AND {where}"
            )
            pieces.append(pd.read_sql_query(query, connection, params=[*batch, *params]))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def load_player_histories(
    db_path: Path, player_ids: Sequence[int], end: pd.Timestamp
) -> tuple[dict[int, pd.DataFrame], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rounds = sqlite_rows_for_players(
        db_path,
        "player_rounds",
        [
            "dg_id",
            "round_date",
            "sg_ott_adj",
            "sg_app_adj",
            "sg_arg_adj",
            "sg_putt_adj",
        ],
        player_ids,
        "tour='pga' AND round_date < ? AND sg_total_adj IS NOT NULL",
        [end.strftime("%Y-%m-%d")],
    )
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce").dt.normalize()
    histories = {
        int(dg_id): group.sort_values("round_date", kind="mergesort").copy()
        for dg_id, group in rounds.groupby("dg_id", sort=False)
    }

    model = sqlite_rows_for_players(
        db_path,
        "model_results2",
        ["rowid", "dg_id", "Date", "skill_est"],
        player_ids,
        "Date < ? AND skill_est IS NOT NULL",
        [end.strftime("%Y-%m-%d")],
    )
    model["Date"] = pd.to_datetime(model["Date"], errors="coerce").dt.normalize()
    model["skill_est"] = pd.to_numeric(model["skill_est"], errors="coerce")
    model = (
        model.dropna(subset=["Date", "skill_est"])
        .sort_values(["dg_id", "Date", "rowid"], kind="mergesort")
        .drop_duplicates(["dg_id", "Date"], keep="last")
    )
    model_histories: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dg_id, group in model.groupby("dg_id", sort=False):
        dates = group["Date"].to_numpy(dtype="datetime64[ns]")
        values = group["skill_est"].to_numpy(float)
        rows = group["rowid"].to_numpy(int)
        model_histories[int(dg_id)] = (dates, values, rows)
    return histories, model_histories


def latest_model_skill(
    model_histories: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dg_id: int,
    cutoff: pd.Timestamp,
) -> float:
    history = model_histories.get(int(dg_id))
    if history is None:
        return np.nan
    dates, values, _ = history
    index = int(np.searchsorted(dates, np.datetime64(cutoff), side="left") - 1)
    return float(values[index]) if index >= 0 else np.nan


def load_r1_predictions(path: Path, replacements: dict[str, str]) -> pd.DataFrame:
    columns = ["player_name", "event_id", "year", "round_num", "prediction"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame[frame["round_num"].eq(1)].copy()
    frame["player_name"] = frame["player_name"].map(
        lambda value: normalize_name(value, replacements)
    )
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame.dropna(subset=["event_id", "year", "prediction"])
    frame[["event_id", "year"]] = frame[["event_id", "year"]].astype(int)
    return frame.drop_duplicates(["year", "event_id", "player_name"], keep="last")


def load_source_crosswalk(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["dg_id", "source_player_id", "round_date"])
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="coerce")
    frame["source_player_id"] = frame["source_player_id"].map(canonical_player_id)
    frame["round_date"] = pd.to_datetime(frame["round_date"], errors="coerce")
    frame = frame.dropna(subset=["dg_id"]).sort_values("round_date", kind="mergesort")
    frame["dg_id"] = frame["dg_id"].astype(int)
    return frame.drop_duplicates("dg_id", keep="last")[["dg_id", "source_player_id"]]


def build_event_field(
    roster: pd.DataFrame,
    meta: EventMeta,
    r1_predictions: pd.DataFrame,
    model_histories: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_crosswalk: pd.DataFrame,
    event_factor: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        latest_model_skill(model_histories, dg_id, meta.start)
        for dg_id in field.loc[missing, "dg_id"]
    ]
    model_filled = missing & field["skill_est"].notna()
    field.loc[model_filled, "skill_source"] = "dated_model_fallback"

    still_missing = field["skill_est"].isna()
    finite = field.loc[~still_missing, "skill_est"]
    floor = float(finite.quantile(0.10) - 0.50) if len(finite) else -2.0
    field.loc[still_missing, "skill_est"] = floor
    field.loc[still_missing, "skill_source"] = "field_floor_fallback"

    factor_columns = [
        "dg_id",
        "factor_raw",
        "factor_centered",
        "factor_delta_sg_per_round",
        "factor_available",
        "coefficient_sg_per_unit",
        "coefficient_fit_status",
    ]
    factor = event_factor[[column for column in factor_columns if column in event_factor]].copy()
    field = field.merge(factor, on="dg_id", how="left", validate="one_to_one")
    field = recenter_factor_on_archived_roster(field)
    field = field.merge(source_crosswalk, on="dg_id", how="left", validate="many_to_one")
    field["source_player_id"] = field["source_player_id"].fillna("").astype(str)
    field = field.sort_values(["player_name", "dg_id"], kind="mergesort").reset_index(drop=True)
    audit = {
        "roster_players": int(len(field)),
        "r1_prediction_players": int(field["skill_source"].eq("r1_prediction").sum()),
        "dated_model_fallback_players": int(field["skill_source"].eq("dated_model_fallback").sum()),
        "field_floor_fallback_players": int(field["skill_source"].eq("field_floor_fallback").sum()),
        "shot_mean_feature_players": int(field["factor_available"].sum()),
        "shot_source_id_players": int(field["source_player_id"].ne("").sum()),
        "skill_min": float(field["skill_est"].min()),
        "skill_max": float(field["skill_est"].max()),
        "walk_forward_delta_min": float(field["factor_delta_sg_per_round"].min()),
        "walk_forward_delta_max": float(field["factor_delta_sg_per_round"].max()),
        "walk_forward_delta_sum": float(field["factor_delta_sg_per_round"].sum()),
    }
    return field, audit


def recenter_factor_on_archived_roster(field: pd.DataFrame) -> pd.DataFrame:
    """Center the deployable factor on the exact archived pre-event roster.

    The source factor panel is centered on actual starters because that is the
    field available in the shot library.  Archived pre-event fields also carry
    WDs/alternates.  Re-centering the overlapping raw feature here prevents a
    small common mean shift while leaving missing-feature players neutral.
    """
    output = field.copy()
    raw = pd.to_numeric(output.get("factor_raw"), errors="coerce")
    available = raw.notna()
    roster_mean = float(raw.loc[available].mean()) if available.any() else np.nan
    output["factor_available"] = available
    output["factor_roster_mean"] = roster_mean
    output["factor_centered"] = (raw - roster_mean).where(available, 0.0)
    coefficients = pd.to_numeric(
        output.get("coefficient_sg_per_unit"), errors="coerce"
    ).dropna()
    coefficient = float(coefficients.iloc[0]) if len(coefficients) else np.nan
    if len(coefficients) and not np.allclose(coefficients, coefficient, atol=1e-12, rtol=0.0):
        raise ValueError("Shot coefficient is not constant within an event roster")
    output["coefficient_sg_per_unit"] = coefficient
    output["factor_delta_sg_per_round"] = np.where(
        available & np.isfinite(coefficient),
        coefficient * output["factor_centered"],
        0.0,
    )
    return output


def arm_deltas(field: pd.DataFrame, arms: Sequence[str]) -> dict[str, np.ndarray]:
    centered = field["factor_centered"].to_numpy(float)
    walk = field["factor_delta_sg_per_round"].to_numpy(float)
    output: dict[str, np.ndarray] = {"baseline": np.zeros(len(field), dtype=float)}
    for arm in arms:
        if arm == "walk_forward_uncapped":
            delta = walk.copy()
        elif arm == "walk_forward_cap015":
            max_abs = float(np.max(np.abs(walk))) if len(walk) else 0.0
            delta = walk * min(1.0, CAP_ABS_SG / max_abs) if max_abs > 0 else walk.copy()
        elif arm == "fixed_beta_sensitivity":
            delta = FIXED_BETA * centered
        else:
            raise ValueError(f"Unknown arm: {arm}")
        output[arm] = np.asarray(delta, dtype=float)
    return output


def course_shape(
    db_path: Path,
    meta: EventMeta,
    disabled: bool,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    if disabled or not meta.course_counts:
        reason = "disabled" if disabled else "missing_course_id"
        return (
            {category: 1.0 for category in CAT_ORDER},
            BASELINE_SKEW.copy(),
            {"status": f"neutral_{reason}", "courses": []},
        )
    total = float(sum(count for _, count in meta.course_counts))
    mults = {category: 0.0 for category in CAT_ORDER}
    skews = {category: 0.0 for category in CAT_ORDER}
    diagnostics: list[dict[str, Any]] = []
    for course_id, count in meta.course_counts:
        weight = count / total
        course_mults, course_skews, audit = reconstruct_course_shape(
            db_path, int(course_id), meta.start
        )
        for category in CAT_ORDER:
            mults[category] += weight * float(course_mults.get(category, 1.0))
            skews[category] += weight * float(
                course_skews.get(category, BASELINE_SKEW[category])
            )
        diagnostics.append(
            {"course_id": int(course_id), "field_count": int(count), "weight": weight, **audit}
        )
    return mults, skews, {"status": "physical_course_weighted", "courses": diagnostics}


def event_format_exclusion(meta: EventMeta) -> str:
    """Return a predeclared exclusion for formats this stroke-play kernel cannot price."""
    name = normalize_text(meta.event_name)
    if "tour championship" in name:
        return "staggered_start_unmodeled"
    if "barracuda championship" in name:
        return "modified_stableford_unmodeled"
    return ""


def cut_line_for_event(meta: EventMeta, field_size: int) -> tuple[int, str]:
    name = normalize_text(meta.event_name)
    if "masters" in name:
        return min(50, field_size), "masters_top50"
    if "pga championship" in name:
        return min(70, field_size), "pga_championship_top70"
    if "u.s. open" in name or "us open" in name:
        return min(60, field_size), "us_open_top60"
    invitational_cut = any(token in name for token in ("genesis invitational", "arnold palmer", "memorial tournament"))
    if meta.year >= 2024 and invitational_cut and field_size <= 90:
        return min(50, field_size), "signature_invitational_top50"
    known_no_cut = any(
        token in name
        for token in (
            "the sentry",
            "pebble beach pro-am",
            "rbc heritage",
            "wells fargo championship",
            "truist championship",
            "travelers championship",
            "zozo championship",
            "baycurrent classic",
            "hero world challenge",
            "olympic men's golf",
            "fedex st. jude championship",
            "bmw championship",
            "cadillac championship",
        )
    )
    if known_no_cut:
        return field_size, "curated_no_cut"
    if field_size <= 80:
        return field_size, "limited_field_no_cut"
    return min(65, field_size), "standard_top65"


def load_coefficients(path: Path = COEFF_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _r1_coefficients(coefficients: dict[str, Any], name: str) -> list[float]:
    values = coefficients[name]
    return [values["ott"], values["putt"], values["residual"], values["residual2"]]


def _r2_coefficients(coefficients: dict[str, Any], name: str) -> list[float]:
    values = coefficients[name]
    return [
        values["residual"],
        values["residual2"],
        values["residual3"],
        values["avg_ott"],
        values["avg_putt"],
        values["avg_app"],
        values["avg_arg"],
        values["delta_app"],
    ]


def _r3_coefficients(coefficients: dict[str, Any], name: str) -> list[float]:
    values = coefficients[name]
    return [
        values["sg_ott_avg"],
        values["sg_putt_avg"],
        values["sg_app_avg"],
        values["sg_arg_avg"],
        values.get("pos_6_10", 0.0),
    ]


def simulate_scores(
    params: dict[str, np.ndarray],
    field: pd.DataFrame,
    chol: np.ndarray,
    coefficients: dict[str, Any],
    delta: np.ndarray,
    simulations: int,
    seed: int,
    cut_line: int,
    week_latent_sd: float,
) -> np.ndarray:
    import sims_kernel

    n_players = len(field)
    skill = field["skill_est"].to_numpy(float)
    total_mean = np.ascontiguousarray(skill + delta)
    mu = np.asarray(params["mu"], float).copy()
    app_index = list(CAT_ORDER).index("sg_app")
    mu[:, app_index] += delta
    common = (
        total_mean,
        total_mean,
        total_mean,
        total_mean,
        np.zeros(n_players, dtype=float),
        np.zeros(n_players, dtype=float),
        _r1_coefficients(coefficients, "coefficients_r1_high"),
        _r1_coefficients(coefficients, "coefficients_r1_midh"),
        _r1_coefficients(coefficients, "coefficients_r1_midl"),
        _r1_coefficients(coefficients, "coefficients_r1_low"),
        _r2_coefficients(coefficients, "coefficients_r2"),
        _r2_coefficients(coefficients, "coefficients_r2_6_30"),
        _r2_coefficients(coefficients, "coefficients_r2_30_up"),
        _r3_coefficients(coefficients, "coefficients_r3"),
        _r3_coefficients(coefficients, "coefficients_r3_mid"),
        _r3_coefficients(coefficients, "coefficients_r3_high"),
        int(cut_line),
        False,
        int(simulations),
        int(seed),
        float(week_latent_sd),
    )
    result = sims_kernel.run_pretournament(
        np.ascontiguousarray(mu),
        np.ascontiguousarray(params["std_shot"]),
        np.ascontiguousarray(params["skew"]),
        np.ascontiguousarray(chol),
        *common,
    )
    return np.ascontiguousarray(result[0]).astype(np.int64, copy=False)


def finish_probabilities(scores: np.ndarray) -> dict[str, np.ndarray]:
    import sims_kernel

    minimum = scores.min(axis=0)
    tied = scores == minimum[None, :]
    win = (tied / tied.sum(axis=0, keepdims=True)).mean(axis=1)
    _, top_dead_heat, _ = sims_kernel.aggregate_round(
        np.ascontiguousarray(scores, dtype=np.int64)
    )
    top = np.asarray(top_dead_heat, dtype=float)
    return {"win": win, "top_5": top[:, 0], "top_10": top[:, 1], "top_20": top[:, 2]}


def probabilities_for_lines(
    scores: np.ndarray,
    field: pd.DataFrame,
    event_lines: pd.DataFrame,
    event_key: str,
    event_start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    player_ids = field["dg_id"].map(canonical_player_id).tolist()
    player_names = field["player_name"].tolist()
    index = {player_id: offset for offset, player_id in enumerate(player_ids)}
    finishes = finish_probabilities(scores)
    rows: list[dict[str, Any]] = []
    missing = 0
    opportunities = event_lines[KEY_COLS].drop_duplicates()
    for line in opportunities.to_dict(orient="records"):
        selection = canonical_player_id(line["selection_id"])
        opponent = canonical_player_id(line.get("opponent_id"))
        if selection not in index:
            missing += 1
            continue
        market = str(line["market"])
        if market == "tournament_matchup":
            if opponent not in index:
                missing += 1
                continue
            left = scores[index[selection]]
            right = scores[index[opponent]]
            probability = float(np.mean(left < right))
            push = float(np.mean(left == right))
        else:
            probability = float(finishes[market][index[selection]])
            push = 0.0
        rows.append(
            {
                "event_id": event_key,
                "event_key": event_key,
                "event_start": event_start,
                "market": market,
                "selection_id": selection,
                "selection_name": player_names[index[selection]],
                "opponent_id": opponent,
                "probability": probability,
                "push_probability": push,
            }
        )
    return pd.DataFrame(rows), {
        "probability_opportunities": int(len(opportunities)),
        "probability_rows": int(len(rows)),
        "market_players_missing_from_roster": int(missing),
    }


def _drop_settlement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "settlement_status",
        "pnl",
        "return_amount",
        "paid_fraction",
        "is_graded",
    ]
    return frame.drop(columns=[column for column in columns if column in frame], errors="ignore")


def select_family_bets(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
    arm: str,
    market_group: str,
    sizing_name: str,
) -> pd.DataFrame:
    matchup = market_group == "matchup"
    gate = GateConfig(
        min_ev=MATCHUP_GATE if matchup else OUTRIGHT_GATE,
        allowed_markets=("tournament_matchup",) if matchup else MARKETS[1:],
    )
    if sizing_name == "flat_1u":
        sizing = SizingConfig(method="flat", flat_stake=1.0)
        return select_arm_bets(lines, probabilities, arm, gate, sizing, KEY_COLS)
    if sizing_name != "production_style":
        raise ValueError(f"Unknown sizing scenario: {sizing_name}")
    if matchup:
        selected = select_arm_bets(
            lines,
            probabilities,
            arm,
            gate,
            SizingConfig(method="flat", flat_stake=1.0),
            KEY_COLS,
        )
        selected = _drop_settlement_columns(selected)
        selected["stake"] = np.where(
            selected["close_odds"] < 0,
            selected["close_odds"].abs() / 100.0,
            1.0,
        )
        return settle_bets(selected)
    # $10,000 production bankroll / $200 reporting unit = 50 units.
    return select_arm_bets(
        lines,
        probabilities,
        arm,
        gate,
        SizingConfig(
            method="quarter_kelly",
            bankroll_units=50.0,
            kelly_fraction=0.25,
            max_bankroll_fraction=1.0,
            min_stake=0.005,
        ),
        KEY_COLS,
    )


def score_comparison(
    lines: pd.DataFrame,
    baseline_probabilities: pd.DataFrame,
    challenger_probabilities: pd.DataFrame,
    challenger_name: str,
    line_policy: str,
    sizing_name: str,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, pd.DataFrame]:
    bet_pieces: list[pd.DataFrame] = []
    attribution_pieces: list[pd.DataFrame] = []
    for market_group, mask in (
        ("matchup", lines["market"].eq("tournament_matchup")),
        ("outright", ~lines["market"].eq("tournament_matchup")),
    ):
        family_lines = lines.loc[mask].copy()
        base = select_family_bets(
            family_lines, baseline_probabilities, "baseline", market_group, sizing_name
        )
        challenger = select_family_bets(
            family_lines, challenger_probabilities, "challenger", market_group, sizing_name
        )
        for frame in (base, challenger):
            frame["market_group"] = market_group
        attribution = compare_arm_bets(base, challenger, KEY_COLS)
        attribution["market_group"] = market_group
        bet_pieces.extend([base, challenger])
        attribution_pieces.append(attribution)

    bets = pd.concat(bet_pieces, ignore_index=True, sort=False)
    attribution = pd.concat(attribution_pieces, ignore_index=True, sort=False)
    bets["challenger_name"] = challenger_name
    bets["line_policy_name"] = line_policy
    bets["sizing_name"] = sizing_name
    attribution["challenger_name"] = challenger_name
    attribution["line_policy_name"] = line_policy
    attribution["sizing_name"] = sizing_name

    performance_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[pd.DataFrame] = []
    group_inputs = [
        ("all", bets, lines),
        (
            "matchup",
            bets[bets["market_group"].eq("matchup")],
            lines[lines["market"].eq("tournament_matchup")],
        ),
        (
            "outright",
            bets[bets["market_group"].eq("outright")],
            lines[~lines["market"].eq("tournament_matchup")],
        ),
    ]
    for group_name, group, group_lines in group_inputs:
        if group.empty:
            continue
        perf = performance_summary(
            group,
            event_col="event_id",
            event_order_col="event_start",
            starting_bankroll=100.0,
        )
        perf["market_group"] = group_name
        performance_rows.append(perf)
        bootstrap_input = add_zero_bet_event_clusters(group, group_lines)
        boot = event_cluster_bootstrap(
            bootstrap_input,
            event_col="event_id",
            event_order_col="event_start",
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        boot["market_group"] = group_name
        bootstrap_rows.append(boot)

    performance = pd.concat(performance_rows, ignore_index=True)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)
    for frame in (performance, bootstrap):
        frame["challenger_name"] = challenger_name
        frame["line_policy_name"] = line_policy
        frame["sizing_name"] = sizing_name
    attribution_summary = summarize_attribution(attribution)
    attribution_summary["challenger_name"] = challenger_name
    attribution_summary["line_policy_name"] = line_policy
    attribution_summary["sizing_name"] = sizing_name
    return {
        "bets": bets,
        "attribution": attribution,
        "attribution_summary": attribution_summary,
        "performance": performance,
        "bootstrap": bootstrap,
    }


def add_zero_bet_event_clusters(bets: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Include eligible zero-bet tournaments in event-cluster inference only."""
    if lines.empty:
        return bets
    universe = lines[["event_id", "event_start"]].drop_duplicates("event_id")
    arms = bets["arm"].drop_duplicates().astype(str).tolist()
    present = set(zip(bets["event_id"].astype(str), bets["arm"].astype(str)))
    rows = []
    for event_id, event_start in universe.itertuples(index=False, name=None):
        for arm in arms:
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
    return pd.concat([bets, pd.DataFrame(rows)], ignore_index=True, sort=False) if rows else bets


def build_year_market_summary(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    data = bets[bets["pnl"].notna()].copy()
    data["year"] = pd.to_datetime(data["event_start"], errors="coerce").dt.year
    output = (
        data.groupby(
            [
                "challenger_name",
                "line_policy_name",
                "sizing_name",
                "arm",
                "year",
                "market",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            bet_count=("pnl", "size"),
            event_count=("event_id", "nunique"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
        )
        .reset_index()
    )
    output["roi"] = np.divide(
        output["pnl"],
        output["stake"],
        out=np.full(len(output), np.nan),
        where=output["stake"].to_numpy(float) > 0,
    )
    return output


def build_report(
    output_dir: Path,
    config: dict[str, Any],
    coverage: pd.DataFrame,
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    attribution: pd.DataFrame,
) -> str:
    primary = performance[
        performance["challenger_name"].eq(PRIMARY_ARM)
        & performance["line_policy_name"].eq("fixed_book")
        & performance["sizing_name"].eq("flat_1u")
        & performance["market_group"].eq("all")
    ]
    lines = [
        "# Shot-Mean Closing-Market Backtest",
        "",
        "## Contract",
        "",
        "This is a walk-forward reconstruction of today's own-model production core, not a bit-exact replay of historical weekly production. Closing odds are used only for bet selection and settlement. Both arms use the same pre-event roster, current category-first simulation, current shot-variance overlay, neutral weather, course reconstruction, market menu, and Monte Carlo seed.",
        "",
        f"- Window: {config['start']} through {config['end']}",
        f"- Simulations per arm/event: {config['simulations']:,}",
        f"- Eligible simulated events: {int(coverage['simulated'].sum()) if 'simulated' in coverage else 0}",
        f"- Primary execution: fixed-book close, flat 1-unit risk, {MATCHUP_GATE:.0%} H2H EV gate, {OUTRIGHT_GATE:.0%} finish EV gate",
        "- Primary challenger: cross-track APP coefficient refit before every event on strictly earlier events; uncapped",
        f"- Sensitivity: event-wise proportional shrink so max absolute APP delta is {CAP_ABS_SG:.2f} SG/round",
        "- Excluded: R1 and live R2-R4 boards, closing-price market regression, historical weather hindsight, and unarchived manual boosts",
        "",
        "## Primary result",
        "",
    ]
    if len(primary) == 2:
        lookup = primary.set_index("arm")
        for arm in ("baseline", "challenger"):
            if arm in lookup.index:
                row = lookup.loc[arm]
                lines.append(
                    f"- {arm.title()}: {int(row['bet_count'])} bets, {row['pnl']:+.2f}u P&L, {row['roi']:.2%} ROI, {row['max_drawdown']:.2f}u max drawdown"
                )
        delta = bootstrap[
            bootstrap["challenger_name"].eq(PRIMARY_ARM)
            & bootstrap["line_policy_name"].eq("fixed_book")
            & bootstrap["sizing_name"].eq("flat_1u")
            & bootstrap["market_group"].eq("all")
            & bootstrap["arm"].eq("challenger_minus_baseline")
            & bootstrap["metric"].eq("pnl")
        ]
        if len(delta):
            row = delta.iloc[0]
            lines.append(
                f"- Challenger minus baseline: {row['estimate']:+.2f}u P&L; event-bootstrap 95% CI [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]u; P(delta > 0) {row['probability_above_zero']:.1%}"
            )
    else:
        lines.append("Primary result rows were not available.")
    lines.extend(
        [
            "",
            "## Wager attribution",
            "",
            "The row-level exports distinguish wagers retained by both arms from challenger-only new wagers and baseline-only dropped wagers.",
            "",
            "## Interpretation limits",
            "",
            "The archived roster and close are point-in-time, and the shot coefficient is walk-forward. However, the exact 2023-2025 production code/config files, historical weather forecasts, DataGolf SG decomposition blend, and manual overrides were not retained. Current model coefficients are therefore a retrospective reconstruction. DataGolf closing snapshots also do not prove fillability, limits, or liquidity.",
            "",
            "## Files",
            "",
            "- `event_coverage.csv`: every included/excluded event and reason",
            "- `event_audit.parquet`: field, feature, course, cut, and runtime diagnostics",
            "- `probabilities.parquet`: frozen arm probabilities before looking at P&L",
            "- `bets.parquet`: selected and settled wagers",
            "- `performance.csv`: ROI/P&L/drawdown summaries",
            "- `bootstrap.csv`: event-cluster confidence intervals",
            "- `attribution.csv`: overlap/new/dropped wager decomposition",
            "- `year_market_summary.csv`: year and market breakdown",
            "- `manifest.json`: immutable inputs, configuration, and checksums",
            "",
        ]
    )
    report = "\n".join(lines)
    (output_dir / "SHOT_MEAN_CLOSING_BACKTEST.md").write_text(report, encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.simulations <= 0 or args.bootstrap <= 0:
        raise ValueError("simulations and bootstrap must be positive")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    requested_arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    invalid_arms = sorted(set(requested_arms) - set(ALL_ARMS))
    if invalid_arms:
        raise ValueError(f"Unknown arms: {invalid_arms}")
    if not requested_arms:
        raise ValueError("At least one challenger arm is required")
    for path in (
        args.db,
        args.predictions,
        args.crosswalk,
        args.factor_panel,
        args.odds_cache / "manifest.json",
        CORR_PATH,
        COEFF_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    replacements = load_name_replacements()
    week_latent_sd = float(load_sim_input_literal("WEEK_LATENT_SD"))
    factor_panel = pd.read_parquet(args.factor_panel)
    factor_panel["start"] = pd.to_datetime(factor_panel["start"], errors="coerce").dt.normalize()
    factor_panel = factor_panel[factor_panel["start"].between(start, end, inclusive="both")]
    factor_events = set(factor_panel["event_key"])

    print("Reading verified historical odds cache ...", flush=True)
    reader = HistoricalOddsCacheReader(args.odds_cache)
    matchup_lines = normalize_matchup_lines(reader.matchups())
    outright_lines = normalize_outright_lines(reader.outrights())
    raw_lines = pd.concat([matchup_lines, outright_lines], ignore_index=True, sort=False)
    rosters = normalize_prediction_rosters(reader.predictions(), replacements)
    line_sets = select_line_sets(raw_lines)
    fixed_events = set(line_sets["fixed_book"]["event_key"])
    roster_events = set(rosters["event_key"])
    candidates = sorted(factor_events & fixed_events & roster_events)
    if args.event_key:
        candidates = [event for event in candidates if event in set(args.event_key)]
    metadata = read_event_metadata(args.db, candidates)
    candidates = [event for event in candidates if event in metadata]
    format_exclusions = {
        event: event_format_exclusion(metadata[event])
        for event in candidates
        if event_format_exclusion(metadata[event])
    }
    candidates = [event for event in candidates if event not in format_exclusions]
    candidates.sort(key=lambda event: (metadata[event].start, event))
    if args.event_limit:
        candidates = candidates[: args.event_limit]
    selected = set(candidates)

    all_events = sorted(factor_events | fixed_events | roster_events)
    coverage_rows: list[dict[str, Any]] = []
    for event in all_events:
        reasons = []
        if event not in factor_events:
            reasons.append("no_shot_mean_factor")
        if event not in fixed_events:
            reasons.append("no_primary_closing_lines")
        if event not in roster_events:
            reasons.append("no_archived_roster")
        if event in factor_events & fixed_events & roster_events and event not in metadata:
            reasons.append("no_local_event_metadata")
        if event in format_exclusions:
            reasons.append(format_exclusions[event])
        if args.event_key and event not in set(args.event_key):
            reasons.append("event_filter")
        if event in selected:
            reasons = []
        coverage_rows.append(
            {
                "event_key": event,
                "start": metadata[event].start if event in metadata else pd.NaT,
                "has_factor": event in factor_events,
                "has_fixed_lines": event in fixed_events,
                "has_roster": event in roster_events,
                "has_metadata": event in metadata,
                "eligible": not reasons,
                "simulated": False,
                "exclusion_reason": ";".join(reasons),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output_dir / "event_coverage_preflight.csv", index=False)
    print(
        f"Eligible intersection: {len(candidates)} events "
        f"(factor={len(factor_events)}, close={len(fixed_events)}, roster={len(roster_events)})",
        flush=True,
    )
    if not candidates:
        raise ValueError("No eligible events after intersecting factor, odds, roster, and metadata")
    if args.preflight_only:
        print(f"Preflight written to {args.output_dir}", flush=True)
        return 0

    import sims_kernel

    source_files = {
        "backtest_engine": Path(__file__),
        "odds_cache_reader": ARCHIVE / "historical_odds_cache.py",
        "market_scoring": ARCHIVE / "market_backtest_scoring.py",
        "category_parameter_builder": ARCHIVE / "backtest_distribution_engines_2026.py",
        "course_shape_builder": ARCHIVE / "backtest_shot_dispersion_recent.py",
        "shot_feature_builder": ARCHIVE / "build_bmw_shot_features_shadow.py",
        "sim_inputs": REPO / "sim_inputs.py",
    }
    shot_export_files = {
        f"shot_sg_{year}": args.export / f"pga_{year}_shot_sg.parquet"
        for year in range(2020, end.year + 1)
    }
    config = {
        "contract": "shot_mean_closing_market_backtest_v2",
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "simulations": int(args.simulations),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "arms": list(requested_arms),
        "primary_arm": PRIMARY_ARM,
        "fixed_beta_sensitivity": FIXED_BETA,
        "cap_sensitivity_abs_sg_per_round": CAP_ABS_SG,
        "matchup_ev_gate": MATCHUP_GATE,
        "outright_ev_gate": OUTRIGHT_GATE,
        "matchup_books_fixed_priority": MATCHUP_BOOKS,
        "outright_books_fixed_priority": OUTRIGHT_BOOKS,
        "weather": "neutral_common_both_arms",
        "week_latent_sd": week_latent_sd,
        "market_regression": "disabled_to_avoid_closing_price_circularity",
        "shot_variance": "disabled_diagnostic" if args.no_shot_variance else "current_overlay_common_both_arms",
        "course_shape": "disabled_diagnostic" if args.no_course_shape else "strict_prior_physical_course",
        "baseline": "current_own_model_core_retrospective_coefficient_reconstruction",
        "not_exact_historical_production": True,
        "factor_deployment_centering": "available_players_on_archived_pre_event_roster",
        "bootstrap_event_universe": "all_eligible_events_in_market_family_including_zero_bet_events",
        "event_count": len(candidates),
        "event_keys": candidates,
        "format_exclusions": format_exclusions,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sims_kernel": sims_kernel.version(),
        },
        "input_hashes": {
            "factor_panel": sha256_file(args.factor_panel),
            "odds_cache_manifest": sha256_file(args.odds_cache / "manifest.json"),
            "predictions": sha256_file(args.predictions),
            "crosswalk": sha256_file(args.crosswalk),
            "correlation": sha256_file(CORR_PATH),
            "coefficients": sha256_file(COEFF_PATH),
            "historical_database": sha256_file(args.db),
            **{
                f"source::{name}": sha256_file(path)
                for name, path in source_files.items()
            },
            **{
                f"shot_export::{name}": sha256_file(path)
                for name, path in shot_export_files.items()
            },
            "shot_export::manifest": sha256_file(args.export / "manifest.json"),
        },
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()
    config["config_hash"] = config_hash
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash:
            raise ValueError(
                f"Output directory contains a different run_config: {config_path}. "
                "Choose a new output directory; existing research artifacts were preserved."
            )
    else:
        atomic_json(config_path, config)

    union_ids = sorted(
        rosters[rosters["event_key"].isin(candidates)]["dg_id"].dropna().astype(int).unique()
    )
    print(f"Loading strict-prior histories for {len(union_ids)} roster players ...", flush=True)
    histories, model_histories = load_player_histories(
        args.db, union_ids, max(metadata[event].start for event in candidates) + pd.Timedelta(days=1)
    )
    r1_predictions = load_r1_predictions(args.predictions, replacements)
    source_crosswalk = load_source_crosswalk(args.crosswalk)
    corr, chol = load_correlation(CORR_PATH)
    coefficients = load_coefficients()

    prepared_fields: dict[str, pd.DataFrame] = {}
    field_audits: dict[str, dict[str, Any]] = {}
    union_source_ids: set[str] = set()
    for event_key in candidates:
        event_roster = rosters[rosters["event_key"].eq(event_key)]
        event_factor = factor_panel[factor_panel["event_key"].eq(event_key)]
        field, audit = build_event_field(
            event_roster,
            metadata[event_key],
            r1_predictions,
            model_histories,
            source_crosswalk,
            event_factor,
        )
        prepared_fields[event_key] = field
        field_audits[event_key] = audit
        union_source_ids.update(field.loc[field["source_player_id"].ne(""), "source_player_id"])

    if args.no_shot_variance:
        shot_stats = pd.DataFrame()
        valid_shots = 0
    else:
        print(
            f"Loading shot-variance histories for {len(union_source_ids)} source players ...",
            flush=True,
        )
        event_dates = load_event_dates(args.db)
        shot_stats, valid_shots = load_shot_round_stats(
            args.export,
            sorted(union_source_ids),
            event_dates,
            max(metadata[event].start for event in candidates) + pd.Timedelta(days=1),
        )
        print(f"Loaded {len(shot_stats):,} player/category/round rows from {valid_shots:,} shots.", flush=True)

    checkpoint_dir = args.output_dir / "event_probabilities"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    probability_pieces: list[pd.DataFrame] = []
    for event_index, event_key in enumerate(candidates, start=1):
        meta = metadata[event_key]
        checkpoint = checkpoint_dir / f"{event_key.replace(':', '_')}.parquet"
        checkpoint_meta = checkpoint.with_suffix(".meta.json")
        if checkpoint.exists() and checkpoint_meta.exists():
            saved = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
            if saved.get("config_hash") == config_hash and saved.get("complete") is True:
                probability_pieces.append(pd.read_parquet(checkpoint))
                audit_rows.append(saved["event_audit"])
                print(f"[{event_index}/{len(candidates)}] resume {event_key}", flush=True)
                continue

        started = time.perf_counter()
        field = prepared_fields[event_key]
        print(
            f"[{event_index}/{len(candidates)}] {meta.event_name} {meta.start.date()} "
            f"({len(field)} players)",
            flush=True,
        )
        if args.no_shot_variance:
            features = pd.DataFrame({"source_player_id": field["source_player_id"].unique()})
            feature_prior: dict[str, float] = {}
        else:
            shot_input = shot_stats[shot_stats["event_date"].lt(meta.start)]
            feature_ids = (
                field.loc[field["source_player_id"].ne(""), ["source_player_id"]]
                .drop_duplicates("source_player_id")
            )
            features, feature_prior = build_dispersion_features(shot_input, feature_ids)
        mults, skews, course_audit = course_shape(args.db, meta, args.no_course_shape)
        params, distribution_audit = build_category_params(
            histories,
            field,
            meta.start,
            features,
            mults,
            skews,
        )
        cut_line, cut_rule = cut_line_for_event(meta, len(field))
        event_seed = stable_seed(event_key, args.seed)
        event_lines = raw_lines[raw_lines["event_key"].eq(event_key)]
        deltas = arm_deltas(field, requested_arms)
        arm_probabilities: list[pd.DataFrame] = []
        probability_audit: dict[str, int] = {}
        for arm in ("baseline", *requested_arms):
            scores = simulate_scores(
                params,
                field,
                chol,
                coefficients,
                deltas[arm],
                args.simulations,
                event_seed,
                cut_line,
                week_latent_sd,
            )
            probabilities, probability_audit = probabilities_for_lines(
                scores, field, event_lines, event_key, meta.start
            )
            probabilities["arm"] = arm
            arm_probabilities.append(probabilities)
            del scores
            gc.collect()
        event_probabilities = pd.concat(arm_probabilities, ignore_index=True, sort=False)
        atomic_parquet(event_probabilities, checkpoint)
        event_audit = {
            "event_key": event_key,
            "year": meta.year,
            "dg_event_id": meta.event_id,
            "event_name": meta.event_name,
            "start": meta.start,
            "seed": event_seed,
            "simulations": args.simulations,
            "cut_line": cut_line,
            "cut_rule": cut_rule,
            "course_count": len(meta.course_counts),
            "course_shape": course_audit,
            "field": field_audits[event_key],
            "distribution": distribution_audit,
            "shot_variance_category_priors": feature_prior,
            **probability_audit,
            "runtime_seconds": time.perf_counter() - started,
        }
        atomic_json(
            checkpoint_meta,
            {
                "complete": True,
                "config_hash": config_hash,
                "checkpoint_sha256": sha256_file(checkpoint),
                "event_audit": event_audit,
            },
        )
        probability_pieces.append(event_probabilities)
        audit_rows.append(event_audit)
        print(f"  completed in {event_audit['runtime_seconds']:.1f}s", flush=True)

    probabilities = pd.concat(probability_pieces, ignore_index=True, sort=False)
    event_audit = normalize_event_audit_rows(audit_rows)
    simulated_events = set(probabilities["event_key"])
    coverage["simulated"] = coverage["event_key"].isin(simulated_events)
    coverage["eligible"] = coverage["event_key"].isin(selected)
    coverage.to_csv(args.output_dir / "event_coverage.csv", index=False)
    atomic_parquet(probabilities, args.output_dir / "probabilities.parquet")
    atomic_parquet(event_audit, args.output_dir / "event_audit.parquet")

    scoring_outputs: dict[str, list[pd.DataFrame]] = {
        "bets": [],
        "attribution": [],
        "attribution_summary": [],
        "performance": [],
        "bootstrap": [],
    }
    baseline_probabilities = probabilities[probabilities["arm"].eq("baseline")]
    for challenger_index, challenger_name in enumerate(requested_arms):
        challenger_probabilities = probabilities[probabilities["arm"].eq(challenger_name)]
        for line_policy, selected_lines in line_sets.items():
            selected_lines = selected_lines[selected_lines["event_key"].isin(simulated_events)].copy()
            selected_lines = selected_lines.merge(
                coverage[["event_key", "start"]].rename(columns={"start": "event_start"}),
                on="event_key",
                how="left",
                validate="many_to_one",
            )
            for sizing_name in ("flat_1u", "production_style"):
                scored = score_comparison(
                    selected_lines,
                    baseline_probabilities,
                    challenger_probabilities,
                    challenger_name,
                    line_policy,
                    sizing_name,
                    args.bootstrap,
                    args.seed + 1000 * challenger_index,
                )
                for key, frame in scored.items():
                    scoring_outputs[key].append(frame)

    combined = {
        key: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for key, frames in scoring_outputs.items()
    }
    year_market = build_year_market_summary(combined["bets"])
    atomic_parquet(combined["bets"], args.output_dir / "bets.parquet")
    atomic_parquet(combined["attribution"], args.output_dir / "attribution.parquet")
    combined["attribution_summary"].to_csv(
        args.output_dir / "attribution_summary.csv", index=False
    )
    combined["performance"].to_csv(args.output_dir / "performance.csv", index=False)
    combined["bootstrap"].to_csv(args.output_dir / "bootstrap.csv", index=False)
    year_market.to_csv(args.output_dir / "year_market_summary.csv", index=False)

    build_report(
        args.output_dir,
        config,
        coverage,
        combined["performance"],
        combined["bootstrap"],
        combined["attribution"],
    )
    manifest = {
        **config,
        "completed": True,
        "simulated_events": len(simulated_events),
        "valid_shots_loaded": int(valid_shots),
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(f"Backtest complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

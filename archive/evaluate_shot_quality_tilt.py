"""Chronological proper-score evaluation of shot-reward quality tilting.

Research only.  This file does not import or modify the production simulator.
It asks a narrow question: after conditioning on a pre-shot category/lie/
distance state, does reweighting complete reward rows by strictly-prior player
category ability improve the future distribution of per-transition SG?

For donor row ``j`` and target player ``t`` the evaluated whole-row weight is

    exp(-beta * donor_ability_j * donor_reward_j)
    * exp(+beta * target_ability_t * donor_reward_j).

The donor term is normalized within historical event/state before events are
pooled.  A global-normalization sensitivity is emitted as well.  Beta zero is
always present and every arm scores identical held-out rows.

No held-out event/state mean is used.  Events sharing a start date are scored
before any event on that date updates the donor support.  Player abilities are
the latest model_results2 value strictly before event start; missing abilities
receive the explicit neutral prior zero.

The archive's non-green expected-strokes curves are fixed Broadie curves, but
its putting curve was fitted within event.  To keep APP/ARG shots ending on the
green on one prior-only reward scale, this evaluator reconstructs a monotone
putting value curve from the earliest completed event date(s), freezes it once
the declared sample gate is met, backfills those warmup donors, and recomputes
every later green-start/green-end reward with that one frozen curve.  Warmup
dates are not scored.  PUTT remains a labeled secondary sensitivity.  The
evaluation is reward-atom only: it does not validate endpoint, lie, hazard,
drop, penalty-resolution, or recursion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive import audit_shot_field_normalization_inputs as identity_audit  # noqa: E402


DEFAULT_EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")
DEFAULT_DB = Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db")
DEFAULT_OUTPUT = Path("permanent_data/shot_quality_tilt_eval")
PRIMARY_CATEGORIES = ("ott", "app", "arg")
ALL_CATEGORIES = (*PRIMARY_CATEGORIES, "putt")
CATEGORY_ABILITY = {
    "ott": "sg_ott",
    "app": "sg_app",
    "arg": "ag_arg",
    "putt": "ag_putt",
}
PUTTING_BIN_EDGES_FT = np.array(
    [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40,
     50, 60, 75, 100, 125, 150, 200, 300, np.inf],
    dtype=float,
)


class PeakRSSMonitor:
    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = 0.0
        self.peak_rss_bytes = 0

    def start(self) -> None:
        self.started = time.perf_counter()
        self.peak_rss_bytes = int(self._process.memory_info().rss)

        def sample() -> None:
            while not self._stop.wait(0.05):
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes,
                    int(self._process.memory_info().rss),
                )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, float | int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_bytes = max(
            self.peak_rss_bytes,
            int(self._process.memory_info().rss),
        )
        return {
            "elapsed_seconds": float(time.perf_counter() - self.started),
            "peak_rss_bytes": int(self.peak_rss_bytes),
            "peak_rss_mb": float(self.peak_rss_bytes / 1024 ** 2),
            "sampling_interval_seconds": 0.05,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    work = frame.reindex(sorted(frame.columns), axis=1).copy()
    if len(work):
        work = work.sort_values(list(work.columns), kind="mergesort")
    return hashlib.sha256(
        work.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_beta_grid(spec: str) -> np.ndarray:
    pieces = [float(value) for value in spec.split(":")]
    if len(pieces) != 3 or pieces[2] <= 0 or pieces[1] < pieces[0]:
        raise ValueError("--beta-grid must be start:stop:step")
    start, stop, step = pieces
    values = np.arange(start, stop + step / 2.0, step)
    values = np.unique(np.round(np.append(values, 0.0), 10))
    if len(values) < 2:
        raise ValueError("beta grid must contain at least two values")
    return values


def _lie(value: object) -> str:
    text = str(value).casefold().strip()
    if text in {"tee", "fairway", "rough", "sand", "recovery", "green"}:
        return text
    if "bunker" in text:
        return "sand"
    return "other"


def _distance_state(category: str, distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    if category == "ott":
        values = np.rint(distance / 25.0).astype(int) * 25
        return np.array([f"{value:d}yd" for value in values], dtype=object)
    if category == "app":
        return np.select(
            [distance <= 100, distance <= 150, distance <= 215],
            ["<=100", "100-150", "150-215"],
            default=">215",
        ).astype(object)
    if category == "arg":
        values = np.rint(distance / 5.0).astype(int) * 5
        return np.array([f"{value:d}yd" for value in values], dtype=object)
    feet = np.maximum(distance * 3.0, 0.0)
    values = np.floor(np.log1p(feet) / 0.22).astype(int)
    return np.array([f"logft_{value:d}" for value in values], dtype=object)


def _weighted_isotonic(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(index), float(index), float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / total
            blocks.append([left[0], right[1], mean, total])
    result = [0.0] * len(values)
    for start, end, value, _ in blocks:
        for index in range(int(start), int(end) + 1):
            result[index] = value
    return result


@dataclass(frozen=True)
class CurveSnapshot:
    distances_ft: np.ndarray
    expected: np.ndarray
    samples: int
    sha256: str
    latest_source_event_start: pd.Timestamp | None = None

    def value(self, distance_yd: np.ndarray) -> np.ndarray:
        feet = np.maximum(np.asarray(distance_yd, dtype=float) * 3.0, 0.0)
        return np.interp(
            feet,
            self.distances_ft,
            self.expected,
            left=float(self.expected[0]),
            right=float(self.expected[-1]),
        )


@dataclass
class PriorPuttingCurve:
    distance_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(len(PUTTING_BIN_EDGES_FT) - 1)
    )
    remaining_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(len(PUTTING_BIN_EDGES_FT) - 1)
    )
    count: np.ndarray = field(
        default_factory=lambda: np.zeros(len(PUTTING_BIN_EDGES_FT) - 1, dtype=np.int64)
    )
    latest_event_start: pd.Timestamp | None = None

    def update(self, observations: pd.DataFrame, event_start: object | None = None) -> None:
        if event_start is not None:
            event_date = pd.Timestamp(event_start)
            if self.latest_event_start is not None and event_date < self.latest_event_start:
                raise AssertionError("putting observations must be updated chronologically")
            self.latest_event_start = event_date
        if observations.empty:
            return
        distance = pd.to_numeric(observations["green_start_distance_yd"], errors="coerce").to_numpy(float) * 3.0
        remaining = pd.to_numeric(observations["green_remaining_strokes"], errors="coerce").to_numpy(float)
        valid = np.isfinite(distance) & np.isfinite(remaining) & (remaining >= 1)
        distance, remaining = distance[valid], remaining[valid]
        indices = np.searchsorted(PUTTING_BIN_EDGES_FT, distance, side="right") - 1
        indices = np.clip(indices, 0, len(self.count) - 1)
        self.count += np.bincount(indices, minlength=len(self.count)).astype(np.int64)
        self.distance_sum += np.bincount(indices, weights=distance, minlength=len(self.count))
        self.remaining_sum += np.bincount(indices, weights=remaining, minlength=len(self.count))

    def snapshot(
        self,
        minimum_samples: int,
        as_of: object | None = None,
    ) -> CurveSnapshot | None:
        if as_of is not None and self.latest_event_start is not None:
            if self.latest_event_start >= pd.Timestamp(as_of):
                raise AssertionError("putting curve includes a same-date or future event")
        if int(self.count.sum()) < minimum_samples:
            return None
        distances = [0.0]
        expected = [1.0]
        weights = [50.0]
        for index, count in enumerate(self.count):
            if count <= 0:
                continue
            distances.append(float(self.distance_sum[index] / count))
            expected.append(float(self.remaining_sum[index] / count))
            weights.append(float(count))
        order = np.argsort(distances)
        x = np.asarray(distances, dtype=float)[order]
        y = np.asarray(expected, dtype=float)[order]
        w = np.asarray(weights, dtype=float)[order]
        y = np.asarray(_weighted_isotonic(y.tolist(), w.tolist()), dtype=float)
        payload = {
            "distances_ft": np.round(x, 8).tolist(),
            "expected": np.round(y, 8).tolist(),
            "samples": int(self.count.sum()),
            "contract": "completed_prior_holes_monotone_v1",
            "latest_source_event_start": self.latest_event_start,
        }
        return CurveSnapshot(
            x,
            y,
            int(self.count.sum()),
            _json_hash(payload),
            self.latest_event_start,
        )


def load_calendar(db_path: Path, start_season: int, end_season: int) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as db:
        frame = pd.read_sql_query(
            """
            SELECT year AS season, event_id, event_name,
                   MIN(round_date) AS event_start
            FROM player_rounds
            WHERE tour='pga' AND year BETWEEN ? AND ?
            GROUP BY year, event_id, event_name
            """,
            db,
            params=(start_season, end_season),
        )
    frame["event_start"] = pd.to_datetime(frame["event_start"], errors="coerce")
    frame["event_key"] = [
        f"pga:R{int(year)}{int(event_id):03d}"
        for year, event_id in zip(frame["season"], frame["event_id"])
    ]
    return frame.dropna(subset=["event_start"]).drop_duplicates("event_key")


def available_event_keys(export_root: Path, seasons: Iterable[int]) -> set[str]:
    keys: set[str] = set()
    for season in seasons:
        path = export_root / f"pga_{season}_shot_sg.parquet"
        if not path.exists():
            continue
        keys.update(pd.read_parquet(path, columns=["event_key"])["event_key"].dropna().astype(str).unique())
    return keys


def select_calendar(
    calendar: pd.DataFrame,
    available: set[str],
    max_events_per_season: int | None,
) -> pd.DataFrame:
    selected = calendar[calendar["event_key"].isin(available)].copy()
    selected = selected.sort_values(["event_start", "event_key"])
    if max_events_per_season is not None:
        selected = selected.groupby("season", group_keys=False).head(max_events_per_season)
    return selected.reset_index(drop=True)


def iter_event_date_batches(calendar: pd.DataFrame):
    ordered = calendar.sort_values(["event_start", "event_key"], kind="mergesort")
    yield from ordered.groupby("event_start", sort=True)


def load_rosters(export_root: Path, calendar: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    wanted = set(calendar["event_key"].astype(str))
    for season, events in calendar.groupby("season", observed=True):
        path = export_root / f"pga_{int(season)}_player_rounds.parquet"
        frame = pd.read_parquet(
            path,
            columns=["event_key", "round_no", "source_player_id", "player_name"],
            filters=[("event_key", "in", events["event_key"].astype(str).tolist())],
        )
        parts.append(frame[frame["event_key"].isin(wanted)])
    roster = pd.concat(parts, ignore_index=True)
    roster["source_player_id"] = roster["source_player_id"].astype(str)
    roster = roster.sort_values("round_no").drop_duplicates(["event_key", "source_player_id"])
    roster["name_key"] = roster["player_name"].map(identity_audit._name_key)
    return roster


def load_skill_history(db_path: Path, end_season: int) -> pd.DataFrame:
    columns = ", ".join(CATEGORY_ABILITY.values())
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as db:
        frame = pd.read_sql_query(
            f"""
            SELECT Date AS skill_date, dg_id, player_name, {columns}
            FROM model_results2
            WHERE Date >= '2018-01-01' AND Date < ?
            """,
            db,
            params=(f"{end_season + 1}-01-01",),
        )
    frame["skill_date"] = pd.to_datetime(frame["skill_date"], errors="coerce")
    frame["name_key"] = frame["player_name"].map(identity_audit._name_key)
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="coerce").astype("Int64").astype("string")
    for column in CATEGORY_ABILITY.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["skill_date", "dg_id", "name_key"])


def build_strict_snapshots(
    roster: pd.DataFrame,
    calendar: pd.DataFrame,
    skills: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = calendar[["event_key", "event_start", "event_name", "season"]]
    query = roster.merge(events, on="event_key", how="inner", validate="many_to_one")
    identities = skills[["name_key", "dg_id"]].drop_duplicates()
    counts = identities.groupby("name_key", observed=True)["dg_id"].nunique()
    valid_keys = set(counts[counts.eq(1)].index)
    mapping = identities[identities["name_key"].isin(valid_keys)].drop_duplicates("name_key")
    query = query.merge(mapping, on="name_key", how="left", validate="many_to_one")
    query["identity_status"] = np.where(
        query["dg_id"].notna(), "unique_name_to_dg", "missing_or_ambiguous"
    )

    mapped = query[query["dg_id"].notna()].copy()
    history = (
        skills[["dg_id", "skill_date", *CATEGORY_ABILITY.values()]]
        .sort_values("skill_date")
        .drop_duplicates(["dg_id", "skill_date"], keep="last")
    )
    prior = pd.merge_asof(
        mapped.sort_values(["event_start", "dg_id"]),
        history.sort_values(["skill_date", "dg_id"]),
        by="dg_id",
        left_on="event_start",
        right_on="skill_date",
        direction="backward",
        allow_exact_matches=False,
    )
    if prior["skill_date"].ge(prior["event_start"]).fillna(False).any():
        raise AssertionError("strict-prior skill invariant failed")
    payload = prior[
        ["event_key", "source_player_id", "dg_id", "skill_date", *CATEGORY_ABILITY.values()]
    ]
    snapshot = query.drop(columns=["dg_id"], errors="ignore").merge(
        payload,
        on=["event_key", "source_player_id"],
        how="left",
        validate="one_to_one",
    )
    snapshot["snapshot_kind"] = np.where(
        snapshot["skill_date"].notna(), "strict_prior_asof", "neutral_missing_prior"
    )
    coverage_rows: list[dict[str, object]] = []
    for event, group in snapshot.groupby("event_key", observed=True):
        row: dict[str, object] = {
            "event_key": event,
            "event_name": group["event_name"].iloc[0],
            "season": int(group["season"].iloc[0]),
            "event_start": group["event_start"].iloc[0],
            "players": int(len(group)),
            "identity_coverage": float(group["dg_id"].notna().mean()),
        }
        for category, column in CATEGORY_ABILITY.items():
            row[f"{category}_ability_coverage"] = float(group[column].notna().mean())
        coverage_rows.append(row)
    return snapshot, pd.DataFrame(coverage_rows)


def load_year_panel(
    export_root: Path,
    season: int,
    event_keys: list[str],
    snapshot: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sg_columns = [
        "event_key", "round_no", "hole_no", "source_player_id", "stroke_no",
        "sg_category", "start_lie", "start_distance_yd", "end_lie",
        "end_distance_yd", "expected_strokes_start", "expected_strokes_end",
        "sg_raw", "sg_valid", "complete_hole", "state_inferred", "shot_key",
        "model_id",
    ]
    raw_columns = [
        "event_key", "source_player_id", "round_no", "hole_no", "stroke_no",
        "event_type", "ball_holed", "shot_key",
    ]
    sg = pd.read_parquet(
        export_root / f"pga_{season}_shot_sg.parquet",
        columns=sg_columns,
        filters=[("event_key", "in", event_keys)],
    )
    raw = pd.read_parquet(
        export_root / f"pga_{season}_shots.parquet",
        columns=raw_columns,
        filters=[("event_key", "in", event_keys)],
    )
    for frame in (sg, raw):
        frame["source_player_id"] = frame["source_player_id"].astype(str)
    raw_context = raw.drop_duplicates("shot_key")[["shot_key", "event_type", "ball_holed"]]
    sg = sg.merge(raw_context, on="shot_key", how="left", validate="many_to_one")
    hole_keys = ["event_key", "source_player_id", "round_no", "hole_no"]
    completed = raw.groupby(hole_keys, observed=True)["ball_holed"].max().eq(1)
    final_stroke = raw.groupby(hole_keys, observed=True)["stroke_no"].max()
    terminal = pd.concat(
        [completed.rename("raw_hole_complete"), final_stroke.rename("final_stroke_no")],
        axis=1,
    ).reset_index()
    sg = sg.merge(terminal, on=hole_keys, how="left", validate="many_to_one")

    event_meta = calendar[["event_key", "event_start", "event_name", "season"]]
    sg = sg.merge(event_meta, on="event_key", how="inner", validate="many_to_one")
    skills = snapshot[
        ["event_key", "source_player_id", "skill_date", *CATEGORY_ABILITY.values()]
    ]
    sg = sg.merge(skills, on=["event_key", "source_player_id"], how="left", validate="many_to_one")

    green_observations = sg[
        sg["start_lie"].eq("green")
        & sg["raw_hole_complete"].fillna(False)
        & sg["event_type"].eq("STROKE")
        & sg["start_distance_yd"].notna()
    ].copy()
    green_observations["green_start_distance_yd"] = green_observations["start_distance_yd"]
    green_observations["green_remaining_strokes"] = (
        pd.to_numeric(green_observations["final_stroke_no"], errors="coerce")
        - pd.to_numeric(green_observations["stroke_no"], errors="coerce")
        + 1.0
    )
    model_diag = (
        sg.groupby(["event_key", "sg_category"], observed=True)
        .agg(
            rows=("shot_key", "size"),
            model_ids=("model_id", "nunique"),
            green_start=("start_lie", lambda x: int(x.eq("green").sum())),
            green_end=("end_lie", lambda x: int(x.eq("green").sum())),
        )
        .reset_index()
    )
    return sg, green_observations, model_diag


def _reward_candidate_mask(event: pd.DataFrame) -> pd.Series:
    return (
        event["sg_category"].isin(ALL_CATEGORIES)
        & event["sg_valid"].eq(1)
        & event["complete_hole"].eq(1)
        & event["sg_raw"].notna()
        & event["start_distance_yd"].notna()
    )


def prepare_event_rewards(
    event: pd.DataFrame,
    curve: CurveSnapshot | None,
    reward_min: float,
    reward_max: float,
    ability_clip: float,
) -> pd.DataFrame:
    work = event[_reward_candidate_mask(event)].copy()
    if work.empty:
        return work
    start = pd.to_numeric(work["expected_strokes_start"], errors="coerce").to_numpy(float)
    end = pd.to_numeric(work["expected_strokes_end"], errors="coerce").to_numpy(float)
    green_start = work["start_lie"].eq("green").to_numpy()
    green_end = work["end_lie"].eq("green").to_numpy()
    terminal = work["ball_holed"].fillna(0).eq(1).to_numpy() | np.isclose(end, 0.0, equal_nan=False)
    work["green_reward_recomputed"] = green_start | green_end
    work["strict_green_curve_available"] = curve is not None
    work["reward_curve_sha256"] = (
        "unavailable_warmup" if curve is None else curve.sha256
    )
    if curve is not None:
        if green_start.any():
            start[green_start] = curve.value(work.loc[green_start, "start_distance_yd"].to_numpy(float))
        nonterminal_green = green_end & ~terminal
        if nonterminal_green.any():
            end[nonterminal_green] = curve.value(work.loc[nonterminal_green, "end_distance_yd"].to_numpy(float))
    strict_reward = np.isfinite(start) & np.isfinite(end) & (~(green_start | green_end) | (curve is not None))
    reward = start - end - 1.0
    work["reward_sg_unclipped"] = reward
    work["reward_sg"] = reward
    work["strict_reward_scale"] = strict_reward
    work = work[work["strict_reward_scale"] & np.isfinite(work["reward_sg"])].copy()
    work["reward_clipped"] = ~work["reward_sg"].between(reward_min, reward_max, inclusive="both")
    work["reward_sg"] = work["reward_sg"].clip(reward_min, reward_max)
    work["lie_state"] = work["start_lie"].map(_lie)
    work["distance_state"] = ""
    for category in ALL_CATEGORIES:
        mask = work["sg_category"].eq(category)
        work.loc[mask, "distance_state"] = _distance_state(
            category,
            work.loc[mask, "start_distance_yd"].to_numpy(float),
        )
    work["exact_state"] = (
        work["sg_category"].astype(str) + "|" + work["lie_state"].astype(str)
        + "|" + work["distance_state"].astype(str)
    )
    work["distance_backoff"] = work["sg_category"].astype(str) + "|" + work["distance_state"].astype(str)
    ability = np.full(len(work), np.nan)
    for category, column in CATEGORY_ABILITY.items():
        mask = work["sg_category"].eq(category).to_numpy()
        ability[mask] = pd.to_numeric(work.loc[mask, column], errors="coerce")
    work["ability_missing"] = ~np.isfinite(ability)
    ability = np.where(np.isfinite(ability), ability, 0.0)
    work["ability_clipped"] = np.abs(ability) > ability_clip
    work["ability_sg_round"] = np.clip(ability, -ability_clip, ability_clip)
    work["penalty_inferred"] = (
        work["state_inferred"].fillna("").str.contains("penalty", case=False)
        | ~work["event_type"].fillna("").eq("STROKE")
    )
    return work


def reward_input_coverage(
    raw_event: pd.DataFrame,
    prepared: pd.DataFrame,
    curve_available: bool,
) -> pd.DataFrame:
    candidate = raw_event[_reward_candidate_mask(raw_event)].copy()
    candidate["green_affected"] = candidate["start_lie"].eq("green") | candidate["end_lie"].eq("green")
    retained = prepared.copy()
    if len(retained):
        retained["green_affected"] = retained["green_reward_recomputed"]
    rows: list[dict[str, object]] = []
    for category in ALL_CATEGORIES:
        base = candidate[candidate["sg_category"].eq(category)]
        kept = retained[retained["sg_category"].eq(category)]
        eligible_green = int(base["green_affected"].sum()) if len(base) else 0
        retained_green = int(kept["green_affected"].sum()) if len(kept) else 0
        rows.append(
            {
                "category": category,
                "eligible_rows": int(len(base)),
                "retained_rows": int(len(kept)),
                "retained_rate": float(len(kept) / len(base)) if len(base) else math.nan,
                "eligible_green_rows": eligible_green,
                "retained_green_rows": retained_green,
                "green_retained_rate": (
                    float(retained_green / eligible_green) if eligible_green else math.nan
                ),
                "strict_curve_available": bool(curve_available),
            }
        )
    return pd.DataFrame(rows)


def complete_hole_telescoping_audit(
    raw_event: pd.DataFrame,
    prepared: pd.DataFrame,
) -> pd.DataFrame:
    """Test curve replacement only where every raw row in a hole is retained."""
    if raw_event.empty or prepared.empty:
        return pd.DataFrame()
    hole_keys = ["event_key", "source_player_id", "round_no", "hole_no"]
    raw = raw_event.copy()
    raw["reward_candidate"] = _reward_candidate_mask(raw)
    raw_summary = (
        raw.groupby(hole_keys, observed=True)
        .agg(raw_rows=("shot_key", "size"), candidate_rows=("reward_candidate", "sum"))
        .reset_index()
    )
    changed = prepared.assign(
        reward_delta=(
            prepared["reward_sg_unclipped"]
            - pd.to_numeric(prepared["sg_raw"], errors="coerce")
        )
    )
    prepared_summary = (
        changed.groupby(hole_keys, observed=True)
        .agg(
            prepared_rows=("shot_key", "size"),
            reward_delta=("reward_delta", "sum"),
            green_affected=("green_reward_recomputed", "max"),
        )
        .reset_index()
    )
    audit = raw_summary.merge(prepared_summary, on=hole_keys, how="inner", validate="one_to_one")
    audit = audit[
        audit["raw_rows"].eq(audit["candidate_rows"])
        & audit["raw_rows"].eq(audit["prepared_rows"])
        & audit["green_affected"]
    ].copy()
    audit["abs_hole_delta"] = audit["reward_delta"].abs()
    return audit


@dataclass
class HistState:
    event_normalized: np.ndarray
    globally_normalized: np.ndarray
    raw_count: int = 0
    event_groups: int = 0


class TiltAccumulator:
    def __init__(
        self,
        betas: np.ndarray,
        reward_centers: np.ndarray,
        reward_edges: np.ndarray,
    ) -> None:
        self.betas = betas
        self.centers = reward_centers
        self.edges = reward_edges
        self.levels: dict[str, dict[str, HistState]] = {
            "exact": {}, "distance": {}, "category": {}
        }
        self.reward_curve_sha256: str | None = None

    def _add_level(
        self,
        frame: pd.DataFrame,
        level: str,
        key_col: str,
        weights: np.ndarray,
        reward_bin: np.ndarray,
    ) -> None:
        codes, unique = pd.factorize(frame[key_col], sort=False)
        groups = len(unique)
        bins = len(self.centers)
        flat = codes * bins + reward_bin
        sizes = np.bincount(codes, minlength=groups).astype(float)
        aggregate = np.empty((len(self.betas), groups, bins), dtype=float)
        for index in range(len(self.betas)):
            aggregate[index] = np.bincount(
                flat, weights=weights[index], minlength=groups * bins
            ).reshape(groups, bins)
        denominators = aggregate.sum(axis=2)
        scaled = aggregate * np.divide(
            sizes[None, :],
            denominators,
            out=np.zeros_like(denominators),
            where=denominators > 0,
        )[:, :, None]
        for group_index, key in enumerate(unique.astype(str)):
            state = self.levels[level].get(key)
            if state is None:
                state = HistState(
                    np.zeros((len(self.betas), bins), dtype=float),
                    np.zeros((len(self.betas), bins), dtype=float),
                )
                self.levels[level][key] = state
            state.event_normalized += scaled[:, group_index, :]
            state.globally_normalized += aggregate[:, group_index, :]
            state.raw_count += int(sizes[group_index])
            state.event_groups += 1

    def add_event(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        curve_hashes = set(frame["reward_curve_sha256"].dropna().astype(str))
        if len(curve_hashes) != 1 or "unavailable_warmup" in curve_hashes:
            raise AssertionError("donor event does not use one frozen reward curve")
        curve_hash = next(iter(curve_hashes))
        if self.reward_curve_sha256 is None:
            self.reward_curve_sha256 = curve_hash
        elif self.reward_curve_sha256 != curve_hash:
            raise AssertionError("donor reward curve hash changed within accumulator")
        reward = frame["reward_sg"].to_numpy(float)
        ability = frame["ability_sg_round"].to_numpy(float)
        log_weight = np.clip(-self.betas[:, None] * (ability * reward)[None, :], -30.0, 30.0)
        weights = np.exp(log_weight)
        reward_bin = np.searchsorted(self.edges, reward, side="right") - 1
        reward_bin = np.clip(reward_bin, 0, len(self.centers) - 1)
        work = frame.copy()
        work["category_key"] = work["sg_category"].astype(str)
        self._add_level(work, "exact", "exact_state", weights, reward_bin)
        self._add_level(work, "distance", "distance_backoff", weights, reward_bin)
        self._add_level(work, "category", "category_key", weights, reward_bin)

    @staticmethod
    def _probability(state: HistState | None, mode: str) -> np.ndarray | None:
        if state is None or state.raw_count <= 0:
            return None
        matrix = getattr(state, mode)
        total = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, total, out=np.zeros_like(matrix), where=total > 0)

    def base_distribution(
        self,
        category: str,
        exact_key: str,
        distance_key: str,
        mode: str,
        state_prior: float,
        distance_prior: float,
        minimum_global: int,
        variance_floor_ratio: float,
        variance_floor_absolute: float,
        uniform_mass: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
        global_state = self.levels["category"].get(category)
        if global_state is None or global_state.raw_count < minimum_global:
            return None
        exact_state = self.levels["exact"].get(exact_key)
        distance_state = self.levels["distance"].get(distance_key)
        global_p = self._probability(global_state, mode)
        assert global_p is not None
        exact_p = self._probability(exact_state, mode)
        distance_p = self._probability(distance_state, mode)
        exact_n = 0 if exact_state is None else exact_state.raw_count
        distance_n = 0 if distance_state is None else distance_state.raw_count
        exact_mass = exact_n / (exact_n + state_prior) if exact_n else 0.0
        remaining = 1.0 - exact_mass
        distance_mass = remaining * distance_n / (distance_n + distance_prior) if distance_n else 0.0
        global_mass = 1.0 - exact_mass - distance_mass
        p = global_mass * global_p
        if exact_p is not None:
            p = p + exact_mass * exact_p
        if distance_p is not None:
            p = p + distance_mass * distance_p
        p /= p.sum(axis=1, keepdims=True)

        global_mean = global_p @ self.centers
        global_var = np.sum(global_p * (self.centers[None, :] - global_mean[:, None]) ** 2, axis=1)
        floor = np.maximum(variance_floor_absolute, variance_floor_ratio * global_var)
        floor_used = np.zeros(len(self.betas), dtype=float)
        for beta_index in range(len(self.betas)):
            mean = float(np.dot(p[beta_index], self.centers))
            variance = float(np.dot(p[beta_index], (self.centers - mean) ** 2))
            if variance >= floor[beta_index]:
                continue
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mix = (lo + hi) / 2.0
                candidate = (1.0 - mix) * p[beta_index] + mix * global_p[beta_index]
                candidate_mean = float(np.dot(candidate, self.centers))
                candidate_var = float(np.dot(candidate, (self.centers - candidate_mean) ** 2))
                if candidate_var < floor[beta_index]:
                    lo = mix
                else:
                    hi = mix
            floor_used[beta_index] = hi
            p[beta_index] = (1.0 - hi) * p[beta_index] + hi * global_p[beta_index]
        if uniform_mass > 0:
            p = (1.0 - uniform_mass) * p + uniform_mass / len(self.centers)
        p /= p.sum(axis=1, keepdims=True)
        diagnostics = {
            "exact_n": float(exact_n),
            "distance_n": float(distance_n),
            "global_n": float(global_state.raw_count),
            "exact_mass": float(exact_mass),
            "distance_mass": float(distance_mass),
        }
        return p, floor_used, diagnostics


def _crps_rows(probability: np.ndarray, centers: np.ndarray, observed: np.ndarray) -> np.ndarray:
    cumulative_probability_before = np.cumsum(probability, axis=1) - probability
    cumulative_moment_before = np.cumsum(probability * centers[None, :], axis=1) - probability * centers[None, :]
    pair_distance = 2.0 * np.sum(
        probability
        * (centers[None, :] * cumulative_probability_before - cumulative_moment_before),
        axis=1,
    )
    mean_absolute = np.mean(np.abs(centers[:, None] - observed[None, :]), axis=1)
    first = probability @ mean_absolute
    return first - 0.5 * pair_distance


def score_event(
    frame: pd.DataFrame,
    accumulator: TiltAccumulator,
    donor_mode: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    curve_hashes = set(frame["reward_curve_sha256"].dropna().astype(str))
    if len(frame) and (
        len(curve_hashes) != 1
        or "unavailable_warmup" in curve_hashes
        or accumulator.reward_curve_sha256 not in curve_hashes
    ):
        raise AssertionError("target and donor reward curves are not identical")
    beta_zero = int(np.where(np.isclose(accumulator.betas, 0.0))[0][0])
    for category, category_frame in frame.groupby("sg_category", observed=True):
        log_sum = np.zeros(len(accumulator.betas))
        crps_sum = np.zeros(len(accumulator.betas))
        floor_shots = np.zeros(len(accumulator.betas))
        shots = 0
        exact_weighted = distance_weighted = global_weighted = 0.0
        grouped = category_frame.groupby(
            ["exact_state", "distance_backoff"],
            observed=True,
            dropna=False,
        )
        for (exact_key, distance_key), group in grouped:
            base = accumulator.base_distribution(
                str(category), str(exact_key), str(distance_key), donor_mode,
                args.state_prior, args.distance_prior, args.minimum_global_atoms,
                args.variance_floor_ratio, args.variance_floor_absolute,
                args.uniform_mass,
            )
            if base is None:
                continue
            base_probability, floor_used, diagnostics = base
            ability_codes, abilities = pd.factorize(
                group["ability_sg_round"], sort=False
            )
            abilities = np.asarray(abilities, dtype=float)
            target_log = np.clip(
                abilities[:, None, None]
                * accumulator.betas[None, :, None]
                * accumulator.centers[None, None, :],
                -30.0,
                30.0,
            )
            tilted = base_probability[None, :, :] * np.exp(target_log)
            tilted /= tilted.sum(axis=2, keepdims=True)
            observed = group["reward_sg"].to_numpy(float)
            observed_bins = np.searchsorted(accumulator.edges, observed, side="right") - 1
            observed_bins = np.clip(observed_bins, 0, len(accumulator.centers) - 1)
            bin_counts = np.zeros((len(abilities), len(accumulator.centers)), dtype=float)
            np.add.at(bin_counts, (ability_codes, observed_bins), 1.0)
            log_sum += -np.einsum(
                "lbk,lk->b", np.log(np.maximum(tilted, 1e-300)), bin_counts
            )

            absolute_sum = np.zeros_like(bin_counts)
            np.add.at(
                absolute_sum,
                ability_codes,
                np.abs(observed[:, None] - accumulator.centers[None, :]),
            )
            cumulative_probability_before = np.cumsum(tilted, axis=2) - tilted
            cumulative_moment_before = (
                np.cumsum(tilted * accumulator.centers[None, None, :], axis=2)
                - tilted * accumulator.centers[None, None, :]
            )
            pair_distance = 2.0 * np.sum(
                tilted
                * (
                    accumulator.centers[None, None, :] * cumulative_probability_before
                    - cumulative_moment_before
                ),
                axis=2,
            )
            first_sum = np.einsum("lbk,lk->lb", tilted, absolute_sum)
            group_sizes = bin_counts.sum(axis=1)
            crps_sum += np.sum(
                first_sum - 0.5 * pair_distance * group_sizes[:, None], axis=0
            )
            floor_shots += (floor_used > 0).astype(float) * len(group)
            shots += len(group)
            exact_weighted += diagnostics["exact_n"] * len(group)
            distance_weighted += diagnostics["distance_n"] * len(group)
            global_weighted += diagnostics["global_n"] * len(group)
        if shots == 0:
            continue
        for beta_index, beta in enumerate(accumulator.betas):
            rows.append(
                {
                    "category": category,
                    "beta": float(beta),
                    "shots": int(shots),
                    "logloss": float(log_sum[beta_index] / shots),
                    "crps": float(crps_sum[beta_index] / shots),
                    "variance_floor_rate": float(floor_shots[beta_index] / shots),
                    "mean_exact_prior_n": float(exact_weighted / shots),
                    "mean_distance_prior_n": float(distance_weighted / shots),
                    "mean_global_prior_n": float(global_weighted / shots),
                    "beta_zero_index": beta_zero,
                }
            )
    return pd.DataFrame(rows)


def add_baseline_deltas(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return curves
    keys = ["scenario", "donor_normalization", "event_key", "category"]
    baseline = curves[np.isclose(curves["beta"], 0.0)][keys + ["logloss", "crps"]].rename(
        columns={"logloss": "logloss_beta0", "crps": "crps_beta0"}
    )
    out = curves.merge(baseline, on=keys, how="left", validate="many_to_one")
    out["logloss_delta_vs_beta0"] = out["logloss"] - out["logloss_beta0"]
    out["crps_delta_vs_beta0"] = out["crps"] - out["crps_beta0"]
    return out


def rolling_selection(
    curves: pd.DataFrame,
    betas: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    if curves.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = ["scenario", "donor_normalization", "category"]
    for keys, panel in curves.groupby(group_cols, observed=True):
        for test_year in range(args.start_season + 2, args.end_season + 1):
            tune = panel[panel["season"].lt(test_year)]
            test = panel[panel["season"].eq(test_year)]
            tune_events = tune["event_key"].nunique()
            if tune_events < args.minimum_tune_events or test.empty:
                continue
            objective = tune.groupby("beta", observed=True)["logloss_delta_vs_beta0"].mean()
            raw_beta = float(objective.idxmin())
            reliability = tune_events / (tune_events + args.beta_prior_events)
            shrunk_beta_value = raw_beta * reliability
            selected_beta = float(betas[np.argmin(np.abs(betas - shrunk_beta_value))])
            selected = test[np.isclose(test["beta"], selected_beta)].copy()
            if selected.empty:
                continue
            deltas = selected["logloss_delta_vs_beta0"].to_numpy(float)
            crps_deltas = selected["crps_delta_vs_beta0"].to_numpy(float)
            event_count = len(selected)
            se = float(np.std(deltas, ddof=1) / math.sqrt(event_count)) if event_count > 1 else math.nan
            crps_se = float(np.std(crps_deltas, ddof=1) / math.sqrt(event_count)) if event_count > 1 else math.nan
            weights = selected["shots"].to_numpy(float)
            fold_rows.append(
                {
                    "scenario": keys[0],
                    "donor_normalization": keys[1],
                    "category": keys[2],
                    "test_year": test_year,
                    "tune_events": int(tune_events),
                    "test_events": int(event_count),
                    "test_shots": int(selected["shots"].sum()),
                    "raw_best_beta": raw_beta,
                    "beta_reliability": float(reliability),
                    "selected_beta": selected_beta,
                    "equal_event_logloss_delta": float(np.mean(deltas)),
                    "event_cluster_se": se,
                    "ci_low": float(np.mean(deltas) - 1.96 * se) if np.isfinite(se) else math.nan,
                    "ci_high": float(np.mean(deltas) + 1.96 * se) if np.isfinite(se) else math.nan,
                    "shot_weighted_logloss_delta": float(np.average(deltas, weights=weights)),
                    "favorable_logloss_events": int(np.sum(deltas < 0)),
                    "equal_event_crps_delta": float(np.mean(crps_deltas)),
                    "crps_event_cluster_se": crps_se,
                    "shot_weighted_crps_delta": float(np.average(crps_deltas, weights=weights)),
                    "favorable_crps_events": int(np.sum(crps_deltas < 0)),
                }
            )
            selected["test_year"] = test_year
            selected["selected_beta"] = selected_beta
            event_rows.extend(selected.to_dict("records"))
    return pd.DataFrame(fold_rows), pd.DataFrame(event_rows)


def write_report(
    folds: pd.DataFrame,
    coverage: pd.DataFrame,
    additivity: pd.DataFrame,
    reward_coverage: pd.DataFrame,
    output: Path,
    smoke: bool,
) -> Path:
    lines = [
        "# Prior-only shot quality-tilt evaluation",
        "",
        "This is a reward-atom research test, not an endpoint/hazard or recursive hole-simulator validation.",
        "All donor supports and player abilities are strictly prior to each held-out event.",
        "One green-value curve is trained on the earliest completed warmup date(s), then frozen and used for every scored donor and target; warmup dates are not scored.",
        "Beta zero scores the identical rows and is the paired control.",
        "",
    ]
    if smoke:
        lines.extend(["**Bounded smoke run:** results below validate mechanics only.", ""])
    if folds.empty:
        lines.extend(["No rolling fold met the minimum tuning/event support gate.", ""])
    else:
        lines.extend([
            "| Scenario | Donor normalization | Category | Test year | Beta | Events | Log-score delta | 95% event CI | CRPS delta |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in folds.itertuples(index=False):
            label = row.category.upper() + (" (secondary)" if row.category == "putt" else "")
            lines.append(
                f"| {row.scenario} | {row.donor_normalization} | {label} | {int(row.test_year)} | "
                f"{row.selected_beta:+.2f} | {int(row.test_events)} | "
                f"{row.equal_event_logloss_delta:+.6f} | [{row.ci_low:+.6f}, {row.ci_high:+.6f}] | "
                f"{row.equal_event_crps_delta:+.6f} |"
            )
        lines.append("")
    minimum_coverage = float(
        coverage[[f"{category}_ability_coverage" for category in PRIMARY_CATEGORIES]].min().min()
    ) if len(coverage) else math.nan
    lines.extend([
        f"Minimum event/category roster skill coverage was {minimum_coverage:.1%}; missing values use the frozen neutral prior 0.",
        "The primary interpretation is OTT/APP/ARG. PUTT is emitted only as a secondary sensitivity.",
        "`all_valid` retains SG-valid penalty-derived rows; `physical_stroke_only` excludes inferred penalty/drop rows.",
        "`event_normalized` preserves each donor event/state's declared raw mass after de-tilting; `globally_normalized` is the source-mixture sensitivity.",
        "",
    ])
    if len(reward_coverage):
        primary = reward_coverage[
            reward_coverage["category"].isin(("app", "arg"))
            & reward_coverage["strict_curve_available"]
        ]
        if len(primary):
            eligible = primary.groupby("category")["eligible_rows"].sum()
            retained = primary.groupby("category")["retained_rows"].sum()
            rates = (retained / eligible).to_dict()
            lines.append(
                "Strict-curve retained-row rates after warmup: "
                + ", ".join(f"{key.upper()} {value:.1%}" for key, value in rates.items())
                + "."
            )
            lines.append("")
    if len(additivity):
        lines.append(
            f"Complete-hole green-value telescoping audit: median absolute reward delta "
            f"{additivity['abs_hole_delta'].median():.8f}, 99th percentile "
            f"{additivity['abs_hole_delta'].quantile(0.99):.8f}."
        )
        lines.append("")
    lines.extend([
        "Negative score deltas favor quality tilting. Intervals cluster at event edition and are descriptive.",
        "No held-out event/state nuisance mean, target-event putting curve, endpoint, lie, or hazard information enters a forecast.",
    ])
    path = output / "QUALITY_TILT_EVALUATION.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--max-events-per-season", type=int)
    parser.add_argument("--beta-grid", default="-0.5:2.5:0.1")
    parser.add_argument("--reward-min", type=float, default=-3.0)
    parser.add_argument("--reward-max", type=float, default=3.0)
    parser.add_argument("--reward-bin-width", type=float, default=0.025)
    parser.add_argument("--ability-clip", type=float, default=2.0)
    parser.add_argument("--state-prior", type=float, default=400.0)
    parser.add_argument("--distance-prior", type=float, default=1200.0)
    parser.add_argument("--minimum-global-atoms", type=int, default=3000)
    parser.add_argument("--minimum-putting-samples", type=int, default=500)
    parser.add_argument("--variance-floor-ratio", type=float, default=0.005)
    parser.add_argument("--variance-floor-absolute", type=float, default=0.0001)
    parser.add_argument("--uniform-mass", type=float, default=1e-5)
    parser.add_argument("--beta-prior-events", type=float, default=20.0)
    parser.add_argument("--minimum-tune-events", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-large-file-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_season >= args.end_season:
        raise ValueError("at least two seasons are required")
    if args.reward_bin_width <= 0 or args.reward_max <= args.reward_min:
        raise ValueError("invalid reward grid")
    if args.smoke and args.max_events_per_season is None:
        args.max_events_per_season = 6
    source_sha256_at_start = _sha256(Path(__file__))
    telemetry = PeakRSSMonitor()
    telemetry.start()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    betas = _parse_beta_grid(args.beta_grid)
    edges = np.arange(
        args.reward_min,
        args.reward_max + args.reward_bin_width / 2.0,
        args.reward_bin_width,
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    calendar = load_calendar(args.db_path, args.start_season, args.end_season)
    available = available_event_keys(args.export_root, range(args.start_season, args.end_season + 1))
    calendar = select_calendar(calendar, available, args.max_events_per_season)
    if calendar.empty:
        raise RuntimeError("no selected events have shot-SG exports")
    rosters = load_rosters(args.export_root, calendar)
    skills = load_skill_history(args.db_path, args.end_season)
    snapshots, coverage = build_strict_snapshots(rosters, calendar, skills)

    accumulators = {
        "all_valid": TiltAccumulator(betas, centers, edges),
        "physical_stroke_only": TiltAccumulator(betas, centers, edges),
    }
    putting = PriorPuttingCurve()
    frozen_curve: CurveSnapshot | None = None
    frozen_curve_available_at: pd.Timestamp | None = None
    warmup_raw_events: list[pd.DataFrame] = []
    curve_rows: list[pd.DataFrame] = []
    curve_snapshots: list[dict[str, object]] = []
    model_diagnostics: list[pd.DataFrame] = []
    additivity_rows: list[pd.DataFrame] = []
    reward_coverage_rows: list[pd.DataFrame] = []
    data_counts = {
        "raw_sg_rows": 0,
        "prepared_rows": 0,
        "warmup_events": 0,
        "warmup_raw_rows": 0,
        "backfilled_donor_rows": 0,
        "scored_target_rows": 0,
        "strict_cutoff_violations": 0,
        "reward_clipped_rows": 0,
        "ability_clipped_rows": 0,
    }

    for season, season_events in calendar.groupby("season", sort=True):
        event_keys = season_events["event_key"].astype(str).tolist()
        sg, green_observations, model_diag = load_year_panel(
            args.export_root, int(season), event_keys, snapshots, calendar
        )
        data_counts["raw_sg_rows"] += len(sg)
        model_diag["season"] = int(season)
        model_diagnostics.append(model_diag)
        for event_date, date_events in iter_event_date_batches(season_events):
            if frozen_curve is None:
                candidate_curve = putting.snapshot(
                    args.minimum_putting_samples, as_of=event_date
                )
                if candidate_curve is not None:
                    frozen_curve = candidate_curve
                    frozen_curve_available_at = pd.Timestamp(event_date)
                    for warmup_event in warmup_raw_events:
                        backfilled = prepare_event_rewards(
                            warmup_event,
                            frozen_curve,
                            args.reward_min,
                            args.reward_max,
                            args.ability_clip,
                        )
                        if len(backfilled) and set(
                            backfilled["reward_curve_sha256"].astype(str)
                        ) != {frozen_curve.sha256}:
                            raise AssertionError("warmup donor backfill curve mismatch")
                        accumulators["all_valid"].add_event(backfilled)
                        accumulators["physical_stroke_only"].add_event(
                            backfilled[~backfilled["penalty_inferred"]]
                        )
                        data_counts["backfilled_donor_rows"] += len(backfilled)
                    warmup_raw_events.clear()
            curve = frozen_curve
            curve_snapshots.append(
                {
                    "event_start": event_date,
                    "event_keys": "|".join(sorted(date_events["event_key"].astype(str))),
                    "prior_putting_samples": 0 if curve is None else curve.samples,
                    "curve_sha256": "unavailable_warmup" if curve is None else curve.sha256,
                    "latest_source_event_start": (
                        None if curve is None else curve.latest_source_event_start
                    ),
                    "frozen_curve_available_at": frozen_curve_available_at,
                    "curve_frozen": curve is not None,
                    "strict_prior": True,
                }
            )
            prepared_by_event: dict[str, pd.DataFrame] = {}
            for event_key in date_events["event_key"].astype(str):
                raw_event = sg[sg["event_key"].eq(event_key)].copy()
                prepared = prepare_event_rewards(
                    raw_event, curve, args.reward_min, args.reward_max, args.ability_clip
                )
                if curve is None and len(prepared) and prepared["green_reward_recomputed"].any():
                    raise AssertionError("green-start/end row scored before prior curve warmup")
                data_counts["prepared_rows"] += len(prepared)
                data_counts["reward_clipped_rows"] += int(prepared.get("reward_clipped", pd.Series(dtype=bool)).sum())
                data_counts["ability_clipped_rows"] += int(prepared.get("ability_clipped", pd.Series(dtype=bool)).sum())
                input_coverage = reward_input_coverage(raw_event, prepared, curve is not None)
                input_coverage["event_key"] = event_key
                input_coverage["event_start"] = event_date
                reward_coverage_rows.append(input_coverage)
                telescoping = complete_hole_telescoping_audit(raw_event, prepared)
                if len(telescoping):
                    if float(telescoping["abs_hole_delta"].max()) > 5e-6:
                        raise AssertionError("complete-hole reward telescoping invariant failed")
                    additivity_rows.append(telescoping)
                if curve is None:
                    warmup_raw_events.append(raw_event)
                    data_counts["warmup_events"] += 1
                    data_counts["warmup_raw_rows"] += len(raw_event)
                    continue
                if set(prepared["reward_curve_sha256"].dropna().astype(str)) != {curve.sha256}:
                    raise AssertionError("target event reward curve mismatch")
                prepared_by_event[event_key] = prepared
                data_counts["scored_target_rows"] += len(prepared)
                for scenario, accumulator in accumulators.items():
                    scored_frame = prepared if scenario == "all_valid" else prepared[~prepared["penalty_inferred"]]
                    for donor_mode in ("event_normalized", "globally_normalized"):
                        scored = score_event(scored_frame, accumulator, donor_mode, args)
                        if scored.empty:
                            continue
                        meta = raw_event.iloc[0]
                        scored["scenario"] = scenario
                        scored["donor_normalization"] = donor_mode
                        scored["event_key"] = event_key
                        scored["event_name"] = meta["event_name"]
                        scored["season"] = int(meta["season"])
                        scored["event_start"] = event_date
                        scored["missing_ability_rate"] = float(scored_frame["ability_missing"].mean()) if len(scored_frame) else math.nan
                        scored["green_recomputed_rate"] = float(scored_frame["green_reward_recomputed"].mean()) if len(scored_frame) else math.nan
                        scored["penalty_inferred_rate"] = float(scored_frame["penalty_inferred"].mean()) if len(scored_frame) else math.nan
                        scored["prior_putting_samples"] = 0 if curve is None else curve.samples
                        scored["putting_curve_sha256"] = "unavailable_warmup" if curve is None else curve.sha256
                        curve_rows.append(scored)
            # Same-date events enter the donor and putting pools only after all
            # forecasts for the date have been scored.
            if curve is not None:
                for event_key, prepared in prepared_by_event.items():
                    accumulators["all_valid"].add_event(prepared)
                    accumulators["physical_stroke_only"].add_event(
                        prepared[~prepared["penalty_inferred"]]
                    )
            else:
                same_date_keys = set(date_events["event_key"].astype(str))
                putting.update(
                    green_observations[
                        green_observations["event_key"].isin(same_date_keys)
                    ],
                    event_start=event_date,
                )
        del sg, green_observations
        print(
            f"PROGRESS season={int(season)} events={len(season_events)} "
            f"prepared_rows={data_counts['prepared_rows']} "
            f"rss_mb={psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.1f}",
            flush=True,
        )

    if frozen_curve is None:
        raise RuntimeError("putting curve never reached the frozen prior-sample gate")
    if warmup_raw_events:
        raise AssertionError("warmup donor queue was not backfilled")
    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    scored_curve_hashes = set(
        curves.get("putting_curve_sha256", pd.Series(dtype=str)).dropna().astype(str)
    )
    if curves.empty or scored_curve_hashes != {frozen_curve.sha256}:
        raise AssertionError("scored targets do not all use the frozen reward curve")
    for accumulator in accumulators.values():
        if accumulator.reward_curve_sha256 != frozen_curve.sha256:
            raise AssertionError("donor support does not use the frozen reward curve")
    curves = add_baseline_deltas(curves)
    folds, selected_events = rolling_selection(curves, betas, args)
    additivity = pd.concat(additivity_rows, ignore_index=True) if additivity_rows else pd.DataFrame()
    reward_coverage = (
        pd.concat(reward_coverage_rows, ignore_index=True)
        if reward_coverage_rows else pd.DataFrame()
    )
    model_diag = pd.concat(model_diagnostics, ignore_index=True) if model_diagnostics else pd.DataFrame()
    curve_audit = pd.DataFrame(curve_snapshots)

    artifacts: dict[str, Path] = {}
    for name, frame in (
        ("candidate_event_scores.csv", curves),
        ("rolling_fold_results.csv", folds),
        ("selected_event_scores.csv", selected_events),
        ("snapshot_coverage.csv", coverage),
        ("putting_curve_snapshots.csv", curve_audit),
        ("reward_model_diagnostics.csv", model_diag),
        ("reward_input_coverage.csv", reward_coverage),
        ("green_complete_hole_telescoping_audit.csv", additivity),
    ):
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        artifacts[name] = path
    report = write_report(
        folds, coverage, additivity, reward_coverage, args.output_dir, args.smoke
    )
    artifacts[report.name] = report

    input_files = [args.db_path, REPO_ROOT / "sim_inputs.py"]
    for season in sorted(calendar["season"].unique()):
        input_files.extend(
            [
                args.export_root / f"pga_{int(season)}_shot_sg.parquet",
                args.export_root / f"pga_{int(season)}_shots.parquet",
                args.export_root / f"pga_{int(season)}_player_rounds.parquet",
            ]
        )
    input_meta = {}
    for path in input_files:
        stat = path.stat()
        input_meta[str(path.resolve())] = {
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": None if args.skip_large_file_hashes else _sha256(path),
        }
    source_sha256_at_end = _sha256(Path(__file__))
    if source_sha256_at_end != source_sha256_at_start:
        raise AssertionError("evaluator source changed during the run")
    run_telemetry = telemetry.finish()
    manifest = {
        "contract": "strict_prior_whole_reward_atom_tilt_v1",
        "research_only": True,
        "production_enabled": False,
        "primary_categories": list(PRIMARY_CATEGORIES),
        "secondary_categories": ["putt"],
        "putt_interpretation": "secondary despite reconstructed prior-only green value curve",
        "event_start_min": calendar["event_start"].min().isoformat(),
        "event_start_max": calendar["event_start"].max().isoformat(),
        "events": int(calendar["event_key"].nunique()),
        "source_sha256": source_sha256_at_start,
        "source_sha256_at_end": source_sha256_at_end,
        "identity_builder_source_sha256": _sha256(Path(identity_audit.__file__)),
        "snapshot_frame_sha256": _frame_hash(snapshots),
        "calendar_frame_sha256": _frame_hash(calendar),
        "frozen_reward_curve": {
            "sha256": frozen_curve.sha256,
            "samples": frozen_curve.samples,
            "latest_source_event_start": frozen_curve.latest_source_event_start,
            "available_for_scoring_at": frozen_curve_available_at,
            "scored_curve_hashes": sorted(scored_curve_hashes),
            "warmup_dates_scored": False,
        },
        "runtime": run_telemetry,
        "beta_grid": betas.tolist(),
        "reward_grid": {
            "minimum": args.reward_min,
            "maximum": args.reward_max,
            "width": args.reward_bin_width,
        },
        "run_config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "data_counts": data_counts,
        "strict_cutoff_violations": int(
            pd.to_datetime(snapshots["skill_date"], errors="coerce")
            .ge(pd.to_datetime(snapshots["event_start"], errors="coerce"))
            .fillna(False)
            .sum()
        ),
        "input_files": input_meta,
        "artifact_sha256": {name: _sha256(path) for name, path in artifacts.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"Events: {manifest['events']}; prepared reward rows: {data_counts['prepared_rows']:,}")
    print(f"Wrote {report}")
    if len(folds):
        print(folds.to_string(index=False))


if __name__ == "__main__":
    main()

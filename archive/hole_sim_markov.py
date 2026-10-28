"""Leakage-safe recursive golf-hole simulator research prototype.

The prototype learns empirical state transitions from the R2 shot archive and
replays a held-out edition of a recurring physical course.  It is deliberately
standalone: no production simulation code imports this file.

Each transition starts from ``(round, hole, stroke, lie, remaining distance,
category)`` and samples a historically observed next state.  Donors back off
hierarchically from exact hole/round/setup to the physical course and then the
PGA Tour.  Exact-course donors collapse a physical shot and any immediately
following penalty/drop sequence into one transition with an explicit stroke
cost, so penalty strokes are not lost.

Player conditioning is an importance-weighted empirical transition model.  A
player's strictly-prior SG distribution in each category is shrunk to the Tour
distribution, then used to reweight otherwise comparable transition donors.
This changes both expected transition quality and dispersion without inventing
geometry that was not observed.

Example
-------
python archive/hole_sim_markov.py --target-event-key pga:R2026013 --sims 400
python archive/hole_sim_markov.py --target-event-key pga:R2026013 --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")
DEFAULT_DB = Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db")
DEFAULT_OUTPUT = REPO / "permanent_data" / "hole_sim_markov"
DEFAULT_EVENT = "pga:R2026013"  # Wyndham / Sedgefield, seven archived editions

CATEGORIES = ("ott", "app", "arg", "putt")
DISTANCE_EDGES = np.asarray([0.0, 100.0, 150.0, 215.0, np.inf])
SCORE_SUPPORT = np.arange(1, 13, dtype=int)


def parse_event_key(value: str) -> tuple[int, int]:
    parsed = pd.Series([value]).str.extract(r"^[^:]+:R(\d{4})(\d+)$").iloc[0]
    if parsed.isna().any():
        raise ValueError(f"Invalid event key: {value!r}")
    return int(parsed.iloc[0]), int(parsed.iloc[1])


def lie_group(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"tee", "fairway", "rough", "sand", "recovery", "green"}:
        return text
    if text == "hole":
        return "hole"
    return "unknown"


def distance_bucket(distance: float) -> int:
    if not np.isfinite(distance):
        return 0
    return int(np.clip(np.searchsorted(DISTANCE_EDGES, distance, side="right") - 1, 0, 3))


def state_category(lie: str, distance: float, initial_category: str | None = None) -> str:
    if lie == "green":
        return "putt"
    if lie == "tee" and initial_category in {"ott", "app"}:
        return initial_category
    if distance <= 50.0:
        return "arg"
    return "app"


def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 0.15)
    z = (x - mean) / sd
    return np.exp(-0.5 * z * z) / sd


def crps_samples(samples: np.ndarray, actual: float) -> float:
    values = np.sort(np.asarray(samples, dtype=float))
    n = len(values)
    if not n:
        return float("nan")
    first = float(np.mean(np.abs(values - actual)))
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    pair_term = float(np.sum(weights * values)) / (n * n)
    return first - pair_term


def smoothed_log_score(samples: np.ndarray, actual: int, alpha: float = 0.5) -> float:
    clipped = np.clip(np.asarray(samples, dtype=int), SCORE_SUPPORT[0], SCORE_SUPPORT[-1])
    counts = np.bincount(clipped, minlength=SCORE_SUPPORT[-1] + 1)
    probability = (counts[int(np.clip(actual, 1, 12))] + alpha) / (
        len(clipped) + alpha * len(SCORE_SUPPORT)
    )
    return -math.log(float(probability))


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class PlayerCategoryPrior:
    mean: float
    sd: float
    pop_mean: float
    pop_sd: float
    n: int


@dataclass
class TransitionPool:
    start_distance: np.ndarray
    next_distance: np.ndarray
    next_lie: np.ndarray
    sg: np.ndarray
    cost: np.ndarray
    holed: np.ndarray
    sg_weight_valid: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "TransitionPool":
        return cls(
            start_distance=frame["start_distance"].to_numpy(float),
            next_distance=frame["next_distance"].to_numpy(float),
            next_lie=frame["next_lie"].astype(str).to_numpy(),
            sg=frame["sg"].to_numpy(float),
            cost=frame["cost"].to_numpy(np.int16),
            holed=frame["holed"].to_numpy(bool),
            sg_weight_valid=frame["sg_weight_valid"].to_numpy(bool),
        )

    def __len__(self) -> int:
        return len(self.cost)


@dataclass(frozen=True)
class HoleSetup:
    round_no: int
    hole_no: int
    start_distance: float
    initial_category: str


class MarkovHoleModel:
    def __init__(
        self,
        transitions: pd.DataFrame,
        tour_transitions: pd.DataFrame,
        min_exact: int = 16,
        knn: int = 64,
    ) -> None:
        self.min_exact = min_exact
        self.knn = knn
        self.exact: dict[tuple, TransitionPool] = {}
        self.exact_lie: dict[tuple, TransitionPool] = {}
        self.course: dict[tuple, TransitionPool] = {}
        self.course_relaxed: dict[tuple, TransitionPool] = {}
        self.tour: dict[tuple, TransitionPool] = {}
        self.tour_relaxed: dict[tuple, TransitionPool] = {}
        self._build_index(transitions, tour_transitions)
        self.backoff_counts: Counter[str] = Counter()
        self.terminal_counts: Counter[str] = Counter()
        self._knn_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _group_pools(frame: pd.DataFrame, columns: list[str]) -> dict[tuple, TransitionPool]:
        result: dict[tuple, TransitionPool] = {}
        for key, group in frame.groupby(columns, observed=True, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            result[key] = TransitionPool.from_frame(group)
        return result

    def _build_index(self, course: pd.DataFrame, tour: pd.DataFrame) -> None:
        self.exact = self._group_pools(
            course, ["round_no", "hole_no", "category", "lie", "bucket"]
        )
        self.exact_lie = self._group_pools(
            course, ["round_no", "hole_no", "category", "lie"]
        )
        self.course = self._group_pools(course, ["category", "lie", "bucket"])
        self.course_relaxed = self._group_pools(course, ["category", "lie"])
        self.tour = self._group_pools(tour, ["category", "lie", "bucket"])
        self.tour_relaxed = self._group_pools(tour, ["category", "lie"])

    def _pool(
        self, round_no: int, hole_no: int, category: str, lie: str, bucket: int
    ) -> tuple[TransitionPool, str]:
        choices = [
            (self.exact, (round_no, hole_no, category, lie, bucket), "exact_hole"),
            (self.exact_lie, (round_no, hole_no, category, lie), "exact_hole_lie"),
            (self.course, (category, lie, bucket), "course"),
            (self.course_relaxed, (category, lie), "course_relaxed"),
            (self.tour, (category, lie, bucket), "tour"),
            (self.tour_relaxed, (category, lie), "tour_relaxed"),
        ]
        for index, key, label in choices:
            pool = index.get(key)
            threshold = self.min_exact if label.startswith("exact") else 4
            if pool is not None and len(pool) >= threshold:
                return pool, label
        # Last-resort category pool. This should be extremely rare and is
        # intentionally visible in backoff diagnostics.
        candidates = [pool for key, pool in self.tour_relaxed.items() if key[0] == category]
        if not candidates:
            candidates = list(self.tour_relaxed.values())
        if not candidates:
            raise RuntimeError(f"No transition donor for {category}/{lie}/{bucket}")
        return max(candidates, key=len), "tour_last_resort"

    def sample_transition(
        self,
        rng: np.random.Generator,
        round_no: int,
        hole_no: int,
        category: str,
        lie: str,
        distance: float,
        prior: PlayerCategoryPrior | None,
    ) -> tuple[bool, str, float, int]:
        pool, level = self._pool(
            round_no, hole_no, category, lie, distance_bucket(distance)
        )
        self.backoff_counts[level] += 1
        # Half-yard quantization is far below the uncertainty in reported shot
        # distances and turns repeated tournament simulation into cached kNN
        # lookup rather than millions of identical argpartitions.
        cache_key = (id(pool), int(round(max(distance, 0.0) * 2.0)))
        cached = self._knn_cache.get(cache_key)
        if cached is None:
            metric = np.abs(
                np.log1p(np.maximum(pool.start_distance, 0.0))
                - math.log1p(max(distance, 0.0))
            )
            k = min(self.knn, len(pool))
            if k < len(pool):
                neighbours = np.argpartition(metric, k - 1)[:k]
            else:
                neighbours = np.arange(len(pool))
            distance_weights = np.exp(-metric[neighbours] / 0.18)
            self._knn_cache[cache_key] = (neighbours, distance_weights)
        else:
            neighbours, distance_weights = cached
        weights = np.maximum(distance_weights, 1e-5)
        if prior is not None:
            donor_sg = np.clip(pool.sg[neighbours], -3.0, 3.0)
            numerator = normal_pdf(donor_sg, prior.mean, prior.sd)
            denominator = normal_pdf(donor_sg, prior.pop_mean, prior.pop_sd)
            ratio = np.clip(numerator / np.maximum(denominator, 1e-8), 0.05, 20.0)
            # Invalid SG rows remain valid geometric/state transitions, but
            # must not masquerade as evidence about player performance.
            ratio = np.where(pool.sg_weight_valid[neighbours], ratio, 1.0)
            weights *= ratio
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            choice = int(rng.choice(neighbours))
        else:
            choice = int(rng.choice(neighbours, p=weights / total))
        return (
            bool(pool.holed[choice]),
            str(pool.next_lie[choice]),
            float(pool.next_distance[choice]),
            int(pool.cost[choice]),
        )

    def simulate_hole(
        self,
        setup: HoleSetup,
        rng: np.random.Generator,
        simulations: int,
        player_priors: Mapping[str, PlayerCategoryPrior] | None,
        max_score: int = 12,
        diagnostic_label: str = "unknown",
    ) -> np.ndarray:
        scores = np.zeros(simulations, dtype=np.int16)
        for sim in range(simulations):
            lie = "tee"
            distance = setup.start_distance
            score = 0
            terminated = False
            for _ in range(max_score):
                category = state_category(lie, distance, setup.initial_category)
                prior = player_priors.get(category) if player_priors else None
                holed, next_lie, next_distance, cost = self.sample_transition(
                    rng, setup.round_no, setup.hole_no, category, lie, distance, prior
                )
                score += max(cost, 1)
                if holed or next_lie == "hole":
                    self.terminal_counts[f"{diagnostic_label}:holed"] += 1
                    terminated = True
                    break
                if score >= max_score:
                    self.terminal_counts[f"{diagnostic_label}:capped"] += 1
                    terminated = True
                    break
                lie = lie_group(next_lie)
                distance = max(next_distance, 0.0) if np.isfinite(next_distance) else distance
            if not terminated:
                self.terminal_counts[f"{diagnostic_label}:loop_limit"] += 1
            scores[sim] = min(score, max_score)
        return scores


def event_catalog(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            """
            SELECT year, event_id, MIN(round_date) AS start_date,
                   MAX(event_name) AS event_name, MAX(course_name) AS course_name,
                   MAX(course_num) AS course_num
            FROM player_rounds
            WHERE tour='pga'
            GROUP BY year, event_id
            """,
            conn,
        )
    finally:
        conn.close()


def qualifying_events(catalog: pd.DataFrame, target_key: str) -> tuple[pd.DataFrame, pd.Series]:
    year, event_id = parse_event_key(target_key)
    target_rows = catalog[(catalog["year"] == year) & (catalog["event_id"] == event_id)]
    if len(target_rows) != 1:
        raise ValueError(f"Target {target_key} was not uniquely found in player_rounds")
    target = target_rows.iloc[0]
    target_start = pd.Timestamp(target["start_date"])
    prior = catalog[
        catalog["course_num"].eq(target["course_num"])
        & pd.to_datetime(catalog["start_date"]).lt(target_start)
    ].copy()
    prior["event_key"] = "pga:R" + prior["year"].astype(str) + prior["event_id"].astype(str).str.zfill(3)
    return prior.sort_values("start_date"), target


def read_sg_year(export: Path, year: int, columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = export / f"pga_{year}_shot_sg.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def read_raw_year(export: Path, year: int, columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = export / f"pga_{year}_shots.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=list(columns) if columns else None)


SG_COLUMNS = [
    "event_key", "source_player_id", "player_name", "round_no", "hole_no",
    "stroke_no", "sg_category", "start_lie", "start_distance_yd", "end_lie",
    "end_distance_yd", "sg_raw", "sg_valid", "complete_hole", "shot_key",
]
RAW_COLUMNS = [
    "event_key", "source_player_id", "round_no", "hole_no", "stroke_no",
    "event_type", "start_lie", "end_lie", "distance_remaining_yd", "ball_holed",
    "shot_key",
]


def build_course_transitions(
    export: Path, prior_events: pd.DataFrame, target_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = set(prior_events["event_key"].astype(str))
    years = sorted(set(prior_events["year"].astype(int)) | {parse_event_key(target_key)[0]})
    transition_parts: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    target_raw = pd.DataFrame()
    for year in years:
        raw = read_raw_year(export, year, RAW_COLUMNS)
        if raw.empty:
            continue
        if year == parse_event_key(target_key)[0]:
            target_piece = raw[raw["event_key"].eq(target_key)].copy()
            if not target_piece.empty:
                target_raw = target_piece
        raw = raw[raw["event_key"].isin(keys)].copy()
        if raw.empty:
            continue
        sg = read_sg_year(export, year, SG_COLUMNS)
        sg = sg[sg["event_key"].isin(keys)].copy()
        stroke = raw[raw["event_type"].eq("STROKE")].copy()
        stroke = stroke.merge(
            sg[["shot_key", "sg_category", "start_distance_yd", "sg_raw", "sg_valid"]],
            on="shot_key", how="left", validate="one_to_one",
        )
        group_cols = ["event_key", "source_player_id", "round_no", "hole_no"]
        stroke = stroke.sort_values(group_cols + ["stroke_no"])
        # Incomplete/withdrawal sequences must never become artificial
        # hole-outs simply because they have no following shot.
        complete_keys = raw.groupby(group_cols, observed=True)["ball_holed"].max()
        complete_keys = complete_keys[complete_keys.eq(1)].reset_index()[group_cols]
        stroke = stroke.merge(complete_keys.assign(_complete=1), on=group_cols, how="inner")
        stroke["next_stroke_no"] = stroke.groupby(group_cols, observed=True)["stroke_no"].shift(-1)
        stroke["next_lie"] = stroke.groupby(group_cols, observed=True)["start_lie"].shift(-1)
        stroke["next_distance"] = stroke.groupby(group_cols, observed=True)["start_distance_yd"].shift(-1)
        stroke["holed"] = stroke["ball_holed"].eq(1)
        stroke["cost"] = np.where(
            stroke["holed"], 1,
            pd.to_numeric(stroke["next_stroke_no"], errors="coerce") - stroke["stroke_no"],
        )
        stroke["cost"] = pd.to_numeric(stroke["cost"], errors="coerce").fillna(1).clip(1, 4).astype(int)
        stroke["next_lie"] = np.where(stroke["holed"], "hole", stroke["next_lie"])
        stroke["next_distance"] = np.where(stroke["holed"], 0.0, stroke["next_distance"])
        stroke["category"] = stroke["sg_category"].astype(str).str.lower()
        stroke["lie"] = stroke["start_lie"].map(lie_group)
        stroke["start_distance"] = pd.to_numeric(stroke["start_distance_yd"], errors="coerce")
        stroke["sg"] = pd.to_numeric(stroke["sg_raw"], errors="coerce")
        stroke["sg_weight_valid"] = stroke["sg_valid"].eq(1) & stroke["sg"].notna()
        stroke["bucket"] = stroke["start_distance"].map(distance_bucket)
        valid = (
            stroke["category"].isin(CATEGORIES)
            & stroke["lie"].ne("unknown")
            & stroke["start_distance"].notna()
            & stroke["next_distance"].notna()
        )
        transition_parts.append(stroke.loc[valid, [
            "event_key", "source_player_id", "round_no", "hole_no", "category",
            "lie", "bucket", "start_distance", "next_lie", "next_distance",
            "sg", "cost", "holed",
            "sg_weight_valid",
        ]])
        complete = raw.groupby(group_cols, observed=True).agg(
            score=("stroke_no", "max"), holed=("ball_holed", "max")
        ).reset_index()
        score_parts.append(complete[complete["holed"].eq(1)])
    if not transition_parts or target_raw.empty:
        raise ValueError("Course history or target raw shots were unavailable")
    transitions = pd.concat(transition_parts, ignore_index=True)
    transitions["sg"] = transitions.groupby("category", observed=True)["sg"].transform(
        lambda x: x.fillna(x.median())
    )
    return transitions, pd.concat(score_parts, ignore_index=True), target_raw


def load_tour_history(
    export: Path,
    target_start: pd.Timestamp,
    catalog: pd.DataFrame,
    target_key: str,
    cap_per_cell: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[int, float, float]]]:
    eligible = catalog[pd.to_datetime(catalog["start_date"]).lt(target_start)].copy()
    eligible["event_key"] = "pga:R" + eligible["year"].astype(str) + eligible["event_id"].astype(str).str.zfill(3)
    eligible_keys = set(eligible["event_key"].astype(str))
    parts: list[pd.DataFrame] = []
    target_players: set[str] = set()
    target_year = parse_event_key(target_key)[0]
    target_sg = read_sg_year(export, target_year, SG_COLUMNS)
    target_sg = target_sg[target_sg["event_key"].eq(target_key)].copy()
    target_players.update(target_sg["source_player_id"].astype(str))
    player_parts: list[pd.DataFrame] = []
    population_moments: dict[str, list[float]] = {
        category: [0.0, 0.0, 0.0] for category in CATEGORIES
    }
    for year in sorted(eligible["year"].astype(int).unique()):
        sg = read_sg_year(export, int(year), SG_COLUMNS)
        if sg.empty:
            continue
        sg = sg[sg["event_key"].isin(eligible_keys) & sg["sg_category"].isin(CATEGORIES)].copy()
        sg["source_player_id"] = sg["source_player_id"].astype(str)
        population_valid = sg[
            sg["sg_valid"].eq(1) & pd.to_numeric(sg["sg_raw"], errors="coerce").notna()
        ].copy()
        population_valid["_sg"] = pd.to_numeric(
            population_valid["sg_raw"], errors="coerce"
        ).clip(-3.0, 3.0)
        for category, group in population_valid.groupby("sg_category", observed=True):
            values = group["_sg"].to_numpy(float)
            moments = population_moments[str(category)]
            moments[0] += len(values)
            moments[1] += float(values.sum())
            moments[2] += float(values @ values)
        player_parts.append(sg[sg["source_player_id"].isin(target_players)][[
            "event_key", "source_player_id", "round_no", "sg_category", "sg_raw", "sg_valid"
        ]])
        donor = sg[
            sg["start_distance_yd"].notna()
            & sg["end_distance_yd"].notna()
            & sg["start_lie"].notna()
        ].copy()
        donor["category"] = donor["sg_category"].astype(str).str.lower()
        donor["lie"] = donor["start_lie"].map(lie_group)
        donor["start_distance"] = pd.to_numeric(donor["start_distance_yd"], errors="coerce")
        donor["next_distance"] = pd.to_numeric(donor["end_distance_yd"], errors="coerce").fillna(0.0)
        donor["next_lie"] = donor["end_lie"].map(lie_group)
        donor["holed"] = donor["next_lie"].eq("hole")
        donor["cost"] = 1
        donor["sg"] = pd.to_numeric(donor["sg_raw"], errors="coerce")
        donor["sg_weight_valid"] = donor["sg_valid"].eq(1) & donor["sg"].notna()
        donor["bucket"] = donor["start_distance"].map(distance_bucket)
        parts.append(donor[[
            "category", "lie", "bucket", "start_distance", "next_lie",
            "next_distance", "sg", "cost", "holed",
            "sg_weight_valid",
        ]])
    tour = pd.concat(parts, ignore_index=True)
    tour = tour[tour["lie"].ne("unknown") & tour["category"].isin(CATEGORIES)].copy()
    tour["sg"] = tour.groupby("category", observed=True)["sg"].transform(lambda x: x.fillna(x.median()))
    # Deterministic capped reservoir per state cell keeps nearest-neighbour
    # lookup tractable while retaining the full exact-course history.
    sampled = []
    for _, group in tour.groupby(["category", "lie", "bucket"], observed=True, sort=False):
        sampled.append(group.sample(min(len(group), cap_per_cell), random_state=seed))
    frozen_moments = {
        category: (int(values[0]), float(values[1]), float(values[2]))
        for category, values in population_moments.items()
    }
    return (
        pd.concat(sampled, ignore_index=True), pd.concat(player_parts, ignore_index=True),
        target_sg, frozen_moments,
    )


def build_player_priors(
    player_sg: pd.DataFrame,
    population_moments: Mapping[str, tuple[int, float, float]],
) -> tuple[dict[str, dict[str, PlayerCategoryPrior]], dict[str, float]]:
    pop_values = []
    for category, (count, total, sumsq) in population_moments.items():
        mean = total / count
        variance = max((sumsq - count * mean * mean) / max(count - 1, 1), 0.04)
        pop_values.append({"category": category, "mean": mean, "std": math.sqrt(variance), "count": count})
    pop = pd.DataFrame(pop_values).set_index("category")
    priors: dict[str, dict[str, PlayerCategoryPrior]] = defaultdict(dict)
    valid = player_sg[
        player_sg["sg_category"].isin(CATEGORIES)
        & pd.to_numeric(player_sg["sg_raw"], errors="coerce").notna()
        & player_sg["sg_valid"].eq(1)
    ].copy()
    valid["sg"] = pd.to_numeric(valid["sg_raw"], errors="coerce").clip(-3.0, 3.0)
    for (player, category), group in valid.groupby(["source_player_id", "sg_category"], observed=True):
        n = len(group)
        pmean = float(pop.loc[category, "mean"])
        psd = max(float(pop.loc[category, "std"]), 0.2)
        raw_mean = float(group["sg"].mean())
        raw_var = float(group["sg"].var(ddof=1)) if n > 1 else psd * psd
        mean_weight = n / (n + 100.0)
        var_weight = n / (n + 200.0)
        mean = pmean + mean_weight * (raw_mean - pmean)
        variance = psd * psd + var_weight * (raw_var - psd * psd)
        priors[str(player)][str(category)] = PlayerCategoryPrior(
            mean=mean, sd=max(math.sqrt(max(variance, 0.04)), 0.2),
            pop_mean=pmean, pop_sd=psd, n=n,
        )
    round_sg = valid.groupby(
        ["source_player_id", "event_key", "round_no"], observed=True
    )["sg"].sum().reset_index()
    skills = {}
    for player, group in round_sg.groupby("source_player_id", observed=True):
        n = len(group)
        skills[str(player)] = float(group["sg"].mean()) * n / (n + 50.0)
    return dict(priors), skills


def target_setups(target_sg: pd.DataFrame) -> dict[tuple[int, int], HoleSetup]:
    first = target_sg[target_sg["stroke_no"].eq(1)].copy()
    setups: dict[tuple[int, int], HoleSetup] = {}
    for (round_no, hole_no), group in first.groupby(["round_no", "hole_no"], observed=True):
        distance = float(pd.to_numeric(group["start_distance_yd"], errors="coerce").median())
        categories = group["sg_category"].dropna().astype(str).str.lower()
        category = str(categories.mode().iloc[0]) if not categories.empty else "ott"
        setups[(int(round_no), int(hole_no))] = HoleSetup(
            int(round_no), int(hole_no), distance, category
        )
    return setups


def prior_setups(course_transitions: pd.DataFrame) -> dict[tuple[int, int], HoleSetup]:
    """Estimate round-specific tee states without reading the held-out edition."""
    first = course_transitions[
        course_transitions["lie"].eq("tee")
        & course_transitions["category"].isin(["ott", "app"])
    ].copy()
    # A tee state may reappear after stroke-and-distance, but it returns to the
    # same setup; the median therefore remains a robust setup estimate.
    setups: dict[tuple[int, int], HoleSetup] = {}
    for (round_no, hole_no), group in first.groupby(["round_no", "hole_no"], observed=True):
        distance = float(group["start_distance"].median())
        category = str(group["category"].mode().iloc[0])
        setups[(int(round_no), int(hole_no))] = HoleSetup(
            int(round_no), int(hole_no), distance, category
        )
    return setups


def actual_target_scores(target_raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["source_player_id", "round_no", "hole_no"]
    target_raw = target_raw.copy()
    target_raw["source_player_id"] = target_raw["source_player_id"].astype(str)
    actual = target_raw.groupby(cols, observed=True).agg(
        actual=("stroke_no", "max"), holed=("ball_holed", "max")
    ).reset_index()
    return actual[actual["holed"].eq(1)].drop(columns="holed")


def empirical_hole_samples(
    course_scores: pd.DataFrame,
    round_no: int,
    hole_no: int,
    simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = course_scores.loc[
        course_scores["round_no"].eq(round_no) & course_scores["hole_no"].eq(hole_no),
        "score",
    ].to_numpy(int)
    if not len(values):
        values = course_scores.loc[course_scores["hole_no"].eq(hole_no), "score"].to_numpy(int)
    if not len(values):
        values = course_scores["score"].to_numpy(int)
    return rng.choice(values, size=simulations, replace=True)


def normal_round_samples(
    rng: np.random.Generator,
    simulations: int,
    course_scores: pd.DataFrame,
    round_no: int,
    skill: float,
) -> np.ndarray:
    mean, sd = course_round_parameters(course_scores, round_no, skill)
    return rng.normal(mean, sd, size=simulations)


def course_round_parameters(
    course_scores: pd.DataFrame, round_no: int, skill: float
) -> tuple[float, float]:
    per_round = course_scores.groupby(
        ["event_key", "source_player_id", "round_no"], observed=True
    )["score"].sum().reset_index()
    relevant = per_round.loc[per_round["round_no"].eq(round_no), "score"]
    if len(relevant) < 20:
        relevant = per_round["score"]
    mean = float(relevant.mean()) - skill
    sd = max(float(relevant.std(ddof=1)), 2.0)
    return mean, sd


def mean_anchor(samples: np.ndarray, target_mean: float) -> np.ndarray:
    """Move a simulated distribution to a strict-prior mean, preserving shape."""
    values = np.asarray(samples, dtype=float)
    return values - float(np.mean(values)) + float(target_mean)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    catalog = event_catalog(args.dg_db)
    prior_events, target = qualifying_events(catalog, args.target_event_key)
    if prior_events.empty:
        raise ValueError("Target course has no earlier physical-course editions")
    print(
        f"Target: {target['event_name']} ({args.target_event_key}), "
        f"course={target['course_name']}, prior editions={len(prior_events)}"
    )
    course_transitions, course_scores, target_raw = build_course_transitions(
        args.export, prior_events, args.target_event_key
    )
    target_start = pd.Timestamp(target["start_date"])
    tour, player_sg, target_sg, population_moments = load_tour_history(
        args.export, target_start, catalog, args.target_event_key,
        args.tour_cap_per_cell, args.seed,
    )
    priors, player_skills = build_player_priors(player_sg, population_moments)
    shape_priors = {
        player: {
            category: PlayerCategoryPrior(
                mean=prior.pop_mean,
                sd=prior.sd,
                pop_mean=prior.pop_mean,
                pop_sd=prior.pop_sd,
                n=prior.n,
            )
            for category, prior in categories.items()
        }
        for player, categories in priors.items()
    }
    model = MarkovHoleModel(course_transitions, tour, args.min_exact, args.knn)
    if args.setup_source == "prior":
        setups = prior_setups(course_transitions)
    else:
        setups = target_setups(target_sg)
    actual = actual_target_scores(target_raw)
    complete_counts = actual.groupby(["source_player_id", "round_no"], observed=True)["hole_no"].nunique()
    complete = complete_counts[complete_counts.eq(18)].index
    actual = actual.set_index(["source_player_id", "round_no"]).loc[complete].reset_index()
    player_rounds = actual[["source_player_id", "round_no"]].drop_duplicates().sort_values(
        ["round_no", "source_player_id"]
    )
    if args.max_player_rounds:
        player_rounds = player_rounds.head(args.max_player_rounds)
    rng = np.random.default_rng(args.seed)
    hole_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    for row_no, row in enumerate(player_rounds.itertuples(index=False), start=1):
        player = str(row.source_player_id)
        round_no = int(row.round_no)
        actual_round = actual[
            actual["source_player_id"].eq(player) & actual["round_no"].eq(round_no)
        ].sort_values("hole_no")
        method_holes: dict[str, list[np.ndarray]] = defaultdict(list)
        for hole_row in actual_round.itertuples(index=False):
            hole_no = int(hole_row.hole_no)
            setup = setups.get((round_no, hole_no))
            if setup is None:
                continue
            simulations = {
                "course_score_pmf": empirical_hole_samples(
                    course_scores, round_no, hole_no, args.sims, rng
                ),
                "markov_population": model.simulate_hole(
                    setup, rng, args.sims, None, args.max_score,
                    diagnostic_label="markov_population",
                ),
                "markov_player": model.simulate_hole(
                    setup, rng, args.sims, priors.get(player), args.max_score,
                    diagnostic_label="markov_player_raw",
                ),
                "markov_player_shape": model.simulate_hole(
                    setup, rng, args.sims, shape_priors.get(player), args.max_score,
                    diagnostic_label="markov_player_shape",
                ),
            }
            for method, values in simulations.items():
                method_holes[method].append(values)
                hole_rows.append({
                    "player_id": player, "round_no": round_no, "hole_no": hole_no,
                    "actual": int(hole_row.actual), "method": method,
                    "log_score": smoothed_log_score(values, int(hole_row.actual)),
                    "crps": crps_samples(values, int(hole_row.actual)),
                    "pred_mean": float(np.mean(values)), "pred_sd": float(np.std(values)),
                })
        actual_total = int(actual_round["actual"].sum())
        normal = normal_round_samples(
            rng, args.sims, course_scores, round_no, player_skills.get(player, 0.0)
        )
        strict_prior_mean, _ = course_round_parameters(
            course_scores, round_no, player_skills.get(player, 0.0)
        )
        round_rows.append({
            "player_id": player, "round_no": round_no, "actual": actual_total,
            "method": "normal_player_skill", "crps": crps_samples(normal, actual_total),
            "pred_mean": float(np.mean(normal)), "pred_sd": float(np.std(normal)),
        })
        for method, values in method_holes.items():
            if len(values) != 18:
                continue
            total_samples = np.sum(np.vstack(values), axis=0)
            round_rows.append({
                "player_id": player, "round_no": round_no, "actual": actual_total,
                "method": method, "crps": crps_samples(total_samples, actual_total),
                "pred_mean": float(np.mean(total_samples)),
                "pred_sd": float(np.std(total_samples)),
            })
            if method in {"markov_population", "markov_player_shape"}:
                anchored = mean_anchor(total_samples, strict_prior_mean)
                round_rows.append({
                    "player_id": player, "round_no": round_no, "actual": actual_total,
                    "method": f"{method}_mean_anchored",
                    "crps": crps_samples(anchored, actual_total),
                    "pred_mean": float(np.mean(anchored)),
                    "pred_sd": float(np.std(anchored)),
                })
        if row_no % 25 == 0 or row_no == len(player_rounds):
            print(f"Simulated {row_no}/{len(player_rounds)} player-rounds")
    holes = pd.DataFrame(hole_rows)
    rounds = pd.DataFrame(round_rows)
    hole_summary = holes.groupby("method", observed=True).agg(
        observations=("actual", "size"), log_score=("log_score", "mean"),
        crps=("crps", "mean"), mean_bias=("pred_mean", lambda x: float(
            np.mean(x.to_numpy() - holes.loc[x.index, "actual"].to_numpy())
        )),
    ).reset_index()
    round_summary = rounds.groupby("method", observed=True).agg(
        observations=("actual", "size"), crps=("crps", "mean"),
        mean_bias=("pred_mean", lambda x: float(
            np.mean(x.to_numpy() - rounds.loc[x.index, "actual"].to_numpy())
        )),
        pred_sd=("pred_sd", "mean"),
    ).reset_index()
    pivot = rounds.pivot_table(
        index=["player_id", "round_no"], columns="method", values="crps", aggfunc="first"
    )
    paired_rows = []
    bootstrap_rng = np.random.default_rng(args.seed + 991)
    comparisons = [
        ("markov_player_shape_mean_anchored", "markov_population_mean_anchored"),
        ("markov_population_mean_anchored", "normal_player_skill"),
    ]
    for method, baseline in comparisons:
        if method not in pivot or baseline not in pivot:
            continue
        delta = (pivot[method] - pivot[baseline]).dropna().to_numpy(float)
        boot = np.mean(
            bootstrap_rng.choice(delta, size=(5000, len(delta)), replace=True), axis=1
        )
        paired_rows.append({
            "method": method, "baseline": baseline, "observations": len(delta),
            "crps_delta": float(np.mean(delta)),
            "ci_2p5": float(np.quantile(boot, 0.025)),
            "ci_97p5": float(np.quantile(boot, 0.975)),
        })
    paired = pd.DataFrame(paired_rows)
    total_backoffs = sum(model.backoff_counts.values())
    backoff_rates = {
        key: value / total_backoffs for key, value in model.backoff_counts.items()
    }
    terminal_by_method: dict[str, Counter[str]] = defaultdict(Counter)
    for key, value in model.terminal_counts.items():
        method, outcome = key.split(":", 1)
        terminal_by_method[method][outcome] += value
    terminal_rates = {
        method: {
            outcome: count / sum(counts.values())
            for outcome, count in counts.items()
        }
        for method, counts in terminal_by_method.items()
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    holes.to_csv(output / "hole_predictions.csv", index=False)
    rounds.to_csv(output / "round_predictions.csv", index=False)
    hole_summary.to_csv(output / "hole_summary.csv", index=False)
    round_summary.to_csv(output / "round_summary.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    diagnostics = {
        "target_event_key": args.target_event_key,
        "target_event": str(target["event_name"]),
        "target_course": str(target["course_name"]),
        "target_course_num": int(target["course_num"]),
        "prior_editions": prior_events[["year", "event_id", "event_name", "start_date"]].to_dict("records"),
        "archived_training_editions": int(course_transitions["event_key"].nunique()),
        "course_transition_rows": len(course_transitions),
        "tour_transition_rows_capped": len(tour),
        "target_player_priors": len(priors),
        "evaluated_player_rounds": len(player_rounds),
        "evaluated_holes": int(len(holes) / 4),
        "simulations": args.sims,
        "setup_source": args.setup_source,
        "heldout_setup_geometry_used": args.setup_source == "heldout",
        "backoff_counts": dict(model.backoff_counts),
        "backoff_rates": backoff_rates,
        "terminal_counts": dict(model.terminal_counts),
        "terminal_rates": terminal_rates,
        "leakage_policy": "all transition and player-profile rows have event start strictly before target start",
        "penalty_policy": "exact-course physical shot plus following penalty/drop is collapsed; cost=next physical stroke_no-current stroke_no",
        "mean_anchor_policy": "mean-anchored arms preserve simulated shape but use course round mean minus strictly-prior shrunk player shot-SG/round",
    }
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    report_lines = [
        "# Recursive hole simulator - held-out edition",
        "",
        f"Target: **{target['event_name']}** ({args.target_event_key}) at {target['course_name']}.",
        f"The catalog contains {len(prior_events)} earlier physical-course editions; {course_transitions['event_key'].nunique()} have R2 shots in the 2020+ archive. Training used {len(course_transitions):,} exact-course transitions and {len(tour):,} capped Tour fallback transitions.",
        f"Setup source: **{args.setup_source}**. " + ("Held-out first-shot distance/category was used only to define the known setup; no held-out transition entered training." if args.setup_source == "heldout" else "Round/hole starting distance and category were estimated entirely from prior editions; no held-out shot informed prediction."),
        "",
        "## Hole scoring",
        "",
        markdown_table(hole_summary),
        "",
        "## Round scoring",
        "",
        markdown_table(round_summary),
        "",
        "## Paired round CRPS deltas",
        "",
        "Negative deltas favor the first method. Intervals are player-round bootstrap percentiles.",
        "",
        markdown_table(paired),
        "",
        "## Transition diagnostics",
        "",
        f"Backoff rates: `{json.dumps(backoff_rates, sort_keys=True)}`",
        f"Terminal rates by arm: `{json.dumps(terminal_rates, sort_keys=True)}`",
        "",
        "## Interpretation limits",
        "",
        "- This is a leave-one-edition-out predictive test, not a production integration.",
        "- Exact-hole setup is approximated by stable hole number and round number; tee/pin geometry is not yet aligned across editions.",
        "- Course transitions preserve observed endpoint lie and continuous remaining distance, but resampling does not create a new XY endpoint or explicit club/strategy choice.",
        "- Penalties are explicit in exact-course transitions. Tour fallback rows come from shot-SG state pairs and carry no extra penalty cost; fallback usage is reported.",
        "- Hole draws are conditionally independent when combined into round scores, so shared weather/form correlation is absent.",
        "- Player modifiers are aggressively shrunk category-level SG likelihood ratios, not yardage-by-lie player effects yet.",
        "- `markov_player` is deliberately retained as an uncalibrated diagnostic: applying its mean edge independently on every transition compounds skill. `markov_player_shape` varies only category dispersion. The `_mean_anchored` round arms are the decision-relevant comparison.",
        "- Player category SG priors are unconditional within category while donors are state-conditional; mean anchoring prevents compounding, but a future model should estimate hierarchical distance-by-lie conditional residuals directly.",
        "- This is a distance/lie Markov prototype, not yet the proposed geometric tee model: it does not transport XY landing distributions for player distance and lateral-dispersion traits.",
        "- There is no shared player-week/weather latent, and this study does not replay the current category-first production simulator as a benchmark.",
    ]
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("\nHole summary")
    print(hole_summary.to_string(index=False))
    print("\nRound summary")
    print(round_summary.to_string(index=False))
    print(f"\nWrote {output}")
    return diagnostics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--target-event-key", default=DEFAULT_EVENT)
    result.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    result.add_argument("--dg-db", type=Path, default=DEFAULT_DB)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--sims", type=int, default=400)
    result.add_argument("--max-player-rounds", type=int, default=0)
    result.add_argument("--tour-cap-per-cell", type=int, default=5000)
    result.add_argument("--min-exact", type=int, default=16)
    result.add_argument("--knn", type=int, default=64)
    result.add_argument("--max-score", type=int, default=12)
    result.add_argument(
        "--setup-source", choices=["prior", "heldout"], default="prior",
        help="prior is the strict leakage-free setup; heldout is a labeled setup sensitivity",
    )
    result.add_argument("--seed", type=int, default=20260820)
    result.add_argument("--smoke", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.smoke:
        args.sims = min(args.sims, 60)
        args.max_player_rounds = args.max_player_rounds or 8
        args.tour_cap_per_cell = min(args.tour_cap_per_cell, 1000)
        args.output = args.output.parent / f"{args.output.name}_smoke"
    if args.sims < 20:
        raise ValueError("Use at least 20 simulations")
    evaluate(args)


if __name__ == "__main__":
    main()

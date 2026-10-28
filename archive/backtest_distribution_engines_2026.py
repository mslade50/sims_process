"""Full-season 2026 comparison of pre-event golf distribution engines.

All arms share the same dated, leakage-safe ``model_results2.skill_est``.  The
study changes only the one-round distribution wrapped around that mean:

* ``category_current``: production category-first EMA distributions.
* ``category_shot``: category-first plus the frozen full-strength shot overlay.
* ``normal_2p8``: a pure Gaussian with the production 2.8 SG/round constant.
* ``normal_player``: a Gaussian using the legacy ``(dated std_dev + 2.8) / 2``.

Category and shot histories are capped strictly before each event.  The main
round test grades every available round using the pre-event distribution; the
R1-only view is the clean operational matchup test.  A static four-round
tournament sensitivity is also produced.  It intentionally omits live skill
updates and historical weather, which are common mean effects rather than
distribution-engine inputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ARCHIVE))

from build_bmw_shot_features_shadow import (  # noqa: E402
    _load_name_replacements,
    build_dispersion_features,
    load_event_dates,
    load_shot_round_stats,
)
from backtest_shot_dispersion_recent import (  # noqa: E402
    BASELINE_SKEW,
    CAT_ORDER,
    EPS,
    SHOT_CAT,
    WEIGHTS,
    markdown_table,
    reconstruct_course_shape,
)


DB_CAT = {
    "sg_ott": "sg_ott_adj",
    "sg_app": "sg_app_adj",
    "sg_arg": "sg_arg_adj",
    "sg_putt": "sg_putt_adj",
}
ARMS = ("category_current", "category_shot", "normal_2p8", "normal_player")
EMA_SPAN = 50
MIN_CAT_OBS = 12
NORMAL_SD = 2.8
CLIP_CAT = (-8.0, 8.0)


@dataclass(frozen=True)
class Event:
    event_id: int
    event_name: str
    start: pd.Timestamp
    end: pd.Timestamp
    course_id: int
    course_count: int
    field_size: int

    @property
    def key(self) -> str:
        return f"{self.start.date()}|{self.event_id}|{self.event_name}"


def normalise_name(value: object, replacements: dict[str, str]) -> str:
    key = str(value).casefold().strip()
    return replacements.get(key, key)


def read_database(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Event]]:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rounds = pd.read_sql_query(
            "SELECT player_name, dg_id, event_id, event_name, year, round_num, "
            "round_date, course_num, score, sg_total_adj, finish_pos, "
            "sg_ott_adj, sg_app_adj, sg_arg_adj, sg_putt_adj "
            "FROM player_rounds WHERE tour='pga' AND sg_total_adj IS NOT NULL "
            "AND round_date >= '2019-01-01' AND round_date < '2026-08-20'",
            connection,
        )
        model = pd.read_sql_query(
            "SELECT rowid, dg_id, Date, skill_est, std_dev FROM model_results2 "
            "WHERE skill_est IS NOT NULL AND dg_id IS NOT NULL "
            "AND Date >= '2025-01-01' AND Date < '2026-08-20'",
            connection,
        )

    rounds["round_date"] = pd.to_datetime(rounds["round_date"]).dt.normalize()
    for column in [*DB_CAT.values(), "score", "sg_total_adj", "finish_pos"]:
        rounds[column] = pd.to_numeric(rounds[column], errors="coerce")
    rounds["dg_id"] = pd.to_numeric(rounds["dg_id"], errors="coerce").astype("Int64")
    model["Date"] = pd.to_datetime(model["Date"]).dt.normalize()
    model["dg_id"] = pd.to_numeric(model["dg_id"], errors="coerce").astype("Int64")
    model = (
        model.sort_values(["Date", "dg_id", "rowid"])
        .drop_duplicates(["Date", "dg_id"], keep="last")
        .drop(columns="rowid")
    )

    season = rounds[(rounds["year"] == 2026) & rounds["round_num"].eq(1)].copy()
    event_rows = []
    for (event_id, event_name), group in season.groupby(["event_id", "event_name"], sort=False):
        all_event = rounds[
            (rounds["year"] == 2026)
            & rounds["event_id"].eq(event_id)
            & rounds["event_name"].eq(event_name)
        ]
        counts = group["course_num"].dropna().astype(int).value_counts()
        event_rows.append(
            Event(
                event_id=int(event_id),
                event_name=str(event_name),
                start=group["round_date"].min(),
                end=all_event["round_date"].max(),
                course_id=int(counts.index[0]) if len(counts) else -1,
                course_count=int(group["course_num"].nunique()),
                field_size=int(group["dg_id"].nunique()),
            )
        )
    events = sorted(event_rows, key=lambda event: (event.start, event.event_name))
    return rounds, model, rounds[rounds["year"].eq(2026)].copy(), events


def attach_dated_skill(field: pd.DataFrame, model: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    query = field[["dg_id", "player_name"]].drop_duplicates("dg_id").copy()
    query["cutoff"] = cutoff
    merged = pd.merge_asof(
        query.sort_values(["cutoff", "dg_id"]),
        model.sort_values(["Date", "dg_id"]),
        left_on="cutoff",
        right_on="Date",
        by="dg_id",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.dropna(subset=["skill_est", "std_dev"]).reset_index(drop=True)


def ema_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.clip(np.asarray(values, float), CLIP_CAT[0], CLIP_CAT[1])
    n = len(values)
    alpha = 2.0 / (EMA_SPAN + 1.0)
    weights = (1.0 - alpha) ** np.arange(n - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    mean = float(weights @ values)
    centered = values - mean
    variance = float(weights @ (centered**2))
    std = math.sqrt(max(variance, 1e-12))
    skew = float(weights @ (centered**3) / std**3) if std > 1e-8 else 0.0
    n_eff = float(1.0 / np.sum(weights**2))
    return mean, std, skew, n_eff


def build_category_params(
    histories: dict[int, pd.DataFrame],
    field: pd.DataFrame,
    cutoff: pd.Timestamp,
    features: pd.DataFrame,
    mults: dict[str, float],
    course_skew: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rows: list[dict[str, float | int]] = []
    for dg_id in field["dg_id"].astype(int):
        history = histories.get(dg_id)
        if history is None:
            history = pd.DataFrame()
        else:
            history = history[history["round_date"].lt(cutoff)]
        row: dict[str, float | int] = {"dg_id": dg_id}
        for cat, source in DB_CAT.items():
            values = history[source].dropna().to_numpy(float) if source in history else np.array([])
            if len(values):
                mean, std, skew, n_eff = ema_stats(values)
                row.update({f"{cat}_mean": mean, f"{cat}_std": std, f"{cat}_skew": skew, f"{cat}_neff": n_eff, f"{cat}_n": len(values)})
            else:
                row.update({f"{cat}_mean": np.nan, f"{cat}_std": np.nan, f"{cat}_skew": np.nan, f"{cat}_neff": 0.0, f"{cat}_n": len(values)})
        rows.append(row)
    dist = pd.DataFrame(rows).set_index("dg_id")

    # Exact histories cover nearly every established player.  The field prior
    # only fills the residual qualifier cells and never creates shot evidence.
    for cat in CAT_ORDER:
        mean_prior = float(dist[f"{cat}_mean"].mean())
        std_prior = float(dist[f"{cat}_std"].median())
        skew_prior = float(dist[f"{cat}_skew"].median())
        if not np.isfinite(mean_prior) or not np.isfinite(std_prior):
            raise ValueError(f"No established historical category prior for {cat} at {cutoff.date()}")
        skew_prior = skew_prior if np.isfinite(skew_prior) else 0.0
        low = dist[f"{cat}_n"].lt(MIN_CAT_OBS)
        reliability = (dist[f"{cat}_n"].astype(float) / MIN_CAT_OBS).clip(0.0, 1.0)
        observed_mean = dist[f"{cat}_mean"].fillna(mean_prior)
        observed_std = dist[f"{cat}_std"].fillna(std_prior)
        blended_mean = reliability * observed_mean + (1.0 - reliability) * mean_prior
        blended_var = (
            reliability * (observed_std**2 + (observed_mean - blended_mean) ** 2)
            + (1.0 - reliability) * (std_prior**2 + (mean_prior - blended_mean) ** 2)
        )
        dist.loc[low, f"{cat}_mean"] = blended_mean[low]
        dist.loc[low, f"{cat}_std"] = np.sqrt(blended_var[low].clip(lower=1e-12))
        dist.loc[low, f"{cat}_skew"] = (
            reliability[low] * dist.loc[low, f"{cat}_skew"].fillna(skew_prior)
            + (1.0 - reliability[low]) * skew_prior
        )

    ordered_ids = field["dg_id"].astype(int).tolist()
    dist = dist.loc[ordered_ids]
    mu = dist[[f"{cat}_mean" for cat in CAT_ORDER]].to_numpy(float)
    skill = field["skill_est"].to_numpy(float)
    mu += ((skill - mu.sum(axis=1)) / 4.0)[:, None]

    base_var: dict[str, np.ndarray] = {}
    shot_var: dict[str, np.ndarray] = {}
    feature_map = features.copy()
    feature_map["source_player_id"] = feature_map["source_player_id"].astype(str)
    feature_map = feature_map.set_index("source_player_id")
    coverage, scales = {}, {}
    for cat in CAT_ORDER:
        base = dist[f"{cat}_std"].to_numpy(float) ** 2
        feature_col = f"{SHOT_CAT[cat]}_shot_indep_var50_shrunk"
        shot = np.asarray([
            feature_map.at[str(source_id), feature_col]
            if pd.notna(source_id) and str(source_id) in feature_map.index and feature_col in feature_map.columns
            else np.nan
            for source_id in field["source_player_id"]
        ], float)
        valid = np.isfinite(shot) & (shot > 0)
        scale = float(base[valid].mean() / shot[valid].mean()) if valid.any() else 1.0
        base_var[cat], shot_var[cat] = base, shot
        coverage[cat], scales[cat] = int(valid.sum()), scale

    std_current, std_shot, effective_skew = [], [], []
    for row_index in range(len(field)):
        row_current, row_shot, row_skew = [], [], []
        for cat in CAT_ORDER:
            old_var = base_var[cat][row_index]
            candidate = shot_var[cat][row_index]
            if np.isfinite(candidate) and candidate > 0:
                new_var = (1.0 - WEIGHTS[cat]) * old_var + WEIGHTS[cat] * candidate * scales[cat]
            else:
                new_var = old_var
            row_current.append(math.sqrt(old_var) * float(mults.get(cat, 1.0)))
            row_shot.append(math.sqrt(new_var) * float(mults.get(cat, 1.0)))
            neff = float(dist.iloc[row_index][f"{cat}_neff"])
            player_skew = float(dist.iloc[row_index][f"{cat}_skew"])
            blend = 0.5 * min(neff / 100.0, 1.0)
            row_skew.append((1.0 - blend) * float(course_skew.get(cat, BASELINE_SKEW[cat])) + blend * player_skew)
        std_current.append(row_current)
        std_shot.append(row_shot)
        effective_skew.append(row_skew)

    return {
        "mu": np.ascontiguousarray(mu),
        "std_current": np.ascontiguousarray(np.asarray(std_current, float)),
        "std_shot": np.ascontiguousarray(np.asarray(std_shot, float)),
        "skew": np.ascontiguousarray(np.asarray(effective_skew, float)),
    }, {
        "coverage": coverage,
        "scales": scales,
        "category_min_history_coverage": {
            cat: int((dist[f"{cat}_n"] >= MIN_CAT_OBS).sum()) for cat in CAT_ORDER
        },
    }


def load_correlation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, index_col=0)
    matrix = frame.reindex(index=CAT_ORDER, columns=CAT_ORDER).to_numpy(float)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        matrix = 0.95 * matrix + 0.05 * np.eye(4)
        chol = np.linalg.cholesky(matrix)
    return np.ascontiguousarray(matrix), np.ascontiguousarray(chol)


def draw_rounds(
    params: dict[str, np.ndarray],
    field: pd.DataFrame,
    chol: np.ndarray,
    sims: int,
    seed: int,
) -> dict[str, np.ndarray]:
    import sims_kernel

    n = len(field)
    skill = np.ascontiguousarray(field["skill_est"].to_numpy(float))
    zeros = np.zeros(n, float)
    expected = np.zeros(n, float)
    common = [params["mu"], params["skew"], chol, skill, zeros, expected, int(sims)]
    current, _ = sims_kernel.run_single_round(
        common[0], params["std_current"], *common[1:], int(seed)
    )
    shot, _ = sims_kernel.run_single_round(
        common[0], params["std_shot"], *common[1:], int(seed)
    )

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, sims))
    normal_2p8 = np.rint(-(skill[:, None] + NORMAL_SD * z)).clip(-12, 12).astype(np.int16)
    legacy_sd = (field["std_dev"].to_numpy(float) + NORMAL_SD) / 2.0
    normal_player = np.rint(-(skill[:, None] + legacy_sd[:, None] * z)).clip(-12, 12).astype(np.int16)
    return {
        "category_current": np.ascontiguousarray(current).astype(np.int16),
        "category_shot": np.ascontiguousarray(shot).astype(np.int16),
        "normal_2p8": normal_2p8,
        "normal_player": normal_player,
    }


def score_pmf(scores: np.ndarray) -> np.ndarray:
    bins = np.arange(-12, 13)
    pmf = np.zeros((scores.shape[0], len(bins)), float)
    for player in range(scores.shape[0]):
        pmf[player] = np.bincount(scores[player] + 12, minlength=len(bins))[: len(bins)]
    return pmf / scores.shape[1]


def pair_probabilities(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pmf = score_pmf(scores)
    n = pmf.shape[0]
    left, right = np.triu_indices(n, k=1)
    cdf = np.cumsum(pmf, axis=1)
    wins = np.sum(pmf[left] * (1.0 - cdf[right]), axis=1)
    ties = np.sum(pmf[left] * pmf[right], axis=1)
    p = np.divide(wins, 1.0 - ties, out=np.full(len(wins), 0.5), where=(1.0 - ties) > 1e-12)
    return left, right, p


def binary_sums(prob: np.ndarray, outcome: np.ndarray) -> tuple[float, float, int]:
    p = np.clip(np.asarray(prob, float), EPS, 1.0 - EPS)
    y = np.asarray(outcome, float)
    return (
        float(np.sum((p - y) ** 2)),
        float(np.sum(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))),
        int(len(y)),
    )


def sample_diagnostics(samples: np.ndarray, outcomes: np.ndarray, sims: int) -> dict[str, np.ndarray]:
    bins = np.arange(-12, 13, dtype=float)
    pmf = score_pmf(samples)
    first = np.sum(pmf * np.abs(bins[None, :] - outcomes[:, None]), axis=1)
    distances = np.abs(bins[:, None] - bins[None, :])
    second = 0.5 * np.einsum("ij,jk,ik->i", pmf, distances, pmf)
    cdf = np.cumsum(pmf, axis=1)
    rounded = np.rint(outcomes).clip(-12, 12).astype(int) + 12
    exact = pmf[np.arange(len(pmf)), rounded]
    smooth_pmf = (exact * sims + 0.5) / (sims + 12.5)
    pit_index = np.searchsorted(bins, outcomes, side="right") - 1
    pit_index = np.clip(pit_index, 0, len(bins) - 1)
    pit = cdf[np.arange(len(cdf)), pit_index]

    def quantile(probability: float) -> np.ndarray:
        return bins[np.argmax(cdf >= probability, axis=1)]

    return {
        "crps": first - second,
        "logpmf": -np.log(smooth_pmf),
        "pit": pit,
        "q05": quantile(0.05), "q10": quantile(0.10), "q25": quantile(0.25),
        "q75": quantile(0.75), "q90": quantile(0.90), "q95": quantile(0.95),
    }


def static_tournament_probs(rounds: list[np.ndarray], cut_size: int, seed: int) -> dict[str, np.ndarray]:
    n_players, sims = rounds[0].shape
    total = rounds[0].astype(np.int16) + rounds[1].astype(np.int16)
    if cut_size < n_players:
        keep = np.argpartition(total, cut_size - 1, axis=0)[:cut_size]
        made = np.zeros((n_players, sims), dtype=bool)
        made[keep, np.arange(sims)[None, :]] = True
    else:
        made = np.ones((n_players, sims), dtype=bool)
    total = total.astype(np.int16) + rounds[2].astype(np.int16) + rounds[3].astype(np.int16)
    ranked_total = total.astype(float)
    ranked_total[~made] = 100.0
    # Random tie breaks make each top-N set exactly N players while preserving
    # symmetry among tied integer tournament scores.
    jitter = np.random.default_rng(seed).uniform(-0.49, 0.49, size=ranked_total.shape)
    order = np.argsort(ranked_total + jitter, axis=0)
    output = {}
    for label, top_n in (("win", 1), ("top5", 5), ("top10", 10), ("top20", 20)):
        chosen = order[: min(top_n, n_players)]
        counts = np.bincount(chosen.ravel(), minlength=n_players)
        output[label] = counts / sims
    return output


def latest_finishes(event_rounds: pd.DataFrame, field: pd.DataFrame) -> np.ndarray:
    latest = (
        event_rounds.dropna(subset=["finish_pos"])
        .sort_values(["round_num", "round_date"])
        .drop_duplicates("dg_id", keep="last")
        .set_index("dg_id")["finish_pos"]
    )
    return field["dg_id"].map(latest).to_numpy(float)


def cluster_bootstrap_delta(
    frame: pd.DataFrame,
    arm: str,
    loss: str,
    draws: int = 10_000,
    seed: int = 20260820,
) -> tuple[float, float, float]:
    use = frame[frame["arm"].isin(["category_current", arm])].copy()
    pivot_sum = use.pivot_table(index="event_key", columns="arm", values=f"{loss}_sum", aggfunc="sum")
    pivot_n = use.pivot_table(index="event_key", columns="arm", values="n", aggfunc="sum")
    common = pivot_sum.dropna().index.intersection(pivot_n.dropna().index)
    pivot_sum, pivot_n = pivot_sum.loc[common], pivot_n.loc[common]
    if not len(common):
        return np.nan, np.nan, np.nan
    point = pivot_sum[arm].sum() / pivot_n[arm].sum() - pivot_sum["category_current"].sum() / pivot_n["category_current"].sum()
    rng = np.random.default_rng(seed)
    values = np.empty(draws, float)
    for draw in range(draws):
        idx = rng.integers(0, len(common), len(common))
        a_sum = pivot_sum[arm].to_numpy()[idx].sum()
        a_n = pivot_n[arm].to_numpy()[idx].sum()
        b_sum = pivot_sum["category_current"].to_numpy()[idx].sum()
        b_n = pivot_n["category_current"].to_numpy()[idx].sum()
        values[draw] = a_sum / a_n - b_sum / b_n
    low, high = np.quantile(values, [0.025, 0.975])
    return float(point), float(low), float(high)


def source_crosswalk(export: Path, replacements: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_parquet(
        export / "pga_2026_player_rounds.parquet",
        columns=["source_player_id", "player_name"],
    )
    frame["player_key"] = frame["player_name"].map(lambda value: normalise_name(value, replacements))
    frame["source_player_id"] = frame["source_player_id"].astype(str)
    return frame.sort_values("source_player_id").drop_duplicates("player_key")[["player_key", "source_player_id"]]


def event_field(
    event: Event,
    season_rounds: pd.DataFrame,
    model: pd.DataFrame,
    crosswalk: pd.DataFrame,
    replacements: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = season_rounds[
        season_rounds["event_id"].eq(event.event_id)
        & season_rounds["event_name"].eq(event.event_name)
    ].copy()
    r1 = actual[actual["round_num"].eq(1)][["dg_id", "player_name"]].drop_duplicates("dg_id")
    field = attach_dated_skill(r1, model, event.start)
    field["player_key"] = field["player_name"].map(lambda value: normalise_name(value, replacements))
    field = field.merge(crosswalk, on="player_key", how="left", validate="one_to_one")
    field = field.sort_values("player_key").reset_index(drop=True)
    return field, actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"))
    parser.add_argument("--export", type=Path, default=Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"))
    parser.add_argument("--sims", type=int, default=30_000)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-20")
    parser.add_argument("--event-limit", type=int)
    parser.add_argument(
        "--event-name",
        action="append",
        help="Optional exact event-name filter; repeat for multiple events.",
    )
    parser.add_argument(
        "--save-case-pmfs",
        action="store_true",
        help="Persist player/round PMFs and independently frozen outcomes/skills.",
    )
    parser.add_argument("--no-course-shape", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO / "permanent_data" / "distribution_engine_backtest_2026")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rounds, model, season_rounds, events = read_database(args.db)
    events = [event for event in events if pd.Timestamp(args.start) <= event.start < pd.Timestamp(args.end)]
    if args.event_name:
        requested = {value.casefold() for value in args.event_name}
        events = [event for event in events if event.event_name.casefold() in requested]
    if args.event_limit:
        events = events[: args.event_limit]
    if not events:
        raise ValueError("No events selected")
    replacements = _load_name_replacements(REPO / "sim_inputs.py")
    crosswalk = source_crosswalk(args.export, replacements)
    histories = {
        int(dg_id): group.sort_values("round_date").copy()
        for dg_id, group in rounds.groupby("dg_id", sort=False)
        if pd.notna(dg_id)
    }
    corr, chol = load_correlation(REPO / "permanent_data" / "sg_cat_corr_tour_within_player_pearson.csv")

    prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    union_source_ids: set[str] = set()
    for event in events:
        field, actual = event_field(event, season_rounds, model, crosswalk, replacements)
        prepared[event.key] = (field, actual)
        union_source_ids.update(field["source_player_id"].dropna().astype(str))
    event_dates = load_event_dates(args.db)
    shot_stats, valid_shots = load_shot_round_stats(
        args.export,
        sorted(union_source_ids),
        event_dates,
        max(event.start for event in events) + pd.Timedelta(days=1),
    )

    matchup_rows: list[dict[str, object]] = []
    round_score_rows: list[dict[str, object]] = []
    tournament_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    case_pmf_rows: list[dict[str, object]] = []

    for event_index, event in enumerate(events):
        print(f"\n[{event_index + 1}/{len(events)}] {event.event_name} ({event.start.date()})", flush=True)
        field, actual = prepared[event.key]
        shot_input = shot_stats[shot_stats["event_date"].lt(event.start)].copy()
        feature_ids = (
            field[["source_player_id"]]
            .dropna(subset=["source_player_id"])
            .drop_duplicates("source_player_id")
        )
        features, _ = build_dispersion_features(shot_input, feature_ids)
        if args.no_course_shape or event.course_id < 0:
            mults = {cat: 1.0 for cat in CAT_ORDER}
            skews = BASELINE_SKEW.copy()
            course_meta = {"history_rows": 0, "years": []}
        else:
            mults, skews, course_meta = reconstruct_course_shape(args.db, event.course_id, event.start)
        params, diagnostics = build_category_params(histories, field, event.start, features, mults, skews)

        round_draws = {arm: [] for arm in ARMS}
        for round_index in range(4):
            draws = draw_rounds(params, field, chol, args.sims, 7100 + event_index * 10 + round_index)
            for arm in ARMS:
                round_draws[arm].append(draws[arm])

        # Pre-event round matchup prices: R1 is the operational primary; all
        # rounds provide repeated outcome sensitivity using the same frozen prior.
        left, right, base_prob = pair_probabilities(round_draws["category_current"][0])
        arm_prob = {"category_current": base_prob}
        for arm in ARMS[1:]:
            _, _, arm_prob[arm] = pair_probabilities(round_draws[arm][0])
        market_like = base_prob >= 0.35
        market_like &= base_prob <= 0.65

        field_index = {int(value): i for i, value in enumerate(field["dg_id"])}
        for round_num in range(1, 5):
            actual_round = actual[actual["round_num"].eq(round_num)].drop_duplicates("dg_id")
            actual_scores = np.full(len(field), np.nan)
            for dg_id, score in actual_round[["dg_id", "score"]].itertuples(index=False):
                if pd.notna(dg_id) and int(dg_id) in field_index:
                    actual_scores[field_index[int(dg_id)]] = float(score)
            valid_pair = np.isfinite(actual_scores[left]) & np.isfinite(actual_scores[right])
            valid_pair &= actual_scores[left] != actual_scores[right]
            use = valid_pair & market_like
            outcome = (actual_scores[left[use]] < actual_scores[right[use]]).astype(float)
            for arm in ARMS:
                p = arm_prob[arm][use]
                brier_sum, logloss_sum, n = binary_sums(p, outcome)
                matchup_rows.append({
                    "event_key": event.key, "event": event.event_name, "start": event.start,
                    "round_num": round_num, "arm": arm, "brier_sum": brier_sum,
                    "logloss_sum": logloss_sum, "n": n,
                    "mean_abs_delta_pp": float(100.0 * np.mean(np.abs(p - base_prob[use]))) if n else np.nan,
                })

            # Conditional round-score test.  The realized common scoring level
            # is profiled out; this cannot favor an arm because all share it.
            valid_players = np.flatnonzero(np.isfinite(actual_scores))
            if len(valid_players):
                nuisance_base = float(np.mean(actual_scores[valid_players] + field.iloc[valid_players]["skill_est"].to_numpy(float)))
                outcome_relative = actual_scores[valid_players] - nuisance_base
                if args.save_case_pmfs:
                    score_bins = np.arange(-12, 13, dtype=int)
                    for arm in ARMS:
                        arm_pmf = score_pmf(round_draws[arm][round_num - 1])
                        for player_index in valid_players:
                            metadata = field.iloc[player_index]
                            probabilities = arm_pmf[player_index]
                            for score_value, probability in zip(score_bins, probabilities):
                                if probability <= 0.0:
                                    continue
                                case_pmf_rows.append({
                                    "event_key": event.key,
                                    "event": event.event_name,
                                    "event_start": event.start,
                                    "dg_id": int(metadata["dg_id"]),
                                    "source_player_id": str(metadata["source_player_id"]),
                                    "player_name": str(metadata["player_key"]),
                                    "round_num": int(round_num),
                                    "arm": arm,
                                    "score_value_relative": int(score_value),
                                    "probability": float(probability),
                                    "actual_strokes": float(actual_scores[player_index]),
                                    "actual_relative": float(
                                        actual_scores[player_index] - nuisance_base
                                    ),
                                    "field_scoring_center": nuisance_base,
                                    "target_sg": float(metadata["skill_est"]),
                                    "skill_date": pd.Timestamp(metadata["Date"]),
                                })
                for arm in ARMS:
                    samples = round_draws[arm][round_num - 1][valid_players]
                    distribution = sample_diagnostics(samples, outcome_relative, args.sims)
                    round_score_rows.append({
                        "event_key": event.key, "event": event.event_name, "start": event.start,
                        "round_num": round_num, "arm": arm, "crps_sum": float(distribution["crps"].sum()),
                        "logpmf_sum": float(distribution["logpmf"].sum()), "n": int(len(valid_players)),
                        "pit_mean": float(distribution["pit"].mean()),
                        "cover50_n": int(np.sum((outcome_relative >= distribution["q25"]) & (outcome_relative <= distribution["q75"]))),
                        "cover80_n": int(np.sum((outcome_relative >= distribution["q10"]) & (outcome_relative <= distribution["q90"]))),
                        "cover90_n": int(np.sum((outcome_relative >= distribution["q05"]) & (outcome_relative <= distribution["q95"]))),
                    })

        # Static pre-event tournament test.  This is deliberately simpler than
        # the live production cascade and isolates the four-round tail implied
        # by each frozen one-round distribution.
        actual_r3 = actual[actual["round_num"].eq(3)]["dg_id"].nunique()
        cut_size = len(field) if actual_r3 >= len(field) - 1 else max(1, min(int(actual_r3), len(field)))
        finishes = latest_finishes(actual, field)
        for arm in ARMS:
            probs = static_tournament_probs(round_draws[arm], cut_size, 9100 + event_index)
            for metric, top_n in (("win", 1), ("top5", 5), ("top10", 10), ("top20", 20)):
                valid = np.isfinite(finishes)
                outcome = (finishes[valid] <= top_n).astype(float)
                brier_sum, logloss_sum, n = binary_sums(probs[metric][valid], outcome)
                tournament_rows.append({
                    "event_key": event.key, "event": event.event_name, "start": event.start,
                    "metric": metric, "arm": arm, "brier_sum": brier_sum,
                    "logloss_sum": logloss_sum, "n": n,
                })

        audit_rows.append({
            "event_key": event.key, "event": event.event_name, "start": event.start,
            "event_id": event.event_id, "course_id": event.course_id,
            "course_count": event.course_count, "raw_field": event.field_size,
            "skill_field": len(field), "skill_coverage": len(field) / event.field_size,
            "source_id_coverage": int(field["source_player_id"].notna().sum()),
            "min_shot_coverage": min(diagnostics["coverage"].values()),
            "min_category_coverage": min(diagnostics["category_min_history_coverage"].values()),
            "cut_size": cut_size, "course_history_rows": int(course_meta.get("history_rows", 0)),
            "course_history_years": json.dumps(course_meta.get("years", [])),
            "course_mults": json.dumps(mults, sort_keys=True),
        })
        print(
            f"  skill={len(field)}/{event.field_size}; source={audit_rows[-1]['source_id_coverage']}; "
            f"shot min={audit_rows[-1]['min_shot_coverage']}; cat min={audit_rows[-1]['min_category_coverage']}; "
            f"course={event.course_id} x{event.course_count}",
            flush=True,
        )
        del round_draws
        gc.collect()

    matchup = pd.DataFrame(matchup_rows)
    round_score = pd.DataFrame(round_score_rows)
    tournament = pd.DataFrame(tournament_rows)
    audit = pd.DataFrame(audit_rows)
    matchup.to_csv(args.output / "round_matchup_event_metrics.csv", index=False)
    round_score.to_csv(args.output / "round_score_event_metrics.csv", index=False)
    tournament.to_csv(args.output / "tournament_event_metrics.csv", index=False)
    audit.to_csv(args.output / "event_audit.csv", index=False)
    if args.save_case_pmfs:
        case_pmfs = pd.DataFrame(case_pmf_rows)
        if case_pmfs.empty:
            raise ValueError("--save-case-pmfs requested but no case PMFs were built")
        case_pmfs.to_parquet(args.output / "round_case_pmfs.parquet", index=False)

    comparison_rows = []
    for scope, frame, losses in (
        ("r1_matchups", matchup[matchup["round_num"].eq(1)], ("brier", "logloss")),
        ("all_round_matchups", matchup, ("brier", "logloss")),
        ("all_round_scores", round_score, ("crps", "logpmf")),
    ):
        for arm in ARMS:
            for loss in losses:
                value = frame.loc[frame["arm"].eq(arm), f"{loss}_sum"].sum() / frame.loc[frame["arm"].eq(arm), "n"].sum()
                if arm == "category_current":
                    delta, low, high = 0.0, 0.0, 0.0
                else:
                    delta, low, high = cluster_bootstrap_delta(frame, arm, loss)
                comparison_rows.append({
                    "scope": scope, "metric": loss, "arm": arm, "value": value,
                    "delta_vs_current": delta, "cluster95_low": low, "cluster95_high": high,
                    "events": int(frame["event_key"].nunique()),
                    "n": int(frame.loc[frame["arm"].eq(arm), "n"].sum()),
                })
    for metric in ("win", "top5", "top10", "top20"):
        frame = tournament[tournament["metric"].eq(metric)]
        for arm in ARMS:
            for loss in ("brier", "logloss"):
                value = frame.loc[frame["arm"].eq(arm), f"{loss}_sum"].sum() / frame.loc[frame["arm"].eq(arm), "n"].sum()
                if arm == "category_current":
                    delta, low, high = 0.0, 0.0, 0.0
                else:
                    delta, low, high = cluster_bootstrap_delta(frame, arm, loss)
                comparison_rows.append({
                    "scope": f"static_tournament_{metric}", "metric": loss, "arm": arm,
                    "value": value, "delta_vs_current": delta,
                    "cluster95_low": low, "cluster95_high": high,
                    "events": int(frame["event_key"].nunique()),
                    "n": int(frame.loc[frame["arm"].eq(arm), "n"].sum()),
                })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.output / "pooled_comparison.csv", index=False)

    coverage = round_score.groupby("arm", as_index=False).agg(
        n=("n", "sum"), cover50_n=("cover50_n", "sum"),
        cover80_n=("cover80_n", "sum"), cover90_n=("cover90_n", "sum"),
    )
    for level in (50, 80, 90):
        coverage[f"cover{level}"] = coverage[f"cover{level}_n"] / coverage["n"]
    coverage.to_csv(args.output / "round_score_coverage.csv", index=False)

    headline = comparison[
        comparison["arm"].ne("category_current")
        & comparison["scope"].isin(["r1_matchups", "all_round_matchups", "all_round_scores", "static_tournament_win", "static_tournament_top10"])
    ].copy()
    report = [
        "# 2026 distribution-engine backtest", "",
        f"Completed {len(events)} PGA events from {events[0].start.date()} through {events[-1].start.date()}, using {args.sims:,} simulations per arm/round and {valid_shots:,} valid shot-SG rows scanned for the union field.", "",
        "## Design", "",
        "Every arm uses the identical dated `model_results2.skill_est`, built only from rounds before the prediction date. The exact-skill sample covers the established-player portion of each historical field; players without a dated estimate are excluded instead of receiving a hindsight approximation. Category histories, course shapes, and shot variances are all capped strictly before the event.", "",
        "`category_current` is the reconstructed category-first production distribution. `category_shot` applies the BMW production weights (OTT 90%, APP 90%, ARG 80%, PUTT 95%) after field-mean variance scaling. `normal_2p8` is a simple Gaussian at 2.8 SG/round. `normal_player` is the legacy player-specific Gaussian scale `(dated std_dev + 2.8) / 2`.", "",
        "R1 matchup scoring is the clean operational primary. The all-round matchup and round-score panels reuse the frozen pre-event distribution for R2-R4, increasing outcome count but not reconstructing live skill updates. The static tournament layer uses four frozen rounds and the realized number of players advancing to R3; it is a distribution-tail sensitivity, not a bit-exact replay of the production cascade.", "",
        "Negative loss deltas favor the alternative over `category_current`. Confidence intervals resample whole events, preserving within-event dependence.", "",
        "## Headline comparisons", "", markdown_table(headline, 7), "",
        "## Round interval coverage", "", markdown_table(coverage[["arm", "n", "cover50", "cover80", "cover90"]], 5), "",
        "## Event coverage audit", "", markdown_table(audit[["event", "start", "raw_field", "skill_field", "skill_coverage", "source_id_coverage", "min_shot_coverage", "min_category_coverage", "course_count"]], 4), "",
        "## Interpretation limits", "",
        "Historical weather forecasts and live between-round skill updates were not archived for the full season. They affect means shared by all arms, so omitting them is appropriate for the primary distribution comparison, but the static tournament results should not be read as exact historical board prices. Multi-course events use the modal R1 course shape and are retained in the all-season result; the saved event audit allows a single-course sensitivity.", "",
    ]
    (args.output / "DISTRIBUTION_ENGINE_BACKTEST_2026.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "events": [event.key for event in events],
        "sims_per_arm_round": args.sims,
        "arms": list(ARMS),
        "weights": WEIGHTS,
        "normal_sd": NORMAL_SD,
        "db": str(args.db),
        "export": str(args.export),
        "correlation": corr.tolist(),
        "event_name_filter": args.event_name,
        "case_pmfs_saved": bool(args.save_case_pmfs),
        "case_pmfs_sha256": (
            hashlib.sha256((args.output / "round_case_pmfs.parquet").read_bytes()).hexdigest()
            if args.save_case_pmfs else None
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output / 'DISTRIBUTION_ENGINE_BACKTEST_2026.md'}")


if __name__ == "__main__":
    main()

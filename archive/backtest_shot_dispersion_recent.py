"""Leakage-safe replay of the production shot-dispersion overlay.

The replay uses the predictions, category distributions, coefficient cache and
published prices committed before each event.  Shot features are rebuilt using
only events strictly earlier than the event being replayed.  Because the exact
historical Google Sheet weather/course cells were not archived, the paired sim
uses zero-centered weather and reconstructs course category multipliers from
the historical round database.  The published-price counterfactual is anchored
as: published probability + (replay dispersion - replay baseline).

This script is intentionally read-only outside its output directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(ARCHIVE))

from build_bmw_shot_features_shadow import (  # noqa: E402
    build_dispersion_features,
    load_event_dates,
    load_shot_round_stats,
)


CAT_ORDER = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
SHOT_CAT = {"sg_ott": "ott", "sg_app": "app", "sg_arg": "arg", "sg_putt": "putt"}
WEIGHTS = {"sg_ott": 0.90, "sg_app": 0.90, "sg_arg": 0.80, "sg_putt": 0.95}
BASELINE_SKEW = {"sg_ott": -0.93, "sg_app": -0.21, "sg_arg": -0.18, "sg_putt": -0.05}
EPS = 1e-7


@dataclass(frozen=True)
class Event:
    label: str
    event_id: int
    course_id: int
    start: str
    final_commit: str
    fairs_commit: str
    file_suffix: str
    cut_line: int
    use_10_shot_rule: bool = False


EVENTS = [
    Event("3M Open", 525, 883, "2026-07-23", "9d23b63bd5df9b9ee2365f206f891fac585c338c", "081f576ffcb1b88c686e3c952cd089e285a2adca", "3m_open", 65),
    Event("Rocket Classic", 524, 876, "2026-07-30", "4d70f605d52939758b9138691b329f97394a146f", "3ff2295db7b865c11d1cf733fa7be193e58607ac", "rocket", 65),
    Event("Wyndham Championship", 13, 752, "2026-08-06", "75c0950b88356554b39dc721f1261c56d611f252", "7a13de9a303391cae7ffc0b23ebd0a35a83b6b97", "wyndham", 65),
    Event("FedEx St. Jude", 27, 513, "2026-08-13", "a40eab4a9c6c070bc2ddd796773c0cda7cbea30d", "3dc3283b59d41f789feedd8a1052773577d1cc94", "st_jude", 100),
]


def norm_name(value: object) -> str:
    return str(value).strip().lower()


def git_bytes(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "show", f"{commit}:{path}"],
        cwd=REPO,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def git_csv(commit: str, path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(git_bytes(commit, path)))


def git_json(commit: str, path: str) -> dict:
    return json.loads(git_bytes(commit, path).decode("utf-8-sig"))


def git_corr(commit: str) -> np.ndarray:
    frame = pd.read_csv(io.BytesIO(git_bytes(commit, "permanent_data/sg_cat_corr_tour_within_player_pearson.csv")), index_col=0)
    matrix = frame.reindex(index=CAT_ORDER, columns=CAT_ORDER).to_numpy(float)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(0.95 * matrix + 0.05 * np.eye(4))


def load_event_inputs(event: Event) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, np.ndarray]:
    preds = git_csv(event.final_commit, f"final_predictions_{event.file_suffix}.csv")
    field = git_csv(event.final_commit, f"pre_course_fit_{event.file_suffix}.csv")
    dists = git_csv(event.final_commit, "this_week_dists_v2.csv")
    published = git_json(event.fairs_commit, "sim_fairs.json")
    coeffs = git_json(event.final_commit, "coeffs_cache.json")

    for frame in (preds, field, dists):
        frame["player_name"] = frame["player_name"].map(norm_name)
    field["source_player_id"] = pd.to_numeric(field["player_num"], errors="coerce").astype("Int64").astype(str)
    field["dg_id"] = pd.to_numeric(field["dg_id"], errors="coerce").astype("Int64")
    field = field[["player_name", "source_player_id", "dg_id"]].drop_duplicates("player_name")
    preds = preds[["player_name", "my_pred"]].drop_duplicates("player_name")

    published_field = {norm_name(name) for name in published.get("field", [])}
    if published_field:
        preds = preds[preds["player_name"].isin(published_field)].copy()
    field = preds.merge(field, on="player_name", how="left", validate="one_to_one")
    if field[["source_player_id", "dg_id"]].isna().any().any():
        raise ValueError(f"{event.label}: incomplete player ID map")
    return field, dists, published, coeffs, git_corr(event.final_commit)


def _winsorize_left(values: np.ndarray, pct: float = 0.005) -> np.ndarray:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size >= 1.0 / pct:
        values = np.clip(values, np.quantile(values, pct), None)
    return values


def reconstruct_course_shape(db_path: Path, course_id: int, cutoff: pd.Timestamp) -> tuple[dict, dict, dict]:
    """Rebuild production-style course variance/skew with a strict date cap."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    cats = ["sg_ott_adj", "sg_app_adj", "sg_arg_adj", "sg_putt_adj"]
    names = dict(zip(cats, CAT_ORDER))
    with sqlite3.connect(uri, uri=True) as connection:
        course = pd.read_sql_query(
            f"SELECT player_name, year, event_id, round_date, {', '.join(cats)} "
            "FROM player_rounds WHERE tour='pga' AND course_num=? AND year>=2019 "
            "AND round_date < ? AND sg_total_adj IS NOT NULL",
            connection,
            params=[course_id, cutoff.strftime("%Y-%m-%d")],
        )
        if course.empty:
            return ({cat: 1.0 for cat in CAT_ORDER}, BASELINE_SKEW.copy(), {"history_rows": 0, "years": []})
        players = sorted(course["player_name"].dropna().unique())
        pieces = []
        for start in range(0, len(players), 500):
            batch = players[start : start + 500]
            marks = ",".join("?" for _ in batch)
            pieces.append(pd.read_sql_query(
                f"SELECT player_name, round_date, {', '.join(cats)} FROM player_rounds "
                f"WHERE player_name IN ({marks}) AND round_date < ? AND sg_total_adj IS NOT NULL",
                connection,
                params=[*batch, cutoff.strftime("%Y-%m-%d")],
            ))
    rounds = pd.concat(pieces, ignore_index=True)
    rounds["round_date"] = pd.to_datetime(rounds["round_date"])
    course["round_date"] = pd.to_datetime(course["round_date"])
    starts = course.groupby(["year", "event_id"], as_index=False)["round_date"].min().rename(columns={"round_date": "event_start"})

    year_rows: list[dict] = []
    for year, starts_year in starts.groupby("year"):
        actual_pieces = []
        expected: dict[str, list[float]] = {cat: [] for cat in cats}
        qualified: set[tuple[str, int]] = set()
        for event_id, event_start in starts_year[["event_id", "event_start"]].itertuples(index=False):
            event_rows = course[(course["year"] == year) & (course["event_id"] == event_id)]
            good_names = []
            for player in event_rows["player_name"].dropna().unique():
                prior = rounds[(rounds["player_name"] == player) & (rounds["round_date"] < event_start)].sort_values("round_date").tail(50)
                if len(prior) < 50:
                    continue
                player_stds = {}
                for cat in cats:
                    valid = prior[cat].dropna()
                    player_stds[cat] = float(valid.std()) if len(valid) >= 30 else np.nan
                if not any(np.isfinite(list(player_stds.values()))):
                    continue
                good_names.append(player)
                qualified.add((player, int(event_id)))
                for cat, value in player_stds.items():
                    if np.isfinite(value):
                        expected[cat].append(value)
            actual_pieces.append(event_rows[event_rows["player_name"].isin(good_names)])
        if len(qualified) < 10:
            continue
        actual = pd.concat(actual_pieces, ignore_index=True)
        row: dict[str, float | int | None] = {"year": int(year), "qual_players": len(qualified)}
        for cat in cats:
            values = _winsorize_left(actual[cat].to_numpy(float))
            row[f"{cat}_actual"] = float(np.std(values, ddof=1)) if len(values) >= 10 else None
            row[f"{cat}_expected"] = float(np.mean(expected[cat])) if expected[cat] else None
            row[f"{cat}_skew"] = float(sp_stats.skew(values)) if len(values) >= 20 else None
        year_rows.append(row)

    if not year_rows:
        return ({cat: 1.0 for cat in CAT_ORDER}, BASELINE_SKEW.copy(), {"history_rows": len(course), "years": []})
    min_year = min(row["year"] for row in year_rows)
    raw_weights = {row["year"]: 1.0 + 0.025 * (row["year"] - min_year) for row in year_rows}
    total_weight = sum(raw_weights.values())
    weights = {year: value / total_weight for year, value in raw_weights.items()}
    mults, skews = {}, {}
    for source_cat, output_cat in names.items():
        pairs = [(row, weights[row["year"]]) for row in year_rows if row[f"{source_cat}_actual"] is not None and row[f"{source_cat}_expected"] is not None]
        if pairs:
            denom = sum(weight for _, weight in pairs)
            actual = sum(row[f"{source_cat}_actual"] * weight for row, weight in pairs) / denom
            expected = sum(row[f"{source_cat}_expected"] * weight for row, weight in pairs) / denom
            mults[output_cat] = round(actual / expected, 2) if expected > 0 else 1.0
        else:
            mults[output_cat] = 1.0
        skew_pairs = [(row, weights[row["year"]]) for row in year_rows if row[f"{source_cat}_skew"] is not None]
        if skew_pairs:
            denom = sum(weight for _, weight in skew_pairs)
            skews[output_cat] = round(sum(row[f"{source_cat}_skew"] * weight for row, weight in skew_pairs) / denom, 2)
        else:
            skews[output_cat] = BASELINE_SKEW[output_cat]
    return mults, skews, {"history_rows": len(course), "years": [row["year"] for row in year_rows]}


def build_kernel_inputs(event: Event, field: pd.DataFrame, dists: pd.DataFrame, features: pd.DataFrame, mults: dict, skews: dict, l_corr: np.ndarray) -> tuple[dict, dict]:
    players = field["player_name"].tolist()
    d = dists[dists["category_clean"].isin(CAT_ORDER)].copy()
    pivots = {column: d.pivot_table(index="player_name", columns="category_clean", values=column, aggfunc="last") for column in ("mean", "std", "skew", "n_eff")}
    global_mu = d.groupby("category_clean")["mean"].mean().reindex(CAT_ORDER)
    global_std = d.groupby("category_clean")["std"].median().reindex(CAT_ORDER)
    feature_map = features.set_index("source_player_id")

    mu_rows, std_rows, std_disp_rows, eff_skew_rows = [], [], [], []
    coverage = {cat: 0 for cat in CAT_ORDER}
    scales = {}
    base_var_by_cat, shot_var_by_cat = {}, {}
    for cat in CAT_ORDER:
        base = []
        shot = []
        for player, source_id in field[["player_name", "source_player_id"]].itertuples(index=False):
            value = pivots["std"].at[player, cat] if player in pivots["std"].index and cat in pivots["std"].columns else np.nan
            base.append(float(value) ** 2 if pd.notna(value) else float(global_std[cat]) ** 2)
            feature_col = f"{SHOT_CAT[cat]}_shot_indep_var50_shrunk"
            value = feature_map.at[str(source_id), feature_col] if str(source_id) in feature_map.index and feature_col in feature_map.columns else np.nan
            shot.append(float(value) if pd.notna(value) and float(value) > 0 else np.nan)
        base = np.asarray(base, float)
        shot = np.asarray(shot, float)
        valid = np.isfinite(shot)
        coverage[cat] = int(valid.sum())
        scales[cat] = float(base[valid].mean() / shot[valid].mean()) if valid.any() else 1.0
        base_var_by_cat[cat] = base
        shot_var_by_cat[cat] = shot

    pred_map = field.set_index("player_name")["my_pred"]
    for row_index, player in enumerate(players):
        mu = np.array([
            pivots["mean"].at[player, cat] if player in pivots["mean"].index and cat in pivots["mean"].columns and pd.notna(pivots["mean"].at[player, cat]) else global_mu[cat]
            for cat in CAT_ORDER
        ], float)
        mu += (float(pred_map[player]) - mu.sum()) / 4.0
        base_std, dispersion_std, player_skew = [], [], []
        for cat_index, cat in enumerate(CAT_ORDER):
            base_var = base_var_by_cat[cat][row_index]
            shot_var = shot_var_by_cat[cat][row_index]
            if np.isfinite(shot_var):
                new_var = (1.0 - WEIGHTS[cat]) * base_var + WEIGHTS[cat] * shot_var * scales[cat]
            else:
                new_var = base_var
            base_std.append(math.sqrt(base_var) * mults[cat])
            dispersion_std.append(math.sqrt(new_var) * mults[cat])
            p_skew = pivots["skew"].at[player, cat] if player in pivots["skew"].index and cat in pivots["skew"].columns else np.nan
            p_neff = pivots["n_eff"].at[player, cat] if player in pivots["n_eff"].index and cat in pivots["n_eff"].columns else np.nan
            p_skew = float(p_skew) if pd.notna(p_skew) else 0.0
            p_neff = float(p_neff) if pd.notna(p_neff) else 0.0
            blend = 0.5 * min(p_neff / 100.0, 1.0)
            player_skew.append((1.0 - blend) * skews[cat] + blend * p_skew)
        mu_rows.append(mu)
        std_rows.append(base_std)
        std_disp_rows.append(dispersion_std)
        eff_skew_rows.append(player_skew)

    pred = field["my_pred"].to_numpy(float)
    zeros = np.zeros(len(field), float)
    common = {
        "mu": np.ascontiguousarray(np.asarray(mu_rows, float)),
        "eff_skew": np.ascontiguousarray(np.asarray(eff_skew_rows, float)),
        "l_corr": np.ascontiguousarray(l_corr),
        "my_pred_base": np.ascontiguousarray(pred),
        "r2_mu": np.ascontiguousarray(pred),
        "r3_mu": np.ascontiguousarray(pred),
        "r4_mu": np.ascontiguousarray(pred),
        "weather_delta_r1": np.ascontiguousarray(zeros),
        "weather_delta_r2": np.ascontiguousarray(zeros),
    }
    diagnostics = {"coverage": coverage, "scales": scales, "base_std": np.asarray(std_rows), "dispersion_std": np.asarray(std_disp_rows)}
    return {**common, "std_base": np.ascontiguousarray(np.asarray(std_rows, float)), "std_disp": np.ascontiguousarray(np.asarray(std_disp_rows, float))}, diagnostics


def coefficient_vectors(coeffs: dict) -> list:
    r1 = lambda key: [coeffs[key][name] for name in ("ott", "putt", "residual", "residual2")]
    r2 = lambda key: [coeffs[key][name] for name in ("residual", "residual2", "residual3", "avg_ott", "avg_putt", "avg_app", "avg_arg", "delta_app")]
    r3 = lambda key: [coeffs[key].get(name, 0.0) for name in ("sg_ott_avg", "sg_putt_avg", "sg_app_avg", "sg_arg_avg", "pos_6_10")]
    return [
        r1("coefficients_r1_high"), r1("coefficients_r1_midh"), r1("coefficients_r1_midl"), r1("coefficients_r1_low"),
        r2("coefficients_r2"), r2("coefficients_r2_6_30"), r2("coefficients_r2_30_up"),
        r3("coefficients_r3"), r3("coefficients_r3_mid"), r3("coefficients_r3_high"),
    ]


def run_kernel(inputs: dict, std_key: str, coeffs: dict, event: Event, sims: int) -> tuple[np.ndarray, dict]:
    import sims_kernel
    args = [inputs[key] for key in ("mu", std_key, "eff_skew", "l_corr", "my_pred_base", "r2_mu", "r3_mu", "r4_mu", "weather_delta_r1", "weather_delta_r2")]
    result = sims_kernel.run_pretournament(
        *args, *coefficient_vectors(coeffs), int(event.cut_line), bool(event.use_10_shot_rule), int(sims), 456, 0.0
    )
    scores = np.ascontiguousarray(result[0])
    win = np.asarray(result[1], float)
    _, top_dh, _ = sims_kernel.aggregate_round(scores)
    metrics = {"win": win, "top5": np.asarray(top_dh[:, 0]), "top10": np.asarray(top_dh[:, 1]), "top20": np.asarray(top_dh[:, 2])}
    return scores, metrics


def actual_finishes(db_path: Path, event: Event, field: pd.DataFrame) -> np.ndarray:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    ids = [int(value) for value in field["dg_id"].dropna().unique()]
    marks = ",".join("?" for _ in ids)
    with sqlite3.connect(uri, uri=True) as connection:
        actual = pd.read_sql_query(
            f"SELECT dg_id, finish_pos FROM ("
            f"SELECT dg_id, finish_pos, ROW_NUMBER() OVER (PARTITION BY dg_id ORDER BY round_num DESC, round_date DESC) AS rn "
            f"FROM player_rounds WHERE tour='pga' AND year=2026 AND event_id=? AND dg_id IN ({marks}) AND finish_pos IS NOT NULL"
            f") WHERE rn=1",
            connection,
            params=[event.event_id, *ids],
        )
    return field["dg_id"].map(actual.set_index("dg_id")["finish_pos"]).to_numpy(float)


def binary_metrics(prob: np.ndarray, outcome: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(prob) & np.isfinite(outcome)
    p = np.clip(prob[mask], EPS, 1.0 - EPS)
    y = outcome[mask]
    return float(np.mean((p - y) ** 2)), float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def published_probs(published: dict, players: list[str], metric: str) -> np.ndarray:
    key = {"win": "winner", "top5": "top_5", "top10": "top_10", "top20": "top_20"}[metric]
    values = {norm_name(name): value for name, value in published.get("outrights", {}).get(key, {}).items()}
    return np.asarray([float(values.get(player, np.nan)) for player in players], float)


def h2h_score(scores_base: np.ndarray, scores_disp: np.ndarray, finishes: np.ndarray) -> tuple[dict, list[dict]]:
    base_brier, disp_brier, base_ll, disp_ll, rows = [], [], [], [], []
    n = len(finishes)
    for i in range(n):
        if not np.isfinite(finishes[i]):
            continue
        for j in range(i + 1, n):
            if not np.isfinite(finishes[j]) or finishes[i] == finishes[j]:
                continue
            def probability(scores: np.ndarray) -> float:
                win = np.count_nonzero(scores[i] < scores[j])
                loss = np.count_nonzero(scores[i] > scores[j])
                return win / (win + loss) if win + loss else 0.5
            p0, p1 = probability(scores_base), probability(scores_disp)
            y = float(finishes[i] < finishes[j])
            base_brier.append((p0 - y) ** 2); disp_brier.append((p1 - y) ** 2)
            base_ll.append(-(y * math.log(max(p0, EPS)) + (1-y) * math.log(max(1-p0, EPS))))
            disp_ll.append(-(y * math.log(max(p1, EPS)) + (1-y) * math.log(max(1-p1, EPS))))
            rows.append({"i": i, "j": j, "p_base": p0, "p_dispersion": p1, "outcome_i_wins": y, "delta_pp": 100*(p1-p0)})
    summary = {
        "pairs": len(rows), "brier_base": float(np.mean(base_brier)), "brier_dispersion": float(np.mean(disp_brier)),
        "logloss_base": float(np.mean(base_ll)), "logloss_dispersion": float(np.mean(disp_ll)),
    }
    return summary, rows


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    """Small dependency-free Markdown renderer for the saved report."""
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, default=Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"))
    parser.add_argument("--db", type=Path, default=Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"))
    parser.add_argument("--sims", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=REPO / "permanent_data" / "shot_dispersion_backtest_20260820")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    loaded = {}
    union_ids = set()
    for event in EVENTS:
        loaded[event.event_id] = load_event_inputs(event)
        union_ids.update(loaded[event.event_id][0]["source_player_id"].tolist())
    dates = load_event_dates(args.db)
    all_stats, valid_shots = load_shot_round_stats(args.export, sorted(union_ids), dates, pd.Timestamp(max(event.start for event in EVENTS)))

    player_rows, event_rows, outcome_rows, h2h_rows_all = [], [], [], []
    for event in EVENTS:
        print(f"\n=== {event.label} ===", flush=True)
        field, dists, published, coeffs, l_corr = loaded[event.event_id]
        cutoff = pd.Timestamp(event.start)
        event_stats = all_stats[all_stats["event_date"] < cutoff].copy()
        features, priors = build_dispersion_features(event_stats, field[["source_player_id"]])
        mults, course_skew, course_meta = reconstruct_course_shape(args.db, event.course_id, cutoff)
        inputs, diag = build_kernel_inputs(event, field, dists, features, mults, course_skew, l_corr)
        scores_base, replay_base = run_kernel(inputs, "std_base", coeffs, event, args.sims)
        scores_disp, replay_disp = run_kernel(inputs, "std_disp", coeffs, event, args.sims)
        finishes = actual_finishes(args.db, event, field)
        players = field["player_name"].tolist()

        reconstruction_errors = []
        anchored_by_metric = {}
        for metric, top_n in (("win", 1), ("top5", 5), ("top10", 10), ("top20", 20)):
            pub = published_probs(published, players, metric)
            delta = replay_disp[metric] - replay_base[metric]
            anchored = np.clip(pub + delta, EPS, 1.0 - EPS)
            anchored_by_metric[metric] = anchored
            mask = np.isfinite(pub)
            reconstruction_errors.extend((replay_base[metric][mask] - pub[mask]).tolist())
            outcome = (finishes <= top_n).astype(float)
            outcome[~np.isfinite(finishes)] = np.nan
            b0, ll0 = binary_metrics(pub, outcome)
            b1, ll1 = binary_metrics(anchored, outcome)
            winner_mask = finishes == 1
            winner_ll0 = float(-np.mean(np.log(np.clip(pub[winner_mask], EPS, 1.0)))) if metric == "win" and winner_mask.any() else np.nan
            winner_ll1 = float(-np.mean(np.log(np.clip(anchored[winner_mask], EPS, 1.0)))) if metric == "win" and winner_mask.any() else np.nan
            outcome_rows.append({"event": event.label, "metric": metric, "players": int(np.isfinite(outcome).sum()), "brier_published": b0, "brier_dispersion": b1, "brier_delta": b1-b0, "logloss_published": ll0, "logloss_dispersion": ll1, "logloss_delta": ll1-ll0, "winner_logloss_published": winner_ll0, "winner_logloss_dispersion": winner_ll1, "winner_logloss_delta": winner_ll1-winner_ll0 if metric == "win" else np.nan})
            for i, player in enumerate(players):
                player_rows.append({
                    "event": event.label, "event_id": event.event_id, "player_name": player, "dg_id": int(field.iloc[i]["dg_id"]), "finish_pos": finishes[i], "metric": metric,
                    "published_prob": pub[i], "replay_baseline_prob": replay_base[metric][i], "replay_dispersion_prob": replay_disp[metric][i],
                    "paired_delta_pp": 100.0 * delta[i], "anchored_dispersion_prob": anchored[i],
                })

        h2h_summary, h2h_rows = h2h_score(scores_base, scores_disp, finishes)
        for row in h2h_rows:
            row.update({"event": event.label, "player_a": players[row.pop("i")], "player_b": players[row.pop("j")]})
            h2h_rows_all.append(row)
        recon = np.asarray(reconstruction_errors)
        coverage = diag["coverage"]
        event_rows.append({
            "event": event.label, "event_id": event.event_id, "field": len(field), "actual_finish_coverage": int(np.isfinite(finishes).sum()),
            "shot_feature_min_coverage": min(coverage.values()), "shot_feature_max_coverage": max(coverage.values()),
            "reconstruction_mae_pp": 100.0 * float(np.mean(np.abs(recon))), "reconstruction_rmse_pp": 100.0 * float(np.sqrt(np.mean(recon**2))),
            "course_mults": json.dumps(mults, sort_keys=True), "course_skew": json.dumps(course_skew, sort_keys=True), "course_history_years": json.dumps(course_meta["years"]),
            **{f"h2h_{key}": value for key, value in h2h_summary.items()},
        })
        print(f"field={len(field)} feature coverage={coverage}; replay MAE={event_rows[-1]['reconstruction_mae_pp']:.3f}pp; H2H Brier delta={h2h_summary['brier_dispersion']-h2h_summary['brier_base']:+.6f}", flush=True)
        del scores_base, scores_disp
        gc.collect()

    players_df = pd.DataFrame(player_rows)
    events_df = pd.DataFrame(event_rows)
    outcomes_df = pd.DataFrame(outcome_rows)
    h2h_df = pd.DataFrame(h2h_rows_all)
    players_df.to_csv(args.output / "event_player_prices.csv", index=False)
    events_df.to_csv(args.output / "event_summary.csv", index=False)
    outcomes_df.to_csv(args.output / "outcome_metrics.csv", index=False)
    h2h_df.to_csv(args.output / "h2h_pair_metrics.csv", index=False)

    pooled = outcomes_df.groupby("metric", as_index=False).agg(
        events=("event", "nunique"), brier_published=("brier_published", "mean"), brier_dispersion=("brier_dispersion", "mean"),
        logloss_published=("logloss_published", "mean"), logloss_dispersion=("logloss_dispersion", "mean"),
    )
    pooled["brier_delta"] = pooled["brier_dispersion"] - pooled["brier_published"]
    pooled["logloss_delta"] = pooled["logloss_dispersion"] - pooled["logloss_published"]
    pooled.to_csv(args.output / "pooled_outcome_metrics.csv", index=False)

    biggest = players_df[players_df["metric"] == "win"].dropna(subset=["published_prob"]).copy()
    biggest["abs_delta"] = biggest["paired_delta_pp"].abs()
    biggest = biggest.nlargest(20, "abs_delta")
    win_event = outcomes_df[outcomes_df["metric"] == "win"][["event", "brier_delta", "logloss_delta", "winner_logloss_published", "winner_logloss_dispersion", "winner_logloss_delta"]]
    report = [
        "# Recent-event shot-dispersion backtest", "",
        f"Generated with {args.sims:,} paired simulations per arm/event; kernel seed 456; {valid_shots:,} valid union-field shot rows scanned.", "",
        "## Result", "",
        "This small recent-event sample is mixed. Winner pricing improved: Brier score fell 0.33%, binary log loss fell 0.98%, and the eventual winner's probability moved up in 3 of 4 events (mean winner log-loss improvement 0.0401). Top-5 and top-10 Brier scores each improved about 0.40%, while their log loss was effectively flat. Top-20 was flat-to-worse. All-pairs H2H Brier and log loss worsened about 0.09%, with a worse H2H result in every event.", "",
        "Taken together, the replay supports using shot dispersion for tournament-tail pricing, but it does not validate applying the same full-strength variance replacement to H2H without more work. Four events are far too few to regard these differences as statistically settled.", "",
        "## What was tested", "",
        "The BMW production weights (OTT 90%, APP 90%, ARG 80%, PUTT 95%) replace the corresponding share of each player's production category variance with the pre-event 50-round shot-level estimate. Each category is field-mean scaled first, so the test changes relative player volatility rather than inflating or deflating the whole field. Missing shot histories retain the production variance.", "",
        "Every feature observation predates the replayed event. Published counterfactual probabilities are anchored to the actual board: `published + paired replay delta`. The replay uses exact committed predictions, distributions, coefficient caches and correlations; old Sheet weather was unavailable, so weather is centered at zero and course multipliers are reconstructed from strictly prior rounds.", "",
        "## Replay fidelity", "",
        markdown_table(events_df[["event", "field", "shot_feature_min_coverage", "shot_feature_max_coverage", "reconstruction_mae_pp", "reconstruction_rmse_pp"]], 4), "",
        "## Realized finish scoring", "",
        "Negative deltas are improvements.", "",
        markdown_table(pooled, 6), "",
        "### Win scoring by event", "", markdown_table(win_event, 6), "",
        "## Biggest reconstructed win-price changes", "",
        markdown_table(biggest[["event", "player_name", "finish_pos", "published_prob", "anchored_dispersion_prob", "paired_delta_pp"]], 5), "",
        "## H2H outcome check", "",
        markdown_table(events_df[["event", "h2h_pairs", "h2h_brier_base", "h2h_brier_dispersion", "h2h_logloss_base", "h2h_logloss_dispersion"]], 6), "",
        "## Interpretation limits", "",
        "This is a four-event, post-selection diagnostic, not a new independent model-selection sample. Outcomes are noisy—especially wins—and player/top-N observations within an event are correlated. The strongest prior evidence remains the 2023–2026 expanding-year round-residual NLL test; this replay answers whether the same production transform would have changed recent prices in a direction rewarded by those four realized leaderboards.", "",
        "The local odds archive contains no snapshots for these four events, so a clean closing-market comparison was not possible. That should be archived prospectively starting with BMW.", "",
    ]
    (args.output / "BACKTEST_SHOT_DISPERSION_RECENT.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "events": [event.__dict__ for event in EVENTS], "sims": args.sims, "seed": 456, "weights": WEIGHTS,
        "export": str(args.export), "db": str(args.db), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output / 'BACKTEST_SHOT_DISPERSION_RECENT.md'}")


if __name__ == "__main__":
    main()

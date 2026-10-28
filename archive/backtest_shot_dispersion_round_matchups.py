"""Backtest the full-strength shot-dispersion overlay on pre-event R1 H2H.

Historical round means/weather and published all-pairs H2H prices come from
the committed R1 cache metadata.  Shot features and course shapes are rebuilt
with strict pre-event cutoffs.  Counterfactual prices are anchored as the exact
published push-fair probability plus the paired replay delta.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_bmw_shot_features_shadow import _load_name_replacements

from backtest_shot_dispersion_recent import (
    BASELINE_SKEW,
    CAT_ORDER,
    EVENTS,
    EPS,
    REPO,
    SHOT_CAT,
    WEIGHTS,
    build_dispersion_features,
    git_bytes,
    git_corr,
    git_csv,
    git_json,
    load_event_dates,
    load_shot_round_stats,
    markdown_table,
    norm_name,
    reconstruct_course_shape,
)

sys.path.insert(0, str(REPO))


ROUND_SKEW_CAL = {525: False, 524: True, 13: True, 27: True}


def git_parquet(commit: str, path: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(git_bytes(commit, path)))


def source_crosswalk(export: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        export / "pga_2026_player_rounds.parquet",
        columns=["source_player_id", "player_name"],
    )
    frame["player_name"] = frame["player_name"].map(norm_name)
    frame["player_name"] = frame["player_name"].replace(
        _load_name_replacements(REPO / "sim_inputs.py")
    )
    frame["source_player_id"] = frame["source_player_id"].astype(str)
    return frame.drop_duplicates("player_name")


def round_event_inputs(event, export: Path) -> dict:
    commit = event.fairs_commit
    meta = git_json(commit, "round_h2h_r1_meta.json")
    if meta.get("tourney") != event.file_suffix or int(meta.get("round", 0)) != 1:
        raise ValueError(f"{event.label}: retained R1 artifact is stale: {meta.get('tourney')}")
    players = [norm_name(name) for name in meta["pred"].keys()]
    dists = git_csv(commit, "this_week_dists_v2.csv")
    dists["player_name"] = dists["player_name"].map(norm_name)
    archived_h2h = git_parquet(commit, "round_h2h_r1.parquet")
    archived_h2h["player_a"] = archived_h2h["player_a"].map(norm_name)
    archived_h2h["player_b"] = archived_h2h["player_b"].map(norm_name)
    samples = git_parquet(commit, "round_samples.parquet")
    samples.index = samples.index.map(norm_name)
    pred = pd.Series({norm_name(k): float(v) for k, v in meta["pred"].items()})
    common = samples.index.intersection(pred.index)
    expected_avg = round(float((samples.loc[common].mean(axis=1) + pred.loc[common]).mean()), 1)
    crosswalk = source_crosswalk(export)
    field = pd.DataFrame({"player_name": players}).merge(crosswalk, on="player_name", how="left", validate="one_to_one")
    return {
        "meta": meta,
        "players": players,
        "field": field,
        "dists": dists,
        "archived_h2h": archived_h2h,
        "expected_avg": expected_avg,
        "l_corr": git_corr(commit),
    }


def build_round_params(field: pd.DataFrame, dists: pd.DataFrame, features: pd.DataFrame, mults: dict, course_skew: dict) -> tuple[dict, dict]:
    players = field["player_name"].tolist()
    d = dists[dists["category_clean"].isin(CAT_ORDER)].copy()
    pivots = {
        column: d.pivot_table(index="player_name", columns="category_clean", values=column, aggfunc="last")
        for column in ("mean", "std", "skew", "n_eff")
    }
    global_mu = d.groupby("category_clean")["mean"].mean().reindex(CAT_ORDER)
    global_std = d.groupby("category_clean")["std"].median().reindex(CAT_ORDER)
    feature_map = features.set_index("source_player_id")

    base_vars, shot_vars, scales, coverage = {}, {}, {}, {}
    for cat in CAT_ORDER:
        base, shot = [], []
        feature_col = f"{SHOT_CAT[cat]}_shot_indep_var50_shrunk"
        for player, source_id in field[["player_name", "source_player_id"]].itertuples(index=False):
            value = pivots["std"].at[player, cat] if player in pivots["std"].index and cat in pivots["std"].columns else np.nan
            base.append(float(value) ** 2 if pd.notna(value) else float(global_std[cat]) ** 2)
            value = feature_map.at[str(source_id), feature_col] if pd.notna(source_id) and str(source_id) in feature_map.index and feature_col in feature_map.columns else np.nan
            shot.append(float(value) if pd.notna(value) and float(value) > 0 else np.nan)
        base = np.asarray(base, float)
        shot = np.asarray(shot, float)
        valid = np.isfinite(shot)
        base_vars[cat], shot_vars[cat] = base, shot
        coverage[cat] = int(valid.sum())
        scales[cat] = float(base[valid].mean() / shot[valid].mean()) if valid.any() else 1.0

    mu_rows, std_base_rows, std_disp_rows, skew_rows = [], [], [], []
    for i, player in enumerate(players):
        mu_rows.append([
            float(pivots["mean"].at[player, cat])
            if player in pivots["mean"].index and cat in pivots["mean"].columns and pd.notna(pivots["mean"].at[player, cat])
            else float(global_mu[cat])
            for cat in CAT_ORDER
        ])
        std_base, std_disp, effective_skew = [], [], []
        for cat in CAT_ORDER:
            base_var, shot_var = base_vars[cat][i], shot_vars[cat][i]
            new_var = ((1.0 - WEIGHTS[cat]) * base_var + WEIGHTS[cat] * shot_var * scales[cat]) if np.isfinite(shot_var) else base_var
            std_base.append(math.sqrt(base_var) * mults[cat])
            std_disp.append(math.sqrt(new_var) * mults[cat])
            player_skew = pivots["skew"].at[player, cat] if player in pivots["skew"].index and cat in pivots["skew"].columns else np.nan
            neff = pivots["n_eff"].at[player, cat] if player in pivots["n_eff"].index and cat in pivots["n_eff"].columns else np.nan
            player_skew = float(player_skew) if pd.notna(player_skew) else 0.0
            neff = float(neff) if pd.notna(neff) else 0.0
            blend = 0.5 * min(neff / 100.0, 1.0)
            effective_skew.append((1.0 - blend) * course_skew.get(cat, BASELINE_SKEW[cat]) + blend * player_skew)
        std_base_rows.append(std_base)
        std_disp_rows.append(std_disp)
        skew_rows.append(effective_skew)
    return {
        "mu": np.ascontiguousarray(np.asarray(mu_rows, float)),
        "std_base": np.ascontiguousarray(np.asarray(std_base_rows, float)),
        "std_disp": np.ascontiguousarray(np.asarray(std_disp_rows, float)),
        "effective_skew": np.ascontiguousarray(np.asarray(skew_rows, float)),
    }, {"coverage": coverage, "scales": scales}


def run_round(params: dict, std_key: str, l_corr: np.ndarray, players: list[str], meta: dict, expected_avg: float, sims: int, apply_skew_cal: bool) -> np.ndarray:
    import sims_kernel

    pred = np.asarray([float(meta["pred"][player]) for player in players], float)
    wx_map = {norm_name(k): float(v) for k, v in (meta.get("wx") or {}).items()}
    wx = np.asarray([wx_map.get(player, 0.0) for player in players], float)
    scores, _ = sims_kernel.run_single_round(
        params["mu"], params[std_key], params["effective_skew"], np.ascontiguousarray(l_corr),
        np.ascontiguousarray(pred - wx), np.ascontiguousarray(wx),
        np.full(len(players), expected_avg, float), int(sims), 789,
    )
    scores = np.ascontiguousarray(scores).astype(np.int16)
    if apply_skew_cal:
        from skew_calibration import calibrate_round_skew

        sim_dict = {player: scores[i].copy() for i, player in enumerate(players)}
        sim_dict = calibrate_round_skew(sim_dict, verbose=False)
        scores = np.stack([sim_dict[player] for player in players]).astype(np.int16)
    return scores


def all_pair_probs(scores: np.ndarray, players: list[str]) -> pd.DataFrame:
    order = sorted(range(len(players)), key=lambda i: players[i])
    names = [players[i] for i in order]
    matrix = scores[order]
    rows = []
    sims = matrix.shape[1]
    for i, player_a in enumerate(names[:-1]):
        partners = matrix[i + 1 :]
        lower = (matrix[i][None, :] < partners).sum(axis=1) / sims
        ties = (matrix[i][None, :] == partners).sum(axis=1) / sims
        for offset, player_b in enumerate(names[i + 1 :]):
            rows.append((player_a, player_b, float(lower[offset]), float(ties[offset])))
    return pd.DataFrame(rows, columns=["player_a", "player_b", "p_a_lt_b", "p_tie"])


def actual_r1_scores(db_path: Path, event_id: int) -> dict[str, float]:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        frame = pd.read_sql_query(
            "SELECT player_name, score FROM player_rounds WHERE tour='pga' AND year=2026 AND event_id=? AND round_num=1 AND score IS NOT NULL",
            connection,
            params=[event_id],
        )
    frame["player_name"] = frame["player_name"].map(norm_name)
    return frame.drop_duplicates("player_name").set_index("player_name")["score"].astype(float).to_dict()


def score_frame(frame: pd.DataFrame, subset: str) -> dict:
    use = frame[frame["actual_a"].notna() & frame["actual_b"].notna() & (frame["actual_a"] != frame["actual_b"])].copy()
    if subset == "market_like":
        use = use[use["published_push"].between(0.35, 0.65)]
    y = (use["actual_a"] < use["actual_b"]).astype(float).to_numpy()
    p0 = np.clip(use["published_push"].to_numpy(float), EPS, 1.0 - EPS)
    p1 = np.clip(use["anchored_dispersion_push"].to_numpy(float), EPS, 1.0 - EPS)
    b0, b1 = np.mean((p0-y)**2), np.mean((p1-y)**2)
    ll0 = np.mean(-(y*np.log(p0)+(1-y)*np.log(1-p0)))
    ll1 = np.mean(-(y*np.log(p1)+(1-y)*np.log(1-p1)))
    return {
        "subset": subset, "graded_pairs": len(use),
        "brier_published": b0, "brier_dispersion": b1, "brier_delta": b1-b0,
        "logloss_published": ll0, "logloss_dispersion": ll1, "logloss_delta": ll1-ll0,
        "closer_to_outcome_rate": float((np.abs(p1-y) < np.abs(p0-y)).mean()),
        "mean_abs_price_delta_pp": float(use["paired_delta_pp"].abs().mean()),
        "p95_abs_price_delta_pp": float(use["paired_delta_pp"].abs().quantile(0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, default=Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"))
    parser.add_argument("--db", type=Path, default=Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"))
    parser.add_argument("--sims", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=REPO / "permanent_data" / "shot_dispersion_backtest_20260820" / "round_matchups")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    loaded, union_ids = {}, set()
    for event in EVENTS:
        info = round_event_inputs(event, args.export)
        loaded[event.event_id] = info
        if info["field"]["source_player_id"].isna().any():
            missing = info["field"].loc[info["field"]["source_player_id"].isna(), "player_name"].tolist()
            raise ValueError(f"{event.label}: missing R2 IDs: {missing}")
        union_ids.update(info["field"]["source_player_id"].tolist())

    dates = load_event_dates(args.db)
    all_stats, valid_shots = load_shot_round_stats(args.export, sorted(union_ids), dates, pd.Timestamp(max(event.start for event in EVENTS)))
    summary_rows, pair_outputs = [], []
    for event in EVENTS:
        print(f"\n=== {event.label} R1 ===", flush=True)
        info = loaded[event.event_id]
        cutoff = pd.Timestamp(event.start)
        features, _ = build_dispersion_features(all_stats[all_stats["event_date"] < cutoff].copy(), info["field"][["source_player_id"]])
        mults, course_skew, _ = reconstruct_course_shape(args.db, event.course_id, cutoff)
        params, diagnostics = build_round_params(info["field"], info["dists"], features, mults, course_skew)
        scores_base = run_round(params, "std_base", info["l_corr"], info["players"], info["meta"], info["expected_avg"], args.sims, ROUND_SKEW_CAL[event.event_id])
        scores_disp = run_round(params, "std_disp", info["l_corr"], info["players"], info["meta"], info["expected_avg"], args.sims, ROUND_SKEW_CAL[event.event_id])
        replay_base = all_pair_probs(scores_base, info["players"]).rename(columns={"p_a_lt_b": "replay_base_lt", "p_tie": "replay_base_tie"})
        replay_disp = all_pair_probs(scores_disp, info["players"]).rename(columns={"p_a_lt_b": "replay_disp_lt", "p_tie": "replay_disp_tie"})
        archived = info["archived_h2h"].rename(columns={"p_a_lt_b": "published_lt", "p_tie": "published_tie"})
        pairs = archived.merge(replay_base, on=["player_a", "player_b"], validate="one_to_one").merge(replay_disp, on=["player_a", "player_b"], validate="one_to_one")
        pairs["published_push"] = pairs["published_lt"] / (1.0-pairs["published_tie"])
        pairs["replay_base_push"] = pairs["replay_base_lt"] / (1.0-pairs["replay_base_tie"])
        pairs["replay_disp_push"] = pairs["replay_disp_lt"] / (1.0-pairs["replay_disp_tie"])
        pairs["paired_delta_pp"] = 100.0*(pairs["replay_disp_push"]-pairs["replay_base_push"])
        pairs["anchored_dispersion_push"] = np.clip(pairs["published_push"]+pairs["paired_delta_pp"]/100.0, EPS, 1.0-EPS)
        actual = actual_r1_scores(args.db, event.event_id)
        pairs["actual_a"] = pairs["player_a"].map(actual)
        pairs["actual_b"] = pairs["player_b"].map(actual)
        pairs["event"] = event.label
        pairs["event_id"] = event.event_id
        pairs["expected_avg_recovered"] = info["expected_avg"]
        pairs["baseline_reconstruction_error_pp"] = 100.0*(pairs["replay_base_push"]-pairs["published_push"])
        pair_outputs.append(pairs)
        for subset in ("market_like", "all_pairs"):
            row = score_frame(pairs, subset)
            row.update({
                "event": event.label, "event_id": event.event_id, "field": len(info["players"]),
                "expected_avg_recovered": info["expected_avg"],
                "feature_min_coverage": min(diagnostics["coverage"].values()),
                "reconstruction_mae_pp": float(pairs["baseline_reconstruction_error_pp"].abs().mean()),
                "reconstruction_rmse_pp": float(np.sqrt(np.mean(pairs["baseline_reconstruction_error_pp"]**2))),
            })
            summary_rows.append(row)
        print(f"field={len(info['players'])} expected_avg={info['expected_avg']:.1f} coverage={diagnostics['coverage']} reconstruction MAE={summary_rows[-1]['reconstruction_mae_pp']:.3f}pp market-like Brier delta={summary_rows[-2]['brier_delta']:+.6f}", flush=True)
        del scores_base, scores_disp
        gc.collect()

    pairs_df = pd.concat(pair_outputs, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    pairs_df.to_csv(args.output / "r1_pair_prices_and_outcomes.csv", index=False)
    summary.to_csv(args.output / "r1_event_metrics.csv", index=False)

    pooled_rows = []
    for subset in ("market_like", "all_pairs"):
        use = pairs_df[pairs_df["actual_a"].notna() & pairs_df["actual_b"].notna() & (pairs_df["actual_a"] != pairs_df["actual_b"])].copy()
        if subset == "market_like":
            use = use[use["published_push"].between(0.35, 0.65)]
        y = (use["actual_a"] < use["actual_b"]).astype(float).to_numpy()
        p0 = use["published_push"].to_numpy(float); p1 = use["anchored_dispersion_push"].to_numpy(float)
        pooled_rows.append({
            "subset": subset, "graded_pairs": len(use),
            "brier_published": np.mean((p0-y)**2), "brier_dispersion": np.mean((p1-y)**2),
            "logloss_published": np.mean(-(y*np.log(p0)+(1-y)*np.log(1-p0))),
            "logloss_dispersion": np.mean(-(y*np.log(p1)+(1-y)*np.log(1-p1))),
            "mean_abs_price_delta_pp": use["paired_delta_pp"].abs().mean(),
            "p95_abs_price_delta_pp": use["paired_delta_pp"].abs().quantile(0.95),
        })
    pooled = pd.DataFrame(pooled_rows)
    pooled["brier_delta"] = pooled["brier_dispersion"]-pooled["brier_published"]
    pooled["logloss_delta"] = pooled["logloss_dispersion"]-pooled["logloss_published"]
    pooled.to_csv(args.output / "r1_pooled_metrics.csv", index=False)

    market_pairs = pairs_df[
        pairs_df["published_push"].between(0.35, 0.65)
        & pairs_df["actual_a"].notna()
        & pairs_df["actual_b"].notna()
        & (pairs_df["actual_a"] != pairs_df["actual_b"])
    ].copy()
    market_pairs["abs_delta"] = market_pairs["paired_delta_pp"].abs()
    biggest = market_pairs.nlargest(25, "abs_delta")
    event_view = summary[summary["subset"] == "market_like"][["event", "graded_pairs", "reconstruction_mae_pp", "mean_abs_price_delta_pp", "p95_abs_price_delta_pp", "brier_delta", "logloss_delta"]]
    report = [
        "# R1 matchup shot-dispersion backtest", "",
        f"Generated from {len(EVENTS)} events with {args.sims:,} paired simulations per arm/event and {valid_shots:,} valid union-field pre-event shot rows.", "",
        "## Method", "",
        "The historical R1 cache metadata supplies the exact round skill and weather values used by the board. The expected scoring average is recovered from each archived joint sample and rounded to its production tenth. Course shape and the 50-round shot feature are rebuilt using only rounds before the event. Published push-fair H2H is then adjusted by the paired replay delta. Actual R1 ties are treated as pushes and excluded.", "",
        "Because the book matchup menus were not retained, the primary `market_like` sample includes archived board pairs priced 35%-65%; `all_pairs` is a sensitivity check.", "",
        "## Result", "",
        "The full-strength overlay was slightly worse for R1 H2H: market-like Brier increased 0.030% and log loss increased 0.023%. It improved 3M but worsened Rocket, Wyndham, and St. Jude. The average absolute price move was 0.257 percentage points and the 95th percentile was 0.777 points.", "",
        "This is effectively a neutral result, not persuasive evidence of harm. With only four event clusters, an event-level 95% interval for the mean Brier delta spans approximately -0.000214 to +0.000369. It does mean the recent replay supplies no positive evidence for using the tournament-selected full-strength variance replacement in R1 matchups.", "",
        "## Pooled result", "", markdown_table(pooled, 6), "",
        "## Market-like result by event", "", markdown_table(event_view, 6), "",
        "## Largest market-like price changes", "",
        markdown_table(biggest[["event", "player_a", "player_b", "actual_a", "actual_b", "published_push", "anchored_dispersion_push", "paired_delta_pp"]], 5), "",
        "## Limits", "",
        "There are only four independent event clusters. The all-pairs rows within an event are highly correlated, and the 35%-65% filter is a proxy for actual sportsbook menus rather than a recovered offered-matchup list. The archived baseline anchors the board price exactly, while the counterfactual delta inherits modest reconstruction error from unavailable historical Sheet course cells.", "",
    ]
    (args.output / "R1_MATCHUP_BACKTEST.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {args.output / 'R1_MATCHUP_BACKTEST.md'}")


if __name__ == "__main__":
    main()

"""Compare current BMW dispersion production outputs with a preserved baseline.

Read-only with respect to simulator outputs.  Writes CSV and Markdown audit
artifacts beneath an explicit output directory; no APIs, pricing, Sheets,
email, bet storage, or publishing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fair_american(probability: float) -> float:
    probability = float(probability)
    if probability <= 0.0 or probability >= 1.0:
        return np.nan
    if probability < 0.5:
        return 100.0 * (1.0 - probability) / probability
    return -100.0 * probability / (1.0 - probability)


def fmt_odds(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:+.0f}"


def fmt_name(value: str) -> str:
    if "," not in value:
        return value.title()
    last, first = [part.strip() for part in value.split(",", 1)]
    return f"{first.title()} {last.title()}"


def load_finish_probs(path: Path) -> pd.DataFrame:
    rank = pd.read_parquet(path)
    records = []
    for player, group in rank.groupby("player_name", sort=False):
        records.append(
            {
                "player_name": player,
                "win": group.loc[group["rank"] == 1, "prob_u"].sum(),
                "top5": group.loc[group["rank"] <= 5, "prob_u"].sum(),
                "top10": group.loc[group["rank"] <= 10, "prob_u"].sum(),
                "top20": group.loc[group["rank"] <= 20, "prob_u"].sum(),
            }
        )
    return pd.DataFrame(records)


def build_player_comparison(baseline_dir: Path, current_dir: Path) -> pd.DataFrame:
    old = load_finish_probs(baseline_dir / "rank_probs_updated_bmw.parquet")
    new = load_finish_probs(current_dir / "rank_probs_updated_bmw.parquet")
    out = old.merge(new, on="player_name", suffixes=("_old", "_new"), validate="1:1")
    for market in ("win", "top5", "top10", "top20"):
        out[f"{market}_delta_pp"] = 100.0 * (out[f"{market}_new"] - out[f"{market}_old"])
        out[f"{market}_old_american"] = out[f"{market}_old"].map(fair_american)
        out[f"{market}_new_american"] = out[f"{market}_new"].map(fair_american)

    old_names = json.loads((baseline_dir / "bmw" / "player_names.json").read_text())
    new_names = json.loads((current_dir / "bmw" / "player_names.json").read_text())
    if old_names != new_names:
        raise ValueError("Tournament player ordering changed between baseline and dispersion run")
    old_scores = np.load(baseline_dir / "bmw" / "final_scores.npy", mmap_mode="r")
    new_scores = np.load(current_dir / "bmw" / "final_scores.npy", mmap_mode="r")
    score_stats = pd.DataFrame(
        {
            "player_name": old_names,
            "score_mean_old": old_scores.mean(axis=1),
            "score_mean_new": new_scores.mean(axis=1),
            "score_sd_old": old_scores.std(axis=1, ddof=1),
            "score_sd_new": new_scores.std(axis=1, ddof=1),
        }
    )
    score_stats["score_mean_delta"] = score_stats["score_mean_new"] - score_stats["score_mean_old"]
    score_stats["score_sd_delta"] = score_stats["score_sd_new"] - score_stats["score_sd_old"]
    return out.merge(score_stats, on="player_name", validate="1:1")


def orient_h2h_movers(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in comparison.itertuples(index=False):
        delta = float(row.prob_a_new - row.prob_a_old)
        if delta >= 0:
            mover, opponent = row.player_a, row.player_b
            old_prob, new_prob = row.prob_a_old, row.prob_a_new
        else:
            mover, opponent = row.player_b, row.player_a
            old_prob, new_prob = 1.0 - row.prob_a_old, 1.0 - row.prob_a_new
        rows.append(
            {
                "mover": mover,
                "opponent": opponent,
                "mover_prob_old": old_prob,
                "mover_prob_new": new_prob,
                "mover_delta_pp": 100.0 * (new_prob - old_prob),
                "mover_odds_old_american": fair_american(old_prob),
                "mover_odds_new_american": fair_american(new_prob),
                "tie_pct_old": row.tie_pct_old,
                "tie_pct_new": row.tie_pct_new,
            }
        )
    return pd.DataFrame(rows).sort_values("mover_delta_pp", ascending=False)


def build_tournament_h2h(baseline_dir: Path, current_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_parquet(baseline_dir / "h2h_matrix_bmw.parquet")
    new = pd.read_parquet(current_dir / "h2h_matrix_bmw.parquet")
    keys = ["player_a", "player_b"]
    out = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="1:1")
    out["prob_a_delta_pp"] = 100.0 * (out["prob_a_new"] - out["prob_a_old"])
    out["prob_a_old_american"] = out["prob_a_old"].map(fair_american)
    out["prob_a_new_american"] = out["prob_a_new"].map(fair_american)
    return out, orient_h2h_movers(out)


def cache_h2h(cache: pd.DataFrame) -> pd.DataFrame:
    players = cache.index.astype(str).tolist()
    scores = cache.to_numpy(dtype=np.int16, copy=False)
    records = []
    for i, player_a in enumerate(players):
        for j in range(i + 1, len(players)):
            wins = int(np.sum(scores[i] < scores[j]))
            losses = int(np.sum(scores[i] > scores[j]))
            ties = int(np.sum(scores[i] == scores[j]))
            denom = wins + losses
            records.append(
                {
                    "player_a": player_a,
                    "player_b": players[j],
                    "prob_a": wins / denom if denom else np.nan,
                    "tie_pct": ties / (wins + losses + ties),
                }
            )
    return pd.DataFrame(records)


def build_round_h2h(baseline_dir: Path, current_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_cache = pd.read_parquet(baseline_dir / "bmw" / "sim_cache_r1.parquet")
    new_cache = pd.read_parquet(current_dir / "bmw" / "sim_cache_r1.parquet")
    if not old_cache.index.equals(new_cache.index):
        raise ValueError("R1 player ordering changed between baseline and dispersion run")
    old = cache_h2h(old_cache)
    new = cache_h2h(new_cache)
    keys = ["player_a", "player_b"]
    out = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="1:1")
    out["prob_a_delta_pp"] = 100.0 * (out["prob_a_new"] - out["prob_a_old"])
    out["prob_a_old_american"] = out["prob_a_old"].map(fair_american)
    out["prob_a_new_american"] = out["prob_a_new"].map(fair_american)
    return out, orient_h2h_movers(out)


def player_market_table(players: pd.DataFrame, market: str, limit: int = 10) -> list[str]:
    selected = players.reindex(players[f"{market}_delta_pp"].abs().sort_values(ascending=False).index).head(limit)
    lines = ["| Player | Current | Dispersion | Delta | Current fair | New fair |", "|---|---:|---:|---:|---:|---:|"]
    for row in selected.itertuples(index=False):
        old_prob = getattr(row, f"{market}_old")
        new_prob = getattr(row, f"{market}_new")
        delta = getattr(row, f"{market}_delta_pp")
        old_odds = getattr(row, f"{market}_old_american")
        new_odds = getattr(row, f"{market}_new_american")
        lines.append(
            f"| {fmt_name(row.player_name)} | {100*old_prob:.3f}% | {100*new_prob:.3f}% | "
            f"{delta:+.3f} pp | {fmt_odds(old_odds)} | {fmt_odds(new_odds)} |"
        )
    return lines


def h2h_table(movers: pd.DataFrame, limit: int = 12) -> list[str]:
    lines = ["| Probability increased for | Opponent | Current | Dispersion | Delta | Current fair | New fair |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in movers.head(limit).itertuples(index=False):
        lines.append(
            f"| {fmt_name(row.mover)} | {fmt_name(row.opponent)} | "
            f"{100*row.mover_prob_old:.2f}% | {100*row.mover_prob_new:.2f}% | "
            f"{row.mover_delta_pp:+.2f} pp | {fmt_odds(row.mover_odds_old_american)} | "
            f"{fmt_odds(row.mover_odds_new_american)} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    players = build_player_comparison(args.baseline_dir, args.current_dir)
    tournament_h2h, tournament_movers = build_tournament_h2h(args.baseline_dir, args.current_dir)
    r1_h2h, r1_movers = build_round_h2h(args.baseline_dir, args.current_dir)

    players.to_csv(args.output_dir / "bmw_dispersion_player_odds_deltas.csv", index=False)
    tournament_h2h.to_parquet(args.output_dir / "bmw_dispersion_tournament_h2h_deltas.parquet", index=False)
    tournament_movers.to_csv(args.output_dir / "bmw_dispersion_tournament_h2h_biggest.csv", index=False)
    r1_h2h.to_parquet(args.output_dir / "bmw_dispersion_r1_h2h_deltas.parquet", index=False)
    r1_movers.to_csv(args.output_dir / "bmw_dispersion_r1_h2h_biggest.csv", index=False)

    report = [
        "# BMW full-strength shot-dispersion production comparison",
        "",
        "The comparison is the preserved production sim immediately before the overlay versus the sim-only production rerun. The feature changes only player/category variance; model means, category means, correlations, skew inputs, weather logic, course multipliers, and week latent are unchanged.",
        "",
        "Weights on field-mean-scaled shot variance: OTT 90%, APP 90%, ARG 80%, PUTT 95%.",
    ]
    for market, title in (("win", "Outright win"), ("top5", "Top 5"), ("top10", "Top 10"), ("top20", "Top 20")):
        report.extend(["", f"## {title}: biggest absolute player deltas", ""])
        report.extend(player_market_table(players, market))
    report.extend(["", "## Tournament H2H: biggest fair-probability increases", ""])
    report.extend(h2h_table(tournament_movers))
    report.extend(["", "## R1 H2H: biggest fair-probability increases", ""])
    report.extend(h2h_table(r1_movers))
    report.extend(
        [
            "",
            "H2H probabilities exclude ties (ties push). Fair odds are raw no-vig American odds. The Monte Carlo uses 100,000 paired simulations; small changes can include rounding and finite-simulation noise.",
            "",
        ]
    )
    (args.output_dir / "BMW_FULL_DISPERSION_PROD_COMPARISON.md").write_text("\n".join(report), encoding="utf-8")
    print(args.output_dir / "BMW_FULL_DISPERSION_PROD_COMPARISON.md")


if __name__ == "__main__":
    main()

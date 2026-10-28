"""Run paired BMW shot-feature shadow simulations from one frozen fixture.

This script never prices, emails, stores bets, publishes, or mutates the weekly
production cache.  It consumes a fixture captured by ``new_sim.py`` and writes
all comparison artifacts beneath an explicit output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sims_kernel  # noqa: E402
from skew_calibration import calibrate_round_skew, calibrate_total_skew  # noqa: E402


CATS = ("ott", "app", "arg", "putt")
CAT_INDEX = {cat: index for index, cat in enumerate(CATS)}
OOS_VARIANCE_WEIGHTS = {
    "ott": 0.90,
    "app": 0.90,
    "arg": 0.80,
    "putt": 0.95,
}


def _norm(value: object) -> str:
    return str(value).casefold().strip()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_array(value: np.ndarray, dtype: type | None = float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.ascontiguousarray(array)


def load_inputs(
    fixture_path: Path,
    features_path: Path,
    dists_path: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    with np.load(fixture_path, allow_pickle=True) as source:
        fixture = {name: source[name].copy() for name in source.files}
    players = [_norm(value) for value in fixture["player_names"].tolist()]
    if len(players) != len(set(players)):
        raise ValueError("Fixture player names are not unique")

    features = pd.read_csv(features_path)
    features["player_name"] = features["player_name"].map(_norm)
    features = features.drop_duplicates("player_name").set_index("player_name")
    missing = sorted(set(players) - set(features.index))
    if missing:
        raise ValueError(f"Feature file is missing {len(missing)} fixture players: {missing}")
    features = features.loc[players].reset_index()

    dists = pd.read_csv(dists_path)
    dists["player_name"] = dists["player_name"].map(_norm)
    dists["category"] = dists["category_clean"].str.removeprefix("sg_")
    std = dists.pivot(index="player_name", columns="category", values="std")
    missing_dists = sorted(set(players) - set(std.index))
    if missing_dists:
        raise ValueError(
            f"Distribution file is missing {len(missing_dists)} fixture players: {missing_dists}"
        )
    std = std.loc[players, list(CATS)].astype(float)
    if std.isna().any().any() or (std <= 0).any().any():
        raise ValueError("Distribution file has missing/non-positive category standard deviations")
    return fixture, features, std


def build_feature_inputs(features: pd.DataFrame, base_std: pd.DataFrame) -> pd.DataFrame:
    out = features[["player_name"]].copy()
    for cat in CATS:
        shot_col = f"{cat}_shot_indep_var50_shrunk"
        if shot_col not in features:
            raise ValueError(f"Feature file lacks {shot_col}")
        base_var = base_std[cat].to_numpy(float) ** 2
        shot_var = features[shot_col].to_numpy(float)
        if not np.isfinite(shot_var).all() or np.any(shot_var <= 0):
            raise ValueError(f"Feature file has invalid {shot_col}")

        # Match the current BMW field-average category variance.  This isolates
        # the tested player ranking and prevents a hidden field-wide variance
        # inflation from masquerading as a feature effect.
        scale = float(base_var.mean() / shot_var.mean())
        shot_scaled = shot_var * scale
        blend50 = 0.50 * base_var + 0.50 * shot_scaled
        weight = OOS_VARIANCE_WEIGHTS[cat]
        blend_oos = (1.0 - weight) * base_var + weight * shot_scaled

        out[f"{cat}_base_var"] = base_var
        out[f"{cat}_shot_var50_scaled"] = shot_scaled
        out[f"{cat}_blend50_var"] = blend50
        out[f"{cat}_blend_oos_var"] = blend_oos
        out[f"{cat}_shot_scale"] = scale
        out[f"{cat}_oos_weight"] = weight

    delta_candidates = (
        "app_cross_delta_sg_per_round",
        "app_cross_sg_delta",
        "app_dir_rms_cross_sg_delta",
        "app_dir_rms_cross_delta_sg",
    )
    delta_col = next((name for name in delta_candidates if name in features), None)
    if delta_col is None:
        raise ValueError(
            "Feature file lacks a frozen APP SG delta; expected one of "
            + ", ".join(delta_candidates)
        )
    app_delta = features[delta_col].to_numpy(float)
    if not np.isfinite(app_delta).all():
        raise ValueError(f"Feature file has invalid {delta_col}")
    # Preserve the BMW field mean exactly, even if a future builder changes its
    # centering convention.
    app_delta = app_delta - app_delta.mean()
    out["app_cross_sg_delta"] = app_delta
    return out


def arm_arrays(
    fixture: dict[str, np.ndarray],
    feature_inputs: pd.DataFrame,
    *,
    variance: str | None,
    app_cross_scale: float,
) -> dict[str, np.ndarray]:
    arrays = {
        "mu": fixture["mu"].astype(float).copy(),
        "std_course": fixture["std_course"].astype(float).copy(),
        "my_pred_base": fixture["my_pred_base"].astype(float).copy(),
        "r2_mu": fixture["r2_mu"].astype(float).copy(),
        "r3_mu": fixture["r3_mu"].astype(float).copy(),
        "r4_mu": fixture["r4_mu"].astype(float).copy(),
    }
    if variance:
        suffix = "blend50_var" if variance == "50" else "blend_oos_var"
        for cat, index in CAT_INDEX.items():
            base_var = feature_inputs[f"{cat}_base_var"].to_numpy(float)
            new_var = feature_inputs[f"{cat}_{suffix}"].to_numpy(float)
            arrays["std_course"][:, index] *= np.sqrt(new_var / base_var)

    if app_cross_scale:
        delta = (
            feature_inputs["app_cross_sg_delta"].to_numpy(float)
            * float(app_cross_scale)
        )
        # Raise/lower total skill and APP together.  Because both the category
        # sum and total target move by delta, new_sim's category recentering is
        # left unchanged and the increment remains APP-only.
        arrays["mu"][:, CAT_INDEX["app"]] += delta
        arrays["my_pred_base"] += delta
        arrays["r2_mu"] += delta
        arrays["r3_mu"] += delta
        arrays["r4_mu"] += delta
    return arrays


def run_tournament(
    fixture: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray | None]:
    result = sims_kernel.run_pretournament(
        _as_array(arrays["mu"]),
        _as_array(arrays["std_course"]),
        _as_array(fixture["eff_skew"]),
        _as_array(fixture["l_corr"]),
        _as_array(arrays["my_pred_base"]),
        _as_array(arrays["r2_mu"]),
        _as_array(arrays["r3_mu"]),
        _as_array(arrays["r4_mu"]),
        _as_array(fixture["weather_delta_r1"]),
        _as_array(fixture["weather_delta_r2"]),
        fixture["r1_high"].tolist(),
        fixture["r1_midh"].tolist(),
        fixture["r1_midl"].tolist(),
        fixture["r1_low"].tolist(),
        fixture["r2_lt6"].tolist(),
        fixture["r2_6_30"].tolist(),
        fixture["r2_30up"].tolist(),
        fixture["r3_lt6"].tolist(),
        fixture["r3_6_20"].tolist(),
        fixture["r3_30up"].tolist(),
        int(fixture["cut_line"]),
        bool(fixture["use_10_shot_rule"]),
        int(fixture["sims"]),
        int(fixture["seed"]),
        float(fixture.get("week_latent_sd", 0.0)),
    )
    scores = np.ascontiguousarray(result[0]).astype(np.int32)
    cat_means = tuple(np.asarray(item, dtype=float) for item in result[2:6])
    made_cut = (
        np.ascontiguousarray(result[6]).astype(bool)
        if len(result) >= 7
        else None
    )
    calibrate_total_skew(scores, made_cut, verbose=False)
    return scores, cat_means, made_cut


def run_r1(
    fixture: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    expected_score: float,
) -> np.ndarray:
    player_avg = np.full(len(arrays["my_pred_base"]), expected_score, dtype=float)
    # new_sim treats positive weather_delta as a cost and subtracts it.  The
    # single-round kernel adds wx_delta, hence the negative sign here.
    wx_delta = -fixture["weather_delta_r1"].astype(float)
    scores, _ = sims_kernel.run_single_round(
        _as_array(arrays["mu"]),
        _as_array(arrays["std_course"]),
        _as_array(fixture["eff_skew"]),
        _as_array(fixture["l_corr"]),
        _as_array(arrays["my_pred_base"]),
        _as_array(wx_delta),
        _as_array(player_avg),
        int(fixture["sims"]),
        789,
    )
    score_array = np.ascontiguousarray(scores).astype(np.int32)
    as_dict = {str(i): score_array[i] for i in range(score_array.shape[0])}
    calibrate_round_skew(as_dict, verbose=False)
    return np.stack([as_dict[str(i)] for i in range(score_array.shape[0])])


def finish_metrics(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = scores.min(axis=0)
    winner = scores == minimum[None, :]
    win_share = winner / winner.sum(axis=0, keepdims=True)
    win_prob = win_share.mean(axis=1)
    _, top_dh, _ = sims_kernel.aggregate_round(
        np.ascontiguousarray(scores.astype(np.int64))
    )
    return win_prob, np.asarray(top_dh, dtype=float)


def build_player_metrics(
    arm_scores: dict[str, np.ndarray],
    r1_scores: dict[str, np.ndarray],
    players: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, scores in arm_scores.items():
        win, top = finish_metrics(scores)
        r1 = r1_scores[arm]
        for i, player in enumerate(players):
            rows.append(
                {
                    "arm": arm,
                    "player_name": player,
                    "win_prob": win[i],
                    "top5_prob": top[i, 0],
                    "top10_prob": top[i, 1],
                    "top20_prob": top[i, 2],
                    "tournament_score_mean": float(scores[i].mean()),
                    "tournament_score_sd": float(scores[i].std()),
                    "r1_score_mean": float(r1[i].mean()),
                    "r1_score_sd": float(r1[i].std()),
                }
            )
    return pd.DataFrame(rows)


def add_player_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        metrics[metrics["arm"].eq("baseline")]
        .set_index("player_name")
        .drop(columns="arm")
    )
    pieces = []
    for arm, frame in metrics.groupby("arm", sort=False):
        if arm == "baseline":
            continue
        out = frame.set_index("player_name").drop(columns="arm").copy()
        for column in ("win_prob", "top5_prob", "top10_prob", "top20_prob"):
            out[f"{column}_delta_pp"] = 100.0 * (out[column] - baseline[column])
        for column in (
            "tournament_score_mean",
            "tournament_score_sd",
            "r1_score_mean",
            "r1_score_sd",
        ):
            out[f"{column}_delta"] = out[column] - baseline[column]
        out.insert(0, "arm", arm)
        pieces.append(out.reset_index())
    return pd.concat(pieces, ignore_index=True)


def h2h_deltas(
    arm_scores: dict[str, np.ndarray],
    players: list[str],
    scope: str,
) -> pd.DataFrame:
    baseline = arm_scores["baseline"]
    sims = baseline.shape[1]
    rows: list[dict[str, object]] = []
    for i, player_a in enumerate(players):
        for j in range(i + 1, len(players)):
            player_b = players[j]
            base_a = baseline[i]
            base_b = baseline[j]
            base_decisive = base_a != base_b
            base_prob = float((base_a < base_b).sum() / base_decisive.sum())
            base_half = np.where(base_a < base_b, 1.0, np.where(base_a > base_b, 0.0, 0.5))
            for arm, scores in arm_scores.items():
                if arm == "baseline":
                    continue
                a, b = scores[i], scores[j]
                decisive = a != b
                prob = float((a < b).sum() / decisive.sum())
                tie = float((~decisive).mean())
                half = np.where(a < b, 1.0, np.where(a > b, 0.0, 0.5))
                paired = half - base_half
                rows.append(
                    {
                        "scope": scope,
                        "arm": arm,
                        "player_a": player_a,
                        "player_b": player_b,
                        "baseline_prob_a": base_prob,
                        "prob_a": prob,
                        "delta_prob_a_pp": 100.0 * (prob - base_prob),
                        "tie_pct": 100.0 * tie,
                        "paired_mc_se_pp": 100.0 * float(paired.std(ddof=1) / np.sqrt(sims)),
                    }
                )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dep."""
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    feature_inputs: pd.DataFrame,
    deltas: pd.DataFrame,
    h2h: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    main = deltas[deltas["arm"].eq("combined50")].copy()
    win = main.reindex(main["win_prob_delta_pp"].abs().sort_values(ascending=False).index).head(10)
    volatility = main.reindex(
        main["tournament_score_sd_delta"].abs().sort_values(ascending=False).index
    ).head(10)
    main_h2h = h2h[
        h2h["arm"].eq("combined50") & h2h["scope"].eq("tournament")
    ].copy()
    main_h2h = main_h2h.reindex(
        main_h2h["delta_prob_a_pp"].abs().sort_values(ascending=False).index
    ).head(10)

    lines = [
        "# BMW shot-feature shadow — 2026-08-20",
        "",
        "## Design",
        "",
        "The feature snapshot is frozen before R1, but enters every round's category draw. "
        "The primary arm uses a conservative 50% blend of production and field-scale-preserving "
        "shot variance plus 50% of the field-centered APP cross-track mean delta.",
        "",
        f"- Players: {summary['players']}",
        f"- Simulations per arm: {summary['simulations']:,}",
        f"- Full fitted APP delta range: {summary['app_delta_min']:+.4f} to {summary['app_delta_max']:+.4f} SG/round",
        f"- Primary-arm APP delta range: {summary['primary_app_delta_min']:+.4f} to {summary['primary_app_delta_max']:+.4f} SG/round",
        f"- APP delta field mean: {summary['app_delta_mean']:+.8f}",
        "",
        "## Largest combined-arm outright movers",
        "",
        _markdown_table(
            win[["player_name", "win_prob", "win_prob_delta_pp", "top10_prob_delta_pp"]]
        ),
        "",
        "## Largest combined-arm volatility changes",
        "",
        _markdown_table(
            volatility[["player_name", "tournament_score_sd", "tournament_score_sd_delta", "r1_score_sd_delta"]]
        ),
        "",
        "## Largest combined-arm tournament H2H changes",
        "",
        _markdown_table(
            main_h2h[["player_a", "player_b", "baseline_prob_a", "prob_a", "delta_prob_a_pp", "paired_mc_se_pp"]]
        ),
        "",
        "## Integrity checks",
        "",
        f"- Dispersion-only maximum absolute tournament mean shift: {summary['dispersion50_max_abs_score_mean_delta']:.6f} strokes.",
        f"- Combined field-average APP adjustment: {summary['app_delta_mean']:+.8f} SG/round.",
        "- All arms use the same fixture, Rust kernel, random seeds, week latent, course multipliers, and skew calibration.",
        "- No pricing, email, Sheets storage, dashboard publishing, or production-cache write occurred.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r1-expected-score", type=float, default=69.8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture, features, base_std = load_inputs(args.fixture, args.features, args.dists)
    feature_inputs = build_feature_inputs(features, base_std)
    players = [_norm(value) for value in fixture["player_names"].tolist()]

    specs = {
        "baseline": (None, 0.0),
        "dispersion50": ("50", 0.0),
        "dispersion_oos_weight": ("oos", 0.0),
        "app_cross50": (None, 0.50),
        "app_cross_full": (None, 1.0),
        # Primary conservative arm: both feature families are half-shrunk.
        "combined50": ("50", 0.50),
        # Sensitivity ceilings, not production recommendations.
        "combined50_app_full": ("50", 1.0),
        "combined_oos_weight": ("oos", 1.0),
    }
    tournament_scores: dict[str, np.ndarray] = {}
    round_scores: dict[str, np.ndarray] = {}
    cat_mean_rows: list[dict[str, object]] = []
    for arm, (variance, app_cross_scale) in specs.items():
        print(f"running {arm}...")
        arrays = arm_arrays(
            fixture,
            feature_inputs,
            variance=variance,
            app_cross_scale=app_cross_scale,
        )
        scores, cat_means, _ = run_tournament(fixture, arrays)
        tournament_scores[arm] = scores
        round_scores[arm] = run_r1(fixture, arrays, args.r1_expected_score)
        for round_no, means in enumerate(cat_means, start=1):
            for i, player in enumerate(players):
                row: dict[str, object] = {
                    "arm": arm,
                    "round": round_no,
                    "player_name": player,
                }
                for j, cat in enumerate(CATS):
                    row[f"{cat}_mean"] = float(means[i, j])
                cat_mean_rows.append(row)

    metrics = build_player_metrics(tournament_scores, round_scores, players)
    deltas = add_player_deltas(metrics)
    tournament_h2h = h2h_deltas(tournament_scores, players, "tournament")
    r1_h2h = h2h_deltas(round_scores, players, "r1")
    all_h2h = pd.concat([tournament_h2h, r1_h2h], ignore_index=True)

    feature_inputs.to_csv(args.output_dir / "bmw_feature_inputs.csv", index=False)
    metrics.to_csv(args.output_dir / "bmw_arm_player_metrics.csv", index=False)
    deltas.to_csv(args.output_dir / "bmw_arm_deltas_vs_baseline.csv", index=False)
    all_h2h.to_parquet(args.output_dir / "bmw_h2h_deltas.parquet", index=False)
    pd.DataFrame(cat_mean_rows).to_csv(
        args.output_dir / "bmw_arm_category_means.csv", index=False
    )
    np.save(args.output_dir / "combined50_final_scores.npy", tournament_scores["combined50"])
    np.save(args.output_dir / "combined50_r1_scores.npy", round_scores["combined50"])

    dispersion_delta = deltas[deltas["arm"].eq("dispersion50")]
    app_delta = feature_inputs["app_cross_sg_delta"]
    summary = {
        "status": "shadow_only",
        "players": len(players),
        "simulations": int(fixture["sims"]),
        "arms": list(specs),
        "r1_expected_score": args.r1_expected_score,
        "app_delta_min": float(app_delta.min()),
        "app_delta_max": float(app_delta.max()),
        "app_delta_mean": float(app_delta.mean()),
        "primary_app_delta_min": float(0.50 * app_delta.min()),
        "primary_app_delta_max": float(0.50 * app_delta.max()),
        "dispersion50_max_abs_score_mean_delta": float(
            dispersion_delta["tournament_score_mean_delta"].abs().max()
        ),
        "input_hashes": {
            "fixture": _hash(args.fixture),
            "features": _hash(args.features),
            "dists": _hash(args.dists),
        },
        "sims_kernel_version": sims_kernel.version(),
        "variance_oos_weights": OOS_VARIANCE_WEIGHTS,
    }
    (args.output_dir / "bmw_shadow_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(
        args.output_dir / "BMW_SHOT_FEATURE_SHADOW_REPORT.md",
        feature_inputs,
        deltas,
        all_h2h,
        summary,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

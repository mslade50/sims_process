"""Audit BMW shot-feature alignment with market prices without repricing.

This script intentionally separates two questions:

1. Existing gap: did the independently built shot features point in the same
   direction as the sharp-market disagreement already observed by the first
   pass of the production simulator?
2. Subsequent move: did those features predict Betcris movement after the
   production market snapshot?

The first result is descriptive and partly in-sample to this week's workflow:
``mkt_adj`` was constructed from the same market gap, and the paired shadow
simulation starts from predictions that already contain market regression.
The second result uses two timestamped Betcris snapshots, both of which predate
the shot-feature build, but it is still only one selected tournament.

No production inputs, prices, caches, or published artifacts are changed.
Only the requested audit report and joined audit CSV are written.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = (
    REPO
    / "permanent_data"
    / "shadow_logs"
    / "bmw_20260820"
    / "run_20260819_2225"
    / "shadow_comparison"
)
DEFAULT_LATEST = (
    REPO.parent / "golf_scraping" / "data" / "betcris_outrights_latest.json"
)


def _american_to_prob(odds: float) -> float:
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def _corr(x: pd.Series, y: pd.Series) -> dict[str, float]:
    frame = pd.concat([x, y], axis=1).dropna()
    pearson = stats.pearsonr(frame.iloc[:, 0], frame.iloc[:, 1])
    spearman = stats.spearmanr(frame.iloc[:, 0], frame.iloc[:, 1])
    return {
        "n": int(len(frame)),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _partial_corr(
    x: pd.Series, y: pd.Series, controls: pd.DataFrame
) -> dict[str, float]:
    frame = pd.concat([x.rename("x"), y.rename("y"), controls], axis=1).dropna()
    z = np.column_stack([np.ones(len(frame)), frame[controls.columns].to_numpy(float)])
    rx = frame["x"].to_numpy(float) - z @ np.linalg.lstsq(
        z, frame["x"].to_numpy(float), rcond=None
    )[0]
    ry = frame["y"].to_numpy(float) - z @ np.linalg.lstsq(
        z, frame["y"].to_numpy(float), rcond=None
    )[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = len(frame) - len(controls.columns) - 2
    t_value = r * math.sqrt(df / max(1e-15, 1.0 - r * r))
    p_value = float(2.0 * stats.t.sf(abs(t_value), df))
    return {"n": int(len(frame)), "r": r, "p": p_value, "df": int(df)}


def _md_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for header, value in zip(headers, row):
            if pd.isna(value):
                values.append("")
            elif header in formats:
                values.append(formats[header].format(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _load_latest_betcris(path: Path) -> tuple[pd.DataFrame, str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for line in payload.get("lines", []):
        if (
            str(line.get("event_id")) == "28"
            and line.get("market_type") == "winner"
            and line.get("book") == "betcris"
        ):
            odds = float(line["odds"])
            rows.append(
                {
                    "player_name": str(line["player"]).lower().strip(),
                    "betcris_new_american": odds,
                    "betcris_new_raw_prob": _american_to_prob(odds),
                }
            )
    frame = pd.DataFrame(rows)
    if frame["player_name"].duplicated().any():
        raise ValueError("Duplicate BMW winner prices in latest Betcris snapshot")
    return frame, str(payload.get("last_updated", "unknown"))


def build_analysis(latest_betcris: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    details = pd.read_csv(REPO / "final_predictions_bmw_details.csv")
    old_odds = pd.read_csv(REPO / "win_odds.csv")[["player_name", "betcris"]]
    feature_inputs = pd.read_csv(RUN_DIR / "bmw_feature_inputs.csv")
    arm_metrics = pd.read_csv(RUN_DIR / "bmw_arm_player_metrics.csv")
    latest, latest_timestamp = _load_latest_betcris(latest_betcris)

    for frame in (details, old_odds, feature_inputs, arm_metrics, latest):
        frame["player_name"] = frame["player_name"].astype(str).str.lower().str.strip()

    arms = arm_metrics.pivot(
        index="player_name", columns="arm", values="win_prob"
    ).reset_index()
    arms.columns.name = None

    joined = (
        details.merge(old_odds, on="player_name", how="inner", validate="one_to_one")
        .merge(feature_inputs, on="player_name", how="inner", validate="one_to_one")
        .merge(arms, on="player_name", how="inner", validate="one_to_one")
        .merge(latest, on="player_name", how="inner", validate="one_to_one")
    )
    expected = len(details)
    if len(joined) != expected:
        raise ValueError(f"BMW join coverage {len(joined)}/{expected}; expected complete field")

    categories = ("ott", "app", "arg", "putt")
    joined["prod_total_var"] = sum(joined[f"{c}_base_var"] for c in categories)
    joined["shot_total_var_scaled"] = sum(
        joined[f"{c}_shot_var50_scaled"] for c in categories
    )
    joined["shot_prod_var_log_ratio"] = np.log(
        joined["shot_total_var_scaled"] / joined["prod_total_var"]
    )

    # The market target used by mkt_regress is already normalized to 100%.
    joined["market_gap_firstpass_pp"] = joined["win_odds"] - joined["top_1_pct"]
    joined["market_gap_firstpass_log"] = np.log(
        (joined["win_odds"] / 100.0).clip(lower=1e-8)
    ) - np.log((joined["top_1_pct"] / 100.0).clip(lower=1e-8))

    for arm in ("dispersion50", "app_cross50", "combined50"):
        joined[f"{arm}_win_delta_pp"] = 100.0 * (
            joined[arm] - joined["baseline"]
        )
        joined[f"{arm}_win_log_change"] = np.log(
            joined[arm].clip(lower=1e-8)
        ) - np.log(joined["baseline"].clip(lower=1e-8))

    # Apples-to-apples Betcris movement. Normalize each snapshot to remove the
    # changing overround (121.38% -> 124.20% in the captured BMW snapshots).
    joined["betcris_old_raw_prob"] = joined["betcris"]
    joined["betcris_old_prob"] = (
        joined["betcris_old_raw_prob"] / joined["betcris_old_raw_prob"].sum()
    )
    joined["betcris_new_prob"] = (
        joined["betcris_new_raw_prob"] / joined["betcris_new_raw_prob"].sum()
    )
    joined["betcris_move_pp"] = 100.0 * (
        joined["betcris_new_prob"] - joined["betcris_old_prob"]
    )
    joined["betcris_move_log"] = np.log(joined["betcris_new_prob"]) - np.log(
        joined["betcris_old_prob"]
    )

    existing = {
        "combined_vs_firstpass_gap_pp": _corr(
            joined["combined50_win_delta_pp"], joined["market_gap_firstpass_pp"]
        ),
        "combined_vs_mkt_adj": _corr(
            joined["combined50_win_delta_pp"], joined["mkt_adj"]
        ),
        "dispersion_vs_mkt_adj": _corr(
            joined["dispersion50_win_delta_pp"], joined["mkt_adj"]
        ),
        "app_win_vs_mkt_adj": _corr(
            joined["app_cross50_win_delta_pp"], joined["mkt_adj"]
        ),
        "raw_variance_vs_relative_gap": _corr(
            joined["shot_prod_var_log_ratio"], joined["market_gap_firstpass_log"]
        ),
        "app_sg_vs_relative_gap": _corr(
            joined["app_cross_sg_delta"], joined["market_gap_firstpass_log"]
        ),
    }
    subsequent = {
        "combined_vs_move_pp": _corr(
            joined["combined50_win_delta_pp"], joined["betcris_move_pp"]
        ),
        "dispersion_vs_move_pp": _corr(
            joined["dispersion50_win_delta_pp"], joined["betcris_move_pp"]
        ),
        "raw_variance_vs_move_log": _corr(
            joined["shot_prod_var_log_ratio"], joined["betcris_move_log"]
        ),
        "app_sg_vs_move_log": _corr(
            joined["app_cross_sg_delta"], joined["betcris_move_log"]
        ),
    }

    partial = _partial_corr(
        joined["shot_prod_var_log_ratio"],
        joined["market_gap_firstpass_log"],
        joined[["pred", "top_1_pct", "c_adj", "sample"]],
    )

    baseline_error = 100.0 * joined["baseline"] - joined["win_odds"]
    combined_error = 100.0 * joined["combined50"] - joined["win_odds"]
    sign_align = (
        joined["combined50_win_delta_pp"] * joined["market_gap_firstpass_pp"] > 0
    )
    largest_gaps = joined.reindex(
        joined["market_gap_firstpass_pp"].abs().sort_values(ascending=False).index
    ).head(10)

    summary: dict[str, object] = {
        "n": int(len(joined)),
        "latest_timestamp": latest_timestamp,
        "latest_path": str(latest_betcris.resolve()),
        "old_file_mtime": datetime.fromtimestamp(
            (REPO / "win_odds.csv").stat().st_mtime
        ).astimezone().isoformat(),
        "feature_file_mtime": datetime.fromtimestamp(
            (REPO / "bmw_shot_features_shadow.csv").stat().st_mtime
        ).astimezone().isoformat(),
        "old_overround": float(joined["betcris_old_raw_prob"].sum()),
        "new_overround": float(joined["betcris_new_raw_prob"].sum()),
        "existing": existing,
        "subsequent": subsequent,
        "partial_variance_gap": partial,
        "sign_align_n": int(sign_align.sum()),
        "largest_10_sign_align_n": int(
            (
                largest_gaps["combined50_win_delta_pp"]
                * largest_gaps["market_gap_firstpass_pp"]
                > 0
            ).sum()
        ),
        "baseline_mae_pp": float(baseline_error.abs().mean()),
        "combined_mae_pp": float(combined_error.abs().mean()),
        "baseline_rmse_pp": float(np.sqrt(np.mean(baseline_error**2))),
        "combined_rmse_pp": float(np.sqrt(np.mean(combined_error**2))),
    }
    return joined, summary


def render_report(joined: pd.DataFrame, summary: dict[str, object]) -> str:
    existing = summary["existing"]
    subsequent = summary["subsequent"]

    def corr_rows(items: list[tuple[str, dict[str, float]]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Comparison": label,
                    "Pearson r": result["pearson"],
                    "Pearson p": result["pearson_p"],
                    "Spearman rho": result["spearman"],
                    "Spearman p": result["spearman_p"],
                }
                for label, result in items
            ]
        )

    existing_rows = corr_rows(
        [
            ("Combined win delta vs first-pass market gap", existing["combined_vs_firstpass_gap_pp"]),
            ("Combined win delta vs mkt_adj", existing["combined_vs_mkt_adj"]),
            ("Dispersion-only win delta vs mkt_adj", existing["dispersion_vs_mkt_adj"]),
            ("APP-only win delta vs mkt_adj", existing["app_win_vs_mkt_adj"]),
            ("Raw shot/prod variance ratio vs relative gap", existing["raw_variance_vs_relative_gap"]),
            ("Raw APP cross-track SG signal vs relative gap", existing["app_sg_vs_relative_gap"]),
        ]
    )
    move_rows = corr_rows(
        [
            ("Combined win delta vs Betcris move", subsequent["combined_vs_move_pp"]),
            ("Dispersion-only win delta vs Betcris move", subsequent["dispersion_vs_move_pp"]),
            ("Raw shot/prod variance ratio vs Betcris log move", subsequent["raw_variance_vs_move_log"]),
            ("Raw APP cross-track SG signal vs Betcris log move", subsequent["app_sg_vs_move_log"]),
        ]
    )

    top = joined.reindex(
        joined["combined50_win_delta_pp"].abs().sort_values(ascending=False).index
    ).head(12)[
        [
            "player_name",
            "market_gap_firstpass_pp",
            "shot_prod_var_log_ratio",
            "app_cross_sg_delta",
            "dispersion50_win_delta_pp",
            "app_cross50_win_delta_pp",
            "combined50_win_delta_pp",
            "betcris_move_pp",
        ]
    ].copy()
    top.columns = [
        "Player",
        "Initial market gap (pp)",
        "Shot/prod variance log ratio",
        "APP SG delta",
        "Dispersion win delta (pp)",
        "APP win delta (pp)",
        "Combined win delta (pp)",
        "Later Betcris move (pp)",
    ]

    partial = summary["partial_variance_gap"]
    return f"""# BMW shot-feature market-alignment audit

Generated from frozen local artifacts. This audit did not run pricing, publish data, call a live odds API, or alter production outputs.

## Bottom line

The BMW shot feature—especially its dispersion component—strongly agrees with the **market disagreement that already existed** before the shadow run. That is useful corroboration, but it is not independent validation: the market gap itself created `mkt_adj`, the paired shadow baseline already includes market regression, and BMW was inspected after noticing the agreement.

The feature did **not** predict the subsequent Betcris update. That later-move test is cleaner temporally, but it is only one tournament and is not necessarily the right economic expectation: these shot histories existed before the opening snapshot, so an efficient market could have incorporated them already.

## Existing market-gap alignment

{_md_table(existing_rows, {'Pearson r': '{:+.3f}', 'Pearson p': '{:.3g}', 'Spearman rho': '{:+.3f}', 'Spearman p': '{:.3g}'})}

For the most feature-independent comparison, the raw field-scaled shot/prod total-variance log ratio correlates {existing['raw_variance_vs_relative_gap']['pearson']:+.3f} with the first-pass relative market gap. After linearly controlling for production skill, initial win probability, course adjustment, and sample size, partial r is {partial['r']:+.3f} (p={partial['p']:.3g}; df={partial['df']}). This removes the obvious favorite/longshot leverage but does not cure the one-event/post-selection problem.

The conservative combined arm moves in the initial market-gap direction for {summary['sign_align_n']}/{summary['n']} players and for {summary['largest_10_sign_align_n']}/10 of the largest initial disagreements. Relative to the same market target used by `mkt_regress`, absolute win-probability error changes as follows:

| Metric | Paired baseline | Combined50 shadow |
| --- | ---: | ---: |
| MAE (percentage points) | {summary['baseline_mae_pp']:.3f} | {summary['combined_mae_pp']:.3f} |
| RMSE (percentage points) | {summary['baseline_rmse_pp']:.3f} | {summary['combined_rmse_pp']:.3f} |

Do not interpret that error reduction as an out-of-sample market backtest; it uses this week's already-observed market target.

## Subsequent Betcris movement

Old snapshot: `{(REPO / 'win_odds.csv').resolve()}` (filesystem time {summary['old_file_mtime']}).

New snapshot: `{summary['latest_path']}` (payload time {summary['latest_timestamp']}).

Both sides use Betcris winner prices normalized within the same 50-player field. The normalization removes the overround change from {summary['old_overround']:.3f} to {summary['new_overround']:.3f}.

{_md_table(move_rows, {'Pearson r': '{:+.3f}', 'Pearson p': '{:.3g}', 'Spearman rho': '{:+.3f}', 'Spearman p': '{:.3g}'})}

None of the direct feature-to-subsequent-move comparisons is statistically distinguishable from zero. In particular, the raw variance ratio has Pearson r={subsequent['raw_variance_vs_move_log']['pearson']:+.3f}, and the raw APP signal has r={subsequent['app_sg_vs_move_log']['pearson']:+.3f}.

## Largest feature-driven moves

{_md_table(top, {column: '{:+.3f}' for column in top.columns if column != 'Player'})}

## What would count as further validation

1. Freeze the exact feature recipe and strengths before inspecting another tournament.
2. Save the opening market snapshot, production fixture/category distributions, feature snapshot, and closing snapshot every week.
3. Score three separate targets: realized category SG/72-hole performance, closing market residual relative to the untouched baseline, and bet-level log loss/CLV. Never tune on the same target used to declare success.
4. Use event-level folds or leave-one-event-out fitting and event-clustered uncertainty. Player rows within a field are not independent because all market probabilities are normalized together.
5. Benchmark against simple alternatives: a favorite/longshot calibration curve, production category variance alone, trailing category SG variance, and DataGolf skill. The shot feature must add value after those controls.
6. Treat the APP mean and dispersion paths as separate candidates. BMW market corroboration belongs mainly to dispersion; the raw APP cross-track signal has weak existing-gap alignment and no later-move confirmation.

## Historical-data feasibility

The local scraper contains 41 timestamped Betcris snapshots across four PGA events with usable opens/closes: PGA Championship, CJ Cup Byron Nelson, Charles Schwab Challenge, and RBC Canadian Open. The SQLite odds tape plus archived rank distributions overlap roughly twelve March–June events, but most SQLite outright prices are Kalshi/Novig and the exact historical production category distributions/paired-simulation fixtures were not retained. Therefore:

- a raw frozen-feature-to-market/outcome panel is feasible now and should be treated as exploratory;
- exact historical paired-simulator replay is not feasible from current local artifacts;
- prospective archival starting with BMW is the cleanest confirmatory design.

Nominal p-values above are cross-sectional and do not correct the one-event dependence or post-selection.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latest-betcris", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--report", type=Path, default=RUN_DIR / "BMW_SHOT_MARKET_ALIGNMENT.md")
    parser.add_argument("--audit-csv", type=Path, default=RUN_DIR / "bmw_shot_market_alignment.csv")
    args = parser.parse_args()

    joined, summary = build_analysis(args.latest_betcris)
    report = render_report(joined, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    joined.to_csv(args.audit_csv, index=False)
    print(f"[ok] report: {args.report.resolve()}")
    print(f"[ok] audit rows: {len(joined)} -> {args.audit_csv.resolve()}")


if __name__ == "__main__":
    main()

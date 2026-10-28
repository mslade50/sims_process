"""Create robustness tables from the completed 2026 distribution study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(ARCHIVE))
from backtest_distribution_engines_2026 import cluster_bootstrap_delta, markdown_table  # noqa: E402


def pooled_delta(frame: pd.DataFrame, arm: str, loss: str) -> float:
    alt = frame[frame["arm"].eq(arm)]
    base = frame[frame["arm"].eq("category_current")]
    return float(
        alt[f"{loss}_sum"].sum() / alt["n"].sum()
        - base[f"{loss}_sum"].sum() / base["n"].sum()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "permanent_data"
        / "distribution_engine_backtest_2026",
    )
    args = parser.parse_args()
    root = args.input
    audit = pd.read_csv(root / "event_audit.csv")
    matchup = pd.read_csv(root / "round_matchup_event_metrics.csv")
    scores = pd.read_csv(root / "round_score_event_metrics.csv")
    tournament = pd.read_csv(root / "tournament_event_metrics.csv")
    audit["shot_coverage_rate"] = audit["min_shot_coverage"] / audit["skill_field"]
    ordered = audit.sort_values("start")

    filters = {
        "all_events": set(audit["event_key"]),
        "single_course": set(audit.loc[audit["course_count"].eq(1), "event_key"]),
        "shot_coverage_95pct": set(audit.loc[audit["shot_coverage_rate"].ge(0.95), "event_key"]),
        "single_course_and_95pct": set(
            audit.loc[audit["course_count"].eq(1) & audit["shot_coverage_rate"].ge(0.95), "event_key"]
        ),
        "first_17_events": set(ordered.head(17)["event_key"]),
        "last_17_events": set(ordered.tail(17)["event_key"]),
    }
    scope_specs = [
        ("r1_h2h_brier", matchup, "brier", "round_num == 1"),
        ("all_round_h2h_brier", matchup, "brier", None),
        ("r1_score_crps", scores, "crps", "round_num == 1"),
        ("r1_score_logpmf", scores, "logpmf", "round_num == 1"),
        ("all_round_score_crps", scores, "crps", None),
        ("all_round_score_logpmf", scores, "logpmf", None),
        ("static_top5_brier", tournament, "brier", "metric == 'top5'"),
        ("static_win_logloss", tournament, "logloss", "metric == 'win'"),
    ]

    rows: list[dict[str, object]] = []
    for filter_name, keys in filters.items():
        for scope, source, loss, query in scope_specs:
            frame = source[source["event_key"].isin(keys)]
            if query:
                frame = frame.query(query)
            for arm in ("category_shot", "normal_2p8", "normal_player"):
                delta, low, high = cluster_bootstrap_delta(frame, arm, loss, draws=10_000)
                rows.append(
                    {
                        "filter": filter_name,
                        "events": int(frame["event_key"].nunique()),
                        "scope": scope,
                        "arm": arm,
                        "delta_vs_category_current": pooled_delta(frame, arm, loss),
                        "cluster95_low": low,
                        "cluster95_high": high,
                        "n": int(frame.loc[frame["arm"].eq(arm), "n"].sum()),
                    }
                )
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(root / "sensitivity_comparison.csv", index=False)

    directions = []
    for scope, source, loss, query in scope_specs:
        frame = source.query(query) if query else source
        event_rows = []
        for event_key, event in frame.groupby("event_key"):
            event_rows.append((event_key, pooled_delta(event, "category_shot", loss)))
        values = np.asarray([value for _, value in event_rows])
        directions.append(
            {
                "scope": scope,
                "events": len(values),
                "shot_improved_events": int(np.sum(values < 0)),
                "shot_worsened_events": int(np.sum(values > 0)),
                "median_event_delta": float(np.median(values)),
            }
        )
    direction_frame = pd.DataFrame(directions)
    direction_frame.to_csv(root / "event_direction_counts.csv", index=False)

    headline_scopes = {
        "r1_h2h_brier",
        "r1_score_crps",
        "r1_score_logpmf",
        "all_round_score_logpmf",
        "static_top5_brier",
    }
    headline = sensitivity[
        sensitivity["filter"].eq("all_events")
        & sensitivity["scope"].isin(headline_scopes)
    ]
    shot_robustness = sensitivity[
        sensitivity["arm"].eq("category_shot")
        & sensitivity["scope"].isin({"r1_h2h_brier", "all_round_score_logpmf", "static_top5_brier"})
    ]
    report = [
        "# 2026 distribution-engine robustness checks",
        "",
        "Negative deltas improve on reconstructed `category_current`. Confidence intervals resample whole events.",
        "",
        "## Primary arms",
        "",
        markdown_table(headline, 7),
        "",
        "## Shot-dispersion sensitivities",
        "",
        markdown_table(shot_robustness, 7),
        "",
        "## Direction by event",
        "",
        markdown_table(direction_frame, 7),
        "",
        "The round-density result is stable in single-course events, fields with at least 95% shot coverage, and both halves of the season. R1 H2H remains effectively flat in every major cut. Tournament top-5 scoring is directionally better but less stable than the direct round-density result.",
        "",
    ]
    (root / "SENSITIVITY_CHECKS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {root / 'SENSITIVITY_CHECKS.md'}")


if __name__ == "__main__":
    main()

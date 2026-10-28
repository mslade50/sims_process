"""Test great-shot and blowup profiles on future distribution tails."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from shot_feature_predictive_study import (
    CONFIRM_YEAR,
    DISCOVERY_YEARS,
    MatrixBuilder,
    _center_within_event,
    load_event_panel,
)


DEFAULT_SIM_PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
BASE = ["prediction", "std_dev_50"]
GROUPS = {
    "great_rates": [
        "great_app_rate_prior40",
        "great_ott_rate_prior40",
        "great_short_rate_prior40",
        "great_total_rate_prior40",
    ],
    "blowup_profile": [
        "penalty_strokes_trail21_mean",
        "nonpenalty_tail_shot_count_trail8",
        "recovery_shot_count_trail8",
        "short_side_miss_rate_trail24",
    ],
    "great_p95_rates": [
        "great_app_p95_rate_prior40",
        "great_ott_p95_rate_prior40",
        "great_short_p95_rate_prior40",
        "great_total_p95_rate_prior40",
    ],
}
CS = (0.001, 0.01, 0.1, 1.0, 10.0)


def add_tail_labels(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    groups = out.groupby(["year", "event_id"])["r12_mean_residual"]
    lo = groups.transform(lambda values: values.quantile(0.10))
    hi = groups.transform(lambda values: values.quantile(0.90))
    out["positive_tail"] = (out["r12_mean_residual"] >= hi).astype(int)
    out["negative_tail"] = (out["r12_mean_residual"] <= lo).astype(int)
    return out


def fit_probabilities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target: str,
    c_value: float,
) -> np.ndarray:
    builder = MatrixBuilder(columns)
    x_train = builder.fit_transform(train)
    x_test = builder.transform(test)
    model = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs")
    model.fit(x_train, train[target].to_numpy(int))
    return model.predict_proba(x_test)[:, 1]


def score(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    c_value: float,
    years: tuple[int, ...],
) -> dict[str, float]:
    data = _center_within_event(panel, features)
    folds = []
    for year in years:
        train = data[data["year"] < year]
        test = data[data["year"].eq(year)]
        if len(train) < 500 or len(test) < 100:
            continue
        base = fit_probabilities(train, test, BASE, target, c_value)
        candidate = fit_probabilities(
            train, test, list(dict.fromkeys(BASE + features)), target, c_value
        )
        fold = test[["year", "event_id", target]].copy()
        fold["base"] = base
        fold["candidate"] = candidate
        folds.append(fold)
    scored = pd.concat(folds, ignore_index=True)
    y = scored[target].to_numpy(int)
    base_brier = brier_score_loss(y, scored["base"])
    candidate_brier = brier_score_loss(y, scored["candidate"])
    base_log = log_loss(y, scored["base"], labels=[0, 1])
    candidate_log = log_loss(y, scored["candidate"], labels=[0, 1])
    scored["brier_delta"] = (
        (y - scored["base"]) ** 2 - (y - scored["candidate"]) ** 2
    )
    event_delta = scored.groupby(["year", "event_id"])["brier_delta"].mean()
    se = event_delta.std(ddof=1) / np.sqrt(len(event_delta))
    return {
        "n": int(len(scored)),
        "events": int(len(event_delta)),
        "base_brier": float(base_brier),
        "candidate_brier": float(candidate_brier),
        "brier_gain_pct": float(100 * (base_brier - candidate_brier) / base_brier),
        "log_loss_gain_pct": float(100 * (base_log - candidate_log) / base_log),
        "event_t": float(event_delta.mean() / se) if se > 0 else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim-prep", type=Path, default=DEFAULT_SIM_PREP)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    panel, _ = load_event_panel(args.sim_prep)
    panel = add_tail_labels(panel)

    rows = []
    specs = (
        ("great_rates", "positive_tail"),
        ("great_rates", "negative_tail"),
        ("great_p95_rates", "positive_tail"),
        ("great_p95_rates", "negative_tail"),
        ("blowup_profile", "negative_tail"),
        ("blowup_profile", "positive_tail"),
    )
    for group, target in specs:
        features = [feature for feature in GROUPS[group] if feature in panel.columns]
        choices = []
        for c_value in CS:
            discovery = score(panel, features, target, c_value, DISCOVERY_YEARS)
            choices.append((discovery["candidate_brier"], c_value, discovery))
        _, c_value, discovery = min(choices, key=lambda item: item[0])
        confirm = score(panel, features, target, c_value, (CONFIRM_YEAR,))
        rows.append({
            "group": group,
            "target": target,
            "features": "|".join(features),
            "C": c_value,
            **{f"discovery_{key}": value for key, value in discovery.items()},
            **{f"confirm_{key}": value for key, value in confirm.items()},
        })
    results = pd.DataFrame(rows)
    print(results[[
        "group", "target", "C", "discovery_brier_gain_pct",
        "discovery_log_loss_gain_pct", "discovery_event_t",
        "confirm_brier_gain_pct", "confirm_log_loss_gain_pct", "confirm_event_t",
    ]].to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

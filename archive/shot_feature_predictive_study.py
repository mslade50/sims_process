"""Leakage-safe screening of archived shot-level features.

This is a research harness, not part of the weekly pipeline.  It uses the
round-1, already-shifted feature snapshot as the pre-event information set,
tests on future R1/R2 model residuals, and reserves 2026 as a confirmation
holdout after walk-forward discovery on 2022-2025.

The script reads sibling ``sim_prep`` artifacts without changing them.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


DEFAULT_SIM_PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
DISCOVERY_YEARS = (2022, 2023, 2024, 2025)
CONFIRM_YEAR = 2026
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)

BASE_COLUMNS = ["prediction", "std_dev_50"]
FEATURE_GROUPS = {
    "approach_process": [
        "app_prox_core_trail16",
        "opportunities_gained_15ft_trail16",
        "app_dir_rms_along_trail16",
        "app_dir_rms_cross_trail16",
        "app_dir_mean_cross_trail16",
        "pin_high_pct_trail16",
        "on_line_pct_3yd_trail16",
        "dialed_pct_5yd_trail16",
    ],
    "putting_process": [
        "lag_putt_leave_control_trail36",
        "short_putt_make_skill_trail100",
        "burned_edge_count_trail8",
        "speed_ideal_pct_trail24",
        "speed_short_pct_trail24",
        "speed_mean_signed_leave_trail24",
        "speed_sd_signed_leave_trail24",
    ],
    "trouble_avoidance": [
        "nonpenalty_tail_shot_count_trail8",
        "recovery_shot_count_trail8",
        "arg_leave_vs_conversion_trail24",
        "short_side_miss_rate_trail24",
    ],
    "directional_control": [
        "app_dir_rms_along_trail16",
        "app_dir_rms_cross_trail16",
        "app_dir_mean_cross_trail16",
        "on_line_pct_3yd_trail16",
    ],
    "app_dispersion": [
        "sg_app_shot_std_prior8",
        "sg_app_shot_std_prior20",
        "app_dir_rms_along_trail16",
        "app_dir_rms_cross_trail16",
    ],
    "great_shot_rates": [
        "great_app_rate_prior40",
        "great_ott_rate_prior40",
        "great_short_rate_prior40",
        "great_total_rate_prior40",
    ],
}


def _norm_name(value: object) -> str:
    return str(value).lower().strip()


def _rolling_prior_mean(
    frame: pd.DataFrame, column: str, windows: tuple[int, ...]
) -> pd.DataFrame:
    order = frame.sort_values(
        ["dg_id", "round_date", "round_num"], kind="mergesort"
    ).copy()
    shifted = order.groupby("dg_id", sort=False)[column].shift(1)
    for window in windows:
        order[f"{column}_prior{window}"] = shifted.groupby(
            order["dg_id"], sort=False
        ).transform(lambda values: values.rolling(window, min_periods=4).mean())
    return order


def _rolling_prior_rate(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    window: int,
    output: str,
) -> pd.Series:
    order = frame.sort_values(
        ["dg_id", "round_date", "round_num"], kind="mergesort"
    )
    num = order.groupby("dg_id", sort=False)[numerator].shift(1)
    den = order.groupby("dg_id", sort=False)[denominator].shift(1)
    num_roll = num.groupby(order["dg_id"], sort=False).transform(
        lambda values: values.rolling(window, min_periods=8).sum()
    )
    den_roll = den.groupby(order["dg_id"], sort=False).transform(
        lambda values: values.rolling(window, min_periods=8).sum()
    )
    rate = num_roll / den_roll.replace(0, np.nan)
    rate.name = output
    return rate.reindex(frame.index)


def load_event_panel(sim_prep: Path) -> tuple[pd.DataFrame, list[str]]:
    bridge = sim_prep / "shot_bridge"
    resids = pd.read_csv(sim_prep / "work" / "sg_preds_resids.csv", low_memory=False)
    resids["_name"] = resids["player_name"].map(_norm_name)
    resids["round_date"] = pd.to_datetime(resids["round_date"], errors="coerce")

    lib = pd.read_parquet(bridge / "shot_feature_library.parquet")
    lib2 = pd.read_parquet(bridge / "shot_feature_library2.parquet")
    simple = pd.read_parquet(bridge / "shot_round_features.parquet")
    great = pd.read_parquet(bridge / "great_shot_library.parquet")
    for frame in (lib, lib2, simple, great):
        frame["round_date"] = pd.to_datetime(frame["round_date"], errors="coerce")
        frame["_name"] = frame["player_name"].map(_norm_name)

    simple_prior = _rolling_prior_mean(simple, "sg_app_shot_std", (8, 20))
    simple_cols = [
        "_name", "event_id", "year", "round_num",
        "sg_app_shot_std_prior8", "sg_app_shot_std_prior20",
        "penalty_strokes_trail21_mean",
    ]

    great_denominators = {
        "great_app": "n_app_shots",
        "great_ott": "n_ott_shots",
        "great_short": "n_arg_shots",
    }
    for numerator, denominator in great_denominators.items():
        great[f"{numerator}_rate_prior40"] = _rolling_prior_rate(
            great, numerator, denominator, 40, f"{numerator}_rate_prior40"
        )
        p95_numerator = f"{numerator}_p95"
        great[f"{p95_numerator}_rate_prior40"] = _rolling_prior_rate(
            great,
            p95_numerator,
            denominator,
            40,
            f"{p95_numerator}_rate_prior40",
        )
    great["great_total_n"] = (
        great["n_app_shots"] + great["n_ott_shots"]
        + great["n_arg_shots"] + great["n_putt_shots"]
    )
    great["great_total_rate_prior40"] = _rolling_prior_rate(
        great, "great_total", "great_total_n", 40, "great_total_rate_prior40"
    )
    great["great_total_p95"] = great[
        ["great_app_p95", "great_arg_p95", "great_ott_p95", "great_putt_p95"]
    ].sum(axis=1)
    great["great_total_p95_rate_prior40"] = _rolling_prior_rate(
        great,
        "great_total_p95",
        "great_total_n",
        40,
        "great_total_p95_rate_prior40",
    )

    key = ["_name", "event_id", "year", "round_num"]
    feature_frames = []
    for frame in (lib, lib2):
        cols = [column for column in frame.columns if column.endswith(tuple(
            ["trail8", "trail16", "trail20", "trail24", "trail36", "trail100"]
        ))]
        feature_frames.append(frame[key + cols].drop_duplicates(key))
    feature_frames.append(simple_prior[simple_cols].drop_duplicates(key))
    great_cols = [column for column in great.columns if column.endswith("rate_prior40")]
    feature_frames.append(great[key + great_cols].drop_duplicates(key))

    r1_features = None
    for frame in feature_frames:
        frame = frame[frame["round_num"].eq(1)].drop(columns="round_num")
        join = ["_name", "event_id", "year"]
        r1_features = frame if r1_features is None else r1_features.merge(
            frame, on=join, how="outer", validate="one_to_one"
        )

    target_rows = resids[resids["round_num"].isin([1, 2])].copy()
    target_rows["sq_residual"] = target_rows["residual"] ** 2
    target_rows["abs_residual"] = target_rows["residual"].abs()
    event_key = ["_name", "event_id", "year"]
    targets = (
        target_rows.sort_values(event_key + ["round_num"])
        .groupby(event_key, as_index=False)
        .agg(
            r12_mean_residual=("residual", "mean"),
            r12_mean_sq_residual=("sq_residual", "mean"),
            r12_mean_abs_residual=("abs_residual", "mean"),
            prediction=("prediction", "first"),
            std_dev_50=("std_dev_50", "first"),
            rounds_observed=("round_num", "nunique"),
        )
    )
    panel = targets.merge(r1_features, on=event_key, how="inner")
    panel = panel[panel["rounds_observed"].eq(2)].copy()
    candidate_columns = sorted(
        set(panel.columns)
        - set(event_key)
        - {
            "r12_mean_residual", "r12_mean_sq_residual",
            "r12_mean_abs_residual", "prediction", "std_dev_50",
            "rounds_observed",
        }
    )
    return panel, candidate_columns


def _center_within_event(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    groups = out.groupby(["year", "event_id"], sort=False)
    for column in columns:
        out[column] = out[column] - groups[column].transform("median")
    return out


class MatrixBuilder:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.medians: pd.Series | None = None
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].replace([np.inf, -np.inf], np.nan)
        self.medians = values.median()
        values = values.fillna(self.medians)
        self.means = values.mean()
        self.stds = values.std().replace(0, 1.0).fillna(1.0)
        return ((values - self.means) / self.stds).to_numpy(float)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.stds is None:
            raise RuntimeError("MatrixBuilder must be fit first")
        values = frame[self.columns].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(self.medians)
        return ((values - self.means) / self.stds).to_numpy(float)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target: str,
    alpha: float,
) -> np.ndarray:
    builder = MatrixBuilder(columns)
    x_train = builder.fit_transform(train)
    x_test = builder.transform(test)
    model = Ridge(alpha=alpha)
    model.fit(x_train, train[target].to_numpy(float))
    return model.predict(x_test)


def walk_forward_score(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
    test_years: tuple[int, ...],
    base_columns: list[str] | None = None,
) -> dict[str, float]:
    base_columns = base_columns or BASE_COLUMNS
    required = list(dict.fromkeys(base_columns + features))
    data = _center_within_event(panel, features)
    predictions = []
    for year in test_years:
        train = data[data["year"] < year].dropna(subset=[target])
        test = data[data["year"].eq(year)].dropna(subset=[target])
        if len(train) < 500 or len(test) < 100:
            continue
        base = _fit_predict(train, test, base_columns, target, alpha)
        augmented = _fit_predict(train, test, required, target, alpha)
        fold = test[["year", "event_id", target]].copy()
        fold["base"] = base
        fold["augmented"] = augmented
        predictions.append(fold)
    if not predictions:
        return {"n": 0, "base_mse": np.nan, "aug_mse": np.nan, "delta_mse": np.nan}
    scored = pd.concat(predictions, ignore_index=True)
    base_mse = mean_squared_error(scored[target], scored["base"])
    aug_mse = mean_squared_error(scored[target], scored["augmented"])
    scored["loss_delta"] = (
        (scored[target] - scored["base"]) ** 2
        - (scored[target] - scored["augmented"]) ** 2
    )
    event_delta = scored.groupby(["year", "event_id"])["loss_delta"].mean()
    event_se = event_delta.std(ddof=1) / np.sqrt(len(event_delta))
    event_t = float(event_delta.mean() / event_se) if event_se > 0 else np.nan
    return {
        "n": int(len(scored)),
        "events": int(len(event_delta)),
        "base_mse": float(base_mse),
        "aug_mse": float(aug_mse),
        "delta_mse": float(base_mse - aug_mse),
        "delta_mse_pct": float(100.0 * (base_mse - aug_mse) / base_mse),
        "event_t": event_t,
    }


def screen_univariate(
    panel: pd.DataFrame, candidates: list[str], target: str
) -> pd.DataFrame:
    rows = []
    for feature in candidates:
        discovery = walk_forward_score(
            panel, [feature], target, alpha=10.0, test_years=DISCOVERY_YEARS
        )
        confirm = walk_forward_score(
            panel, [feature], target, alpha=10.0, test_years=(CONFIRM_YEAR,)
        )
        coverage = float(panel[feature].notna().mean())
        rows.append({
            "feature": feature,
            "coverage": coverage,
            **{f"discovery_{key}": value for key, value in discovery.items()},
            **{f"confirm_{key}": value for key, value in confirm.items()},
        })
    return pd.DataFrame(rows).sort_values(
        ["discovery_delta_mse", "confirm_delta_mse"], ascending=False
    )


def test_groups(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for name, requested in FEATURE_GROUPS.items():
        features = [feature for feature in requested if feature in panel.columns]
        if not features:
            continue
        alpha_results = []
        for alpha in RIDGE_ALPHAS:
            score = walk_forward_score(
                panel, features, target, alpha=alpha, test_years=DISCOVERY_YEARS
            )
            alpha_results.append((score["delta_mse"], alpha, score))
        _, alpha, discovery = max(alpha_results, key=lambda item: item[0])
        confirm = walk_forward_score(
            panel, features, target, alpha=alpha, test_years=(CONFIRM_YEAR,)
        )
        rows.append({
            "group": name,
            "features": features,
            "alpha": alpha,
            **{f"discovery_{key}": value for key, value in discovery.items()},
            **{f"confirm_{key}": value for key, value in confirm.items()},
        })
    return pd.DataFrame(rows).sort_values("discovery_delta_mse", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim-prep", type=Path, default=DEFAULT_SIM_PREP)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    panel, candidates = load_event_panel(args.sim_prep)
    print(
        f"panel: {len(panel):,} player-events, "
        f"{panel['event_id'].nunique():,} event IDs, "
        f"years {panel['year'].min()}-{panel['year'].max()}"
    )
    print(f"candidate features: {len(candidates)}")

    outputs: dict[str, pd.DataFrame] = {}
    for target in (
        "r12_mean_residual",
        "r12_mean_sq_residual",
        "r12_mean_abs_residual",
    ):
        print(f"\n=== {target}: univariate screen ===")
        screen = screen_univariate(panel, candidates, target)
        outputs[f"{target}_univariate"] = screen
        print(screen.head(15)[[
            "feature", "coverage", "discovery_delta_mse_pct",
            "discovery_event_t", "confirm_delta_mse_pct", "confirm_event_t",
        ]].to_string(index=False))

        print(f"\n=== {target}: grouped models ===")
        groups = test_groups(panel, target)
        outputs[f"{target}_groups"] = groups
        print(groups[[
            "group", "alpha", "discovery_delta_mse_pct",
            "discovery_event_t", "confirm_delta_mse_pct", "confirm_event_t",
        ]].to_string(index=False))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "panel_rows": int(len(panel)),
            "year_min": int(panel["year"].min()),
            "year_max": int(panel["year"].max()),
            "discovery_years": DISCOVERY_YEARS,
            "confirmation_year": CONFIRM_YEAR,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        for name, frame in outputs.items():
            serializable = frame.copy()
            if "features" in serializable:
                serializable["features"] = serializable["features"].map(
                    lambda values: "|".join(values)
                )
            serializable.to_csv(args.output_dir / f"{name}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

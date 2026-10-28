"""Calibrate the ability component of shot-donor field normalization.

Spatial donor normalization removes historical player distance, directional
bias, and dispersion.  It does not by itself remove the fact that stronger
players produce better transitions from the same state.  This research-only
evaluator estimates that second component using strictly pre-event category
skill snapshots and fixed-reference per-shot ``sg_raw``.

The fitted slope has units of shot SG per one SG/round of category ability.
For an exponential donor-atom tilt, the local natural-parameter coefficient is
approximately ``slope / conditional_reward_variance``.  Results are saved for
audit only; this script never changes a production or simulator coefficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXPORT = Path(
    r"C:\Users\McKinley Slade\dev\golf_scraping\output\export"
)
DEFAULT_INPUT = Path("permanent_data/shot_field_normalization_input_audit")
DEFAULT_OUTPUT = Path("permanent_data/shot_quality_normalization_calibration")
CATEGORY_ABILITY = {
    "ott": "sg_ott",
    "app": "sg_app",
    "arg": "ag_arg",
    "putt": "ag_putt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lie(value: object) -> str:
    text = str(value).casefold().strip()
    if text in {"fairway", "tee", "rough", "sand", "recovery", "green"}:
        return text
    if "bunker" in text:
        return "sand"
    return "other"


def _distance_state(category: pd.Series, distance: pd.Series) -> pd.Series:
    category = category.astype(str)
    distance = pd.to_numeric(distance, errors="coerce").to_numpy(float)
    result = np.empty(len(category), dtype=object)
    values = category.to_numpy()
    for cat in CATEGORY_ABILITY:
        mask = values == cat
        x = distance[mask]
        if cat == "ott":
            bucket = np.rint(x / 25.0).astype(int)
            result[mask] = [f"{value * 25:d}yd" for value in bucket]
        elif cat == "app":
            labels = np.select(
                [x <= 100, x <= 150, x <= 215],
                ["<=100", "100-150", "150-215"],
                default=">215",
            )
            result[mask] = labels
        elif cat == "arg":
            bucket = np.rint(x / 5.0).astype(int)
            result[mask] = [f"{value * 5:d}yd" for value in bucket]
        else:
            feet = np.maximum(x * 3.0, 0.0)
            # Fine near the cup, progressively broader for lag putts.
            bucket = np.floor(np.log1p(feet) / 0.22).astype(int)
            result[mask] = [f"logft_{value:d}" for value in bucket]
    return pd.Series(result, index=category.index, dtype="string")


def load_shot_panel(export_root: Path, event_audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_key",
        "round_no",
        "hole_no",
        "source_player_id",
        "sg_category",
        "start_lie",
        "start_distance_yd",
        "sg_raw",
        "sg_valid",
        "complete_hole",
        "shot_key",
    ]
    parts: list[pd.DataFrame] = []
    for season, events in event_audit.groupby("season", observed=True):
        keys = events["event_key"].astype(str).tolist()
        frame = pd.read_parquet(
            export_root / f"pga_{int(season)}_shot_sg.parquet",
            columns=columns,
            filters=[("event_key", "in", keys)],
        )
        frame["season"] = int(season)
        parts.append(frame)
    panel = pd.concat(parts, ignore_index=True)
    panel["source_player_id"] = panel["source_player_id"].astype(str)
    panel = panel[
        panel["sg_category"].isin(CATEGORY_ABILITY)
        & panel["sg_valid"].fillna(False)
        & panel["complete_hole"].fillna(False)
        & panel["sg_raw"].notna()
        & panel["start_distance_yd"].notna()
    ].copy()
    panel["start_lie_state"] = panel["start_lie"].map(_lie)
    panel["distance_state"] = _distance_state(
        panel["sg_category"], panel["start_distance_yd"]
    )
    return panel


def attach_strict_skills(panel: pd.DataFrame, player_audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_key",
        "source_player_id",
        "skill_date",
        "event_start",
        *CATEGORY_ABILITY.values(),
    ]
    snapshots = player_audit[columns].copy()
    snapshots["source_player_id"] = snapshots["source_player_id"].astype(str)
    snapshots["skill_date"] = pd.to_datetime(snapshots["skill_date"], errors="coerce")
    snapshots["event_start"] = pd.to_datetime(snapshots["event_start"], errors="coerce")
    invalid = snapshots["skill_date"].ge(snapshots["event_start"]).fillna(False)
    if invalid.any():
        raise ValueError("Player input contains a non-prior skill snapshot")
    snapshots = snapshots.drop_duplicates(["event_key", "source_player_id"])
    out = panel.merge(
        snapshots,
        on=["event_key", "source_player_id"],
        how="left",
        validate="many_to_one",
    )
    ability = np.full(len(out), np.nan)
    for category, column in CATEGORY_ABILITY.items():
        mask = out["sg_category"].eq(category).to_numpy()
        ability[mask] = pd.to_numeric(out.loc[mask, column], errors="coerce")
    out["ability_sg_round"] = ability
    return out


def residualize_states(panel: pd.DataFrame, minimum_state_shots: int = 4) -> pd.DataFrame:
    keys = [
        "event_key",
        "round_no",
        "hole_no",
        "sg_category",
        "start_lie_state",
        "distance_state",
    ]
    work = panel[panel["ability_sg_round"].notna()].copy()
    groups = work.groupby(keys, observed=True, dropna=False)
    count = groups["sg_raw"].transform("size")
    work = work[count.ge(minimum_state_shots)].copy()
    groups = work.groupby(keys, observed=True, dropna=False)
    work["reward_residual_sg"] = work["sg_raw"] - groups["sg_raw"].transform("mean")
    work["ability_residual_sg_round"] = (
        work["ability_sg_round"]
        - groups["ability_sg_round"].transform("mean")
    )
    return work


def _fit_slope(frame: pd.DataFrame) -> dict[str, float]:
    x = frame["ability_residual_sg_round"].to_numpy(float)
    y = frame["reward_residual_sg"].to_numpy(float)
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return {
            "slope_sg_shot_per_sg_round": math.nan,
            "reward_residual_variance": math.nan,
            "natural_parameter_per_sg_round": math.nan,
            "correlation": math.nan,
        }
    slope = float(np.dot(x, y) / denominator)
    variance = float(np.var(y, ddof=1))
    return {
        "slope_sg_shot_per_sg_round": slope,
        "reward_residual_variance": variance,
        "natural_parameter_per_sg_round": slope / variance,
        "correlation": float(np.corrcoef(x, y)[0, 1]),
    }


def expanding_folds(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, category_panel in panel.groupby("sg_category", observed=True):
        for test_year in sorted(category_panel["season"].unique()):
            train = category_panel[category_panel["season"].lt(test_year)]
            test = category_panel[category_panel["season"].eq(test_year)]
            if train["season"].nunique() < 2 or len(test) < 500:
                continue
            fit = _fit_slope(train)
            slope = fit["slope_sg_shot_per_sg_round"]
            x = test["ability_residual_sg_round"].to_numpy(float)
            y = test["reward_residual_sg"].to_numpy(float)
            base_error = y
            model_error = y - slope * x
            rows.append(
                {
                    "category": category,
                    "test_year": int(test_year),
                    "train_years": int(train["season"].nunique()),
                    "train_shots": int(len(train)),
                    "test_shots": int(len(test)),
                    **fit,
                    "test_mse_baseline": float(np.mean(base_error**2)),
                    "test_mse_normalized": float(np.mean(model_error**2)),
                    "test_mse_delta": float(
                        np.mean(model_error**2) - np.mean(base_error**2)
                    ),
                    "test_cov_xy": float(np.mean(x * y)),
                }
            )
    return pd.DataFrame(rows)


def expanding_event_folds(panel: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Score each held-out event with the coefficient fit before its year."""
    rows: list[dict[str, object]] = []
    for fold in folds.itertuples(index=False):
        test = panel[
            panel["sg_category"].eq(fold.category)
            & panel["season"].eq(fold.test_year)
        ]
        for event_key, event in test.groupby("event_key", observed=True):
            x = event["ability_residual_sg_round"].to_numpy(float)
            y = event["reward_residual_sg"].to_numpy(float)
            model_error = y - float(fold.slope_sg_shot_per_sg_round) * x
            rows.append(
                {
                    "category": fold.category,
                    "test_year": int(fold.test_year),
                    "event_key": event_key,
                    "shots": int(len(event)),
                    "test_mse_baseline": float(np.mean(y**2)),
                    "test_mse_normalized": float(np.mean(model_error**2)),
                    "test_mse_delta": float(np.mean(model_error**2) - np.mean(y**2)),
                }
            )
    return pd.DataFrame(rows)


def category_summary(
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    event_folds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, group in panel.groupby("sg_category", observed=True):
        fit = _fit_slope(group)
        category_folds = folds[folds["category"].eq(category)]
        category_events = event_folds[event_folds["category"].eq(category)]
        rows.append(
            {
                "category": category,
                "shots": int(len(group)),
                "events": int(group["event_key"].nunique()),
                "years": int(group["season"].nunique()),
                **fit,
                "expanding_test_shots": int(category_folds["test_shots"].sum()),
                "expanding_mse_delta_weighted": float(
                    np.average(
                        category_folds["test_mse_delta"],
                        weights=category_folds["test_shots"],
                    )
                ) if len(category_folds) else math.nan,
                "favorable_test_years": int(category_folds["test_mse_delta"].lt(0).sum()),
                "test_years": int(len(category_folds)),
                "favorable_test_events": int(
                    category_events["test_mse_delta"].lt(0).sum()
                ),
                "test_events": int(len(category_events)),
                "equal_event_mse_delta": float(
                    category_events["test_mse_delta"].mean()
                ) if len(category_events) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def state_summary(panel: pd.DataFrame, minimum_shots: int = 750) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["sg_category", "start_lie_state", "distance_state"]
    for state, group in panel.groupby(keys, observed=True, dropna=False):
        if len(group) < minimum_shots:
            continue
        rows.append(
            {
                "category": state[0],
                "start_lie_state": state[1],
                "distance_state": state[2],
                "shots": int(len(group)),
                **_fit_slope(group),
            }
        )
    return pd.DataFrame(rows)


def event_impact(panel: pd.DataFrame, event_audit: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    slopes = summary.set_index("category")["slope_sg_shot_per_sg_round"]
    rounds = panel[["event_key", "source_player_id", "round_no"]].drop_duplicates()
    exposure = (
        panel.groupby(["event_key", "sg_category"], observed=True).size()
        / rounds.groupby("event_key").size()
    ).rename("shots_per_player_round").reset_index()
    fields = event_audit[["event_key", "season", "event_name"]].copy()
    for category, ability_column in CATEGORY_ABILITY.items():
        source = f"field_{ability_column}"
        fields[category] = event_audit[source] if source in event_audit else np.nan
    fields = fields.melt(
        id_vars=["event_key", "season", "event_name"],
        value_vars=list(CATEGORY_ABILITY),
        var_name="sg_category",
        value_name="field_category_skill",
    )
    out = fields.merge(exposure, on=["event_key", "sg_category"], how="left")
    out["slope_sg_shot_per_sg_round"] = out["sg_category"].map(slopes)
    out["implied_field_shift_sg_per_shot"] = (
        out["field_category_skill"] * out["slope_sg_shot_per_sg_round"]
    )
    out["implied_field_shift_sg_per_round"] = (
        out["implied_field_shift_sg_per_shot"] * out["shots_per_player_round"]
    )
    return out


def write_report(summary: pd.DataFrame, folds: pd.DataFrame, impact: pd.DataFrame, output: Path) -> Path:
    rows = []
    for row in summary.sort_values("category").itertuples(index=False):
        rows.append(
            f"| {row.category.upper()} | {row.shots:,} | "
            f"{row.slope_sg_shot_per_sg_round:+.5f} | "
            f"{row.natural_parameter_per_sg_round:+.4f} | "
            f"{row.favorable_test_years}/{row.test_years} | "
            f"{row.favorable_test_events}/{row.test_events} | "
            f"{row.expanding_mse_delta_weighted:+.8f} |"
        )
    travelers = impact[
        impact["event_name"].str.contains("Travelers", case=False)
        & impact["season"].ge(2024)
    ]
    traveler_rows = [
        f"| {int(row.season)} | {row.sg_category.upper()} | "
        f"{row.field_category_skill:+.3f} | {row.shots_per_player_round:.2f} | "
        f"{row.implied_field_shift_sg_per_round:+.3f} |"
        for row in travelers.itertuples(index=False)
    ]
    report = f"""# Shot-quality field-normalization calibration

Player category ability comes from the latest snapshot strictly before each
event.  Shot reward is fixed-reference `sg_raw`, residualized within event,
round, hole, lie, and distance state.  Thus the slope measures player quality
within a comparable transition state rather than field/course difficulty.

| Category | Shots | SG/shot per SG/round | Approx. natural parameter | Favorable OOS years | Favorable OOS events | OOS MSE delta |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Negative OOS MSE delta is favorable.  The natural parameter is the first-order
exponential-tilt mapping `slope / conditional reward variance`; it is not yet
a production coefficient.  It must be validated in the coherent endpoint +
lie + hazard transition model with the same state support.

## Travelers 2024-25 field-composition implication

| Year | Category | Field category SG | Shots/player-round | Implied donor field shift (SG/round) |
|---|---|---:|---:|---:|
{chr(10).join(traveler_rows)}

The implied shift is a diagnostic of what raw donor atoms can inherit from
that edition's field.  It is not a proposed direct score adjustment: donor
ability is removed with whole-atom weights, and target ability is later
reapplied inside the transition kernel.

This study uses only three recurring events and is exploratory.  The primary
spatial-normalization evaluation should keep ability weighting disabled until
these mappings survive a wider rolling course panel.
"""
    path = output / "QUALITY_CALIBRATION.md"
    path.write_text(report, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--through-season", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.input_dir / "event_field_strength.csv"
    player_path = args.input_dir / "player_identity_snapshot.csv"
    event_audit = pd.read_csv(event_path)
    event_audit = event_audit[event_audit["season"].le(args.through_season)].copy()
    player_audit = pd.read_csv(player_path, dtype={"source_player_id": str})
    player_audit = player_audit[player_audit["season"].le(args.through_season)].copy()
    raw = load_shot_panel(args.export_root, event_audit)
    attached = attach_strict_skills(raw, player_audit)
    panel = residualize_states(attached)
    folds = expanding_folds(panel)
    event_folds = expanding_event_folds(panel, folds)
    summary = category_summary(panel, folds, event_folds)
    states = state_summary(panel)
    impact = event_impact(panel, event_audit, summary)

    folds.to_csv(args.output_dir / "expanding_year_folds.csv", index=False)
    event_folds.to_csv(args.output_dir / "expanding_event_folds.csv", index=False)
    summary.to_csv(args.output_dir / "category_quality_mapping.csv", index=False)
    states.to_csv(args.output_dir / "state_quality_mapping.csv", index=False)
    impact.to_csv(args.output_dir / "event_field_quality_impact.csv", index=False)
    report = write_report(summary, folds, impact, args.output_dir)
    manifest = {
        "contract": "strict_prior_transition_quality_v1",
        "through_season": int(args.through_season),
        "raw_shots": int(len(raw)),
        "usable_shots": int(len(panel)),
        "ability_coverage": float(attached["ability_sg_round"].notna().mean()),
        "event_input_sha256": _sha256(event_path),
        "player_input_sha256": _sha256(player_path),
        "source_sha256": _sha256(Path(__file__)),
        "production_enabled": False,
    }
    registry = {
        "registry_version": "shot_quality_natural_parameter_v1",
        "status": "exploratory_disabled_three_course_panel",
        "coefficient_unit": "natural log atom weight per (category SG/round * shot SG)",
        "required_runtime_default": 0.0,
        "categories": {
            str(row.category): {
                "slope_sg_shot_per_sg_round": float(
                    row.slope_sg_shot_per_sg_round
                ),
                "natural_parameter_per_sg_round": float(
                    row.natural_parameter_per_sg_round
                ),
                "favorable_expanding_years": int(row.favorable_test_years),
                "expanding_test_years": int(row.test_years),
                "favorable_expanding_events": int(row.favorable_test_events),
                "expanding_test_events": int(row.test_events),
            }
            for row in summary.itertuples(index=False)
        },
        "input_manifest": manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output_dir / "quality_coefficient_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

"""Evaluate field-strength normalization for recurring-course tee donors.

Research only.  This script does not modify production or the hole-simulator
prototype.  It compares three donor representations on 2026 holdouts at Sony,
Travelers, and Wyndham:

* ``raw``: historical endpoint coordinates unchanged;
* ``event_centered``: each donor edition/setup centered to the equal-edition
  recurring-course setup mean;
* ``player_state_residualized``: invert player distance, lateral bias, and
  dispersion transforms fit strictly before that donor event.

Every normalized donor distribution is then transported through the held-out
player's own strict pre-event profile and scored on endpoint density/error,
remaining-distance CRPS, and lie log loss/Brier.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hole_sim_tee_transition as tee  # noqa: E402


DEFAULT_DB = Path(r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db")
DEFAULT_OUTPUT = Path("permanent_data/shot_field_normalization_eval")
EVENTS = ("Sony Open in Hawaii", "Travelers Championship", "Wyndham Championship")
MODES = ("raw", "event_centered", "player_state_residualized")


def norm_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    tokens = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    # Shot feeds use "last, first" while the skill model is not guaranteed to;
    # token sorting makes the crosswalk order-invariant.
    return " ".join(sorted(tokens))


def load_event_dates(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as connection:
        dates = pd.read_sql_query(
            """
            SELECT year, event_id, event_name, MIN(round_date) AS event_date
            FROM player_rounds
            WHERE tour='pga' AND year BETWEEN 2020 AND 2026
            GROUP BY year, event_id, event_name
            """,
            connection,
        )
    dates["event_date"] = pd.to_datetime(dates["event_date"], errors="coerce")
    dates["event_key"] = [
        f"pga:R{int(year)}{int(event_id):03d}"
        for year, event_id in zip(dates["year"], dates["event_id"])
    ]
    return dates.dropna(subset=["event_date"]).drop_duplicates("event_key")


def load_skill_history(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as connection:
        skills = pd.read_sql_query(
            """
            SELECT Date AS skill_date, player_name, skill_est
            FROM model_results2
            WHERE Date >= '2018-01-01' AND Date < '2027-01-01'
              AND skill_est IS NOT NULL
            """,
            connection,
        )
    skills["skill_date"] = pd.to_datetime(skills["skill_date"], errors="coerce")
    skills["name_key"] = skills["player_name"].map(norm_name)
    return skills.dropna(subset=["skill_date", "skill_est"]).sort_values("skill_date")


def skill_snapshot(
    skills: pd.DataFrame, names: pd.Series, cutoff: pd.Timestamp
) -> pd.DataFrame:
    query = pd.DataFrame({"name_key": names.map(norm_name).drop_duplicates()})
    query["cutoff"] = pd.Timestamp(cutoff).to_datetime64().astype("datetime64[ns]")
    result = pd.merge_asof(
        query.sort_values("cutoff"),
        skills[["name_key", "skill_date", "skill_est"]].sort_values("skill_date"),
        by="name_key",
        left_on="cutoff",
        right_on="skill_date",
        direction="backward",
        allow_exact_matches=False,
    )
    return result[["name_key", "skill_est"]]


def prepare_profile_rows(drives: pd.DataFrame) -> pd.DataFrame:
    """Compute setup-relative sufficient statistics once for dated snapshots."""
    setup = ["event_key", "round_no", "hole_no", "course_id"]
    frame = drives.copy()
    group = frame.groupby(setup, observed=True)
    frame["field_long"] = group["long_yd"].transform("mean")
    frame["field_lat"] = group["lat_yd"].transform("mean")
    long_sd = group["long_yd"].transform("std").replace([np.inf, -np.inf], np.nan)
    lat_sd = group["lat_yd"].transform("std").replace([np.inf, -np.inf], np.nan)
    long_sd = long_sd.fillna(long_sd.median()).clip(8.0, 80.0)
    lat_sd = lat_sd.fillna(lat_sd.median()).clip(6.0, 60.0)
    frame["log_distance_ratio"] = np.log(
        frame["long_yd"] / frame["field_long"]
    ).clip(-0.35, 0.35)
    frame["lat_resid"] = frame["lat_yd"] - frame["field_lat"]
    frame["long_std_resid"] = (frame["long_yd"] - frame["field_long"]) / long_sd
    frame["lat_std_resid"] = frame["lat_resid"] / lat_sd
    return frame


def fit_snapshot(
    prepared: pd.DataFrame,
    cutoff: pd.Timestamp,
    mean_prior: float = 50.0,
    variance_prior: float = 120.0,
) -> pd.DataFrame:
    """Fit player transforms using only shots with event_date before cutoff."""
    history = prepared[prepared["event_date"].lt(cutoff)].copy()
    if history.empty:
        return pd.DataFrame(
            columns=[
                "source_player_id", "n_drives", "distance_ratio",
                "lateral_bias_yd", "long_scale", "lat_scale",
            ]
        )
    player = history.groupby("source_player_id", observed=True)
    history["long_disp"] = (
        history["long_std_resid"] - player["long_std_resid"].transform("mean")
    )
    history["lat_disp"] = (
        history["lat_std_resid"] - player["lat_std_resid"].transform("mean")
    )
    global_long_var = float(np.mean(history["long_disp"] ** 2))
    global_lat_var = float(np.mean(history["lat_disp"] ** 2))
    grouped = history.groupby("source_player_id", observed=True)
    profile = grouped.agg(
        n_drives=("shot_key", "size"),
        distance_sum=("log_distance_ratio", "sum"),
        lateral_sum=("lat_resid", "sum"),
        long_ss=("long_disp", lambda x: float(np.sum(np.asarray(x) ** 2))),
        lat_ss=("lat_disp", lambda x: float(np.sum(np.asarray(x) ** 2))),
    ).reset_index()
    n = profile["n_drives"].to_numpy(float)
    profile["distance_ratio"] = np.exp(
        profile["distance_sum"] / (n + mean_prior)
    ).clip(0.88, 1.12)
    profile["lateral_bias_yd"] = (
        profile["lateral_sum"] / (n + mean_prior)
    ).clip(-12, 12)
    profile["long_scale"] = np.sqrt(
        (profile["long_ss"] + variance_prior * global_long_var)
        / ((n + variance_prior) * global_long_var)
    ).clip(0.65, 1.50)
    profile["lat_scale"] = np.sqrt(
        (profile["lat_ss"] + variance_prior * global_lat_var)
        / ((n + variance_prior) * global_lat_var)
    ).clip(0.65, 1.50)
    profile["source_player_id"] = profile["source_player_id"].astype(str)
    return profile


def neutral_profile() -> pd.Series:
    return pd.Series(
        {
            "n_drives": 0,
            "distance_ratio": 1.0,
            "lateral_bias_yd": 0.0,
            "long_scale": 1.0,
            "lat_scale": 1.0,
        }
    )


def inverse_transport(
    frame: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Map donor endpoints back to the neutral player for that setup."""
    result = frame.copy()
    profile = profiles.set_index("source_player_id")
    modifiers = result[["source_player_id"]].join(profile, on="source_player_id")
    covered = modifiers["distance_ratio"].notna()
    for column, default in (
        ("distance_ratio", 1.0), ("lateral_bias_yd", 0.0),
        ("long_scale", 1.0), ("lat_scale", 1.0),
    ):
        modifiers[column] = modifiers[column].fillna(default)
    setup = ["event_key", "round_no", "hole_no"]
    center_long = result.groupby(setup)["long_yd"].transform("mean")
    center_lat = result.groupby(setup)["lat_yd"].transform("mean")
    result["long_yd"] = center_long + (
        result["long_yd"] - center_long * modifiers["distance_ratio"].to_numpy(float)
    ) / modifiers["long_scale"].to_numpy(float)
    result["lat_yd"] = center_lat + (
        result["lat_yd"] - center_lat - modifiers["lateral_bias_yd"].to_numpy(float)
    ) / modifiers["lat_scale"].to_numpy(float)
    result["donor_profile_coverage"] = covered.astype(int)
    for column in ("distance_ratio", "lateral_bias_yd", "long_scale", "lat_scale"):
        result[f"donor_{column}"] = modifiers[column].to_numpy(float)
    return result


def event_center(course: pd.DataFrame) -> pd.DataFrame:
    """Remove donor-edition setup means while preserving equal-edition center."""
    result = course.copy()
    setup = ["event_key", "round_no", "hole_no"]
    edition_mean = result.groupby(setup)[["long_yd", "lat_yd"]].transform("mean")
    means = (
        result.groupby(setup, as_index=False)[["long_yd", "lat_yd"]]
        .mean()
    )
    reference = means.groupby(["round_no", "hole_no"])[["long_yd", "lat_yd"]].transform("mean")
    ref_map = means[["event_key", "round_no", "hole_no"]].copy()
    ref_map[["ref_long", "ref_lat"]] = reference.to_numpy()
    ref = result.merge(ref_map, on=setup, how="left", validate="many_to_one")
    result["long_yd"] = result["long_yd"] - edition_mean["long_yd"] + ref["ref_long"]
    result["lat_yd"] = result["lat_yd"] - edition_mean["lat_yd"] + ref["ref_lat"]
    return result


def normalize_course_donors(
    course: pd.DataFrame,
    prepared_profiles: pd.DataFrame,
    event_dates: dict[str, pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    raw = course.copy()
    raw["surface_long_yd"] = raw["long_yd"]
    raw["surface_lat_yd"] = raw["lat_yd"]
    centered = event_center(course)
    centered["surface_long_yd"] = course["long_yd"]
    centered["surface_lat_yd"] = course["lat_yd"]
    residual_parts = []
    for event_key, donor in course.groupby("event_key", sort=False):
        cutoff = event_dates[event_key]
        snapshot = fit_snapshot(prepared_profiles, cutoff)
        normalized = inverse_transport(donor, snapshot)
        normalized["surface_long_yd"] = donor["long_yd"]
        normalized["surface_lat_yd"] = donor["lat_yd"]
        normalized["profile_cutoff"] = cutoff
        residual_parts.append(normalized)
    residualized = pd.concat(residual_parts).sort_index()
    audit_columns = [
        "donor_profile_coverage", "donor_distance_ratio",
        "donor_lateral_bias_yd", "donor_long_scale", "donor_lat_scale",
    ]
    for column in audit_columns:
        raw[column] = residualized[column].reindex(raw.index)
        centered[column] = residualized[column].reindex(centered.index)
    return {
        "raw": raw,
        "event_centered": centered,
        "player_state_residualized": residualized,
    }


def deterministic_lie_forecast(
    surface: tee.SurfaceMap,
    points: np.ndarray,
    weights: np.ndarray,
    max_points: int = 180,
) -> np.ndarray:
    """Integrate lie probabilities with common deterministic mixture quantiles."""
    weights = weights / weights.sum()
    if len(points) > max_points:
        uniforms = (np.arange(max_points, dtype=float) + 0.5) / max_points
        chosen = np.searchsorted(np.cumsum(weights), uniforms, side="left")
        query = points[np.minimum(chosen, len(points) - 1)]
        query_weights = np.full(max_points, 1.0 / max_points)
    else:
        query, query_weights = points, weights
    conditional = surface.probabilities(query)
    return np.sum(conditional * query_weights[:, None], axis=0)


def score_mode(
    course_train: pd.DataFrame,
    holdout: pd.DataFrame,
    target_profiles: pd.DataFrame,
    mode: str,
    seed: int,
) -> pd.DataFrame:
    profile_lookup = target_profiles.set_index("source_player_id")
    distributions: dict[tuple[int, int, int], tee.Distribution] = {}
    surfaces: dict[int, tee.SurfaceMap] = {}
    lie_cache: dict[tuple[int, int, str], np.ndarray] = {}
    rows = []
    for record in holdout.itertuples(index=False):
        setup_key = (int(record.round_no), int(record.hole_no), int(record.par))
        if setup_key not in distributions:
            distributions[setup_key] = tee.build_distribution(course_train, *setup_key)
        distribution = distributions[setup_key]
        hole = int(record.hole_no)
        if hole not in surfaces:
            # Surface labels remain attached to their raw historical geometry.
            # Endpoint transports are reclassified against this fixed empirical
            # spatial map rather than carrying stale donor lie labels with them.
            surface_history = course_train[course_train["hole_no"].eq(hole)].copy()
            surface_history["long_yd"] = surface_history["surface_long_yd"]
            surface_history["lat_yd"] = surface_history["surface_lat_yd"]
            surfaces[hole] = tee.SurfaceMap(surface_history)
        player_id = str(record.source_player_id)
        profile = profile_lookup.loc[player_id] if player_id in profile_lookup.index else neutral_profile()
        if isinstance(profile, pd.DataFrame):
            profile = profile.iloc[-1]
        points = tee.adjust_distribution(distribution.points, distribution.weights, profile)
        observed = np.array([record.long_yd, record.lat_yd], dtype=float)
        point_mean = tee._weighted_mean(points, distribution.weights)
        remaining = np.hypot(record.pin_long_yd - points[:, 0], points[:, 1])
        lie_key = (record.round_no, record.hole_no, player_id)
        if lie_key not in lie_cache:
            lie_cache[lie_key] = deterministic_lie_forecast(
                surfaces[hole], points, distribution.weights, 180
            )
        lie_probs = lie_cache[lie_key]
        lie_index = tee.LIE_CLASSES.index(record.lie_class)
        onehot = np.zeros(len(tee.LIE_CLASSES)); onehot[lie_index] = 1.0
        rows.append(
            {
                "mode": mode,
                "event_key": record.event_key,
                "season": record.season,
                "round_no": record.round_no,
                "hole_no": record.hole_no,
                "source_player_id": player_id,
                "player_name": record.player_name,
                "endpoint_nll": tee._kde_nll(points, distribution.weights, observed),
                "endpoint_error_yd": float(np.linalg.norm(observed - point_mean)),
                "remaining_crps": tee._crps(
                    remaining, distribution.weights, record.geom_remaining_yd
                ),
                "lie_logloss": float(-math.log(max(lie_probs[lie_index], 1e-12))),
                "lie_brier": float(np.mean((lie_probs - onehot) ** 2)),
                "long_residual_yd": float(record.long_yd - point_mean[0]),
                "profile_n_drives": int(profile.get("n_drives", 0)),
            }
        )
    return pd.DataFrame(rows)


def donor_diagnostics(
    normalized: dict[str, pd.DataFrame],
    skill_history: pd.DataFrame,
    event_dates: dict[str, pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    round_rows = []
    for mode, frame in normalized.items():
        # Residualize coordinates within recurring-course round/hole so the
        # correlation reflects donor composition, not hole identity.
        reference = frame.groupby(["round_no", "hole_no"])["long_yd"].transform("mean")
        diagnostic = frame.assign(long_course_resid=frame["long_yd"] - reference)
        for event_key, event in diagnostic.groupby("event_key"):
            snapshot = skill_snapshot(
                skill_history, event["player_name"], event_dates[event_key]
            )
            event_rows.append(
                {
                    "mode": mode,
                    "event_key": event_key,
                    "event_date": event_dates[event_key],
                    "field_skill": snapshot["skill_est"].mean(),
                    "skill_coverage": snapshot["skill_est"].notna().mean(),
                    "mean_long_course_resid": event["long_course_resid"].mean(),
                    "fairway_rate": event["lie_class"].eq("fairway").mean(),
                }
            )
        early = diagnostic[diagnostic["round_no"].eq(1)]
        late = diagnostic[diagnostic["round_no"].ge(3)]
        round_rows.append(
            {
                "mode": mode,
                "early_n": len(early),
                "late_n": len(late),
                "late_minus_early_distance_ratio_pp": 100 * (
                    late["donor_distance_ratio"].mean() - early["donor_distance_ratio"].mean()
                ),
                "late_minus_early_long_scale": late["donor_long_scale"].mean() - early["donor_long_scale"].mean(),
                "late_minus_early_lat_scale": late["donor_lat_scale"].mean() - early["donor_lat_scale"].mean(),
                "late_minus_early_fairway_pp": 100 * (
                    late["lie_class"].eq("fairway").mean()
                    - early["lie_class"].eq("fairway").mean()
                ),
            }
        )
    event_diag = pd.DataFrame(event_rows)
    correlation = []
    for mode, group in event_diag.groupby("mode"):
        valid = group.dropna(subset=["field_skill", "mean_long_course_resid"])
        correlation.append(
            {
                "mode": mode,
                "events": len(valid),
                "corr_field_skill_long_residual": valid["field_skill"].corr(
                    valid["mean_long_course_resid"]
                ),
                "slope_long_yd_per_sg": np.polyfit(
                    valid["field_skill"], valid["mean_long_course_resid"], 1
                )[0] if len(valid) >= 3 else np.nan,
                "mean_skill_coverage": valid["skill_coverage"].mean(),
            }
        )
    return event_diag.merge(pd.DataFrame(correlation), on="mode"), pd.DataFrame(round_rows)


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    metrics = ("endpoint_nll", "endpoint_error_yd", "remaining_crps", "lie_logloss", "lie_brier")
    event = scores.groupby(["event_name", "mode"])[list(metrics)].mean().reset_index()
    raw = event[event["mode"].eq("raw")].set_index("event_name")
    rows = []
    for mode in MODES:
        group = event[event["mode"].eq(mode)].set_index("event_name")
        for metric in metrics:
            values = group[metric]
            deltas = values - raw[metric]
            se = float(deltas.std(ddof=1) / math.sqrt(len(deltas))) if mode != "raw" else 0.0
            t_critical = 4.303  # df=2; exploratory only with three events
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "events": len(values),
                    "shots": int(scores[scores["mode"].eq(mode)].shape[0]),
                    "equal_event_value": values.mean(),
                    "delta_vs_raw": deltas.mean(),
                    "event_cluster_se": se,
                    "event_min_delta": deltas.min(),
                    "event_max_delta": deltas.max(),
                    "exploratory_t_ci_low": deltas.mean() - t_critical * se,
                    "exploratory_t_ci_high": deltas.mean() + t_critical * se,
                }
            )
    return pd.DataFrame(rows)


def skill_deciles(scores: pd.DataFrame, skills: pd.DataFrame, cutoffs: dict[str, pd.Timestamp]) -> pd.DataFrame:
    parts = []
    for event_name, event in scores.groupby("event_name"):
        raw_players = event[["player_name"]].drop_duplicates()
        snapshot = skill_snapshot(skills, raw_players["player_name"], cutoffs[event_name])
        raw_players["name_key"] = raw_players["player_name"].map(norm_name)
        raw_players = raw_players.merge(snapshot, on="name_key", how="left")
        raw_players["skill_decile"] = pd.qcut(
            raw_players["skill_est"], 10, labels=False, duplicates="drop"
        )
        part = event.copy(); part["name_key"] = part["player_name"].map(norm_name)
        parts.append(part.merge(raw_players[["name_key", "skill_est", "skill_decile"]], on="name_key", how="left"))
    panel = pd.concat(parts, ignore_index=True)
    metrics = ["endpoint_nll", "remaining_crps", "lie_logloss", "long_residual_yd"]
    grouped = panel.groupby(["mode", "skill_decile"])[metrics].agg(["mean", "size"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    return grouped.reset_index()


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    event_diag: pd.DataFrame,
    round_diag: pd.DataFrame,
    deciles: pd.DataFrame,
) -> Path:
    lines = []
    for row in summary.itertuples(index=False):
        if row.mode == "raw":
            continue
        lines.append(
            f"| {row.mode} | {row.metric} | {row.delta_vs_raw:+.6f} | "
            f"[{row.event_min_delta:+.6f}, {row.event_max_delta:+.6f}] |"
        )
    corr = event_diag.drop_duplicates("mode")
    corr_lines = [
        f"| {row.mode} | {int(row.events)} | {row.corr_field_skill_long_residual:+.3f} | "
        f"{row.slope_long_yd_per_sg:+.3f} | {row.mean_skill_coverage:.1%} |"
        for row in corr.itertuples(index=False)
    ]
    round_lines = [
        f"| {row.mode} | {row.late_minus_early_distance_ratio_pp:+.3f} | "
        f"{row.late_minus_early_long_scale:+.4f} | {row.late_minus_early_lat_scale:+.4f} | "
        f"{row.late_minus_early_fairway_pp:+.3f} |"
        for row in round_diag.itertuples(index=False)
    ]
    decile_lines = []
    raw_deciles = deciles[deciles["mode"].eq("raw")].set_index("skill_decile")
    for mode, group in deciles.groupby("mode"):
        group = group.sort_values("skill_decile")
        correlation = group["skill_decile"].corr(group["long_residual_yd_mean"])
        slope = np.polyfit(group["skill_decile"], group["long_residual_yd_mean"], 1)[0]
        endpoint_delta = (
            group.set_index("skill_decile")["endpoint_nll_mean"]
            - raw_deciles["endpoint_nll_mean"]
        )
        decile_lines.append(
            f"| {mode} | {correlation:+.3f} | {slope:+.3f} | "
            f"{endpoint_delta.min():+.4f} | {endpoint_delta.max():+.4f} |"
        )
    travelers = event_diag[event_diag["event_name"].eq("Travelers Championship")].copy()
    travelers["season"] = travelers["event_key"].str.slice(5, 9).astype(int)
    traveler_lines = []
    for mode, group in travelers.groupby("mode"):
        early = group[group["season"].le(2023)]
        designated = group[group["season"].ge(2024)]
        traveler_lines.append(
            f"| {mode} | {early['field_skill'].mean():+.3f} | {designated['field_skill'].mean():+.3f} | "
            f"{early['mean_long_course_resid'].mean():+.3f} | "
            f"{designated['mean_long_course_resid'].mean():+.3f} |"
        )
    report = f"""# Field-strength normalization evaluation

Three recurring-course 2026 holdouts are scored: Sony, Travelers, and Wyndham.
All donor player transforms are fit with a strict date cutoff before that donor
event. Target transforms and DataGolf skills are likewise pre-event snapshots.
Negative scoring deltas favor normalization over raw course donors.

The score is conditional on the held-out edition's actual tee/pin axis and
yardage. It is therefore a transition-model/known-setup test, not evidence that
the same gain is available in a pre-tournament unknown-setup forecast.

## Held-out scoring versus raw donors

| Mode | Metric | Equal-event delta | Three-event range |
|---|---|---:|---:|
{chr(10).join(lines)}

The CSV also includes an exploratory Student-t interval (df=2). With three
holdouts it must not be interpreted as a production-grade confidence interval.

## Donor field-strength fingerprint

| Mode | Donor events | Corr(field SG, endpoint residual) | Yards per field SG | Skill coverage |
|---|---:|---:|---:|---:|
{chr(10).join(corr_lines)}

### Travelers designated-field natural experiment

Travelers changed from an ordinary field in 2020-2023 to a much stronger
signature field in 2024-2025. This table shows whether the endpoint year cloud
moved with that exogenous field-strength jump.

| Mode | 2020-23 field SG | 2024-25 field SG | 2020-23 endpoint residual (yd) | 2024-25 endpoint residual (yd) |
|---|---:|---:|---:|---:|
{chr(10).join(traveler_lines)}

## R1 versus R3/R4 cut-selection fingerprint

| Mode | Distance-ratio change (pp) | Long-scale change | Lateral-scale change | Fairway-rate change (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(round_lines)}

## Held-out skill-decile fingerprint

| Mode | Corr(decile, long residual) | Residual slope (yd/decile) | Best endpoint-NLL delta | Worst endpoint-NLL delta |
|---|---:|---:|---:|---:|
{chr(10).join(decile_lines)}

The strict player/state method inverts each donor's distance, lateral-bias, and
standardized dispersion profile before constructing the course distribution.
Simple centering removes an edition/setup mean but cannot remove player-level
dispersion or post-cut composition. With only three holdout events, intervals
are descriptive; the donor correlations use all available prior editions.
Lie states are reclassified after transport with a fixed empirical spatial
surface map learned from raw donor coordinates. This is coherent with the
transported endpoint, but it remains an approximation until polygon geometry
or the core geometry classifier is available.

This evaluator intentionally implements an independent estimand. The proposed
core API is `normalize_donor_points(donors, profile_snapshot, mode)` returning
model coordinates plus cutoff/coverage audit columns; equivalence testing
should be required before production integration.

Finally, event/setup centering removes each donor edition's realized mean. It
can remove setup, weather, firmness, strategy, measurement, and field effects
simultaneously, so its gain is evidence for edition/setup de-noising—not proof
of a causal field-strength correction.
"""
    path = output_dir / "FIELD_NORMALIZATION_EVALUATION.md"
    path.write_text(report, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=tee.DEFAULT_EXPORT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summary = pd.read_csv(output / "heldout_metric_summary.csv")
        event_diag = pd.read_csv(output / "donor_field_strength_diagnostics.csv")
        round_diag = pd.read_csv(output / "cut_selection_diagnostics.csv")
        deciles = pd.read_csv(output / "skill_decile_diagnostics.csv")
        if "donor_composition" not in set(round_diag["mode"]):
            round_diag = round_diag[round_diag["mode"].eq("raw")].copy()
            round_diag["mode"] = "donor_composition"
            round_diag.to_csv(output / "cut_selection_diagnostics.csv", index=False)
        print(f"Wrote {write_report(output, summary, event_diag, round_diag, deciles)}")
        return
    dates = load_event_dates(args.db_path)
    date_map = dates.set_index("event_key")["event_date"].to_dict()
    skills = load_skill_history(args.db_path)

    print("Loading all strict-prior player-profile drives...")
    player_drives = tee.load_player_training(args.export_root, 2020, 2026)
    player_drives = player_drives.merge(
        dates[["event_key", "event_date"]], on="event_key", how="left", validate="many_to_one"
    ).dropna(subset=["event_date"])
    prepared = prepare_profile_rows(player_drives)

    all_scores = []
    donor_event_diagnostics = []
    round_diagnostics = []
    target_cutoffs = {}
    for event_name in EVENTS:
        print(f"Evaluating {event_name}")
        keys = tee._event_keys(args.export_root, event_name, 2026)
        course_train, holdout = tee.load_course_panel(args.export_root, keys, 2026, 2020)
        if args.smoke_test:
            holdout = holdout[holdout["hole_no"].isin([1, 2])].head(120)
        donor_dates = {key: date_map[key] for year, key in keys.items() if year < 2026}
        target_cutoff = date_map[keys[2026]]; target_cutoffs[event_name] = target_cutoff
        target_profile = fit_snapshot(prepared, target_cutoff)
        normalized = normalize_course_donors(course_train, prepared, donor_dates)
        event_diag, round_diag = donor_diagnostics(normalized, skills, donor_dates)
        event_diag["event_name"] = event_name; round_diag["event_name"] = event_name
        donor_event_diagnostics.append(event_diag); round_diagnostics.append(round_diag)
        for index, (mode, donors) in enumerate(normalized.items()):
            scored = score_mode(donors, holdout, target_profile, mode, args.seed + index)
            scored["event_name"] = event_name
            all_scores.append(scored)

    scores = pd.concat(all_scores, ignore_index=True)
    event_diag = pd.concat(donor_event_diagnostics, ignore_index=True)
    round_diag = pd.concat(round_diagnostics, ignore_index=True)
    # Recompute field-strength correlation across all 18 donor editions per mode.
    corr = []
    for mode, group in event_diag.groupby("mode"):
        valid = group.dropna(subset=["field_skill", "mean_long_course_resid"])
        no_variation = valid["mean_long_course_resid"].std() < 1e-8
        corr.append(
            {
                "mode": mode,
                "events": len(valid),
                "corr_field_skill_long_residual": 0.0 if no_variation else (valid["field_skill"].corr(valid["mean_long_course_resid"]) if len(valid) >= 2 else np.nan),
                "slope_long_yd_per_sg": 0.0 if no_variation else (np.polyfit(valid["field_skill"], valid["mean_long_course_resid"], 1)[0] if len(valid) >= 3 else np.nan),
                "mean_skill_coverage": valid["skill_coverage"].mean(),
            }
        )
    event_diag = event_diag.drop(columns=[
        "events", "corr_field_skill_long_residual", "slope_long_yd_per_sg", "mean_skill_coverage"
    ]).merge(pd.DataFrame(corr), on="mode")
    round_diag = round_diag.groupby("mode", as_index=False).agg(
        early_n=("early_n", "sum"), late_n=("late_n", "sum"),
        late_minus_early_distance_ratio_pp=("late_minus_early_distance_ratio_pp", "mean"),
        late_minus_early_long_scale=("late_minus_early_long_scale", "mean"),
        late_minus_early_lat_scale=("late_minus_early_lat_scale", "mean"),
        late_minus_early_fairway_pp=("late_minus_early_fairway_pp", "mean"),
    )
    # Cut composition is a property of the donor fields, not a model mode.
    round_diag = round_diag[round_diag["mode"].eq("raw")].copy()
    round_diag["mode"] = "donor_composition"
    summary = summarize_scores(scores)
    deciles = skill_deciles(scores, skills, target_cutoffs)
    suffix = "_smoke" if args.smoke_test else ""
    scores.to_csv(output / f"holdout_shot_scores{suffix}.csv", index=False)
    summary.to_csv(output / f"heldout_metric_summary{suffix}.csv", index=False)
    event_diag.to_csv(output / f"donor_field_strength_diagnostics{suffix}.csv", index=False)
    round_diag.to_csv(output / f"cut_selection_diagnostics{suffix}.csv", index=False)
    deciles.to_csv(output / f"skill_decile_diagnostics{suffix}.csv", index=False)
    report_path = write_report(output, summary, event_diag, round_diag, deciles)
    print(summary.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

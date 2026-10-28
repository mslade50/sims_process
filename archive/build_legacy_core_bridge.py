"""Capture the exact legacy R1 neutral core from the current retrospective model.

This is deliberately a research-only harness.  It calls the canonical
``sim_prep.weekly_model.build_baseline_model`` in an isolated working directory
and uses a return trace to retain the in-memory Step 2 / Step 7 decomposition
that the production script otherwise discards.  Nothing in sim_prep/work is
written.

The resulting bridge supports the identity

    replacement_prediction = baseline_prediction
                             + new_core
                             - legacy_core_pred_step2

which changes only the neutral core and preserves course shape, rest/course
experience, course history, majors history, and links history exactly as they
entered the current-code retrospective baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_PREP = REPO_ROOT.parent / "sim_prep"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "permanent_data"
    / "research"
    / "base_skill_rebuild_20200101_20260822"
    / "core_bridge"
)
DEFAULT_ODDS = (
    REPO_ROOT
    / "permanent_data"
    / "research"
    / "dg_matchup_cache_20200101_20260822"
    / "matchups"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_input(source: Path, destination: Path) -> None:
    """Stage a read-only input, preferring a hard link for the large CSV."""
    if destination.exists():
        raise FileExistsError(f"isolated capture input already exists: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _capture_current_model(
    sim_prep: Path, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    work = sim_prep / "work"
    capture_dir = output / "isolated_weekly_model_capture"
    suffix = 1
    while capture_dir.exists():
        suffix += 1
        capture_dir = output / f"isolated_weekly_model_capture_{suffix}"
    capture_dir.mkdir(parents=True)

    inputs = {
        "field_adjusted_sg.csv": work / "field_adjusted_sg.csv",
        "course_random_effects.csv": work / "course_random_effects.csv",
        "field_updates.csv": work / "field_updates.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing canonical inputs: " + ", ".join(missing))
    for name, source in inputs.items():
        _stage_input(source, capture_dir / name)

    # Prevent the diagnostic coefficient fit from touching Google Sheets.
    os.environ["SIM_PREP_UPDATE_COEFFICIENTS_SHEET"] = "0"
    sys.path.insert(0, str(sim_prep))
    import weekly_model  # type: ignore

    captured: dict[str, pd.DataFrame] = {}

    detail_columns = [
        "player_name",
        "dg_id",
        "event_id",
        "event_name",
        "year",
        "tour",
        "round_date",
        "round_num",
        "course_num",
        "sample",
        "ema_adj_skill_est",
        "pred_step1",
        "pred_step2",
        "pred_step4",
        "pred_step5",
        "pred_step6",
        "pred_step7",
        "course_adjustment",
        "relative_dsle",
        "c_exp",
        "course_history",
        "std_dev_50",
    ]
    final_columns = [
        "player_name",
        "event_id",
        "year",
        "round_date",
        "round_num",
        "prediction",
        "majors_history",
        "links_adj",
    ]

    target_code = weekly_model.build_baseline_model.__code__

    def tracer(frame: Any, event: str, arg: Any):
        if frame.f_code is target_code and event == "return":
            low = frame.f_locals.get("low_skill")
            high = frame.f_locals.get("high_skill")
            final = frame.f_locals.get("all_rounds_flat")
            # CPython also emits a return trace while unwinding an exception.
            # Do not mask the original exception if the model did not reach its
            # final frames.
            if not isinstance(low, pd.DataFrame) or not isinstance(high, pd.DataFrame):
                return tracer
            if not isinstance(final, pd.DataFrame):
                return tracer

            detail_parts = []
            for label, part in (("LOW", low), ("HIGH", high)):
                missing_detail = sorted(set(detail_columns) - set(part.columns))
                if missing_detail:
                    raise RuntimeError(f"{label} detail frame is missing {missing_detail}")
                keep = part.loc[:, detail_columns].copy()
                keep["legacy_skill_bucket"] = label
                detail_parts.append(keep)
            captured["detail"] = pd.concat(detail_parts, ignore_index=True)

            missing_final = sorted(set(final_columns) - set(final.columns))
            if missing_final:
                raise RuntimeError(f"final frame is missing {missing_final}")
            captured["final"] = final.loc[:, final_columns].copy()
        return tracer

    original_cwd = Path.cwd()
    try:
        os.chdir(capture_dir)
        field_adjusted = pd.read_csv("field_adjusted_sg.csv", low_memory=False)
        sys.settrace(tracer)
        weekly_model.build_baseline_model(field_adjusted)
    finally:
        sys.settrace(None)
        os.chdir(original_cwd)

    if set(captured) != {"detail", "final"}:
        raise RuntimeError("failed to capture the canonical model decomposition")
    return captured["detail"], captured["final"], capture_dir


def _make_bridge(detail: pd.DataFrame, final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["player_name", "event_id", "year", "round_date", "round_num"]
    for frame in (detail, final):
        frame["round_date"] = pd.to_datetime(frame["round_date"], errors="raise").dt.normalize()

    detail_r1 = detail.loc[detail["round_num"].eq(1)].copy()
    final_r1 = final.loc[final["round_num"].eq(1)].copy()
    if detail_r1.duplicated(key).any() or final_r1.duplicated(key).any():
        raise RuntimeError("R1 model frames are not unique on player/event/year/date/round")

    bridge = detail_r1.merge(final_r1, on=key, how="inner", validate="one_to_one")
    if len(bridge) != len(detail_r1) or len(bridge) != len(final_r1):
        raise RuntimeError(
            f"decomposition join lost rows: detail={len(detail_r1)} "
            f"final={len(final_r1)} joined={len(bridge)}"
        )

    bridge["component_age"] = bridge["pred_step2"] - bridge["pred_step1"]
    bridge["component_course_shape"] = bridge["pred_step4"] - bridge["pred_step2"]
    bridge["component_relative_dsle"] = bridge["pred_step5"] - bridge["pred_step4"]
    bridge["component_course_experience"] = bridge["pred_step6"] - bridge["pred_step5"]
    bridge["component_course_history"] = bridge["pred_step7"] - bridge["pred_step6"]
    bridge["component_major"] = bridge["majors_history"].fillna(0.0)
    bridge["component_links"] = bridge["links_adj"].fillna(0.0)
    component_columns = [
        "component_course_shape",
        "component_relative_dsle",
        "component_course_experience",
        "component_course_history",
        "component_major",
        "component_links",
    ]
    bridge["downstream_preserved_total"] = bridge[component_columns].sum(axis=1)
    bridge["prediction_reconstructed"] = bridge["pred_step2"] + bridge["downstream_preserved_total"]
    bridge["reconstruction_error"] = bridge["prediction"] - bridge["prediction_reconstructed"]
    bridge["event_start"] = bridge["round_date"]
    bridge["event_key"] = (
        bridge["year"].astype(int).astype(str)
        + "__"
        + pd.to_numeric(bridge["event_id"], errors="raise").astype(int).astype(str)
        + "__"
        + bridge["event_start"].dt.strftime("%Y%m%d")
    )
    bridge["player_name_normalized"] = bridge["player_name"].astype(str).str.strip().str.lower()
    bridge["legacy_core_pred_step2"] = bridge["pred_step2"]
    bridge["baseline_r1_prediction"] = bridge["prediction"]
    bridge["replacement_formula"] = "baseline_r1_prediction + new_core - legacy_core_pred_step2"
    bridge["retrospective_current_coefficients"] = True
    bridge["historical_prod_snapshot"] = False

    bridge = bridge.loc[bridge["year"].between(2020, 2026)].copy()
    bridge.sort_values(["event_start", "event_id", "dg_id"], inplace=True)
    bridge.reset_index(drop=True, inplace=True)

    max_abs_error = float(bridge["reconstruction_error"].abs().max())
    if max_abs_error > 1e-10:
        raise RuntimeError(f"final prediction decomposition is not exact: max error {max_abs_error}")

    event_meta = (
        detail.assign(round_date=pd.to_datetime(detail["round_date"]).dt.normalize())
        .loc[lambda x: x["year"].between(2020, 2026)]
        .groupby(["year", "event_id"], as_index=False)
        .agg(
            event_start=("round_date", "min"),
            event_end=("round_date", "max"),
            event_name=("event_name", "first"),
            tour=("tour", "first"),
            course_num=("course_num", "first"),
            n_round_rows=("round_num", "size"),
        )
    )
    starters = (
        bridge.groupby(["year", "event_id", "event_start"], as_index=False)
        .agg(n_starters=("dg_id", "nunique"))
    )
    event_meta = event_meta.merge(
        starters,
        on=["year", "event_id", "event_start"],
        how="left",
        validate="one_to_one",
    )
    event_meta["event_key"] = (
        event_meta["year"].astype(int).astype(str)
        + "__"
        + pd.to_numeric(event_meta["event_id"], errors="raise").astype(int).astype(str)
        + "__"
        + event_meta["event_start"].dt.strftime("%Y%m%d")
    )
    return bridge, event_meta


def _read_odds(odds_dir: Path) -> pd.DataFrame:
    paths = sorted(odds_dir.glob("*.jsonl"))
    if not paths:
        return pd.DataFrame()
    columns = [
        "query_year",
        "event_id",
        "event_name",
        "event_completed",
        "p1_dg_id",
        "p2_dg_id",
    ]
    frames = []
    for path in paths:
        frame = pd.read_json(path, lines=True)
        frames.append(frame.loc[:, [c for c in columns if c in frame.columns]])
    return pd.concat(frames, ignore_index=True)


def _build_odds_crosswalk(
    bridge: pd.DataFrame,
    event_meta: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    """Map local events to odds events by date plus participant overlap.

    Event id equality is retained as a diagnostic only and is never sufficient
    for a match, because DataGolf event ids have drifted across historical eras.
    """
    if odds.empty:
        out = event_meta.copy()
        out["odds_match_found"] = False
        out["crosswalk_method"] = "odds_cache_unavailable"
        return out

    odds = odds.copy()
    odds["query_year"] = pd.to_numeric(odds["query_year"], errors="coerce").astype("Int64")
    odds["event_id"] = pd.to_numeric(odds["event_id"], errors="coerce").astype("Int64")
    odds["event_completed"] = pd.to_datetime(odds["event_completed"], errors="coerce").dt.normalize()
    odds = odds.dropna(subset=["query_year", "event_id", "event_completed"])

    odds_groups: list[dict[str, Any]] = []
    odds_key = ["query_year", "event_id", "event_name", "event_completed"]
    for keys, group in odds.groupby(odds_key, dropna=False, sort=False):
        ids = set(
            pd.concat([group["p1_dg_id"], group["p2_dg_id"]])
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
            .astype(int)
        )
        odds_groups.append(
            {
                "odds_query_year": int(keys[0]),
                "odds_event_id": int(keys[1]),
                "odds_event_name": str(keys[2]),
                "odds_event_completed": pd.Timestamp(keys[3]),
                "odds_player_ids": ids,
            }
        )

    local_sets = {
        key: set(pd.to_numeric(group["dg_id"], errors="coerce").dropna().astype(int))
        for key, group in bridge.groupby("event_key")
    }
    rows: list[dict[str, Any]] = []
    for local in event_meta.itertuples(index=False):
        local_ids = local_sets.get(local.event_key, set())
        candidates: list[dict[str, Any]] = []
        for odds_event in odds_groups:
            if odds_event["odds_query_year"] != int(local.year):
                continue
            date_delta = abs((odds_event["odds_event_completed"] - local.event_end).days)
            if date_delta > 8:
                continue
            odds_ids = odds_event["odds_player_ids"]
            overlap = len(local_ids & odds_ids)
            if not overlap:
                continue
            union = len(local_ids | odds_ids)
            jaccard = overlap / union if union else 0.0
            local_coverage = overlap / len(local_ids) if local_ids else 0.0
            odds_coverage = overlap / len(odds_ids) if odds_ids else 0.0
            name_similarity = SequenceMatcher(
                None,
                str(local.event_name).lower(),
                odds_event["odds_event_name"].lower(),
            ).ratio()
            id_equal = int(local.event_id) == odds_event["odds_event_id"]
            score = (
                5.0 * odds_coverage
                + 2.0 * jaccard
                + 0.75 * local_coverage
                + 0.50 * name_similarity
                + 0.35 * float(id_equal)
                + 0.50 * np.exp(-date_delta / 2.0)
            )
            candidates.append(
                {
                    **odds_event,
                    "date_delta_days": int(date_delta),
                    "player_overlap_n": int(overlap),
                    "local_player_n": int(len(local_ids)),
                    "odds_player_n": int(len(odds_ids)),
                    "local_player_coverage": float(local_coverage),
                    "odds_player_coverage": float(odds_coverage),
                    "player_jaccard": float(jaccard),
                    "event_name_similarity": float(name_similarity),
                    "event_id_equal": bool(id_equal),
                    "crosswalk_score": float(score),
                }
            )
        candidates.sort(key=lambda row: row["crosswalk_score"], reverse=True)
        best = candidates[0] if candidates else None
        runner_up_score = candidates[1]["crosswalk_score"] if len(candidates) > 1 else np.nan
        accepted = bool(
            best
            and best["player_overlap_n"] >= 2
            and best["odds_player_coverage"] >= 0.50
            and best["date_delta_days"] <= 4
        )
        base = {name: getattr(local, name) for name in event_meta.columns}
        if best:
            best = {k: v for k, v in best.items() if k != "odds_player_ids"}
            base.update(best)
        base["runner_up_score"] = runner_up_score
        base["score_gap"] = (
            best["crosswalk_score"] - runner_up_score
            if best and np.isfinite(runner_up_score)
            else np.nan
        )
        base["odds_match_found"] = accepted
        base["crosswalk_method"] = "participant_overlap_plus_date" if accepted else "unmatched"
        base["event_id_drift"] = bool(accepted and not best["event_id_equal"]) if best else False
        base["crosswalk_ambiguous"] = bool(
            accepted and np.isfinite(base["score_gap"]) and base["score_gap"] < 0.5
        )
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-prep", type=Path, default=DEFAULT_SIM_PREP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--odds-dir", type=Path, default=DEFAULT_ODDS)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail, final, capture_dir = _capture_current_model(args.sim_prep.resolve(), output)
    bridge, event_meta = _make_bridge(detail, final)
    odds = _read_odds(args.odds_dir.resolve())
    crosswalk = _build_odds_crosswalk(bridge, event_meta, odds)

    bridge_path = output / "legacy_core_bridge.parquet"
    event_path = output / "local_event_identity.parquet"
    crosswalk_path = output / "odds_event_crosswalk.parquet"
    bridge.to_parquet(bridge_path, index=False)
    event_meta.to_parquet(event_path, index=False)
    crosswalk.to_parquet(crosswalk_path, index=False)

    target = bridge.loc[bridge["year"].between(2023, 2026)]
    source_baseline = args.sim_prep / "work" / "sg_preds_resids.csv"
    captured_baseline = capture_dir / "sg_preds_resids.csv"
    source_baseline_hash = _sha256(source_baseline)
    captured_baseline_hash = _sha256(captured_baseline)
    unmatched = crosswalk.loc[
        ~crosswalk["odds_match_found"],
        ["year", "event_id", "event_name", "event_start", "event_end", "event_key"],
    ].copy()
    unmatched["event_start"] = unmatched["event_start"].astype(str)
    unmatched["event_end"] = unmatched["event_end"].astype(str)
    manifest = {
        "artifact_label": "CURRENT-CODE RETROSPECTIVE LEGACY DECOMPOSITION",
        "purpose": "research-only exact neutral-core bridge for the current-code retrospective baseline",
        "formula": "replacement = baseline_r1_prediction + new_core - legacy_core_pred_step2",
        "source_code": str((args.sim_prep / "weekly_model.py").resolve()),
        "source_code_sha256": _sha256(args.sim_prep / "weekly_model.py"),
        "source_field_adjusted_sha256": _sha256(args.sim_prep / "work" / "field_adjusted_sg.csv"),
        "source_sg_preds_resids_sha256": source_baseline_hash,
        "captured_baseline_sha256": captured_baseline_hash,
        "captured_baseline_bitwise_identical_to_source": captured_baseline_hash == source_baseline_hash,
        "successful_isolated_capture": str(capture_dir),
        "rows_2020_2026": int(len(bridge)),
        "rows_2023_2026": int(len(target)),
        "events_2023_2026": int(target["event_key"].nunique()),
        "date_coverage_2020_2026": [
            str(bridge["event_start"].min().date()),
            str(bridge["event_start"].max().date()),
        ],
        "date_coverage_2023_2026": [
            str(target["event_start"].min().date()),
            str(target["event_start"].max().date()),
        ],
        "max_abs_reconstruction_error": float(bridge["reconstruction_error"].abs().max()),
        "duplicate_player_event_rows": int(
            bridge.duplicated(["event_key", "dg_id"]).sum()
        ),
        "odds_crosswalk_matches": int(crosswalk.get("odds_match_found", pd.Series(dtype=bool)).sum()),
        "odds_crosswalk_id_drifts": int(crosswalk.get("event_id_drift", pd.Series(dtype=bool)).sum()),
        "odds_crosswalk_ambiguous": int(crosswalk.get("crosswalk_ambiguous", pd.Series(dtype=bool)).sum()),
        "odds_crosswalk_unmatched": unmatched.to_dict(orient="records"),
        "point_in_time": {
            "feature_rows": "lagged/shifted by the canonical weekly model before each historical round",
            "coefficient_fit": "retrospective current coefficients fit through the latest source date",
            "historical_prod_snapshot": False,
            "warning": (
                "This exactly decomposes the current-code retrospective baseline; it is not an "
                "archive of the coefficients or manual overrides that were live on each historical date."
            ),
        },
        "outputs": {
            path.name: {"rows": int(len(frame)), "sha256": _sha256(path)}
            for path, frame in (
                (bridge_path, bridge),
                (event_path, event_meta),
                (crosswalk_path, crosswalk),
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

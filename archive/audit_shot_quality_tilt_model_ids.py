"""Persist the source reward-model ID mix for a completed quality-tilt run.

This is a provenance supplement only.  It does not rescore rows or modify the
authoritative evaluator manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args()

    parent_manifest_path = args.result_dir / "manifest.json"
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    selected = pd.read_csv(
        args.result_dir / "reward_input_coverage.csv",
        usecols=["event_key", "event_start"],
    ).drop_duplicates("event_key")
    selected["season"] = pd.to_datetime(selected["event_start"]).dt.year

    parts: list[pd.DataFrame] = []
    for season, events in selected.groupby("season", observed=True):
        path = args.export_root / f"pga_{int(season)}_shot_sg.parquet"
        frame = pd.read_parquet(
            path,
            columns=["event_key", "sg_category", "model_id", "start_lie", "end_lie"],
            filters=[("event_key", "in", events["event_key"].astype(str).tolist())],
        )
        frame["model_id"] = frame["model_id"].fillna("<missing>").astype(str)
        frame["green_start"] = frame["start_lie"].eq("green").astype(int)
        frame["green_end"] = frame["end_lie"].eq("green").astype(int)
        mix = (
            frame.groupby(
                ["event_key", "sg_category", "model_id"],
                observed=True,
                dropna=False,
            )
            .agg(rows=("model_id", "size"), green_start=("green_start", "sum"), green_end=("green_end", "sum"))
            .reset_index()
        )
        mix.insert(0, "season", int(season))
        parts.append(mix)

    detail = pd.concat(parts, ignore_index=True).sort_values(
        ["season", "event_key", "sg_category", "model_id"], kind="mergesort"
    )
    summary = (
        detail.groupby(["season", "sg_category"], observed=True, dropna=False)
        .agg(
            events=("event_key", "nunique"),
            distinct_model_ids=("model_id", "nunique"),
            event_model_pairs=("model_id", "size"),
            rows=("rows", "sum"),
            green_start=("green_start", "sum"),
            green_end=("green_end", "sum"),
        )
        .reset_index()
    )

    output = args.result_dir / "model_id_provenance"
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "model_id_mix.csv"
    summary_path = output / "model_id_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    source_path = Path(__file__).resolve()
    manifest = {
        "contract": "shot_quality_tilt_model_id_provenance_v1",
        "authoritative_scores_modified": False,
        "parent_manifest_sha256": sha256(parent_manifest_path),
        "parent_evaluator_source_sha256": parent["source_sha256"],
        "source_sha256": sha256(source_path),
        "selected_events": int(selected["event_key"].nunique()),
        "model_id_rows": int(len(detail)),
        "parent_input_files": parent["input_files"],
        "artifacts": {
            detail_path.name: sha256(detail_path),
            summary_path.name: sha256(summary_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

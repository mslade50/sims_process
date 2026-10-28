"""Combine frozen full-chain event panels with an auditable hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from evaluate_spatial_chain_rounds import load_complete_round_particles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = [path.resolve() for path in args.input]
    if len(set(inputs)) != len(inputs):
        raise ValueError("Input particle paths must be unique")
    frames = [load_complete_round_particles(path) for path in inputs]
    combined = pd.concat(frames, ignore_index=True)
    duplicate_key = [
        "event", "source_player_id", "candidate_arm", "round_num", "particle_id"
    ]
    if combined.duplicated(duplicate_key).any():
        raise ValueError("Combined panel has overlapping event/particle keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    verified = load_complete_round_particles(args.output)
    if len(verified) != len(combined):
        raise AssertionError("Persisted combined panel changed row count")

    event_counts = (
        verified.groupby("event", observed=True)
        .size()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    manifest = {
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "rows": int(len(verified)),
        "events": event_counts,
        "players_by_event": {
            str(event): int(group["source_player_id"].nunique())
            for event, group in verified.groupby("event", observed=True)
        },
        "candidate_arms": sorted(verified["candidate_arm"].unique().tolist()),
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "loader_contract_passed": True,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

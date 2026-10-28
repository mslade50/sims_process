"""Reuse unaffected v2 checkpoints after the major-cut-rule correction.

The only simulation behavior changed between the source and destination runs is
the cut line for PGA Championships and U.S. Opens.  This helper copies the
other event checkpoints into a new output directory, verifies their hashes,
and updates only their run-config hash.  The five affected events are omitted
so the main backtest runner must simulate them again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


AFFECTED_EVENTS = {
    "pga:R2024033",
    "pga:R2025033",
    "pga:R2026033",
    "pga:R2025026",
    "pga:R2026026",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--backtest-source",
        type=Path,
        default=Path(__file__).with_name("backtest_shot_mean_closing_markets.py"),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    source_config_path = source / "run_config.json"
    source_checkpoints = source / "event_probabilities"
    if not source_config_path.is_file() or not source_checkpoints.is_dir():
        raise FileNotFoundError("Source run is missing run_config.json or checkpoints")

    old_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    old_hash = str(old_config.pop("config_hash"))
    old_source_hash = str(old_config["input_hashes"]["source::backtest_engine"])
    new_source_hash = sha256_file(args.backtest_source.resolve())
    if new_source_hash == old_source_hash:
        raise ValueError("Backtest source hash did not change; migration is unnecessary")
    old_config["input_hashes"]["source::backtest_engine"] = new_source_hash
    new_hash = hashlib.sha256(
        json.dumps(old_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    new_config = {**old_config, "config_hash": new_hash}

    destination_checkpoints = destination / "event_probabilities"
    destination_checkpoints.mkdir(parents=True)
    (destination / "run_config.json").write_text(
        json.dumps(new_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    copied = 0
    skipped = set()
    for metadata_path in sorted(source_checkpoints.glob("*.meta.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        event_key = str(metadata.get("event_audit", {}).get("event_key", ""))
        checkpoint_path = metadata_path.with_suffix("").with_suffix(".parquet")
        if event_key in AFFECTED_EVENTS:
            skipped.add(event_key)
            continue
        if metadata.get("complete") is not True:
            raise ValueError(f"Incomplete source checkpoint: {metadata_path}")
        if metadata.get("config_hash") != old_hash:
            raise ValueError(f"Unexpected source config hash: {metadata_path}")
        if sha256_file(checkpoint_path) != metadata.get("checkpoint_sha256"):
            raise ValueError(f"Source checkpoint hash mismatch: {checkpoint_path}")

        destination_checkpoint = destination_checkpoints / checkpoint_path.name
        shutil.copy2(checkpoint_path, destination_checkpoint)
        metadata["config_hash"] = new_hash
        metadata["checkpoint_reused_from_config_hash"] = old_hash
        metadata["checkpoint_reuse_reason"] = (
            "Unaffected by explicit PGA Championship/U.S. Open cut-line correction"
        )
        (destination_checkpoints / metadata_path.name).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        copied += 1

    if skipped != AFFECTED_EVENTS:
        raise ValueError(
            f"Affected checkpoint set mismatch: expected={sorted(AFFECTED_EVENTS)} "
            f"observed={sorted(skipped)}"
        )
    if copied + len(skipped) != int(new_config["event_count"]):
        raise ValueError("Migrated plus invalidated checkpoint count does not match event_count")
    print(f"Copied {copied} unaffected checkpoints; invalidated {len(skipped)}")
    print(f"Old config hash: {old_hash}")
    print(f"New config hash: {new_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

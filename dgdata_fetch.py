"""Fetch the latest dg_historical snapshot from Cloudflare R2 (read-only).

Vendored minimal reader for sim_prep/dgdata.py's snapshot layout (protocol 2):
canonical/manifest.json -> databases[name].manifest_key -> per-snapshot
manifest {object_key, sha256, snapshot_id, uploaded_at} -> zstd sqlite.
Falls back to {name}/manifest.json when no root manifest exists.

Credentials from env: CF_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
DGDATA_BUCKET (default dg-golf-data). Used by build-archetypes.yml; point
DG_HISTORICAL_DB at the output so sg_diagnostic reads the snapshot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT_MANIFEST_KEY = "canonical/manifest.json"
PROTOCOL_VERSION = 2


def fetch_snapshot(name: str = "dg_historical", out: str | None = None) -> Path:
    import boto3
    import zstandard as zstd

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['CF_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    bucket = os.getenv("DGDATA_BUCKET", "dg-golf-data")

    def read_json(key: str) -> dict:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())

    manifest_key = f"{name}/manifest.json"
    try:
        root = read_json(ROOT_MANIFEST_KEY)
        if int(root.get("protocol_version", 0)) != PROTOCOL_VERSION:
            raise RuntimeError(
                f"Unsupported canonical manifest protocol {root.get('protocol_version')!r}"
            )
        manifest_key = root["databases"][name]["manifest_key"]
    except client.exceptions.NoSuchKey:
        pass
    manifest = read_json(manifest_key)

    out_path = Path(out or f"{name}_snapshot.db")
    compressed = out_path.with_suffix(".zst.part")
    staged = out_path.with_suffix(".part")
    try:
        client.download_file(bucket, manifest["object_key"], str(compressed))
        with compressed.open("rb") as src, staged.open("wb") as dst:
            zstd.ZstdDecompressor().copy_stream(src, dst)
        digest = hashlib.sha256()
        with staged.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest["sha256"]:
            raise RuntimeError(
                f"Snapshot checksum mismatch for {name}/{manifest['snapshot_id']}"
            )
        staged.replace(out_path)
    finally:
        compressed.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)

    print(
        f"{name}: snapshot {manifest['snapshot_id']} "
        f"(uploaded {manifest.get('uploaded_at', '?')}) -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest dgdata snapshot from R2")
    parser.add_argument("--name", default="dg_historical")
    parser.add_argument("--out", help="Output path (default {name}_snapshot.db)")
    args = parser.parse_args()
    fetch_snapshot(args.name, args.out)

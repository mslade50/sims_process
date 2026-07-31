"""Read-only access to dgdata snapshots in Cloudflare R2.

Vendored minimal reader for sim_prep/dgdata.py's snapshot layout (protocol 2):
canonical/manifest.json -> databases[name].manifest_key -> per-snapshot
manifest {object_key, sha256, snapshot_id, uploaded_at} -> zstd sqlite.
Falls back to {name}/manifest.json when no root manifest exists.

Credentials: CF_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
DGDATA_BUCKET (default dg-golf-data) — environment first, then the local
"dgdata" keyring (same store sim_prep uses).

Two entry points:
  fetch_snapshot(name, out)   — download latest to an explicit path (CI)
  cached_snapshot_path(name)  — latest via the shared local cache
                                (%LOCALAPPDATA%/etr-golf/cache, same layout as
                                sim_prep, so the two tools share downloads);
                                falls back to the newest cached copy offline
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT_MANIFEST_KEY = "canonical/manifest.json"
PROTOCOL_VERSION = 2


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import keyring

        return keyring.get_password("dgdata", name)
    except Exception:
        return None


def _bucket() -> str:
    return _secret("DGDATA_BUCKET") or "dg-golf-data"


def _client():
    import boto3
    from botocore.config import Config

    account = _secret("CF_ACCOUNT_ID")
    key_id = _secret("R2_ACCESS_KEY_ID")
    secret = _secret("R2_SECRET_ACCESS_KEY")
    if not (account and key_id and secret):
        raise RuntimeError("R2 credentials unavailable (checked env and dgdata keyring)")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(connect_timeout=5, read_timeout=120, retries={"max_attempts": 2}),
    )


def _latest_manifest(client, name: str) -> dict:
    def read_json(key: str) -> dict:
        return json.loads(client.get_object(Bucket=_bucket(), Key=key)["Body"].read())

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
    return read_json(manifest_key)


def _download_verified(client, manifest: dict, out_path: Path) -> None:
    import zstandard as zstd

    compressed = out_path.with_suffix(".zst.part")
    staged = out_path.with_suffix(".part")
    try:
        client.download_file(_bucket(), manifest["object_key"], str(compressed))
        with compressed.open("rb") as src, staged.open("wb") as dst:
            zstd.ZstdDecompressor().copy_stream(src, dst)
        digest = hashlib.sha256()
        with staged.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest["sha256"]:
            raise RuntimeError(
                f"Snapshot checksum mismatch for {manifest['database']}/{manifest['snapshot_id']}"
            )
        staged.replace(out_path)
    finally:
        compressed.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


def fetch_snapshot(name: str = "dg_historical", out: str | None = None) -> Path:
    """Download the latest snapshot to an explicit path (used by CI)."""
    client = _client()
    manifest = _latest_manifest(client, name)
    out_path = Path(out or f"{name}_snapshot.db")
    _download_verified(client, manifest, out_path)
    print(
        f"{name}: snapshot {manifest['snapshot_id']} "
        f"(uploaded {manifest.get('uploaded_at', '?')}) -> {out_path}"
    )
    return out_path


def cached_snapshot_path(name: str = "dg_historical") -> Path:
    """Latest snapshot via the shared local cache; newest cached copy offline."""
    base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    cache_dir = base / "etr-golf" / "cache" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = _client()
        manifest = _latest_manifest(client, name)
        db_path = cache_dir / f"{manifest['snapshot_id']}.db"
        sha_path = cache_dir / f"{manifest['snapshot_id']}.sha256"
        cache_valid = (
            db_path.is_file()
            and sha_path.is_file()
            and sha_path.read_text(encoding="ascii").strip() == manifest["sha256"]
        )
        if not cache_valid:
            print(f"[dgdata] downloading {name} snapshot {manifest['snapshot_id']}...")
            _download_verified(client, manifest, db_path)
            sha_path.write_text(manifest["sha256"], encoding="ascii")
        return db_path
    except Exception as exc:
        cached = sorted(cache_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cached:
            print(f"[dgdata] R2 unavailable ({exc}) — using cached {cached[0].name}")
            return cached[0]
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest dgdata snapshot from R2")
    parser.add_argument("--name", default="dg_historical")
    parser.add_argument("--out", help="Output path (default {name}_snapshot.db)")
    args = parser.parse_args()
    fetch_snapshot(args.name, args.out)

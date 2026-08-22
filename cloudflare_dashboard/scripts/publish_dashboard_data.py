"""Export the latest dashboard JSON and publish it atomically to Cloudflare R2.

All data objects are uploaded before ``manifest.json``. The manifest is the
dashboard's freshness signal, so publishing it last prevents clients from
observing a partially refreshed snapshot.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


SITE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SITE_ROOT.parent
PUBLISH_ROOT = SITE_ROOT / ".publish" / "data"
DEFAULT_BUCKET = "golf-model-dashboard-data"

if str(SITE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(SITE_ROOT / "scripts"))

from export_dashboard_data import export  # noqa: E402


def order_snapshot_files(files: list[Path], root: Path) -> list[Path]:
    """Return snapshot paths with the manifest last for atomic publication."""
    files = sorted(files)
    manifests = [path for path in files if path.relative_to(root).as_posix() == "manifest.json"]
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one manifest.json in {root}, found {len(manifests)}")
    manifest = manifests[0]
    return [path for path in files if path != manifest] + [manifest]


def snapshot_files(root: Path) -> list[Path]:
    return order_snapshot_files(
        [path for path in root.rglob("*.json") if path.is_file()], root
    )


def object_key(path: Path, root: Path) -> str:
    return f"data/{path.relative_to(root).as_posix()}"


def cache_control(path: Path, root: Path) -> str:
    if path.relative_to(root).as_posix() == "manifest.json":
        return "no-cache"
    return "public, max-age=300, stale-while-revalidate=3600"


def _r2_client():
    import boto3

    account_id = os.environ["CF_ACCOUNT_ID"]
    access_key = os.getenv("DASHBOARD_R2_ACCESS_KEY_ID") or os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.getenv("DASHBOARD_R2_SECRET_ACCESS_KEY") or os.environ["R2_SECRET_ACCESS_KEY"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def upload_with_s3(files: list[Path], root: Path, bucket: str) -> None:
    client = _r2_client()
    for path in files:
        key = object_key(path, root)
        client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={
                "ContentType": "application/json; charset=utf-8",
                "CacheControl": cache_control(path, root),
            },
        )
        remote = client.head_object(Bucket=bucket, Key=key)
        if int(remote.get("ContentLength", -1)) != path.stat().st_size:
            raise RuntimeError(f"R2 size check failed for {key}")
        print(f"  + r2://{bucket}/{key} ({path.stat().st_size / 1024:.1f} KB)")


def upload_with_wrangler(files: list[Path], root: Path, bucket: str) -> None:
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise RuntimeError("Could not find npx for the authenticated Wrangler upload")
    for path in files:
        key = object_key(path, root)
        command = [
            npx,
            "wrangler",
            "r2",
            "object",
            "put",
            f"{bucket}/{key}",
            "--file",
            str(path),
            "--content-type",
            "application/json; charset=utf-8",
            "--cache-control",
            cache_control(path, root),
            "--remote",
            "--force",
        ]
        result = subprocess.run(
            command,
            cwd=SITE_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Wrangler upload failed for {key}: {detail[-500:]}")
        print(f"  + r2://{bucket}/{key} ({path.stat().st_size / 1024:.1f} KB)")


def publish(
    bucket: str, output: Path, dry_run: bool = False, reuse_export: bool = False
) -> None:
    if reuse_export:
        print(f"\n  Reusing dashboard JSON from {output}...")
    else:
        print("\n  Exporting Cloudflare dashboard JSON...")
        export(output)
    files = snapshot_files(output)
    total_mb = sum(path.stat().st_size for path in files) / (1024 * 1024)
    print(f"\n  Prepared {len(files)} JSON objects ({total_mb:.1f} MB); manifest publishes last.")
    if dry_run:
        print(f"  [DRY RUN] Would upload to r2://{bucket}/data/")
        return

    has_dashboard_credentials = all(
        os.getenv(name)
        for name in (
            "CF_ACCOUNT_ID",
            "DASHBOARD_R2_ACCESS_KEY_ID",
            "DASHBOARD_R2_SECRET_ACCESS_KEY",
        )
    )
    has_legacy_credentials = all(
        os.getenv(name)
        for name in ("CF_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    )
    has_s3_credentials = has_dashboard_credentials or has_legacy_credentials
    if has_s3_credentials:
        try:
            upload_with_s3(files, output, bucket)
            return
        except Exception as exc:
            if not os.getenv("CLOUDFLARE_API_TOKEN"):
                raise
            print(f"  S3 upload unavailable ({exc}); retrying with the Cloudflare API token.")
    upload_with_wrangler(files, output, bucket)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish dashboard JSON to Cloudflare R2")
    parser.add_argument("--bucket", default=os.getenv("DASHBOARD_R2_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--output", type=Path, default=PUBLISH_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="Upload the existing output directory without exporting again",
    )
    args = parser.parse_args()
    publish(
        args.bucket,
        args.output.resolve(),
        dry_run=args.dry_run,
        reuse_export=args.reuse_export,
    )


if __name__ == "__main__":
    main()

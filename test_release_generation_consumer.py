import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("COEFFS_FROM_CACHE", "1")
os.environ.setdefault("MAKER_SHADOW", "1")

import kalshi_maker as km


def _sealed(payload):
    payload = dict(payload)
    payload["manifest_sha256"] = km._sealed_manifest_id(payload)
    return payload


def test_maker_resolves_manifest_bound_release_generation(monkeypatch):
    km._SIM_FAIRS_META = {
        "source": "github",
        "release_generation": "generation-1",
        "tourney": "test_event",
        "event_id": "99",
        "round": 4,
        "simulation_manifest_sha256": "sim-id",
        "live_tournament_manifest_sha256": "live-id",
    }
    manifest = _sealed(
        {
            "schema_version": "complete-live-package/v1",
            "generation": "generation-1",
            "tourney": "test_event",
            "event_id": "99",
            "round": 4,
            "simulation_manifest_sha256": "sim-id",
            "live_tournament_manifest_sha256": "live-id",
            "release_assets": {},
            "git_files": {},
        }
    )
    monkeypatch.setattr(
        km, "_fetch_published_json_github", lambda filename: manifest
    )
    assert km._load_release_generation_manifest() == manifest

    bad = {**manifest, "event_id": "another-event"}
    monkeypatch.setattr(km, "_fetch_published_json_github", lambda filename: bad)
    assert km._load_release_generation_manifest() is None


def test_maker_downloads_exact_versioned_matchup_asset(tmp_path, monkeypatch):
    import requests

    table = pa.Table.from_pandas(
        pd.DataFrame([[270, 271], [271, 270]], index=["alpha", "beta"]),
        preserve_index=True,
    )
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"tourney": b"test_event",
            b"event_id": b"99",
            b"sim_run_at": b"2026-08-23 12:00:00 UTC",
            b"simulation_manifest_sha256": b"sim-id",
            b"live_tournament_manifest_sha256": b"live-id",
        }
    )
    tape = tmp_path / "source.parquet"
    pq.write_table(table, tape)
    data = tape.read_bytes()
    name = "matchup_scores_live.generation-1.abc.parquet"
    binding = {
        "name": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }
    manifest = _sealed(
        {
            "schema_version": "complete-live-package/v1",
            "generation": "generation-1",
            "tourney": "test_event",
            "event_id": "99",
            "round": 4,
            "simulation_manifest_sha256": "sim-id",
            "live_tournament_manifest_sha256": "live-id",
            "release_assets": {"matchup_scores_live": binding},
            "git_files": {},
        }
    )
    km._SIM_FAIRS_META = {
        "source": "github",
        "release_generation": "generation-1",
        "tourney": "test_event",
        "event_id": "99",
        "round": 4,
        "sim_run_at": "2026-08-23 12:00:00 UTC",
        "simulation_manifest_sha256": "sim-id",
        "live_tournament_manifest_sha256": "live-id",
    }
    monkeypatch.setattr(km, "tourney", "test_event")
    monkeypatch.setattr(km, "_load_release_generation_manifest", lambda: manifest)
    monkeypatch.setattr(
        km, "_MATCHUP_TAPE_CACHE", str(tmp_path / "cache.parquet")
    )
    monkeypatch.setattr(
        km, "_MATCHUP_TAPE_CACHE_META", str(tmp_path / "cache.json")
    )

    class Response:
        def __init__(self, payload=None, content=b""):
            self._payload = payload
            self.content = content

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    def get(url, **_kwargs):
        if "/releases/tags/" in url:
            return Response(
                payload={
                    "assets": [
                        {
                            "id": 7,
                            "name": name,
                            "updated_at": "2026-08-23T12:00:00Z",
                            "url": "https://api.github.test/assets/7",
                        }
                    ]
                }
            )
        return Response(content=data)

    monkeypatch.setattr(requests, "get", get)
    scores, names = km._fetch_matchup_tape_release()
    assert scores.shape == (2, 2)
    assert names == ["alpha", "beta"]
    cache_meta = json.loads((tmp_path / "cache.json").read_text())
    assert cache_meta["generation"] == "generation-1"
    assert cache_meta["sha256"] == binding["sha256"]

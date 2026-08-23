"""Upload release-bound odds + fair prices to Cloudflare R2 for the odds screen.

Loads the strict simulation release committed at the checked-out Git revision,
joins its model probabilities to event/round-scoped book quotes, and either
writes publish-ready files or uploads them to R2.

Usage:
    python push_odds_screen.py             # Upload all markets
    python push_odds_screen.py --dry-run   # Print JSON to stdout, skip upload
    python push_odds_screen.py --output-dir /tmp/odds-screen

Env vars:
    CF_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY  (for R2 upload)
    GH_TOKEN or GITHUB_TOKEN  (for golf_scraping repo fetch)
    GOOGLE_CREDS_JSON or credentials.json  (for sheet_config)
"""

import json
import hashlib
import logging
import math
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
R2_BUCKET = "golf-odds-data"
R2_PREFIX = "odds_data"
RELEASE_MANIFEST_SCHEMA = "complete-live-package/v1"
ODDS_GENERATION_SCHEMA = "odds-screen-generation/v1"
RELEASE_MANIFEST_PATH = PROJECT_ROOT / "sim_release_manifest.json"
SIM_FAIRS_PATH = PROJECT_ROOT / "sim_fairs.json"
PROVENANCE_KEYS = (
    "event_id",
    "tourney",
    "round",
    "release_generation",
    "release_manifest_sha256",
    "simulation_manifest_sha256",
    "live_tournament_manifest_sha256",
    "source_git_sha",
)

GITHUB_REPO = "mslade50/golf_scraping"
GITHUB_BRANCH = "master"
GITHUB_DATA_PATH = "data"

SCRAPED_LOCAL = PROJECT_ROOT / "permanent_data" / "scraped_odds"


class OddsScreenContractError(RuntimeError):
    """The committed model package is not safe to activate on the odds screen."""


def _require_recent_timestamp(value, *, label: str) -> datetime:
    """Apply the simulation health gate's production age/future-skew limits."""
    from sim_health_gate import DEFAULT_MAX_AGE_HOURS

    try:
        stamped = datetime.strptime(
            str(value), "%Y-%m-%d %H:%M:%S UTC"
        ).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise OddsScreenContractError(f"{label} timestamp is missing or invalid") from exc
    age = datetime.now(timezone.utc) - stamped
    if age < -timedelta(minutes=5):
        raise OddsScreenContractError(f"{label} timestamp is too far in the future")
    if age > timedelta(hours=float(DEFAULT_MAX_AGE_HOURS)):
        raise OddsScreenContractError(
            f"{label} is stale ({age.total_seconds() / 3600:.1f}h old)"
        )
    return stamped


def _safe_release_path(project_root: Path, relative: str) -> Path:
    """Resolve one manifest path without allowing absolute or parent traversal."""
    relative = str(relative or "")
    posix = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or posix.is_absolute()
        or ".." in posix.parts
        or any(":" in part for part in posix.parts)
        or str(posix) != relative
    ):
        raise OddsScreenContractError(
            f"unsafe path in sim release manifest: {relative!r}"
        )
    root = Path(project_root).resolve()
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise OddsScreenContractError(
            f"release manifest path escapes the project root: {relative!r}"
        ) from exc
    return path


def _git_command(project_root: Path, *args: str, text: bool = False):
    root = Path(project_root).resolve()
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            *args,
        ],
        capture_output=True,
        text=text,
    )


def _git_head_sha(project_root: Path) -> str:
    result = _git_command(project_root, "rev-parse", "HEAD", text=True)
    sha = result.stdout.strip()
    if result.returncode != 0 or len(sha) != 40 or any(
        ch not in "0123456789abcdefABCDEF" for ch in sha
    ):
        raise OddsScreenContractError("could not resolve the committed source Git SHA")
    return sha.lower()


def _git_blob_bytes(project_root: Path, source_git_sha: str, relative: str) -> bytes:
    result = _git_command(project_root, "show", f"{source_git_sha}:{relative}")
    if result.returncode != 0:
        raise OddsScreenContractError(
            f"release input is not present at {source_git_sha[:12]}: {relative}"
        )
    return result.stdout


def _require_git_equivalent_worktree_file(
    project_root: Path, source_git_sha: str, relative: str
) -> None:
    """Require the checked-out file to map to the exact source-revision blob.

    `git diff` applies attributes/EOL rules, so a clean Windows CRLF checkout is
    accepted while any consumer-visible change still fails closed.
    """
    result = _git_command(
        project_root,
        "diff",
        "--quiet",
        "--no-ext-diff",
        source_git_sha,
        "--",
        relative,
    )
    if result.returncode != 0:
        raise OddsScreenContractError(
            f"declared sim release file differs from {source_git_sha[:12]}: {relative}"
        )


def _release_required_files(tourney: str, sim_round: int) -> set[str]:
    """Files a strict live release must bind for the odds-screen generation."""
    return {
        "sim_fairs.json",
        "round_samples.parquet",
        f"round_h2h_r{sim_round}.parquet",
        f"round_h2h_r{sim_round}_meta.json",
        f"round_h2h_r{sim_round}_health.json",
        f"round_3ball_r{sim_round}.parquet",
        f"round_3ball_r{sim_round}_meta.json",
        f"round_3ball_r{sim_round}_contract.json",
        "tournament_samples.parquet",
        "tournament_made_cut.parquet",
        f"tournament_live_{tourney}_health.json",
    }


def _require_probability_map(
    value, *, label: str, field: set[str] | None = None
) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise OddsScreenContractError(f"committed sim fairs have empty {label}")
    result = {}
    for raw_name, raw_probability in value.items():
        name = str(raw_name or "").strip().lower()
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise OddsScreenContractError(
                f"committed sim fairs have invalid probability in {label}: {raw_name!r}"
            ) from exc
        if not name or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise OddsScreenContractError(
                f"committed sim fairs have invalid probability in {label}: {raw_name!r}"
            )
        if field is not None and name not in field:
            raise OddsScreenContractError(
                f"committed sim fairs contain an off-field player in {label}: {name}"
            )
        if name in result:
            raise OddsScreenContractError(
                f"committed sim fairs contain duplicate canonical names in {label}: {name}"
            )
        result[name] = probability
    if field is not None and set(result) != field:
        raise OddsScreenContractError(
            f"committed sim fairs do not provide exact field coverage in {label}"
        )
    return result


def _validate_sim_fairs_semantics(fairs: dict) -> None:
    """Require the model families the screen exposes, not merely valid JSON."""
    field_values = fairs.get("field")
    if not isinstance(field_values, list) or not field_values:
        raise OddsScreenContractError("committed sim fairs have an empty field")
    field = {str(name or "").strip().lower() for name in field_values}
    if "" in field or len(field) != len(field_values):
        raise OddsScreenContractError(
            "committed sim fairs have blank or duplicate field names"
        )

    outrights = fairs.get("outrights") or {}
    for market in ("winner", "top_5", "top_10", "top_20", "make_cut"):
        _require_probability_map(
            outrights.get(market), label=f"outrights.{market}", field=field
        )
    outrights_nodh = fairs.get("outrights_nodh") or {}
    for market in ("top_5", "top_10", "top_20"):
        _require_probability_map(
            outrights_nodh.get(market),
            label=f"outrights_nodh.{market}",
            field=field,
        )

    matchups = fairs.get("matchups")
    if not isinstance(matchups, list) or not matchups:
        raise OddsScreenContractError("committed sim fairs have empty tournament matchups")
    matchup_pairs = set()
    for row in matchups:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise OddsScreenContractError("committed tournament matchup row is malformed")
        p1 = str(row[0] or "").strip().lower()
        p2 = str(row[1] or "").strip().lower()
        try:
            probability = float(row[2])
        except (TypeError, ValueError, OverflowError) as exc:
            raise OddsScreenContractError(
                "committed tournament matchup probability is malformed"
            ) from exc
        if (
            p1 not in field
            or p2 not in field
            or p1 == p2
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise OddsScreenContractError(
                "committed tournament matchup is outside the sealed field"
            )
        pair = tuple(sorted((p1, p2)))
        if pair in matchup_pairs:
            raise OddsScreenContractError(
                "committed tournament matchups contain a duplicate canonical pair"
            )
        matchup_pairs.add(pair)
    expected_pairs = len(field) * (len(field) - 1) // 2
    if len(matchup_pairs) != expected_pairs:
        raise OddsScreenContractError(
            "committed tournament matchups do not cover the sealed field"
        )

    round_scores = fairs.get("round_scores")
    if not isinstance(round_scores, dict) or not round_scores:
        raise OddsScreenContractError("committed sim fairs have empty round score PMFs")
    if set(str(name).strip().lower() for name in round_scores) != field:
        raise OddsScreenContractError(
            "committed round score PMFs do not cover the sealed field"
        )
    for player, pmf in round_scores.items():
        if not isinstance(pmf, dict) or not pmf:
            raise OddsScreenContractError(f"round score PMF is empty for {player}")
        total = 0.0
        for raw_score, raw_probability in pmf.items():
            try:
                score = float(raw_score)
                probability = float(raw_probability)
            except (TypeError, ValueError, OverflowError) as exc:
                raise OddsScreenContractError(
                    f"round score PMF is malformed for {player}"
                ) from exc
            if (
                not math.isfinite(score)
                or not math.isfinite(probability)
                or probability < 0.0
            ):
                raise OddsScreenContractError(
                    f"round score PMF is malformed for {player}"
                )
            total += probability
        if not math.isclose(total, 1.0, abs_tol=1e-5):
            raise OddsScreenContractError(
                f"round score PMF does not sum to one for {player}: {total:.8f}"
            )


def _require_release_health(
    payload: dict,
    *,
    kind: str,
    simulation_id: str,
    identity: dict[str, str],
    label: str,
) -> dict:
    from sim_health_gate import seal_manifest

    if (
        payload.get("kind") != kind
        or seal_manifest(payload).get("manifest_sha256")
        != payload.get("manifest_sha256")
    ):
        raise OddsScreenContractError(f"{label} health manifest is invalid")
    _require_recent_timestamp(payload.get("generated_at"), label=f"{label} health")
    source = payload.get("simulation_manifest") or {}
    event = source.get("event") or {}
    if (
        str(source.get("manifest_sha256")) != simulation_id
        or seal_manifest(source).get("manifest_sha256")
        != source.get("manifest_sha256")
        or (source.get("approval") or {}).get("status") != "approved"
        or not (source.get("checks") or {}).get("passed")
        or str(event.get("event_id")) != identity["event_id"]
        or str(event.get("tourney")) != identity["tourney"]
        or str(event.get("round")) != identity["round"]
    ):
        raise OddsScreenContractError(
            f"{label} health manifest references a different or unapproved simulation"
        )
    source_times = source.get("source") or {}
    _require_recent_timestamp(
        source_times.get("generated_at"), label=f"{label} simulation"
    )
    _require_recent_timestamp(
        source_times.get("root_generated_at") or source_times.get("generated_at"),
        label=f"{label} root simulation",
    )
    return source


def _parquet_metadata(path: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    return {
        key.decode("utf-8", errors="replace"): value.decode(
            "utf-8", errors="replace"
        )
        for key, value in metadata.items()
    }


def _require_parquet_identity(
    path: Path,
    *,
    identity: dict[str, str],
    simulation_id: str | None = None,
    live_id: str | None = None,
) -> dict[str, str]:
    metadata = _parquet_metadata(path)
    for key in ("event_id", "tourney"):
        if metadata.get(key) != identity[key]:
            raise OddsScreenContractError(f"{path.name} does not bind release {key}")
    if simulation_id is not None and metadata.get(
        "simulation_manifest_sha256"
    ) != simulation_id:
        raise OddsScreenContractError(
            f"{path.name} does not bind the release simulation manifest"
        )
    if live_id is not None and metadata.get(
        "live_tournament_manifest_sha256"
    ) != live_id:
        raise OddsScreenContractError(
            f"{path.name} does not bind the live tournament manifest"
        )
    return metadata


def _validate_release_assets(manifest: dict) -> None:
    expected = {
        "tournament_samples_full",
        "tournament_made_cut_full",
        "matchup_scores_live",
    }
    assets = manifest.get("release_assets") or {}
    if set(assets) != expected:
        raise OddsScreenContractError("sim release manifest asset set is incomplete")
    generation = str(manifest.get("generation") or "")
    for label, binding in assets.items():
        try:
            name = str(binding["name"])
            digest = str(binding["sha256"])
            size = int(binding["size"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise OddsScreenContractError(
                f"sim release asset binding is malformed: {label}"
            ) from exc
        if (
            PurePosixPath(name).name != name
            or name != f"{label}.{generation}.{digest[:16]}.parquet"
            or len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
            or size <= 0
        ):
            raise OddsScreenContractError(
                f"sim release asset binding is incomplete: {label}"
            )


def _require_threeball_group_binding(
    *,
    threeball_meta: dict,
    tee_source: dict,
    status: str,
    active_field: set[str],
    priced_groups: set[tuple[str, str, str]],
) -> None:
    """Bind priced threesomes to the exact event-scoped tee-time contract."""
    if threeball_meta.get("tee_group_source") != tee_source:
        raise OddsScreenContractError(
            "round 3-ball metadata and contract tee-group sources disagree"
        )
    if tee_source.get("status") != status:
        raise OddsScreenContractError(
            "round 3-ball status disagrees with its tee-group source"
        )
    raw_groups = tee_source.get("groups")
    if not isinstance(raw_groups, list):
        raise OddsScreenContractError("round 3-ball tee-group source has no groups")
    source_groups = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, (list, tuple)) or len(raw_group) != 3:
            raise OddsScreenContractError("round 3-ball source group is malformed")
        canonical = tuple(
            sorted(str(player or "").strip().lower() for player in raw_group)
        )
        if (
            len(set(canonical)) != 3
            or not set(canonical).issubset(active_field)
            or canonical in source_groups
        ):
            raise OddsScreenContractError(
                "round 3-ball tee-group source is outside the sealed field"
            )
        source_groups.add(canonical)
    if source_groups != priced_groups:
        raise OddsScreenContractError(
            "round 3-ball priced groups do not match the tee-group source"
        )


def _validate_non_h2h_release_artifacts(
    project_root: Path,
    *,
    manifest: dict,
    fairs: dict,
    identity: dict[str, str],
) -> dict:
    """Validate every non-H2H artifact the strict generation declares as core."""
    import numpy as np

    from sim_health_gate import file_sha256

    _validate_release_assets(manifest)
    simulation_id = str(manifest["simulation_manifest_sha256"])
    live_id = str(manifest["live_tournament_manifest_sha256"])
    sim_round = int(identity["round"])
    active_field = {
        str(player).strip().lower() for player in fairs["round_scores"]
    }
    tournament_field = {str(player).strip().lower() for player in fairs["field"]}

    round_samples_path = project_root / "round_samples.parquet"
    round_samples = pd.read_parquet(round_samples_path)
    round_metadata = _require_parquet_identity(
        round_samples_path, identity=identity
    )
    if (
        round_metadata.get("round") != identity["round"]
        or not round_metadata.get("sim_run_at")
    ):
        raise OddsScreenContractError("round_samples.parquet has the wrong round")
    round_players = {
        str(player).strip().lower() for player in round_samples.index
    }
    if (
        round_samples.empty
        or round_players != active_field
        or len(round_samples.index) != len(round_players)
        or len(round_samples.columns) != len(set(round_samples.columns))
        or not np.isfinite(round_samples.to_numpy(dtype=float)).all()
    ):
        raise OddsScreenContractError(
            "round_samples.parquet does not cover the sealed active field"
        )

    tournament_path = project_root / "tournament_samples.parquet"
    made_cut_path = project_root / "tournament_made_cut.parquet"
    tournament = pd.read_parquet(tournament_path)
    made_cut = pd.read_parquet(made_cut_path)
    tournament_meta = _require_parquet_identity(
        tournament_path,
        identity=identity,
        simulation_id=simulation_id,
        live_id=live_id,
    )
    made_cut_meta = _require_parquet_identity(
        made_cut_path,
        identity=identity,
        simulation_id=simulation_id,
        live_id=live_id,
    )
    if tournament_meta.get("source") != "final_scores_live" or made_cut_meta.get(
        "source"
    ) != "made_cut_live":
        raise OddsScreenContractError(
            "tournament git tapes do not identify their live sources"
        )
    if (
        not tournament_meta.get("sim_run_at")
        or tournament_meta.get("sim_run_at") != made_cut_meta.get("sim_run_at")
    ):
        raise OddsScreenContractError(
            "tournament git tapes do not bind one simulation timestamp"
        )
    tournament_players = {
        str(player).strip().lower() for player in tournament.index
    }
    if (
        tournament.empty
        or tournament_players != tournament_field
        or len(tournament.index) != len(tournament_players)
        or len(tournament.columns) != len(set(tournament.columns))
        or list(tournament.index) != list(made_cut.index)
        or list(tournament.columns) != list(made_cut.columns)
        or not np.isfinite(tournament.to_numpy(dtype=float)).all()
        or not np.isin(made_cut.to_numpy(), (0, 1)).all()
    ):
        raise OddsScreenContractError(
            "tournament sample and made-cut tapes are not one aligned live joint"
        )

    live_health_path = project_root / f"tournament_live_{identity['tourney']}_health.json"
    try:
        live_health = json.loads(live_health_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OddsScreenContractError(
            "live tournament health manifest is unreadable"
        ) from exc
    _require_release_health(
        live_health,
        kind="live_tournament_tape",
        simulation_id=simulation_id,
        identity=identity,
        label="live tournament",
    )
    if str(live_health.get("manifest_sha256")) != live_id:
        raise OddsScreenContractError(
            "live tournament health manifest has the wrong release identity"
        )
    expected_live_files = {
        "final_scores",
        "player_names",
        "made_cut",
        "finish_probs",
        "finish_probs_event",
    }
    live_files = live_health.get("files") or {}
    if set(live_files) != expected_live_files:
        raise OddsScreenContractError(
            "live tournament health manifest file set is incomplete"
        )
    expected_live_paths = {
        "final_scores": f"final_scores_live_{identity['tourney']}.npy",
        "player_names": f"player_names_live_{identity['tourney']}.json",
        "made_cut": f"made_cut_live_{identity['tourney']}.npy",
        "finish_probs": "simulated_probs_live.csv",
        "finish_probs_event": f"top_finish_probs_live_{identity['tourney']}.csv",
    }
    for label, binding in live_files.items():
        digest = str((binding or {}).get("sha256") or "")
        path_name = str((binding or {}).get("path") or "")
        if (
            PurePosixPath(path_name).name != path_name
            or path_name != expected_live_paths[label]
            or len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
        ):
            raise OddsScreenContractError(
                f"live tournament health binding is incomplete: {label}"
            )

    threeball_path = project_root / f"round_3ball_r{sim_round}.parquet"
    threeball_meta_path = project_root / f"round_3ball_r{sim_round}_meta.json"
    threeball_contract_path = (
        project_root / f"round_3ball_r{sim_round}_contract.json"
    )
    try:
        threeball = pd.read_parquet(threeball_path)
        threeball_meta = json.loads(threeball_meta_path.read_text(encoding="utf-8"))
        threeball_contract = json.loads(
            threeball_contract_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise OddsScreenContractError("round 3-ball package is unreadable") from exc
    for key, expected_value in identity.items():
        if str(threeball_meta.get(key)) != expected_value:
            raise OddsScreenContractError(f"round 3-ball metadata does not bind {key}")
    _require_release_health(
        threeball_contract,
        kind="published_round_3ball",
        simulation_id=simulation_id,
        identity=identity,
        label="round 3-ball",
    )
    contract_files = threeball_contract.get("files") or {}
    for label, path in (
        ("threeball_parquet", threeball_path),
        ("threeball_meta", threeball_meta_path),
    ):
        binding = contract_files.get(label) or {}
        if binding.get("path") != path.name or binding.get("sha256") != file_sha256(path):
            raise OddsScreenContractError(
                f"round 3-ball contract does not bind {path.name}"
            )
    extra = threeball_contract.get("extra") or {}
    status = str(threeball_meta.get("status") or "")
    if (
        str(extra.get("event_id")) != identity["event_id"]
        or str(extra.get("round")) != identity["round"]
        or extra.get("status") != status
        or status not in ("groups", "no_groups_offered")
    ):
        raise OddsScreenContractError("round 3-ball contract identity is invalid")
    tee_source = extra.get("tee_group_source") or {}
    if (
        str(tee_source.get("round")) != identity["round"]
        or str(tee_source.get("requested_event_id")) != identity["event_id"]
        or tee_source.get("event_identity_verified") is not True
        or float(tee_source.get("simulation_field_overlap") or 0.0) != 1.0
        or float(tee_source.get("simulation_tee_time_coverage") or 0.0) != 1.0
        or (
            tee_source.get("source_event_id") not in (None, "")
            and str(tee_source.get("source_event_id")) != identity["event_id"]
        )
    ):
        raise OddsScreenContractError(
            "round 3-ball contract has no verified event/round tee-group source"
        )
    required_threeball = {
        "player_a",
        "player_b",
        "player_c",
        "p_a",
        "p_b",
        "p_c",
    }
    if not required_threeball.issubset(threeball.columns):
        raise OddsScreenContractError("round 3-ball table is malformed")
    seen_groups = set()
    if status == "no_groups_offered":
        if (
            not threeball.empty
            or int(extra.get("num_groups") or 0) != 0
            or int(threeball_meta.get("num_groups") or 0) != 0
        ):
            raise OddsScreenContractError(
                "round 3-ball no-groups contract contains priced rows"
            )
    elif threeball.empty:
        raise OddsScreenContractError("round 3-ball groups contract has no rows")
    else:
        grouped_players = set()
        for row in threeball[list(required_threeball)].itertuples(index=False):
            players = (str(row.player_a), str(row.player_b), str(row.player_c))
            probabilities = (float(row.p_a), float(row.p_b), float(row.p_c))
            canonical = tuple(sorted(player.strip().lower() for player in players))
            if (
                len(set(canonical)) != 3
                or not set(canonical).issubset(active_field)
                or canonical in seen_groups
                or not set(canonical).isdisjoint(grouped_players)
                or not all(
                    math.isfinite(probability) and 0.0 <= probability <= 1.0
                    for probability in probabilities
                )
                or not math.isclose(sum(probabilities), 1.0, abs_tol=2e-4)
            ):
                raise OddsScreenContractError(
                    "round 3-ball table has invalid group probabilities"
                )
            seen_groups.add(canonical)
            grouped_players.update(canonical)
        if (
            len(seen_groups) != int(extra.get("num_groups") or -1)
            or len(seen_groups) != int(threeball_meta.get("num_groups") or -1)
        ):
            raise OddsScreenContractError(
                "round 3-ball group count does not match its contract"
            )
    _require_threeball_group_binding(
        threeball_meta=threeball_meta,
        tee_source=tee_source,
        status=status,
        active_field=active_field,
        priced_groups=seen_groups,
    )
    return {"tournament_samples": tournament}


def _load_committed_release_contract(
    *,
    expected_tourney: str,
    expected_event_id,
    expected_round: int,
    project_root: Path = PROJECT_ROOT,
    source_git_sha: str | None = None,
    verify_git: bool = True,
) -> dict:
    """Load one exact, manifest-bound model generation from the checked-out commit."""
    from sim_health_gate import file_sha256, seal_manifest

    project_root = Path(project_root)
    manifest_path = project_root / RELEASE_MANIFEST_PATH.name
    fairs_path = project_root / SIM_FAIRS_PATH.name
    if not manifest_path.is_file() or not fairs_path.is_file():
        raise OddsScreenContractError(
            "odds screen requires committed sim_release_manifest.json and sim_fairs.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fairs = json.loads(fairs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OddsScreenContractError(f"committed sim release JSON is unreadable: {exc}") from exc

    if manifest.get("schema_version") != RELEASE_MANIFEST_SCHEMA:
        raise OddsScreenContractError("sim release manifest schema is unsupported")
    if seal_manifest(manifest).get("manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        raise OddsScreenContractError("sim release manifest content hash is invalid")
    _require_recent_timestamp(
        manifest.get("generated_at"), label="sim release manifest"
    )

    identity = {
        "tourney": str(expected_tourney or "").strip(),
        "event_id": str(expected_event_id),
        "round": str(int(expected_round)),
    }
    for key, expected in identity.items():
        if str(manifest.get(key)) != expected:
            raise OddsScreenContractError(
                f"committed sim release {key}={manifest.get(key)!r}, expected {expected!r}"
            )
    if not manifest.get("generation") or not manifest.get("manifest_sha256"):
        raise OddsScreenContractError("sim release manifest has no generation identity")
    simulation_id = str(manifest.get("simulation_manifest_sha256") or "")
    live_id = str(manifest.get("live_tournament_manifest_sha256") or "")
    for label, digest in (
        ("simulation manifest", simulation_id),
        ("live tournament manifest", live_id),
    ):
        if len(digest) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in digest
        ):
            raise OddsScreenContractError(f"sim release {label} identity is invalid")
    expected_generation = (
        f"event-{identity['event_id']}-r{identity['round']}-"
        f"{simulation_id[:12]}-{live_id[:12]}"
    )
    if manifest.get("generation") != expected_generation:
        raise OddsScreenContractError(
            "sim release generation does not bind its event, round, and manifests"
        )

    git_files = manifest.get("git_files") or {}
    required = _release_required_files(identity["tourney"], int(identity["round"]))
    missing = sorted(required - set(git_files))
    if missing:
        raise OddsScreenContractError(
            "sim release manifest omits required odds-screen artifacts: "
            + ", ".join(missing)
        )

    if verify_git:
        source_git_sha = source_git_sha or _git_head_sha(project_root)
        committed_manifest = _git_blob_bytes(
            project_root, source_git_sha, RELEASE_MANIFEST_PATH.name
        )
        _require_git_equivalent_worktree_file(
            project_root, source_git_sha, RELEASE_MANIFEST_PATH.name
        )
        try:
            committed_manifest_json = json.loads(committed_manifest.decode("utf-8"))
        except Exception as exc:
            raise OddsScreenContractError(
                "committed sim release manifest is unreadable"
            ) from exc
        if committed_manifest_json != manifest:
            raise OddsScreenContractError(
                "sim release manifest differs semantically from the checked-out Git commit"
            )
    else:
        source_git_sha = source_git_sha or "test-source"

    for relative, binding in git_files.items():
        path = _safe_release_path(project_root, relative)
        if not path.is_file():
            raise OddsScreenContractError(f"declared sim release file is missing: {relative}")
        try:
            expected_size = int(binding.get("size"))
            expected_hash = str(binding.get("sha256") or "")
        except (AttributeError, TypeError, ValueError) as exc:
            raise OddsScreenContractError(
                f"declared sim release binding is malformed: {relative}"
            ) from exc
        if (
            expected_size <= 0
            or len(expected_hash) != 64
            or any(ch not in "0123456789abcdefABCDEF" for ch in expected_hash)
        ):
            raise OddsScreenContractError(
                f"declared sim release binding is incomplete: {relative}"
            )
        if verify_git:
            committed_blob = _git_blob_bytes(project_root, source_git_sha, relative)
            if (
                len(committed_blob) != expected_size
                or hashlib.sha256(committed_blob).hexdigest() != expected_hash.lower()
            ):
                raise OddsScreenContractError(
                    f"committed Git blob does not match release binding: {relative}"
                )
            _require_git_equivalent_worktree_file(
                project_root, source_git_sha, relative
            )
        elif path.stat().st_size != expected_size or file_sha256(path) != expected_hash:
            raise OddsScreenContractError(
                f"declared sim release file changed after sealing: {relative}"
            )

    core_bindings = {
        "release_generation": manifest.get("generation"),
        "simulation_manifest_sha256": manifest.get("simulation_manifest_sha256"),
        "live_tournament_manifest_sha256": manifest.get(
            "live_tournament_manifest_sha256"
        ),
        "event_id": manifest.get("event_id"),
        "tourney": manifest.get("tourney"),
        "round": manifest.get("round"),
        "generated_at": manifest.get("generated_at"),
    }
    for fairs_key, expected in core_bindings.items():
        if str(fairs.get(fairs_key)) != str(expected):
            raise OddsScreenContractError(
                f"sim_fairs.json does not bind release field {fairs_key}"
            )
    if fairs.get("outrights_source") != "live" or fairs.get(
        "matchups_source"
    ) != "final_scores_live":
        raise OddsScreenContractError(
            "odds screen requires live-conditioned outright and tournament matchup fairs"
        )
    _validate_sim_fairs_semantics(fairs)

    h2h_path = project_root / f"round_h2h_r{expected_round}.parquet"
    h2h_meta_path = project_root / f"round_h2h_r{expected_round}_meta.json"
    h2h_health_path = project_root / f"round_h2h_r{expected_round}_health.json"
    try:
        h2h = pd.read_parquet(h2h_path)
        h2h_meta = json.loads(h2h_meta_path.read_text(encoding="utf-8"))
        h2h_health = json.loads(h2h_health_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OddsScreenContractError(f"committed round H2H package is unreadable: {exc}") from exc
    required_h2h = {"player_a", "player_b", "p_a_lt_b", "p_tie"}
    if h2h.empty or not required_h2h.issubset(h2h.columns):
        raise OddsScreenContractError("committed round H2H table is empty or malformed")
    for key, expected in identity.items():
        if str(h2h_meta.get(key)) != expected:
            raise OddsScreenContractError(f"round H2H metadata does not bind {key}")
    if str(h2h_meta.get("source_manifest_sha256")) != str(
        manifest.get("simulation_manifest_sha256")
    ):
        raise OddsScreenContractError(
            "round H2H metadata references a different simulation manifest"
        )
    if (
        h2h_health.get("kind") != "published_round_h2h"
        or seal_manifest(h2h_health).get("manifest_sha256")
        != h2h_health.get("manifest_sha256")
    ):
        raise OddsScreenContractError("round H2H health manifest is invalid")
    health_files = h2h_health.get("files") or {}
    for label, path in (
        ("h2h_parquet", h2h_path),
        ("h2h_meta", h2h_meta_path),
    ):
        binding = health_files.get(label) or {}
        if (
            binding.get("path") != path.name
            or binding.get("sha256") != file_sha256(path)
        ):
            raise OddsScreenContractError(
                f"round H2H health manifest does not bind {path.name}"
            )
    _require_release_health(
        h2h_health,
        kind="published_round_h2h",
        simulation_id=simulation_id,
        identity=identity,
        label="round H2H",
    )

    seen_pairs = set()
    h2h_players = set()
    for row in h2h[list(required_h2h)].itertuples(index=False):
        player_a = str(row.player_a or "").strip().lower()
        player_b = str(row.player_b or "").strip().lower()
        try:
            p_a_lt_b = float(row.p_a_lt_b)
            p_tie = float(row.p_tie)
        except (TypeError, ValueError, OverflowError) as exc:
            raise OddsScreenContractError(
                "committed round H2H table has non-numeric probabilities"
            ) from exc
        pair = (player_a, player_b)
        if (
            not player_a
            or player_a >= player_b
            or pair in seen_pairs
            or not math.isfinite(p_a_lt_b)
            or not math.isfinite(p_tie)
            or not 0.0 <= p_a_lt_b <= 1.0
            or not 0.0 <= p_tie <= 1.0
            or p_a_lt_b + p_tie > 1.00002
        ):
            raise OddsScreenContractError(
                "committed round H2H table has an invalid or duplicate pair"
            )
        seen_pairs.add(pair)
        h2h_players.update(pair)
    expected_players = int(h2h_meta.get("num_players") or 0)
    expected_pairs = expected_players * (expected_players - 1) // 2
    if (
        expected_players < 2
        or len(h2h_players) != expected_players
        or len(seen_pairs) != expected_pairs
        or h2h_players != {
            str(player).strip().lower() for player in fairs["round_scores"]
        }
    ):
        raise OddsScreenContractError(
            "committed round H2H table does not cover the sealed round-score field"
        )

    non_h2h = _validate_non_h2h_release_artifacts(
        project_root,
        manifest=manifest,
        fairs=fairs,
        identity=identity,
    )

    return {
        "manifest": manifest,
        "fairs": fairs,
        "round_h2h": h2h,
        "round_h2h_meta": h2h_meta,
        **non_h2h,
        "source_git_sha": source_git_sha,
    }


# ─── helpers ────────────────────────────────────────────────────────────────

def _get_r2_client():
    """Return a boto3 S3 client configured for Cloudflare R2."""
    import boto3
    account_id = os.environ["CF_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _upload_json(client, key: str, data: dict):
    """Upload a JSON object to R2."""
    body = json.dumps(data, default=str, allow_nan=False)
    client.put_object(
        Bucket=R2_BUCKET,
        Key=f"{R2_PREFIX}/{key}",
        Body=body.encode(),
        ContentType="application/json",
    )
    logger.info(f"Uploaded {key} ({len(body)} bytes)")


def _write_payload_files(payloads: dict, output_dir: Path) -> list[Path]:
    """Atomically write publish-ready JSON files for an external uploader.

    GitHub Actions uses this path with Wrangler's scoped Cloudflare API token.
    Keeping generation separate from transport means a failed upload cannot be
    mistaken for a successful odds-screen refresh.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, data in payloads.items():
        target = output_dir / key
        fd, temp_name = tempfile.mkstemp(
            dir=str(output_dir), prefix=f".{key}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    data,
                    handle,
                    default=str,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        written.append(target)
        logger.info(f"Wrote {target} ({target.stat().st_size} bytes)")
    return written


def _json_bytes(data: dict) -> bytes:
    return (
        json.dumps(
            data,
            default=str,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _validate_odds_screen_payloads(payloads: dict) -> None:
    """Fail before staging if a generation is complete only at the byte level."""
    required_files = {
        "round_matchups.json",
        "tournament_matchups.json",
        "score_lines.json",
        "outrights.json",
        "meta.json",
    }
    missing = sorted(required_files - set(payloads))
    if missing:
        raise OddsScreenContractError(
            "odds-screen generation omits required payloads: " + ", ".join(missing)
        )
    meta = payloads.get("meta.json")
    if not isinstance(meta, dict):
        raise OddsScreenContractError("odds-screen meta payload is malformed")
    for key in PROVENANCE_KEYS:
        if meta.get(key) in (None, ""):
            raise OddsScreenContractError(f"odds-screen generation is missing {key}")
    for key, length in (
        ("release_manifest_sha256", 64),
        ("simulation_manifest_sha256", 64),
        ("live_tournament_manifest_sha256", 64),
        ("source_git_sha", 40),
    ):
        value = str(meta.get(key) or "")
        if len(value) != length or any(
            char not in "0123456789abcdefABCDEF" for char in value
        ):
            raise OddsScreenContractError(
                f"odds-screen generation has invalid provenance hash {key}"
            )
    try:
        if int(meta["round"]) not in (1, 2, 3, 4):
            raise ValueError
    except (TypeError, ValueError):
        raise OddsScreenContractError("odds-screen generation has an invalid round")
    for name, payload in payloads.items():
        if not isinstance(payload, dict):
            raise OddsScreenContractError(f"odds-screen payload is malformed: {name}")
        for key in PROVENANCE_KEYS:
            if str(payload.get(key)) != str(meta.get(key)):
                raise OddsScreenContractError(
                    f"odds-screen payload {name} is not release-aligned on {key}"
                )

    round_matchups = payloads["round_matchups.json"].get("matchups")
    if not isinstance(round_matchups, list) or not round_matchups:
        raise OddsScreenContractError("odds-screen round matchups have no model rows")
    for row in round_matchups:
        fair = row.get("fair") if isinstance(row, dict) else None
        if not isinstance(fair, dict) or fair.get("p1_prob") is None or fair.get(
            "p2_prob"
        ) is None:
            raise OddsScreenContractError(
                "odds-screen round matchup row has no committed model fair"
            )
        _validate_two_way_fair(
            fair,
            "round matchup",
            tie_is_loss=str(row.get("ties") or "").strip().lower()
            == "separate bet offered",
        )
    if not _has_valid_book_quote(round_matchups, ("p1", "p2")):
        raise OddsScreenContractError(
            "odds-screen round matchups have no usable current book quote; "
            "retaining prior generation"
        )

    tournament_matchups = payloads["tournament_matchups.json"].get("matchups")
    if not isinstance(tournament_matchups, list) or not tournament_matchups:
        raise OddsScreenContractError(
            "odds-screen tournament matchups have no model rows"
        )
    for row in tournament_matchups:
        fair = row.get("fair") if isinstance(row, dict) else None
        if not isinstance(fair, dict) or fair.get("p1_prob") is None or fair.get(
            "p2_prob"
        ) is None:
            raise OddsScreenContractError(
                "odds-screen tournament matchup row has no committed model fair"
            )
        _validate_two_way_fair(
            fair,
            "tournament matchup",
            tie_is_loss=str(row.get("ties") or "").strip().lower()
            == "separate bet offered",
        )
    if not _has_valid_book_quote(tournament_matchups, ("p1", "p2")):
        raise OddsScreenContractError(
            "odds-screen tournament matchups have no usable current book quote; "
            "retaining prior generation"
        )

    score_lines = payloads["score_lines.json"].get("lines")
    if not isinstance(score_lines, list) or not score_lines:
        raise OddsScreenContractError("odds-screen score lines have no model rows")
    for row in score_lines:
        if not isinstance(row, dict) or not _is_finite_number(row.get("pred")):
            raise OddsScreenContractError(
                "odds-screen score row has no committed model prediction"
            )
        if row.get("books") and not row.get("fair"):
            raise OddsScreenContractError(
                "odds-screen offered score row has book odds but no model fair"
            )
        if row.get("books"):
            if not _is_finite_number(row.get("line")):
                raise OddsScreenContractError(
                    "odds-screen offered score row has no finite market line"
                )
            if not _is_half_stroke_line(row.get("line")):
                raise OddsScreenContractError(
                    "odds-screen score lines must be half-strokes so no push mass is hidden"
                )
            fair = row["fair"]
            _validate_two_way_fair(
                {
                    "p1_prob": fair.get("over_prob"),
                    "p2_prob": fair.get("under_prob"),
                },
                "score line",
            )
    if not _has_valid_book_quote(score_lines, ("over", "under")):
        raise OddsScreenContractError(
            "odds-screen score lines have no usable current book quote; "
            "retaining prior generation"
        )

    markets = payloads["outrights.json"].get("markets")
    if not isinstance(markets, dict):
        raise OddsScreenContractError("odds-screen outright markets are malformed")
    for market in ("winner", "top_5", "top_10", "top_20"):
        rows = markets.get(market)
        if not isinstance(rows, list) or not rows:
            raise OddsScreenContractError(
                f"odds-screen outright market has no model rows: {market}"
            )
        for row in rows:
            try:
                probability = float(row.get("sim_prob"))
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise OddsScreenContractError(
                    f"odds-screen outright market contains a books-only row: {market}"
                ) from exc
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise OddsScreenContractError(
                    f"odds-screen outright market has invalid model probability: {market}"
                )
        if not _has_valid_book_quote(rows, ("yes",)):
            raise OddsScreenContractError(
                f"odds-screen outright market has no usable current book quote: {market}; "
                "retaining prior generation"
            )


def _has_valid_book_quote(rows: list, sides: tuple[str, ...]) -> bool:
    """Whether at least one row has a complete, finite American-odds quote."""
    for row in rows:
        for quote in (row.get("books") or {}).values():
            if not isinstance(quote, dict):
                continue
            if all(_parse_american_odds(quote.get(side)) is not None for side in sides):
                return True
    return False


def _parse_american_odds(value) -> int | None:
    """Return one usable integer American quote, rejecting zero/phantom prices."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or (-100.0 < numeric < 100.0):
        return None
    return int(round(numeric))


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _is_half_stroke_line(value) -> bool:
    try:
        doubled = float(value) * 2.0
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(doubled) or not math.isclose(
        doubled, round(doubled), abs_tol=1e-9
    ):
        return False
    return int(round(doubled)) % 2 == 1


def _validate_two_way_fair(
    fair: dict, label: str, *, tie_is_loss: bool = False
) -> None:
    try:
        p1 = float(fair["p1_prob"])
        p2 = float(fair["p2_prob"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise OddsScreenContractError(
            f"odds-screen {label} has invalid model probabilities"
        ) from exc
    total = p1 + p2
    if tie_is_loss:
        try:
            tie_probability = float(fair["tie_prob"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise OddsScreenContractError(
                f"odds-screen {label} has invalid tie probability"
            ) from exc
        total += tie_probability
        tie_valid = math.isfinite(tie_probability) and 0.0 <= tie_probability <= 1.0
    else:
        tie_valid = True
    if (
        not math.isfinite(p1)
        or not math.isfinite(p2)
        or not 0.0 <= p1 <= 1.0
        or not 0.0 <= p2 <= 1.0
        or not tie_valid
        or not math.isclose(total, 1.0, abs_tol=1e-4)
    ):
        raise OddsScreenContractError(
            f"odds-screen {label} has invalid model probabilities"
        )


def _build_atomic_payload_bundle(payloads: dict, generation: str | None = None):
    """Build immutable odds-screen objects plus the pointer readers fetch first.

    Market objects are never overwritten in place.  A failed upload leaves the
    old root ``meta.json`` pointer untouched, so readers keep one complete prior
    generation instead of combining new markets with stale metadata.
    """
    _validate_odds_screen_payloads(payloads)
    encoded = {name: _json_bytes(data) for name, data in sorted(payloads.items())}
    content_digest = hashlib.sha256(
        b"".join(name.encode("utf-8") + b"\0" + data for name, data in encoded.items())
    ).hexdigest()
    if generation is None:
        raw_time = str((payloads.get("meta.json") or {}).get("last_updated") or "")
        time_part = "".join(ch for ch in raw_time if ch.isdigit())[:14] or "unstamped"
        generation = f"{time_part}-{content_digest[:16]}"
    safe_generation_chars = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    )
    if not generation or any(ch not in safe_generation_chars for ch in generation):
        raise ValueError("odds-screen generation must contain only letters, digits, '-' or '_'")

    base = f"generations/{generation}"
    file_bindings = {
        name: {
            "key": f"{base}/{name}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        for name, data in encoded.items()
    }
    meta = dict(payloads.get("meta.json") or {})
    pointer = {
        **meta,
        "schema_version": ODDS_GENERATION_SCHEMA,
        "generation": generation,
        "content_sha256": content_digest,
        "files": file_bindings,
    }
    return {
        "generation": generation,
        "encoded": encoded,
        "pointer": pointer,
        "pointer_bytes": _json_bytes(pointer),
    }


def _write_atomic_payload_bundle(
    payloads: dict, output_dir: Path, generation: str | None = None
) -> list[Path]:
    """Write versioned objects first and root meta.json publication pointer last."""
    bundle = _build_atomic_payload_bundle(payloads, generation=generation)
    generation_dir = Path(output_dir) / "generations" / bundle["generation"]
    # Reuse the fsync + os.replace writer for each immutable generation object.
    decoded = {
        name: json.loads(data.decode("utf-8"))
        for name, data in bundle["encoded"].items()
    }
    written = _write_payload_files(decoded, generation_dir)
    written.extend(_write_payload_files({"meta.json": bundle["pointer"]}, output_dir))
    return written


def _atomic_upload_plan(output_dir: Path) -> list[tuple[str, Path]]:
    """Verify and enumerate every pointer binding, with meta.json strictly last."""
    output_dir = Path(output_dir)
    pointer_path = output_dir / "meta.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema_version") != ODDS_GENERATION_SCHEMA:
        raise RuntimeError("odds-screen pointer schema is missing or invalid")
    generation = str(pointer.get("generation") or "")
    if not generation or any(
        char
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in generation
    ):
        raise RuntimeError("odds-screen pointer generation is unsafe")
    prefix = f"generations/{generation}/"
    files = pointer.get("files") or {}
    if not files:
        raise RuntimeError("odds-screen pointer declares no generation files")
    plan = []
    declared_paths = set()
    decoded = {}
    digest_parts = []
    for name, binding in sorted(files.items()):
        key = str(binding.get("key") or "")
        if key != f"{prefix}{name}" or PurePosixPath(name).name != name:
            raise RuntimeError(f"unsafe odds-screen pointer binding: {name}")
        path = output_dir / Path(key)
        if not path.is_file():
            raise RuntimeError(f"declared odds-screen generation file is missing: {key}")
        data = path.read_bytes()
        if len(data) != int(binding.get("size") or -1):
            raise RuntimeError(f"odds-screen generation size mismatch: {key}")
        if hashlib.sha256(data).hexdigest() != binding.get("sha256"):
            raise RuntimeError(f"odds-screen generation hash mismatch: {key}")
        try:
            decoded[name] = json.loads(data.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"odds-screen generation JSON is invalid: {key}") from exc
        digest_parts.append(name.encode("utf-8") + b"\0" + data)
        declared_paths.add(path.resolve())
        plan.append((key, path))
    actual_paths = {
        path.resolve()
        for path in (output_dir / "generations" / generation).glob("*.json")
    }
    if actual_paths != declared_paths:
        raise RuntimeError("odds-screen generation contains undeclared or omitted JSON files")
    content_digest = hashlib.sha256(b"".join(digest_parts)).hexdigest()
    if pointer.get("content_sha256") != content_digest:
        raise RuntimeError("odds-screen generation content digest mismatch")
    _validate_odds_screen_payloads(decoded)
    generation_meta = decoded["meta.json"]
    if any(pointer.get(key) != value for key, value in generation_meta.items()):
        raise RuntimeError("odds-screen pointer metadata does not match its generation")
    plan.append(("meta.json", pointer_path))
    return plan


def _upload_atomic_payload_bundle(client, payloads: dict, generation: str | None = None):
    """Stage one complete R2 generation, then atomically advance root meta.json."""
    bundle = _build_atomic_payload_bundle(payloads, generation=generation)
    for name, data in bundle["encoded"].items():
        key = f"generations/{bundle['generation']}/{name}"
        client.put_object(
            Bucket=R2_BUCKET,
            Key=f"{R2_PREFIX}/{key}",
            Body=data,
            ContentType="application/json",
        )
        logger.info(f"Uploaded {key} ({len(data)} bytes)")
    # Consumer-visible commit point. Never execute this if any staged write fails.
    client.put_object(
        Bucket=R2_BUCKET,
        Key=f"{R2_PREFIX}/meta.json",
        Body=bundle["pointer_bytes"],
        ContentType="application/json",
    )
    logger.info(
        f"Published odds-screen generation {bundle['generation']} via meta.json"
    )
    return bundle["pointer"]


def _fetch_scraped_guarded(
    market: str, *, round=None, event_id=None
) -> "dict | None":
    """Fetch a scraped market JSON via odds_loader (GitHub->local) and apply the
    SAME guards the sim pricer uses: freshness (MAX_AGE_HOURS), event_id scope,
    file-level round, and per-row round filtering. Returns a safe dict or None.

    `market` is the bare market name ('round_matchups', 'tournament_matchups',
    'round_scores') — odds_loader._fetch_scraped_json appends '_latest.json'. This
    replaces push_odds_screen's old un-guarded fetch so the odds screen can never
    price a stale / wrong-event / wrong-round file the sim itself would reject."""
    from odds_loader import (
        MAX_AGE_HOURS,
        _fetch_scraped_json as _ol_fetch,
        guard_scraped_data,
    )

    data = _ol_fetch(market)
    target = _event_id_token(event_id)
    event_ids = {target} if target is not None else None
    guarded = guard_scraped_data(
        data, market, round=round, event_ids=event_ids
    )
    if not guarded:
        return None

    timestamp = str(guarded.get("last_updated") or "").strip()
    try:
        updated = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S UTC").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        logger.warning(
            "Scraped %s has no parseable freshness timestamp; strict activation rejects it",
            market,
        )
        return None
    age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600.0
    if age_hours > MAX_AGE_HOURS or age_hours < -0.25:
        logger.warning(
            "Scraped %s timestamp is outside the strict freshness window (%.2fh)",
            market,
            age_hours,
        )
        return None

    if target is None:
        return None
    file_event = _event_id_token(guarded.get("event_id"))
    rows_key = "match_list" if "match_list" in guarded else (
        "lines" if "lines" in guarded else None
    )
    if rows_key is None:
        logger.warning("Scraped %s has no attributable quote rows", market)
        return None
    rows = guarded.get(rows_key) or []
    tagged = [row for row in rows if _event_id_token(row.get("event_id")) is not None]
    if tagged:
        rows = [row for row in rows if _event_id_token(row.get("event_id")) == target]
    elif file_event != target:
        logger.warning(
            "Scraped %s has no exact event evidence for event %s", market, target
        )
        return None
    if not rows:
        return None
    return {**guarded, rows_key: rows}


def _event_id_token(value) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError, OverflowError):
        pass
    return text or None


def _load_name_replacements() -> dict:
    try:
        from sim_inputs import name_replacements
        return name_replacements
    except ImportError:
        return {}


def _norm(name: str, replacements: dict) -> str:
    """Lowercase + apply name_replacements."""
    name = name.strip().lower()
    return replacements.get(name, name)


def _american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds >= 100:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def _prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return round(-prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)


def _edge_pct(fair_prob: float, market_odds: float) -> float:
    """Edge % = fair_prob - implied_prob (positive = value)."""
    implied = _american_to_prob(market_odds)
    return round((fair_prob - implied) * 100, 1)


# ─── market builders ────────────────────────────────────────────────────────

def _oriented_h2h_probability(lookup: dict, p1: str, p2: str):
    """Return push and tie-loss probabilities in the offered player order."""
    a, b = (p1, p2) if p1 <= p2 else (p2, p1)
    hit = lookup.get((a, b))
    if hit is None:
        return None
    p_a_lt_b, p_tie = hit
    # The producer stores independently rounded float32 components. At an exact
    # boundary they may sum a few millionths above one; normalize that approved
    # rounding envelope before deriving the opposite side.
    p_a_lt_b = max(0.0, p_a_lt_b)
    p_b_lt_a = max(0.0, 1.0 - p_a_lt_b - p_tie)
    p_tie = max(0.0, p_tie)
    total = p_a_lt_b + p_b_lt_a + p_tie
    if total <= 0.0:
        return None
    p_a_lt_b /= total
    p_b_lt_a /= total
    p_tie /= total
    p1_lt = p_a_lt_b if p1 <= p2 else p_b_lt_a
    p2_lt = p_b_lt_a if p1 <= p2 else p_a_lt_b
    non_tie = p1_lt + p2_lt
    if non_tie <= 0:
        return 0.5, 0.5, p1_lt, p2_lt, p_tie
    return p1_lt / non_tie, p2_lt / non_tie, p1_lt, p2_lt, p_tie


def _round_h2h_lookup(release: dict) -> dict:
    lookup = {}
    for row in release["round_h2h"].itertuples(index=False):
        a = str(row.player_a).strip().lower()
        b = str(row.player_b).strip().lower()
        lookup[(a, b)] = (float(row.p_a_lt_b), float(row.p_tie))
    return lookup


def _build_round_matchups(
    tourney: str,
    sim_round: int,
    repl: dict,
    *,
    event_id,
    release: dict,
) -> list:
    """Join fresh offered odds to the manifest-bound current-round H2H table."""
    lookup = _round_h2h_lookup(release)
    scraped = _fetch_scraped_guarded(
        "round_matchups", round=sim_round, event_id=event_id
    )
    rows = (scraped or {}).get("match_list")
    if not isinstance(rows, list) or not rows:
        raise OddsScreenContractError(
            "fresh event/round-scoped matchup odds are unavailable; retaining prior generation"
        )

    matchups = {}
    missing_fairs = set()
    for match in rows:
        p1 = _norm(match.get("p1_player_name", ""), repl)
        p2 = _norm(match.get("p2_player_name", ""), repl)
        if not p1 or not p2 or p1 == p2:
            continue
        probabilities = _oriented_h2h_probability(lookup, p1, p2)
        if probabilities is None:
            missing_fairs.add((p1, p2))
            continue
        p1_push, p2_push, p1_tie_loss, p2_tie_loss, p_tie = probabilities
        ties = match.get("ties")
        tie_is_loss = str(ties or "").strip().lower() == "separate bet offered"
        p1_prob = p1_tie_loss if tie_is_loss else p1_push
        p2_prob = p2_tie_loss if tie_is_loss else p2_push
        key = (p1, p2, tie_is_loss)
        rec = matchups.setdefault(
            key,
            {
                "p1": p1,
                "p2": p2,
                "ties": ties,
                "books": {},
                "fair": {
                    "p1": _prob_to_american(p1_prob),
                    "p2": _prob_to_american(p2_prob),
                    "p1_prob": round(p1_prob, 5),
                    "p2_prob": round(p2_prob, 5),
                    "tie_prob": round(p_tie, 5),
                },
                "edge": {},
            },
        )
        for book, odds in (match.get("odds") or {}).items():
            if book == "datagolf" or not isinstance(odds, dict):
                continue
            p1_odds = odds.get("p1")
            p2_odds = odds.get("p2")
            if book == "kalshi" and odds.get("p1_mid") is not None:
                p1_odds = odds["p1_mid"]
                p2_odds = odds.get("p2_mid", p2_odds)
            if p1_odds is None or p2_odds is None:
                continue
            p1_odds = _parse_american_odds(p1_odds)
            p2_odds = _parse_american_odds(p2_odds)
            if p1_odds is None or p2_odds is None:
                continue
            rec["books"][book] = {"p1": p1_odds, "p2": p2_odds}
            rec["edge"][f"{book}_p1"] = _edge_pct(p1_prob, p1_odds)
            rec["edge"][f"{book}_p2"] = _edge_pct(p2_prob, p2_odds)

    if missing_fairs:
        raise OddsScreenContractError(
            f"{len(missing_fairs)} fresh offered round matchup pair(s) lack a "
            "sealed model fair; retaining prior generation"
        )
    if not matchups:
        raise OddsScreenContractError(
            "no fresh offered round matchup could be joined to the sealed H2H table"
        )
    if not _has_valid_book_quote(list(matchups.values()), ("p1", "p2")):
        raise OddsScreenContractError(
            "fresh round matchup payload has no usable book quote; "
            "retaining prior generation"
        )
    for rec in matchups.values():
        if rec["edge"]:
            rec["best_edge"] = max(rec["edge"].values())
    return sorted(matchups.values(), key=lambda x: x.get("best_edge", 0), reverse=True)


def _tournament_h2h_lookup(release: dict) -> dict:
    lookup = {}
    for raw_p1, raw_p2, raw_probability in release["fairs"]["matchups"]:
        p1 = str(raw_p1).strip().lower()
        p2 = str(raw_p2).strip().lower()
        probability = float(raw_probability)
        if p1 <= p2:
            lookup[(p1, p2)] = probability
        else:
            lookup[(p2, p1)] = 1.0 - probability
    return lookup


def _tournament_score_lookup(release: dict) -> dict:
    samples = release.get("tournament_samples")
    if not isinstance(samples, pd.DataFrame) or samples.empty:
        raise OddsScreenContractError(
            "committed tournament sample joint is unavailable for settlement"
        )
    return {
        str(player).strip().lower(): samples.loc[player].to_numpy(dtype=float)
        for player in samples.index
    }


def _tournament_probabilities(
    lookup: dict, score_lookup: dict, p1: str, p2: str
):
    """Return tie-push and tie-loss probabilities in the offered player order."""
    a, b = (p1, p2) if p1 <= p2 else (p2, p1)
    probability = lookup.get((a, b))
    p1_scores = score_lookup.get(p1)
    p2_scores = score_lookup.get(p2)
    if probability is None or p1_scores is None or p2_scores is None:
        return None
    p1_push = probability if p1 <= p2 else 1.0 - probability
    p2_push = 1.0 - p1_push
    p1_raw = float((p1_scores < p2_scores).mean())
    p2_raw = float((p2_scores < p1_scores).mean())
    p_tie = float((p1_scores == p2_scores).mean())
    return p1_push, p2_push, p1_raw, p2_raw, p_tie


def _model_matchup_row(
    p1: str,
    p2: str,
    p1_prob: float,
    p2_prob: float,
    tie_prob: float,
    ties,
) -> dict:
    return {
        "p1": p1,
        "p2": p2,
        "ties": ties,
        "books": {},
        "fair": {
            "p1": _prob_to_american(p1_prob),
            "p2": _prob_to_american(p2_prob),
            "p1_prob": round(p1_prob, 5),
            "p2_prob": round(p2_prob, 5),
            "tie_prob": round(tie_prob, 5),
        },
        "edge": {},
    }


def _build_tournament_matchups(
    tourney: str, repl: dict, *, event_id, release: dict
) -> list:
    """Build tournament matchups only from the sealed live tournament joint."""
    lookup = _tournament_h2h_lookup(release)
    score_lookup = _tournament_score_lookup(release)
    scraped = _fetch_scraped_guarded("tournament_matchups", event_id=event_id)
    offered = (scraped or {}).get("match_list")
    if not isinstance(offered, list) or not offered:
        raise OddsScreenContractError(
            "fresh event-scoped tournament matchup odds are unavailable; "
            "retaining prior generation"
        )

    matchups = {}
    missing_fairs = set()
    for match in offered:
        p1 = _norm(match.get("p1_player_name", ""), repl)
        p2 = _norm(match.get("p2_player_name", ""), repl)
        probabilities = _tournament_probabilities(lookup, score_lookup, p1, p2)
        if probabilities is None:
            missing_fairs.add((p1, p2))
            continue
        p1_push, p2_push, p1_tie_loss, p2_tie_loss, p_tie = probabilities
        ties = match.get("ties")
        tie_is_loss = str(ties or "").strip().lower() == "separate bet offered"
        p1_prob = p1_tie_loss if tie_is_loss else p1_push
        p2_prob = p2_tie_loss if tie_is_loss else p2_push
        key = (p1, p2, tie_is_loss)
        rec = matchups.setdefault(
            key,
            _model_matchup_row(p1, p2, p1_prob, p2_prob, p_tie, ties),
        )
        for book, odds in (match.get("odds") or {}).items():
            if book == "datagolf" or not isinstance(odds, dict):
                continue
            p1_odds = odds.get("p1")
            p2_odds = odds.get("p2")
            if book == "kalshi" and odds.get("p1_mid") is not None:
                p1_odds = odds["p1_mid"]
                p2_odds = odds.get("p2_mid", p2_odds)
            if p1_odds is None or p2_odds is None:
                continue
            p1_odds = _parse_american_odds(p1_odds)
            p2_odds = _parse_american_odds(p2_odds)
            if p1_odds is None or p2_odds is None:
                continue
            rec["books"][book] = {"p1": p1_odds, "p2": p2_odds}
            rec["edge"][f"{book}_p1"] = _edge_pct(p1_prob, p1_odds)
            rec["edge"][f"{book}_p2"] = _edge_pct(p2_prob, p2_odds)
    if missing_fairs:
        raise OddsScreenContractError(
            f"{len(missing_fairs)} fresh offered tournament matchup pair(s) "
            "lack a sealed model fair; retaining prior generation"
        )
    if not matchups:
        raise OddsScreenContractError(
            "fresh tournament matchup odds do not join the sealed tournament model"
        )
    if not _has_valid_book_quote(list(matchups.values()), ("p1", "p2")):
        raise OddsScreenContractError(
            "fresh tournament matchup payload has no usable book quote; "
            "retaining prior generation"
        )
    for rec in matchups.values():
        if rec["edge"]:
            rec["best_edge"] = max(rec["edge"].values())
    return sorted(matchups.values(), key=lambda x: x.get("best_edge", 0), reverse=True)


def _build_score_lines(
    tourney: str,
    sim_round: int,
    repl: dict,
    *,
    event_id,
    release: dict,
) -> list:
    """Price score lines from the round PMFs embedded in sealed sim_fairs.json."""
    pmfs = {
        _norm(player, repl): {float(score): float(prob) for score, prob in pmf.items()}
        for player, pmf in release["fairs"]["round_scores"].items()
    }
    scraped = _fetch_scraped_guarded(
        "round_scores", round=sim_round, event_id=event_id
    )
    scraped_lines = {}
    if scraped:
        missing_fairs = set()
        for item in scraped.get("lines", []):
            player = _norm(item.get("player_name", ""), repl)
            odds = item.get("odds", {})
            books = {}
            for book, book_odds in odds.items():
                if isinstance(book_odds, dict):
                    over = book_odds.get("over")
                    under = book_odds.get("under")
                    over = _parse_american_odds(over)
                    under = _parse_american_odds(under)
                    if over is not None and under is not None:
                        books[book] = {"over": over, "under": under}
            if not books or not player:
                continue
            try:
                line = float(item.get("line"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise OddsScreenContractError(
                    f"fresh offered score quote has an invalid line for {player}"
                ) from exc
            if not math.isfinite(line) or not _is_half_stroke_line(line):
                raise OddsScreenContractError(
                    f"fresh offered score quote is not a half-stroke line for {player}"
                )
            if player not in pmfs:
                missing_fairs.add(player)
                continue
            scraped_lines[player] = {"line": line, "books": books}
        if missing_fairs:
            raise OddsScreenContractError(
                f"{len(missing_fairs)} fresh offered score player(s) lack a sealed "
                "model PMF; retaining prior generation"
            )
        logger.info(f"Loaded {len(scraped_lines)} scraped score lines")

    results = []
    for player in sorted(pmfs):
        rec = {"player": player, "books": {}, "fair": {}, "edge": {}}
        sl = scraped_lines.get(player, {})
        rec["books"] = sl.get("books", {})
        line = sl.get("line")
        pmf = pmfs[player]
        rec["pred"] = round(sum(score * probability for score, probability in pmf.items()), 2)
        if line is not None:
            rec["line"] = line
            fair_under_prob = sum(
                probability for score, probability in pmf.items() if score < line
            )
            fair_over_prob = 1.0 - fair_under_prob
            rec["fair"] = {
                "over": _prob_to_american(fair_over_prob),
                "under": _prob_to_american(fair_under_prob),
                "over_prob": round(fair_over_prob, 5),
                "under_prob": round(fair_under_prob, 5),
            }
            for book, odds in rec["books"].items():
                rec["edge"][f"{book}_over"] = _edge_pct(
                    fair_over_prob, odds["over"]
                )
                rec["edge"][f"{book}_under"] = _edge_pct(
                    fair_under_prob, odds["under"]
                )
        if rec["edge"]:
            rec["best_edge"] = max(rec["edge"].values())
        results.append(rec)

    results = sorted(results, key=lambda x: x.get("best_edge", 0), reverse=True)
    if not _has_valid_book_quote(results, ("over", "under")):
        raise OddsScreenContractError(
            "fresh round score payload has no usable book quote; "
            "retaining prior generation"
        )
    return results


def _decimal_to_american(dec: float) -> int:
    """Convert decimal odds to American odds."""
    if not math.isfinite(dec) or dec <= 1.0:
        raise ValueError("decimal odds must be finite and greater than one")
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    else:
        return int(round(-100 / (dec - 1)))


def _fetch_dg_outrights(market_name: str, repl: dict, *, event_id=None) -> dict:
    """Fetch outright odds from DataGolf API. Returns {player: {book: american_odds}}."""
    api_key = os.getenv("DATAGOLF_API_KEY")
    if not api_key:
        return {}
    try:
        resp = requests.get(
            "https://feeds.datagolf.com/betting-tools/outrights",
            params={
                "tour": "pga", "market": market_name,
                "odds_format": "decimal", "file_format": "json", "key": api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"DG outright API failed for {market_name}: {e}")
        return {}

    response_event_id = _event_id_token(data.get("event_id"))
    target_event_id = _event_id_token(event_id)
    if target_event_id is None or response_event_id != target_event_id:
        logger.warning(
            "DG outrights (%s) lack exact event identity (%s != %s); dropping quotes",
            market_name,
            response_event_id,
            target_event_id,
        )
        return {}

    result = {}
    for entry in data.get("odds", []):
        if not isinstance(entry, dict):
            continue
        player = _norm(entry.get("player_name", ""), repl)
        if not player:
            continue
        books = {}
        for book in ["pinnacle", "betcris", "betonline", "draftkings", "fanduel",
                      "betmgm", "caesars", "bovada", "unibet"]:
            dec = entry.get(book)
            if dec is not None:
                try:
                    american = _parse_american_odds(
                        _decimal_to_american(float(dec))
                    )
                except (ValueError, ZeroDivisionError, OverflowError):
                    american = None
                if american is not None:
                    books[book] = american
        if books:
            result[player] = books
    logger.info(f"DG outrights ({market_name}): {len(result)} players")
    return result


def _build_outrights(
    tourney: str, repl: dict, *, event_id, release: dict
) -> dict:
    """Build outright records from sealed sim fairs plus current/frozen book odds."""
    sealed_outrights = release["fairs"]["outrights"]
    sealed_nodh = release["fairs"].get("outrights_nodh") or {}
    fair_by_market = {
        market: {
            _norm(player, repl): float(probability)
            for player, probability in (sealed_outrights.get(market) or {}).items()
        }
        for market in ("winner", "top_5", "top_10", "top_20")
    }
    nodh_by_market = {
        market: {
            _norm(player, repl): float(probability)
            for player, probability in (sealed_nodh.get(market) or {}).items()
        }
        for market in ("top_5", "top_10", "top_20")
    }

    # Kalshi/NoVig outrights are no longer published as scraped JSON (the sim
    # fetches them live); kept here so the dead-heat split still applies if they return.
    kalshi_by_market = {}

    # Fetch DataGolf API outrights (pinnacle, betcris, betonline, etc.)
    dg_by_market = {}
    for dg_market in ["win", "top_5", "top_10", "top_20"]:
        dg_by_market[dg_market] = _fetch_dg_outrights(
            dg_market, repl, event_id=event_id
        )

    # Overlay event-tagged, freshness-stamped Betcris quotes. Calling the legacy
    # DataFrame loader here would discard the source timestamp and reintroduce an
    # implicit sim_inputs event scope, so strict activation consumes guarded JSON.
    betcris_scraped = {}
    betcris_payload = _fetch_scraped_guarded(
        "betcris_outrights", event_id=event_id
    )
    for line in (betcris_payload or {}).get("lines", []):
        market = str(line.get("market_type") or "").strip().lower()
        raw_player = str(line.get("player") or line.get("player_name") or "")
        player = _norm(raw_player, repl)
        american = _parse_american_odds(line.get("odds"))
        if market in ("winner", "top_5", "top_10", "top_20") and player and american:
            betcris_scraped.setdefault(market, {})[player] = american
    if betcris_scraped:
        logger.info(
            "Loaded strict Betcris outright markets: %s",
            ", ".join(sorted(betcris_scraped)),
        )

    DG_MARKET_MAP = {"winner": "win", "top_5": "top_5", "top_10": "top_10", "top_20": "top_20"}
    NODH_BOOKS = {"kalshi", "novig"}   # these don't dead-heat → use the _nodh fair

    markets = {}
    for market_key in ("winner", "top_5", "top_10", "top_20"):
        records = []
        dg_market_name = DG_MARKET_MAP.get(market_key, market_key)
        dg_odds = dg_by_market.get(dg_market_name, {})
        dh_fairs = fair_by_market.get(market_key) or {}
        nodh_fairs = nodh_by_market.get(market_key) or {}
        offered_players = {
            player for player, books in dg_odds.items() if books
        } | set(betcris_scraped.get(market_key, {}))
        missing_fairs = offered_players - set(dh_fairs)
        if missing_fairs:
            raise OddsScreenContractError(
                f"{len(missing_fairs)} attributable {market_key} quote player(s) "
                "lack a sealed outright fair; retaining prior generation"
            )

        for player in sorted(dh_fairs):
            rec = {"player": player, "books": {}, "edge": {}}
            dh_prob = dh_fairs[player]
            nodh_prob = nodh_fairs.get(player)

            # Display fair = dead-heat-resolved (what most books pay).
            rec["sim_prob"] = round(dh_prob, 5)
            rec["fair_odds"] = _prob_to_american(dh_prob)
            if nodh_prob is not None:
                rec["sim_prob_nodh"] = round(nodh_prob, 5)

            # DataGolf book odds: standard books dead-heat → dh fair; (none of the
            # DG books are in NODH_BOOKS today, but the split is applied per-book).
            player_dg = dict(dg_odds.get(player, {}))
            # Prefer the fresh scraped Betcris over DataGolf's frozen live-play quote
            # (and add Betcris for players DG dropped entirely during live play).
            # Both sources were normalized through the release name map above.
            bc_live = betcris_scraped.get(market_key, {}).get(player)
            if bc_live is not None:
                player_dg["betcris"] = bc_live
            for book, american in player_dg.items():
                american = _parse_american_odds(american)
                if american is None:
                    continue
                rec["books"][book] = {"yes": american}
                fair = nodh_prob if book in NODH_BOOKS else dh_prob
                if fair is not None:
                    rec["edge"][book] = _edge_pct(fair, american)

            # Kalshi overlay (no dead-heat → no-DH fair).
            kalshi_odds = kalshi_by_market.get(market_key, {}).get(player)
            if kalshi_odds:
                yes_odds = _parse_american_odds(kalshi_odds.get("yes"))
                if yes_odds is not None:
                    rec["books"]["kalshi"] = {"yes": yes_odds}
                    fair = nodh_prob if nodh_prob is not None else dh_prob
                    if fair is not None:
                        rec["edge"]["kalshi"] = _edge_pct(fair, yes_odds)

            if rec["edge"]:
                rec["best_edge"] = max(rec["edge"].values())
            records.append(rec)

        market_rows = sorted(
            records, key=lambda x: x.get("best_edge", x.get("sim_prob", 0)), reverse=True
        )
        if not _has_valid_book_quote(market_rows, ("yes",)):
            raise OddsScreenContractError(
                f"outright quote sources have no usable {market_key} quote; "
                "retaining prior generation"
            )
        markets[market_key] = market_rows

    return markets


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload odds screen data to R2")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--dry-run", action="store_true", help="Print JSON, skip upload"
    )
    output_mode.add_argument(
        "--output-dir",
        type=Path,
        help="Write publish-ready JSON files here and skip the built-in R2 upload",
    )
    args = parser.parse_args()

    # The reprice job and the odds screen must resolve one identical event/round.
    # Sheet config is the operational expectation; the committed strict release
    # is the model authority. Any disagreement leaves the prior R2 pointer active.
    try:
        from sheet_config import load_config
        config = load_config()
        expected_tourney = str(config["tourney"]).strip()
        expected_event_id = config["event_id"]
        round_num = int(config["round_num"])
    except Exception as e:
        raise OddsScreenContractError(
            f"cannot bind odds-screen publication to Sheet config: {e}"
        ) from e
    if not expected_tourney or expected_event_id in (None, "", 0, "0"):
        raise OddsScreenContractError("Sheet config has no publishable event identity")
    sim_round = round_num + 1 if round_num < 4 else 4
    release = _load_committed_release_contract(
        expected_tourney=expected_tourney,
        expected_event_id=expected_event_id,
        expected_round=sim_round,
    )
    manifest = release["manifest"]
    fairs = release["fairs"]
    tourney = manifest["tourney"]
    event_id = manifest["event_id"]
    event_name = fairs.get("event_name") or tourney.replace("_", " ").title()

    repl = _load_name_replacements()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    provenance = {
        "event_id": str(event_id),
        "tourney": str(tourney),
        "round": int(sim_round),
        "release_generation": manifest["generation"],
        "release_manifest_sha256": manifest["manifest_sha256"],
        "simulation_manifest_sha256": manifest["simulation_manifest_sha256"],
        "live_tournament_manifest_sha256": manifest[
            "live_tournament_manifest_sha256"
        ],
        "source_git_sha": release["source_git_sha"],
    }

    logger.info(
        "Building odds screen data: %s event %s R%s from release %s @ %s",
        tourney,
        event_id,
        sim_round,
        manifest["generation"],
        release["source_git_sha"][:12],
    )

    # Build all markets
    round_mu = _build_round_matchups(
        tourney,
        sim_round,
        repl,
        event_id=event_id,
        release=release,
    )
    tourn_mu = _build_tournament_matchups(
        tourney, repl, event_id=event_id, release=release
    )
    score_lines = _build_score_lines(
        tourney,
        sim_round,
        repl,
        event_id=event_id,
        release=release,
    )
    outrights = _build_outrights(
        tourney, repl, event_id=event_id, release=release
    )

    payloads = {
        "round_matchups.json": {
            **provenance,
            "event_name": event_name, "last_updated": now, "round": sim_round,
            "matchups": round_mu,
        },
        "tournament_matchups.json": {
            **provenance,
            "event_name": event_name, "last_updated": now,
            "matchups": tourn_mu,
        },
        "score_lines.json": {
            **provenance,
            "event_name": event_name, "last_updated": now, "round": sim_round,
            "lines": score_lines,
        },
        "outrights.json": {
            **provenance,
            "event_name": event_name, "last_updated": now,
            "markets": outrights,
        },
        "meta.json": {
            **provenance,
            "event_name": event_name, "tourney": tourney,
            "round": sim_round, "last_updated": now,
        },
    }

    _validate_odds_screen_payloads(payloads)

    for key, data in payloads.items():
        count = len(data.get("matchups", data.get("lines", data.get("markets", {}))))
        logger.info(f"  {key}: {count} records")

    if args.dry_run:
        for key, data in payloads.items():
            print(f"\n{'='*60}")
            print(f"  {key}")
            print(f"{'='*60}")
            print(json.dumps(data, indent=2, default=str)[:2000])
        return

    if args.output_dir:
        written = _write_atomic_payload_bundle(payloads, args.output_dir)
        logger.info(
            f"Done — staged {len(written) - 1} immutable files and wrote meta.json last"
        )
        return

    # Upload to R2
    client = _get_r2_client()
    pointer = _upload_atomic_payload_bundle(client, payloads)
    logger.info(f"Done — published atomic generation {pointer['generation']}")


if __name__ == "__main__":
    main()

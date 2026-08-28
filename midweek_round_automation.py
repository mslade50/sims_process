"""
Advance and run the live round pipeline when the odds board is ready.

The golf_scraping odds board already dispatches ``sentinel-scrape-complete`` to
this repository after committing fresh sim-facing odds. The accompanying GitHub
Actions workflow fetches that JSON and invokes this coordinator.

The coordinator is intentionally fail-closed on inputs the simulation consumes:
  1. Require fresh, event-scoped next-round prices from BetCris and BetOnline.
  2. Require completed-round DataGolf stats and posted next-round tee times.
  3. Before R2, require every active player in the rich-shot archive and enough
     valid approaches for the pin-high feature. A harmless gap elsewhere in a
     player's shot sequence is warned on, but does not block the simulation.
  4. Fetch all remaining-round Open-Meteo dewpoint + AI multi-model wind.
  5. Batch-write weather, advance the Sheet round pointer exactly once, and
     leave a durable Sheet status so a failed run can resume.
  6. Run live_stats_engine.py --automation, then the full round_sim.py.

Normal "not ready yet" checks exit 0 because the odds board will dispatch again.
Pipeline failures after a transition starts exit non-zero and are retried on the
next dispatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from round_matchup_coverage import (
    DEFAULT_REQUIRED_BOOKS,
    resolve_required_matchup_books,
)
from score_centering import (
    CENTERING_VERSION,
    validate_field_relative_predictions,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_ODDS_FILE = (
    ROOT / "permanent_data" / "scraped_odds" / "round_matchups_latest.json"
)
REQUIRED_BOOKS = DEFAULT_REQUIRED_BOOKS
MIN_PIN_HIGH_APPROACHES = 6


class NotReady(RuntimeError):
    """Expected preflight result; a future odds-board dispatch may be ready."""


class PipelineFailure(RuntimeError):
    """A transition began or an invariant failed and the workflow should fail."""


@dataclass(frozen=True)
class OddsReadiness:
    target_round: int
    last_updated: str
    counts: dict[str, int]
    scoped_rows: int


def _parse_utc_timestamp(value: str) -> datetime:
    value = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise NotReady("Odds payload has no parseable UTC last_updated timestamp")


def _valid_two_way_price(price) -> bool:
    if not isinstance(price, dict):
        return False
    p1, p2 = price.get("p1"), price.get("p2")
    try:
        p1_num = int(str(p1).strip().replace("+", ""))
        p2_num = int(str(p2).strip().replace("+", ""))
    except (TypeError, ValueError):
        return False
    return p1_num != 0 and p2_num != 0


def evaluate_odds_readiness(
    data: dict,
    event_id,
    *,
    min_book_matchups: int = 5,
    max_age_hours: float = 3.0,
    now: datetime | None = None,
    required_books: tuple[str, ...] | None = None,
) -> OddsReadiness:
    """Validate and count the board's nested DataGolf-format matchup payload."""
    required_books = (
        tuple(required_books) if required_books is not None else REQUIRED_BOOKS
    )
    if not required_books:
        raise ValueError("At least one required matchup book must be configured")
    if not isinstance(data, dict):
        raise NotReady("Odds payload is not a JSON object")

    try:
        target_round = int(data.get("round"))
    except (TypeError, ValueError):
        raise NotReady("Odds payload has no target round")
    if target_round not in (2, 3, 4):
        raise NotReady(f"R{target_round} does not advance a completed live round")

    updated_at = _parse_utc_timestamp(data.get("last_updated"))
    now = now or datetime.now(timezone.utc)
    age_hours = (now - updated_at).total_seconds() / 3600.0
    if age_hours < -0.25:
        raise NotReady(f"Odds payload timestamp is {abs(age_hours):.1f}h in the future")
    if age_hours > max_age_hours:
        raise NotReady(
            f"Odds payload is stale ({age_hours:.1f}h > {max_age_hours:.1f}h)"
        )

    target_event = str(event_id).strip()
    file_event = str(data.get("event_id") or "").strip()
    counts = {book: set() for book in required_books}
    scoped_rows = 0

    for row in data.get("match_list") or []:
        if not isinstance(row, dict):
            continue
        row_event = str(row.get("event_id") or "").strip()
        if row_event:
            if row_event != target_event:
                continue
        elif not file_event or file_event != target_event:
            continue

        row_round = row.get("round")
        try:
            row_round = int(row_round) if row_round not in (None, "") else 0
        except (TypeError, ValueError):
            continue
        if row_round not in (0, target_round):
            continue

        scoped_rows += 1
        p1 = str(row.get("p1_player_name") or "").strip().lower()
        p2 = str(row.get("p2_player_name") or "").strip().lower()
        pair = tuple(sorted((p1, p2)))
        if not all(pair):
            continue
        odds = row.get("odds") or {}
        for book in required_books:
            if _valid_two_way_price(odds.get(book)):
                counts[book].add(pair)

    final_counts = {book: len(pairs) for book, pairs in counts.items()}
    missing = {
        book: count
        for book, count in final_counts.items()
        if count < min_book_matchups
    }
    if missing:
        detail = ", ".join(
            f"{book}={count}/{min_book_matchups}"
            for book, count in missing.items()
        )
        raise NotReady(
            f"R{target_round} sharp-book coverage is not ready ({detail})"
        )

    return OddsReadiness(
        target_round=target_round,
        last_updated=str(data["last_updated"]),
        counts=final_counts,
        scoped_rows=scoped_rows,
    )


def transition_action(
    sheet_round: int,
    target_round: int,
    automation_target,
    automation_status: str,
    automation_event=None,
    event_id=None,
    force=False,
) -> str:
    """Return advance, resume, complete, or raise for an unsafe pointer gap."""
    same_event = (
        event_id is None
        or str(automation_event or "").strip() == str(event_id).strip()
    )
    try:
        prior_target = int(automation_target) if same_event else None
    except (TypeError, ValueError):
        prior_target = None
    status = str(automation_status or "").strip().lower() if same_event else ""

    if prior_target == target_round and status == "complete":
        if force and sheet_round == target_round - 1:
            return "resume"
        return "complete"
    if sheet_round == target_round - 2:
        return "advance"
    if sheet_round == target_round - 1:
        return "resume"
    if sheet_round < target_round - 2:
        raise PipelineFailure(
            f"Sheet round is R{sheet_round}, but odds are already for R{target_round}; "
            "more than one transition is missing"
        )
    raise NotReady(
        f"Sheet round R{sheet_round} is already beyond the R{target_round} transition"
    )


def _load_course_coordinates(course_id, sheet_lat=None, sheet_lon=None):
    if sheet_lat is not None and sheet_lon is not None:
        latitude, longitude = float(sheet_lat), float(sheet_lon)
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            print(
                f"  Course coordinates from Sheet: "
                f"{latitude:.5f},{longitude:.5f}"
            )
            return latitude, longitude

    try:
        import sim_inputs

        lat_override = getattr(sim_inputs, "lat_override", None)
        lon_override = getattr(sim_inputs, "lon_override", None)
        if lat_override is not None and lon_override is not None:
            return float(lat_override), float(lon_override)
    except Exception:
        pass

    path = ROOT / "permanent_data" / "course_coordinates.csv"
    if not path.exists():
        raise PipelineFailure(f"Course coordinates file is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                if int(float(row.get("course_id", ""))) == int(course_id):
                    return float(row["lat"]), float(row["lon"])
            except (TypeError, ValueError, KeyError):
                continue
    raise PipelineFailure(f"course_id {course_id} is missing from {path.name}")


def _check_datagolf_ready(api_key: str, completed_round: int, target_round: int,
                          min_rows: int = 10):
    from api_utils import fetch_field_updates, fetch_live_stats

    stats = fetch_live_stats(completed_round, api_key)
    valid_stats = 0
    if stats is not None:
        if "sg_total" in stats.columns:
            valid_stats = int(stats["sg_total"].notna().sum())
        else:
            valid_stats = len(stats)
    if stats is None or valid_stats < min_rows:
        raise NotReady(
            f"DataGolf R{completed_round} stats are not ready "
            f"({valid_stats}/{min_rows} players)"
        )

    tee_col = f"r{target_round}_teetime"
    field = fetch_field_updates(
        api_key,
        teetime_col=tee_col,
        include_course=True,
        fill_missing_teetimes=False,
    )
    if field is None or tee_col not in field.columns:
        raise NotReady(f"DataGolf has not posted R{target_round} tee times")
    tee_times = field[tee_col].replace("", None)
    posted = int(tee_times.notna().sum())
    if posted < min_rows:
        raise NotReady(
            f"DataGolf R{target_round} tee times are not ready "
            f"({posted}/{min_rows} players)"
        )

    courses = []
    if "course" in field.columns:
        courses = sorted(
            str(value).strip()
            for value in field.loc[tee_times.notna(), "course"].dropna().unique()
            if str(value).strip()
        )
    if len(courses) > 1:
        raise NotReady(
            "Automated scoring currently requires a single target-round course; "
            f"R{target_round} has {courses}. Update the per-course scoring inputs manually."
        )

    print(
        f"  DataGolf ready: {valid_stats} R{completed_round} stat rows, "
        f"{posted} R{target_round} tee times"
    )
    return field


def _normalized_player_names(values) -> set[str]:
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip() and str(value).strip().lower() != "nan"
    }


def _active_target_round_players(field: pd.DataFrame, target_round: int) -> set[str]:
    tee_col = f"r{target_round}_teetime"
    if "player_name" not in field.columns or tee_col not in field.columns:
        raise PipelineFailure(
            f"DataGolf R{target_round} field is missing player_name or {tee_col}"
        )
    tee_times = field[tee_col].replace("", None)
    return _normalized_player_names(field.loc[tee_times.notna(), "player_name"])


def _check_shot_collector_ready(
    config: dict,
    field: pd.DataFrame,
    completed_round: int,
    target_round: int,
) -> set[str]:
    """Require an archive row for every next-round player.

    ``complete`` describes the integrity of the player's entire round, while
    the R2 simulation consumes only the derived pin-high approach feature.
    Sequence gaps therefore warn here and are judged by the feature-specific
    approach coverage check after the artifact is generated.
    """
    from api_utils import fetch_img_player_rounds
    from generate_pin_high_r1 import build_event_key

    active_players = _active_target_round_players(field, target_round)
    if not active_players:
        raise NotReady(f"DataGolf R{target_round} active field is empty")

    event_ids = config.get("event_ids") or [config.get("event_id")]
    if len(event_ids) != 1 or event_ids[0] is None:
        raise PipelineFailure(
            f"Shot-collector readiness requires one active event: {event_ids}"
        )
    season = datetime.now().year
    event_key = build_event_key(config.get("tour", "pga"), event_ids[0], season)
    archive = fetch_img_player_rounds(
        event_ids[0],
        completed_round,
        db_path=os.getenv("IMG_SHOT_DB_PATH"),
        tour=config.get("tour", "pga"),
        season=season,
        event_key=event_key,
    )
    if archive is None or archive.empty:
        raise NotReady(
            f"Shot collector has no R{completed_round} archive for {event_key}"
        )
    if "player_name" not in archive.columns or "complete" not in archive.columns:
        raise PipelineFailure(
            f"Shot collector R{completed_round} archive is missing player_name/complete"
        )

    archive = archive.copy()
    archive["player_key"] = archive["player_name"].map(
        lambda value: str(value).strip().lower()
    )
    complete_values = archive["complete"].map(
        lambda value: value is True
        or str(value).strip().lower() in {"1", "true", "yes"}
    )
    archived_players = set(archive["player_key"])
    missing = sorted(active_players - archived_players)
    if missing:
        preview = ", ".join(missing[:12])
        if len(missing) > 12:
            preview += f", +{len(missing) - 12} more"
        raise NotReady(
            f"Shot collector R{completed_round} is missing R{target_round} active "
            f"players ({len(active_players) - len(missing)}/{len(active_players)} "
            f"archived): {preview}"
        )

    complete_players = set(archive.loc[complete_values, "player_key"])
    incomplete = sorted(active_players - complete_players)
    if incomplete:
        preview = ", ".join(incomplete[:12])
        if len(incomplete) > 12:
            preview += f", +{len(incomplete) - 12} more"
        print(
            f"::warning::Shot collector R{completed_round} has "
            f"{len(incomplete)} incomplete active-player round(s): {preview}. "
            "Continuing to the pin-high approach coverage check."
        )
    else:
        print(
            f"  Shot collector ready: {len(active_players)}/{len(active_players)} "
            f"active players complete for {event_key} R{completed_round}"
        )
    return active_players


def _validate_pin_high_coverage(
    path: Path,
    active_players: set[str],
    event_key: str,
    minimum_approaches: int = MIN_PIN_HIGH_APPROACHES,
) -> None:
    """Fail closed only when usable pin-high input coverage is insufficient."""
    if not path.exists() or path.stat().st_size == 0:
        raise PipelineFailure(f"{path.name} was not created")
    frame = pd.read_csv(path)
    required = {"event_key", "player_name", "n_approaches", "coverage_status"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise PipelineFailure(
            f"{path.name} is missing coverage columns: {missing_columns}"
        )
    event_keys = set(frame["event_key"].dropna().astype(str).str.strip())
    if event_keys != {event_key}:
        raise PipelineFailure(
            f"{path.name} is for {sorted(event_keys)}, expected {event_key}"
        )

    frame = frame.copy()
    frame["player_key"] = frame["player_name"].map(
        lambda value: str(value).strip().lower()
    )
    approaches = pd.to_numeric(frame["n_approaches"], errors="coerce").fillna(0)
    covered_players = set(
        frame.loc[approaches.ge(int(minimum_approaches)), "player_key"]
    )
    missing_players = sorted(active_players - covered_players)
    if missing_players:
        preview = ", ".join(missing_players[:12])
        if len(missing_players) > 12:
            preview += f", +{len(missing_players) - 12} more"
        raise PipelineFailure(
            f"Pin-high coverage is below {int(minimum_approaches)} valid approaches "
            f"for one or more active players ({len(covered_players & active_players)}/"
            f"{len(active_players)}): {preview}"
        )
    print(
        f"  Pin-high coverage ready: {len(active_players)}/{len(active_players)} "
        f"active players have at least {int(minimum_approaches)} valid approaches"
    )


def _prepare_optional_pin_high(
    config: dict,
    field: pd.DataFrame,
    completed_round: int,
    target_round: int,
    *,
    dry_run: bool = False,
) -> str | None:
    """Prepare R1 pin-high input without making it a core-sim dependency.

    The adjustment is deliberately optional in ``live_stats_engine.py``. Keep
    its stricter feature-quality checks, but disable it for this process when
    the collector, generator, or artifact is not healthy enough. This also
    prevents a stale artifact from being consumed after a generation failure.
    """
    try:
        active_players = _check_shot_collector_ready(
            config, field, completed_round, target_round
        )
        if dry_run:
            print("  [dry-run] Would regenerate and validate full-field R1 pin-high")
            return None

        _run(
            [sys.executable, "generate_pin_high_r1.py"],
            "R1 pin-high adjustment generation",
        )
        from generate_pin_high_r1 import build_event_key

        event_ids = config.get("event_ids") or [config.get("event_id")]
        event_key = build_event_key(
            config.get("tour", "pga"), event_ids[0], datetime.now().year
        )
        _validate_pin_high_coverage(
            ROOT / "pin_high_r1.csv", active_players, event_key
        )
        return None
    except Exception as exc:  # noqa: BLE001 - optional feature must fail open
        warning = str(exc).strip() or exc.__class__.__name__
        os.environ["LIVE_PIN_HIGH_ADJ"] = "0"
        print(
            "::warning::Optional R1 pin-high adjustment disabled for this "
            f"R{target_round} run: {warning}. Continuing with the core sim."
        )
        return warning


def _format_array(values) -> str:
    return ",".join(f"{float(value):g}" for value in values)


def _run(cmd, label):
    print(f"\n  Running {label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise PipelineFailure(
            f"{label} failed with exit code {result.returncode}"
        )


def _verify_predictions(target_round: int):
    prediction = ROOT / f"model_predictions_r{target_round}.csv"
    if not prediction.exists() or prediction.stat().st_size == 0:
        raise PipelineFailure(f"{prediction.name} was not created")
    frame = pd.read_csv(prediction)
    header = set(frame.columns)
    required = {
        "my_pred" if target_round == 1 else f"my_pred{target_round}",
        f"scores_r{target_round}",
        f"weather_sg_r{target_round}",
        f"wind_adj{target_round}",
        f"dew_adj{target_round}",
        "field_skill_mean",
        "centering_version",
        "centering_group",
    }
    missing = sorted(required - header)
    if missing:
        raise PipelineFailure(
            f"{prediction.name} is skill-only or incomplete; missing {missing}"
        )
    versions = set(frame["centering_version"].dropna())
    if versions != {CENTERING_VERSION}:
        raise PipelineFailure(
            f"{prediction.name} has unsupported centering: {versions}"
        )
    groups = set(frame["centering_group"].dropna())
    if len(groups) != 1:
        raise PipelineFailure(
            f"{prediction.name} has inconsistent centering groups: {groups}"
        )
    group_col = next(iter(groups))
    if group_col == "field":
        group_col = None
    try:
        skill_col = "my_pred" if target_round == 1 else f"my_pred{target_round}"
        validate_field_relative_predictions(
            frame,
            skill_col=skill_col,
            score_col=f"scores_r{target_round}",
            weather_col=f"weather_sg_r{target_round}",
            group_col=group_col,
        )
    except ValueError as exc:
        raise PipelineFailure(str(exc)) from exc


def _verify_outputs(target_round: int, tourney: str, started_at: float | None = None):
    meta_path = ROOT / f"round_h2h_r{target_round}_meta.json"
    if not meta_path.exists():
        raise PipelineFailure(f"{meta_path.name} was not created")
    if started_at is not None and meta_path.stat().st_mtime < started_at - 2:
        raise PipelineFailure(
            f"{meta_path.name} was not refreshed by the current simulation"
        )
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if int(meta.get("round", 0)) != target_round:
        raise PipelineFailure(
            f"{meta_path.name} is stamped R{meta.get('round')}, "
            f"expected R{target_round}"
        )
    if str(meta.get("tourney")) != str(tourney):
        raise PipelineFailure(
            f"{meta_path.name} is for {meta.get('tourney')}, expected {tourney}"
        )


def _mark_status(spreadsheet, event_id, target_round: int, status: str,
                 message: str = ""):
    from sheets_storage import update_round_config_params

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    updates = {
        "automation_event_id": event_id,
        "automation_target_round": target_round,
        "automation_status": status,
        "automation_updated_at": now,
    }
    if message:
        updates["automation_message"] = message[:300]
    update_round_config_params(
        updates,
        notes={
            "automation_status": "Managed by midweek_round_automation.py",
        },
        spreadsheet=spreadsheet,
    )


def _load_gate_context(args):
    """Run the cheap odds + Sheet idempotency gate (no DataGolf or weather)."""
    load_dotenv()

    odds_path = Path(args.odds_file).resolve()
    if not odds_path.exists():
        raise NotReady(f"Odds file is missing: {odds_path}")
    with odds_path.open(encoding="utf-8") as handle:
        odds_data = json.load(handle)

    from sheet_config import load_config

    config = load_config()
    sheet_round = int(config["round_num"])
    event_id = config["event_id"]
    tourney = config["tourney"]

    required_books = resolve_required_matchup_books()
    print("  Required matchup books: " + ", ".join(required_books))
    readiness = evaluate_odds_readiness(
        odds_data,
        event_id,
        min_book_matchups=args.min_book_matchups,
        max_age_hours=args.max_age_hours,
        required_books=required_books,
    )
    target_round = readiness.target_round
    completed_round = target_round - 1
    print(
        f"  Odds ready for {tourney} R{target_round}: "
        + ", ".join(
            f"{book}={count}" for book, count in readiness.counts.items()
        )
    )

    from sheets_storage import get_round_config_params, get_spreadsheet

    spreadsheet = get_spreadsheet()
    raw_params = get_round_config_params(spreadsheet=spreadsheet)
    action = transition_action(
        sheet_round,
        target_round,
        raw_params.get("automation_target_round"),
        raw_params.get("automation_status", ""),
        raw_params.get("automation_event_id"),
        event_id,
        force=args.force,
    )
    return {
        "action": action,
        "config": config,
        "readiness": readiness,
        "spreadsheet": spreadsheet,
    }


def run_pipeline(args) -> int:
    gate = _load_gate_context(args)
    action = gate["action"]
    config = gate["config"]
    readiness = gate["readiness"]
    spreadsheet = gate["spreadsheet"]
    target_round = readiness.target_round
    completed_round = target_round - 1
    event_id = config["event_id"]
    tourney = config["tourney"]
    course_id = config["course_id"]

    if action == "complete":
        print(f"  R{target_round} automation is already complete; repricer handles new prices")
        return 0

    try:
        import sim_inputs

        sim_tourney = getattr(sim_inputs, "tourney", None)
        if sim_tourney and tourney and sim_tourney != tourney:
            raise PipelineFailure(
                f"Split-brain config: Sheet tourney '{tourney}' != "
                f"sim_inputs.tourney '{sim_tourney}'"
            )
    except PipelineFailure:
        raise
    except Exception as exc:
        raise PipelineFailure(f"Could not validate sim_inputs: {exc}") from exc

    api_key = os.getenv("DATAGOLF_API_KEY", "").strip()
    if not api_key:
        raise PipelineFailure("DATAGOLF_API_KEY is not configured")
    field = _check_datagolf_ready(
        api_key, completed_round, target_round, min_rows=args.min_datagolf_rows
    )

    pin_high_warning = None
    if (
        completed_round == 1
        and os.environ.get("LIVE_PIN_HIGH_ADJ") == "1"
    ):
        pin_high_warning = _prepare_optional_pin_high(
            config,
            field,
            completed_round,
            target_round,
            dry_run=args.dry_run,
        )

    from api_utils import fetch_event_weather_forecast, get_round_dates

    lat, lon = _load_course_coordinates(
        course_id,
        config.get("course_latitude"),
        config.get("course_longitude"),
    )
    forecast = fetch_event_weather_forecast(
        lat, lon, get_round_dates(), timezone="auto"
    )
    if not forecast:
        raise PipelineFailure("Open-Meteo event weather forecast is unavailable")
    for rnd in range(target_round, 5):
        values = forecast["rounds"].get(rnd) or {}
        if len(values.get("wind") or []) != 15 or len(values.get("dew") or []) != 15:
            raise PipelineFailure(
                f"Open-Meteo returned incomplete R{rnd} weather "
                f"(wind={len(values.get('wind') or [])}, "
                f"dew={len(values.get('dew') or [])})"
            )
        ai_hours = int((forecast.get("ai_hours_by_round") or {}).get(rnd, 0))
        if ai_hours < 12:
            raise PipelineFailure(
                f"Open-Meteo AI wind coverage for R{rnd} is incomplete "
                f"({ai_hours}/15 hours)"
            )
    print(
        f"  Weather ready: {forecast['provider']} "
        f"({forecast['ai_hours']} AI-blended hours, "
        f"timezone={forecast['timezone']})"
    )

    if args.dry_run:
        print(
            f"  [dry-run] Would {action} Sheet to completed R{completed_round}, "
            f"write R{target_round}-R4 weather, run live stats, and simulate R{target_round}"
        )
        return 0

    from sheets_storage import (
        update_round_config_params,
        update_round_config_weather,
    )

    target_weather = forecast["rounds"][target_round]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        update_round_config_weather(
            forecast["rounds"], target_round, spreadsheet=spreadsheet
        )
        update_round_config_params(
            {
                "round": completed_round,
                "wind": _format_array(target_weather["wind"]),
                "dew": _format_array(target_weather["dew"]),
                "weather_provider": forecast["provider"],
                "course_timezone": forecast["timezone"],
                "weather_updated_at": now,
                "automation_target_round": target_round,
                "automation_event_id": event_id,
                "automation_status": "running",
                "automation_odds_updated_at": readiness.last_updated,
                "automation_book_counts": ",".join(
                    f"{book}:{count}"
                    for book, count in readiness.counts.items()
                ),
                "automation_message": "",
            },
            notes={
                "round": f"Advanced automatically for R{target_round} markets",
                "wind": f"Automated R{target_round} AI multi-model forecast",
                "dew": f"Automated R{target_round} dewpoint forecast",
                "weather_provider": "Last automated weather source",
                "course_timezone": "Course-local timezone resolved from lat/lon",
                "automation_status": "Managed by midweek_round_automation.py",
            },
            spreadsheet=spreadsheet,
        )
        print(
            f"  Sheet transition {action}: completed round R{completed_round}, "
            f"target sim R{target_round}"
        )

        _run(
            [sys.executable, "live_stats_engine.py", "--automation"],
            "live_stats_engine.py --automation",
        )
        _verify_predictions(target_round)
        sim_started_at = datetime.now(timezone.utc).timestamp()
        _run([sys.executable, "round_sim.py"], "round_sim.py")
        _verify_outputs(target_round, tourney, started_at=sim_started_at)
        completion_message = (
            f"R{target_round} predictions, simulation, and fairs completed"
        )
        if pin_high_warning:
            completion_message += f"; optional pin-high disabled: {pin_high_warning}"
        _mark_status(
            spreadsheet,
            event_id,
            target_round,
            "complete",
            completion_message,
        )
    except Exception as exc:
        try:
            _mark_status(
                spreadsheet, event_id, target_round, "failed", str(exc)
            )
        except Exception as status_exc:
            print(f"  Could not persist failed status: {status_exc}")
        if isinstance(exc, PipelineFailure):
            raise
        raise PipelineFailure(str(exc)) from exc

    print(f"\n  Midweek automation complete for {tourney} R{target_round}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Advance and run the next-round live golf pipeline"
    )
    parser.add_argument(
        "--odds-file",
        default=str(DEFAULT_ODDS_FILE),
        help="Path to round_matchups_latest.json",
    )
    parser.add_argument(
        "--min-book-matchups",
        type=int,
        default=int(os.getenv("MIN_NEXT_ROUND_MATCHUPS_PER_BOOK", "5")),
        help="Minimum valid event/round matchups required from each sharp book",
    )
    parser.add_argument(
        "--min-datagolf-rows",
        type=int,
        default=int(os.getenv("MIN_DATAGOLF_READY_ROWS", "10")),
        help="Minimum completed stats and posted tee-time rows",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=float(os.getenv("MAX_NEXT_ROUND_ODDS_AGE_HOURS", "3")),
        help="Maximum accepted age of the odds-board payload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all preflight checks without changing the Sheet or running sims",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help=(
            "Run only the cheap odds/Sheet gate; exit 0 when a full run is "
            "needed and 78 when this dispatch is a normal no-op"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the current completed target round during a manual repair run",
    )
    args = parser.parse_args()

    try:
        if args.gate_only:
            gate = _load_gate_context(args)
            if gate["action"] == "complete":
                raise NotReady(
                    f"R{gate['readiness'].target_round} automation is already complete"
                )
            print(
                f"\n  Gate ready: run the full R{gate['readiness'].target_round} "
                "midweek pipeline"
            )
            return 0
        return run_pipeline(args)
    except NotReady as exc:
        print(f"\n  Not ready: {exc}")
        return 78 if args.gate_only else 0
    except PipelineFailure as exc:
        print(f"\n  FAILED: {exc}")
        return 1
    except Exception as exc:
        print(f"\n  UNEXPECTED FAILURE: {exc}")
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

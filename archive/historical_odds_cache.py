"""Resumable local cache for DataGolf historical odds and archived predictions.

The downloader is deliberately separate from production code.  It is the only
piece that touches the network or writes files; ``HistoricalOddsCacheReader``
is read-only and is the intended interface for backtests.

Examples
--------
Smoke-test one completed event (four API requests at most)::

    python archive/historical_odds_cache.py --cache-dir PATH --smoke

Build the full default 2023-08-22 through 2026-08-22 cache::

    python archive/historical_odds_cache.py --cache-dir PATH

The API key is read from ``DATAGOLF_API_KEY`` (and, for the CLI only, ``.env``).
It is never accepted as a command-line argument, written to disk, or included
in exceptions/log messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = 1
DATAGOLF_BASE = "https://feeds.datagolf.com"
DEFAULT_START_DATE = date(2023, 8, 22)
DEFAULT_END_DATE = date(2026, 8, 22)
DEFAULT_BOOKS = ("betonline", "betcris", "pinnacle")
KNOWN_UNSUPPORTED_BOOKS = {
    # Verified against historical-odds/matchups on 2026-08-22. DataGolf uses
    # other published book codes; the literal ``bookmaker`` returns HTTP 400.
    "bookmaker": "DataGolf historical-odds API rejects this book code (HTTP 400)",
}
DEFAULT_OUTRIGHT_MARKETS = (
    "win",
    "top_5",
    "top_10",
    "top_20",
    "make_cut",
    "mc",
)
# The published limit is 45 requests/minute across all DataGolf endpoints.
# Forty-two leaves headroom for another process using the same account.
DEFAULT_REQUESTS_PER_MINUTE = 42
MAX_REQUESTS_PER_MINUTE = 44


class CacheError(RuntimeError):
    """Base error for cache download/read failures."""


class ConfigurationError(CacheError):
    """Raised for invalid configuration without exposing any credential."""


class DataGolfDownloadError(CacheError):
    """Raised for a sanitized DataGolf request failure."""


class RateLimitError(DataGolfDownloadError):
    """Raised on 403/429; callers must stop rather than retry the suspension."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if abs(seconds) > 10_000_000_000:
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _coerce_number(value: Any) -> Any:
    if value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return value


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _json_copy(value: Any) -> Any:
    """Return an independent JSON-compatible copy of an API value."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _record_id(*parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_slug(value: Any) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-.")
    return slug or "none"


@dataclass(frozen=True)
class RequestPlan:
    family: str
    endpoint: str
    year: int | None = None
    event_id: str | None = None
    book: str | None = None
    market: str | None = None
    tour: str | None = "pga"

    def safe_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "file_format": "json",
        }
        if self.tour is not None:
            params["tour"] = self.tour
        if self.year is not None:
            params["year"] = self.year
        if self.event_id is not None:
            params["event_id"] = str(self.event_id)
        if self.book is not None:
            params["book"] = self.book
        if self.market is not None:
            params["market"] = self.market
        if self.family in {"matchups", "outrights"}:
            params["odds_format"] = "american"
        if self.family == "predictions":
            params["odds_format"] = "percent"
        return params

    @property
    def request_id(self) -> str:
        pieces = [self.family]
        for value in (self.year, self.event_id, self.book, self.market):
            if value is not None:
                pieces.append(_safe_slug(value))
        return "__".join(pieces)


class PacedRateLimiter:
    """Evenly space requests below DataGolf's global 45/minute limit."""

    def __init__(
        self,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= requests_per_minute <= MAX_REQUESTS_PER_MINUTE:
            raise ConfigurationError(
                f"requests_per_minute must be between 1 and {MAX_REQUESTS_PER_MINUTE}"
            )
        self.requests_per_minute = requests_per_minute
        self.minimum_interval = 60.0 / requests_per_minute
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: float | None = None

    def acquire(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = self.minimum_interval - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
                now = self._monotonic()
        self._last_request_at = now


class DataGolfClient:
    """Small credential-safe HTTP client used only by the downloader."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: Any | None = None,
        limiter: PacedRateLimiter | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        self._api_key = api_key or os.getenv("DATAGOLF_API_KEY")
        if not self._api_key:
            raise ConfigurationError("DATAGOLF_API_KEY is not set")
        # ``None`` delegates transport creation to api_utils; injected sessions
        # are used by deterministic tests.
        self._session = session
        self._limiter = limiter or PacedRateLimiter()
        self._timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return "DataGolfClient(api_key=<redacted>)"

    def fetch_json(self, plan: RequestPlan) -> Any:
        # Keep api_utils lazy so the read-only cache reader can be imported and
        # used without initializing any production configuration dependencies.
        try:
            from api_utils import DataGolfRequestError, fetch_datagolf_json
        except ModuleNotFoundError as exc:
            if exc.name != "api_utils":
                raise
            # Direct ``python archive/...py`` execution places archive/ rather
            # than the repository root on sys.path.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from api_utils import DataGolfRequestError, fetch_datagolf_json

        self._limiter.acquire()
        try:
            return fetch_datagolf_json(
                plan.endpoint,
                params=plan.safe_params(),
                api_key=self._api_key,
                session=self._session,
                timeout=self._timeout_seconds,
            )
        except DataGolfRequestError as exc:
            status = exc.status_code
            if status in (403, 429):
                raise RateLimitError(
                    f"DataGolf rate-limited {plan.request_id} (HTTP {status}); stop and resume later"
                ) from None
            raise DataGolfDownloadError(
                f"DataGolf request failed for {plan.request_id}"
            ) from None


def iter_event_dicts(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield event objects from both list and event-id-keyed API responses.

    DataGolf's ``event_id=all`` response is a dict keyed by event id (for
    example ``{"14": {"odds": [...]}}``), while single-event responses are
    ordinary event dicts.  This adapter intentionally handles both.
    """
    if isinstance(payload, list):
        if payload and all(
            isinstance(item, Mapping)
            and any(key in item for key in ("p1_dg_id", "dg_id", "player_name"))
            for item in payload
        ):
            yield {"odds": _json_copy(payload)}
            return
        for item in payload:
            if isinstance(item, Mapping):
                if "odds" in item or "event_id" in item:
                    yield dict(item)
                else:
                    yield from iter_event_dicts(item)
        return
    if not isinstance(payload, Mapping):
        return

    if "odds" in payload:
        yield dict(payload)
        return
    for envelope in ("data", "events", "results"):
        value = payload.get(envelope)
        if isinstance(value, (Mapping, list)):
            yield from iter_event_dicts(value)
            return

    for key, value in payload.items():
        if isinstance(value, Mapping):
            event = dict(value)
            event.setdefault("event_id", key)
            if "odds" in event or "event_name" in event:
                yield event
            else:
                yield from iter_event_dicts({key: event})
        elif isinstance(value, list):
            yield {"event_id": key, "odds": _json_copy(value)}


def _odds_list(event: Mapping[str, Any], *, market: str | None, book: str | None) -> list[dict[str, Any]]:
    candidates = [event.get("odds")]
    if market:
        candidates.append(event.get(market))
    for candidate in candidates:
        if isinstance(candidate, list):
            return [dict(row) for row in candidate if isinstance(row, Mapping)]
        if isinstance(candidate, Mapping):
            for key in (book, market, "odds", "data"):
                if key is not None and isinstance(candidate.get(key), list):
                    return [
                        dict(row) for row in candidate[key] if isinstance(row, Mapping)
                    ]
    return []


def _event_date(event: Mapping[str, Any], row: Mapping[str, Any]) -> date | None:
    for value in (
        event.get("event_completed"),
        event.get("event_date"),
        event.get("end_date"),
        event.get("start_date"),
        row.get("close_time"),
        row.get("open_time"),
    ):
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def _in_window(
    event: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    query_year: int | None,
    start_date: date,
    end_date: date,
) -> bool:
    observed = _event_date(event, row)
    if observed is not None:
        return start_date <= observed <= end_date
    # Missing dates are safe only when the entire queried calendar year lies
    # inside the requested window. Boundary-year unknowns are excluded.
    if query_year is None:
        return False
    return date(query_year, 1, 1) >= start_date and date(query_year, 12, 31) <= end_date


def normalize_matchups(
    payload: Any,
    plan: RequestPlan,
    *,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in iter_event_dicts(payload):
        event_meta = {key: value for key, value in event.items() if key != "odds"}
        for source in _odds_list(event, market=plan.market, book=plan.book):
            if not _in_window(
                event, source, query_year=plan.year,
                start_date=start_date, end_date=end_date,
            ):
                continue
            event_id = _first(event, "event_id", default=plan.event_id)
            book = _first(source, "book", default=_first(event, "book", default=plan.book))
            bet_type = _first(source, "bet_type", "market", default=plan.market)
            normalized: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "matchup",
                "tour": plan.tour or "pga",
                "query_year": plan.year,
                "event_id": None if event_id is None else str(event_id),
                "event_name": _first(event, "event_name", "name"),
                "event_completed": _first(event, "event_completed", "event_date", "end_date"),
                "season": _coerce_number(_first(event, "season", "year", default=plan.year)),
                "book": book,
                "bet_type": bet_type,
                "open_time": _first(source, "open_time", "open_timestamp"),
                "close_time": _first(source, "close_time", "close_timestamp"),
                "tie_rule": _first(source, "tie_rule", default=event.get("tie_rule")),
            }
            for side in ("p1", "p2"):
                normalized[f"{side}_dg_id"] = _first(source, f"{side}_dg_id", f"{side}_id")
                normalized[f"{side}_player_name"] = _first(
                    source, f"{side}_player_name", f"{side}_name"
                )
                normalized[f"{side}_open"] = _coerce_number(
                    _first(source, f"{side}_open", f"{side}_open_odds")
                )
                normalized[f"{side}_close"] = _coerce_number(
                    _first(source, f"{side}_close", f"{side}_close_odds")
                )
                normalized[f"{side}_outcome"] = _coerce_number(
                    _first(source, f"{side}_outcome")
                )
                normalized[f"{side}_outcome_text"] = _first(
                    source, f"{side}_outcome_text"
                )
            normalized["record_id"] = _record_id(
                "matchup", plan.year, normalized["event_id"], book, bet_type,
                normalized["open_time"], normalized["close_time"],
                normalized["p1_dg_id"], normalized["p2_dg_id"],
            )
            normalized["source_event"] = _json_copy(event_meta)
            normalized["source_odds"] = _json_copy(source)
            rows.append(normalized)
    return rows


def normalize_outrights(
    payload: Any,
    plan: RequestPlan,
    *,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in iter_event_dicts(payload):
        event_meta = {key: value for key, value in event.items() if key != "odds"}
        for source in _odds_list(event, market=plan.market, book=plan.book):
            if not _in_window(
                event, source, query_year=plan.year,
                start_date=start_date, end_date=end_date,
            ):
                continue
            event_id = _first(event, "event_id", default=plan.event_id)
            book = _first(source, "book", default=_first(event, "book", default=plan.book))
            market = _first(source, "market", default=_first(event, "market", default=plan.market))
            normalized: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "outright",
                "tour": plan.tour or "pga",
                "query_year": plan.year,
                "event_id": None if event_id is None else str(event_id),
                "event_name": _first(event, "event_name", "name"),
                "event_completed": _first(event, "event_completed", "event_date", "end_date"),
                "season": _coerce_number(_first(event, "season", "year", default=plan.year)),
                "book": book,
                "market": market,
                "dg_id": _first(source, "dg_id", "player_id"),
                "player_name": _first(source, "player_name", "name"),
                "open_time": _first(source, "open_time", "open_timestamp"),
                "close_time": _first(source, "close_time", "close_timestamp"),
                "open_odds": _coerce_number(_first(source, "open_odds", "open")),
                "close_odds": _coerce_number(_first(source, "close_odds", "close")),
                "outcome": _first(source, "outcome", "fin_text"),
                "bet_outcome_numeric": _coerce_number(
                    _first(source, "bet_outcome_numeric", "outcome_numeric")
                ),
                "bet_outcome_text": _first(source, "bet_outcome_text", "outcome_text"),
            }
            normalized["record_id"] = _record_id(
                "outright", plan.year, normalized["event_id"], book, market,
                normalized["dg_id"], normalized["open_time"], normalized["close_time"],
            )
            normalized["source_event"] = _json_copy(event_meta)
            normalized["source_odds"] = _json_copy(source)
            rows.append(normalized)
    return rows


def normalize_predictions(
    payload: Any,
    plan: RequestPlan,
    *,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    event = dict(payload)
    if not _in_window(
        event, {}, query_year=plan.year,
        start_date=start_date, end_date=end_date,
    ):
        return []
    event_id = _first(event, "event_id", default=plan.event_id)
    event_meta = {
        key: value
        for key, value in event.items()
        if key not in {"baseline", "baseline_history_fit"}
    }
    rows: list[dict[str, Any]] = []
    for model in ("baseline", "baseline_history_fit"):
        model_rows = event.get(model)
        if not isinstance(model_rows, list):
            continue
        for source_value in model_rows:
            if not isinstance(source_value, Mapping):
                continue
            source = dict(source_value)
            normalized: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "pre_tournament_prediction",
                "model": model,
                "tour": "pga",
                "query_year": plan.year,
                "event_id": None if event_id is None else str(event_id),
                "event_name": _first(event, "event_name", "name"),
                "event_completed": _first(event, "event_completed", "event_date", "end_date"),
                "models_available": _json_copy(event.get("models_available", [])),
                "dg_id": _first(source, "dg_id", "player_id"),
                "player_name": _first(source, "player_name", "name"),
                "finish_text": _first(source, "fin_text", "finish_text", "outcome"),
                "outcome": _first(source, "fin_text", "finish_text", "outcome"),
            }
            # Preserve every archived probability column at top level for easy
            # DataFrame use. The untouched source row remains nested as well.
            for key, value in source.items():
                if key not in {"dg_id", "player_name", "name", "fin_text", "finish_text"}:
                    normalized[key] = _coerce_number(value)
            normalized["record_id"] = _record_id(
                "prediction", plan.year, normalized["event_id"], model, normalized["dg_id"]
            )
            normalized["source_event"] = _json_copy(event_meta)
            normalized["source_player"] = _json_copy(source)
            rows.append(normalized)
    return rows


def normalize_event_list(payload: Any, plan: RequestPlan) -> list[dict[str, Any]]:
    source_rows: Iterable[Any]
    if isinstance(payload, list):
        source_rows = payload
    elif isinstance(payload, Mapping):
        for envelope in ("data", "events", "results"):
            if isinstance(payload.get(envelope), list):
                source_rows = payload[envelope]
                break
        else:
            source_rows = payload.values()
    else:
        source_rows = []
    rows: list[dict[str, Any]] = []
    for source_value in source_rows:
        if not isinstance(source_value, Mapping):
            continue
        source = dict(source_value)
        event_id = _first(source, "event_id", "id")
        year = _coerce_number(_first(source, "calendar_year", "year"))
        row = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "event_list",
            "tour": plan.tour or "pga",
            "calendar_year": year,
            "event_id": None if event_id is None else str(event_id),
            "event_name": _first(source, "event_name", "name"),
            "archived_preds": _first(source, "archived_preds"),
            "matchups": _first(source, "matchups"),
            "outrights": _first(source, "outrights"),
            "source_event": _json_copy(source),
        }
        row["record_id"] = _record_id("event_list", year, row["event_id"])
        rows.append(row)
    return rows


def normalize_payload(
    payload: Any,
    plan: RequestPlan,
    *,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    if plan.family == "matchups":
        return normalize_matchups(payload, plan, start_date=start_date, end_date=end_date)
    if plan.family == "outrights":
        return normalize_outrights(payload, plan, start_date=start_date, end_date=end_date)
    if plan.family == "predictions":
        return normalize_predictions(payload, plan, start_date=start_date, end_date=end_date)
    if plan.family == "event_list":
        return normalize_event_list(payload, plan)
    raise ConfigurationError(f"Unsupported cache family: {plan.family}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part-{uuid.uuid4().hex}")
    with part.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part-{uuid.uuid4().hex}")
    with part.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def _shard_paths(cache_dir: Path, plan: RequestPlan, start_date: date, end_date: date) -> tuple[Path, Path]:
    window = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    data_path = cache_dir / plan.family / f"{plan.request_id}__{window}.jsonl"
    meta_path = data_path.with_suffix(data_path.suffix + ".meta.json")
    return data_path, meta_path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _valid_shard(
    data_path: Path,
    meta_path: Path,
    plan: RequestPlan,
    start_date: date,
    end_date: date,
) -> dict[str, Any] | None:
    if not data_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = _read_json(meta_path)
        if (
            meta.get("complete") is not True
            or meta.get("schema_version") != SCHEMA_VERSION
            or meta.get("request_id") != plan.request_id
            or meta.get("safe_params") != plan.safe_params()
            or meta.get("start_date") != start_date.isoformat()
            or meta.get("end_date") != end_date.isoformat()
            or meta.get("sha256") != _sha256(data_path)
        ):
            return None
        return meta
    except (OSError, ValueError, TypeError):
        return None


def _write_shard(
    cache_dir: Path,
    plan: RequestPlan,
    rows: Sequence[Mapping[str, Any]],
    *,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    data_path, meta_path = _shard_paths(cache_dir, plan, start_date, end_date)
    _atomic_jsonl(data_path, rows)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "request_id": plan.request_id,
        "family": plan.family,
        "endpoint": plan.endpoint,
        "safe_params": plan.safe_params(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "row_count": len(rows),
        "sha256": _sha256(data_path),
        "data_path": data_path.relative_to(cache_dir).as_posix(),
        "downloaded_at_utc": _utc_now(),
    }
    _atomic_json(meta_path, meta)
    return meta


def _download_plan(
    cache_dir: Path,
    client: DataGolfClient,
    plan: RequestPlan,
    *,
    start_date: date,
    end_date: date,
) -> tuple[dict[str, Any], bool]:
    data_path, meta_path = _shard_paths(cache_dir, plan, start_date, end_date)
    existing = _valid_shard(data_path, meta_path, plan, start_date, end_date)
    if existing is not None:
        return existing, True
    payload = client.fetch_json(plan)
    rows = normalize_payload(
        payload, plan, start_date=start_date, end_date=end_date
    )
    return _write_shard(
        cache_dir, plan, rows, start_date=start_date, end_date=end_date
    ), False


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise CacheError(f"Invalid JSONL in {path.name} at line {line_number}") from exc
            if isinstance(value, dict):
                yield value


def _eligible_prediction_events(
    event_rows: Sequence[Mapping[str, Any]],
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in event_rows:
        year_value = row.get("calendar_year")
        try:
            year = int(float(year_value))
        except (TypeError, ValueError):
            continue
        if not start_date.year <= year <= end_date.year:
            continue
        if str(row.get("archived_preds", "")).strip().lower() not in {"yes", "true", "1"}:
            continue
        if row.get("event_id") is None:
            continue
        selected.append(dict(row))
    return sorted(selected, key=lambda row: (int(float(row["calendar_year"])), str(row["event_id"])))


def _smoke_event(events: Sequence[Mapping[str, Any]], start_date: date, end_date: date) -> dict[str, Any]:
    interior = [
        event for event in events
        if start_date.year < int(float(event["calendar_year"])) < end_date.year
    ]
    candidates = interior or list(events)
    if not candidates:
        raise CacheError("No archived prediction event is available for the smoke download")
    return dict(candidates[0])


def download_historical_cache(
    cache_dir: str | Path,
    *,
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
    books: Sequence[str] = DEFAULT_BOOKS,
    outright_markets: Sequence[str] = DEFAULT_OUTRIGHT_MARKETS,
    include_matchups: bool = True,
    include_outrights: bool = True,
    include_predictions: bool = True,
    smoke: bool = False,
    client: DataGolfClient | None = None,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    progress_callback: Callable[[int, int, Mapping[str, Any], bool], None] | None = None,
) -> dict[str, Any]:
    """Download/cache the requested window and return its credential-free manifest.

    Completed shards are checksum-validated and skipped, so an interrupted run
    can be restarted safely.  A 403/429 immediately stops the run without
    writing a complete manifest.
    """
    if start_date > end_date:
        raise ConfigurationError("start_date must be on or before end_date")
    clean_books = tuple(dict.fromkeys(str(book).strip().lower() for book in books if str(book).strip()))
    clean_markets = tuple(
        dict.fromkeys(str(market).strip().lower() for market in outright_markets if str(market).strip())
    )
    if (include_matchups or include_outrights) and not clean_books:
        raise ConfigurationError("At least one sportsbook is required")
    if include_outrights and not clean_markets:
        raise ConfigurationError("At least one outright market is required")

    target = Path(cache_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if client is None:
        client = DataGolfClient(
            limiter=PacedRateLimiter(requests_per_minute=requests_per_minute)
        )

    active: list[dict[str, Any]] = []
    downloaded = 0
    resumed = 0

    event_plan = RequestPlan(
        family="event_list",
        endpoint="historical-odds/event-list",
        tour="pga",
    )
    event_meta, was_resumed = _download_plan(
        target, client, event_plan, start_date=start_date, end_date=end_date
    )
    active.append(event_meta)
    resumed += int(was_resumed)
    downloaded += int(not was_resumed)
    event_rows = list(_read_jsonl(target / event_meta["data_path"]))
    prediction_events = _eligible_prediction_events(event_rows, start_date, end_date)

    years = list(range(start_date.year, end_date.year + 1))
    market_event_id = "all"
    if smoke:
        chosen = _smoke_event(prediction_events, start_date, end_date)
        years = [int(float(chosen["calendar_year"]))]
        market_event_id = str(chosen["event_id"])
        prediction_events = [chosen]
        clean_books = clean_books[:1]
        clean_markets = clean_markets[:1]

    plans: list[RequestPlan] = []
    if include_matchups:
        plans.extend(
            RequestPlan(
                family="matchups",
                endpoint="historical-odds/matchups",
                year=year,
                event_id=market_event_id,
                book=book,
                tour="pga",
            )
            for year in years for book in clean_books
        )
    if include_outrights:
        plans.extend(
            RequestPlan(
                family="outrights",
                endpoint="historical-odds/outrights",
                year=year,
                event_id=market_event_id,
                book=book,
                market=market,
                tour="pga",
            )
            for year in years for book in clean_books for market in clean_markets
        )
    if include_predictions:
        plans.extend(
            RequestPlan(
                family="predictions",
                endpoint="preds/pre-tournament-archive",
                year=int(float(event["calendar_year"])),
                event_id=str(event["event_id"]),
                # This endpoint is PGA-only and does not accept tour.
                tour=None,
            )
            for event in prediction_events
        )

    for plan_index, plan in enumerate(plans, start=1):
        meta, was_resumed = _download_plan(
            target, client, plan, start_date=start_date, end_date=end_date
        )
        active.append(meta)
        resumed += int(was_resumed)
        downloaded += int(not was_resumed)
        if progress_callback is not None:
            progress_callback(plan_index, len(plans), meta, was_resumed)

    family_rows: dict[str, int] = {}
    for meta in active:
        family = str(meta["family"])
        family_rows[family] = family_rows.get(family, 0) + int(meta["row_count"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "mode": "smoke" if smoke else "full",
        "tour": "pga",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "books": list(clean_books),
        "known_unsupported_books": KNOWN_UNSUPPORTED_BOOKS,
        "outright_markets": list(clean_markets),
        "created_at_utc": _utc_now(),
        "downloaded_shards": downloaded,
        "resumed_shards": resumed,
        "row_counts": family_rows,
        "active_shards": [
            {
                "request_id": meta["request_id"],
                "family": meta["family"],
                "data_path": meta["data_path"],
                "row_count": meta["row_count"],
                "sha256": meta["sha256"],
            }
            for meta in active
        ],
    }
    _atomic_json(target / "manifest.json", manifest)
    return manifest


class HistoricalOddsCacheReader:
    """Read-only access to one complete cache manifest."""

    FAMILIES = {"event_list", "matchups", "outrights", "predictions"}

    def __init__(self, cache_dir: str | Path, *, verify_checksums: bool = True) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        manifest_path = self.cache_dir / "manifest.json"
        try:
            self.manifest = _read_json(manifest_path)
        except (OSError, ValueError) as exc:
            raise CacheError(f"Cannot read complete cache manifest at {manifest_path}") from exc
        if self.manifest.get("complete") is not True:
            raise CacheError("Cache manifest is incomplete")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise CacheError("Unsupported cache schema version")
        self.verify_checksums = verify_checksums

    def iter_rows(self, family: str, **filters: Any) -> Iterator[dict[str, Any]]:
        if family not in self.FAMILIES:
            raise ConfigurationError(f"Unknown cache family: {family}")
        # Event IDs are deliberately reusable across calendar years. Include
        # query_year in the de-duplication token for compatibility with v1
        # prediction shards whose record_id did not yet encode the year.
        seen: set[tuple[Any, str]] = set()
        for shard in self.manifest.get("active_shards", []):
            if shard.get("family") != family:
                continue
            path = (self.cache_dir / str(shard["data_path"])).resolve()
            try:
                path.relative_to(self.cache_dir)
            except ValueError as exc:
                raise CacheError("Manifest contains a path outside the cache directory") from exc
            if self.verify_checksums and _sha256(path) != shard.get("sha256"):
                raise CacheError(f"Checksum mismatch for cache shard {path.name}")
            for row in _read_jsonl(path):
                record_id = str(row.get("record_id", ""))
                dedupe_token = (row.get("query_year"), record_id)
                if record_id and dedupe_token in seen:
                    continue
                if all(row.get(key) == expected for key, expected in filters.items()):
                    if record_id:
                        seen.add(dedupe_token)
                    yield row

    def rows(self, family: str, **filters: Any) -> list[dict[str, Any]]:
        return list(self.iter_rows(family, **filters))

    def matchups(self, **filters: Any) -> list[dict[str, Any]]:
        return self.rows("matchups", **filters)

    def outrights(self, **filters: Any) -> list[dict[str, Any]]:
        return self.rows("outrights", **filters)

    def predictions(self, **filters: Any) -> list[dict[str, Any]]:
        return self.rows("predictions", **filters)


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _cli_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=_cli_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=_cli_date, default=DEFAULT_END_DATE)
    parser.add_argument("--books", type=_csv_values, default=DEFAULT_BOOKS)
    parser.add_argument(
        "--outright-markets", type=_csv_values, default=DEFAULT_OUTRIGHT_MARKETS
    )
    parser.add_argument("--requests-per-minute", type=int, default=DEFAULT_REQUESTS_PER_MINUTE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-matchups", action="store_true")
    parser.add_argument("--skip-outrights", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Loading .env is intentionally a CLI concern; importing this module does
    # not mutate the process environment.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    except ImportError:
        pass

    def show_progress(index, total, meta, was_resumed):
        if index == 1 or index % 10 == 0 or index == total:
            action = "resumed" if was_resumed else "downloaded"
            print(
                f"[{index}/{total}] {action} {meta['request_id']} "
                f"({meta['row_count']} rows)",
                flush=True,
            )

    try:
        manifest = download_historical_cache(
            args.cache_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            books=args.books,
            outright_markets=args.outright_markets,
            include_matchups=not args.skip_matchups,
            include_outrights=not args.skip_outrights,
            include_predictions=not args.skip_predictions,
            smoke=args.smoke,
            requests_per_minute=args.requests_per_minute,
            progress_callback=show_progress,
        )
    except CacheError as exc:
        print(f"Cache download failed: {exc}")
        return 1
    print(
        f"Cache complete ({manifest['mode']}): "
        f"{manifest['row_counts']} at {Path(args.cache_dir).resolve()}"
    )
    print(
        f"Shards downloaded: {manifest['downloaded_shards']}; "
        f"resumed: {manifest['resumed_shards']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

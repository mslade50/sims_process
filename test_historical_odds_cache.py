import json
from datetime import date

import pytest

from archive.historical_odds_cache import (
    CacheError,
    DataGolfClient,
    DataGolfDownloadError,
    HistoricalOddsCacheReader,
    PacedRateLimiter,
    RequestPlan,
    download_historical_cache,
    iter_event_dicts,
    normalize_matchups,
    normalize_outrights,
    normalize_predictions,
)


START = date(2023, 8, 22)
END = date(2026, 8, 22)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payloads, status_code=200):
        self.payloads = payloads
        self.status_code = status_code
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        endpoint = url.split("feeds.datagolf.com/", 1)[-1]
        return FakeResponse(self.payloads.get(endpoint), self.status_code)


def _matchup_event(event_completed="2024-04-14"):
    return {
        "event_completed": event_completed,
        "event_name": "Masters Tournament",
        "book": "betonline",
        "season": 2024,
        "year": 2024,
        "odds": [
            {
                "bet_type": "72-hole Match",
                "open_time": "2024-04-09 15:00",
                "close_time": "2024-04-11 08:00",
                "tie_rule": "Bets are void in event of tie",
                "p1_dg_id": 1,
                "p1_player_name": "Player, One",
                "p1_open": "-110",
                "p1_close": "-115",
                "p1_outcome": 1.0,
                "p1_outcome_text": "win",
                "p2_dg_id": 2,
                "p2_player_name": "Player, Two",
                "p2_open": "-110",
                "p2_close": "+105",
                "p2_outcome": 0.0,
                "p2_outcome_text": "loss",
            }
        ],
    }


def _outright_event():
    return {
        "event_completed": "2024-04-14",
        "event_name": "Masters Tournament",
        "book": "betonline",
        "market": "win",
        "season": 2024,
        "year": 2024,
        "odds": [
            {
                "dg_id": 1,
                "player_name": "Player, One",
                "open_time": "2024-04-08 12:00",
                "close_time": "2024-04-11 08:00",
                "open_odds": "+1200",
                "close_odds": "+1000",
                "outcome": "1",
                "bet_outcome_numeric": 1,
                "bet_outcome_text": "win",
            }
        ],
    }


def _prediction_event():
    player = {
        "dg_id": 1,
        "player_name": "Player, One",
        "fin_text": "1",
        "win": 8.5,
        "top_5": 30.0,
        "top_10": 50.0,
        "top_20": 70.0,
        "make_cut": 92.0,
        "first_round_leader": 4.0,
    }
    return {
        "event_completed": "2024-04-14",
        "event_id": "14",
        "event_name": "Masters Tournament",
        "models_available": ["baseline", "baseline_history_fit"],
        "baseline": [player],
        "baseline_history_fit": [{**player, "win": 9.0}],
    }


def test_event_id_all_dict_payload_is_normalized_without_losing_settlement_fields():
    payload = {
        "14": _matchup_event(),
        "999": _matchup_event(event_completed="2023-08-20"),
    }
    events = list(iter_event_dicts(payload))
    assert [event["event_id"] for event in events] == ["14", "999"]

    plan = RequestPlan(
        family="matchups",
        endpoint="historical-odds/matchups",
        year=2024,
        event_id="all",
        book="betonline",
    )
    rows = normalize_matchups(payload, plan, start_date=START, end_date=END)

    assert len(rows) == 1
    row = rows[0]
    assert row["event_id"] == "14"
    assert row["open_time"] == "2024-04-09 15:00"
    assert row["close_time"] == "2024-04-11 08:00"
    assert row["tie_rule"] == "Bets are void in event of tie"
    assert row["p1_close"] == -115.0
    assert row["p1_outcome"] == 1.0
    assert row["p1_outcome_text"] == "win"
    assert row["p2_outcome"] == 0.0
    assert row["source_odds"]["p2_outcome_text"] == "loss"


def test_outright_and_both_prediction_models_preserve_outcomes_and_probabilities():
    outright_plan = RequestPlan(
        family="outrights",
        endpoint="historical-odds/outrights",
        year=2024,
        event_id="all",
        book="betonline",
        market="win",
    )
    outright = normalize_outrights(
        {"14": _outright_event()}, outright_plan, start_date=START, end_date=END
    )[0]
    assert outright["close_odds"] == 1000.0
    assert outright["outcome"] == "1"
    assert outright["bet_outcome_numeric"] == 1
    assert outright["bet_outcome_text"] == "win"

    prediction_plan = RequestPlan(
        family="predictions",
        endpoint="preds/pre-tournament-archive",
        year=2024,
        event_id="14",
        tour=None,
    )
    predictions = normalize_predictions(
        _prediction_event(), prediction_plan, start_date=START, end_date=END
    )
    assert {row["model"] for row in predictions} == {
        "baseline",
        "baseline_history_fit",
    }
    baseline = next(row for row in predictions if row["model"] == "baseline")
    history_fit = next(
        row for row in predictions if row["model"] == "baseline_history_fit"
    )
    assert baseline["dg_id"] == 1
    assert baseline["player_name"] == "Player, One"
    assert baseline["finish_text"] == "1"
    assert baseline["win"] == 8.5
    assert history_fit["win"] == 9.0
    assert baseline["source_player"]["make_cut"] == 92.0

    reused_event = _prediction_event()
    reused_event["event_completed"] = "2025-04-13"
    reused_plan = RequestPlan(
        family="predictions",
        endpoint="preds/pre-tournament-archive",
        year=2025,
        event_id="14",
        tour=None,
    )
    next_year = normalize_predictions(
        reused_event, reused_plan, start_date=START, end_date=END
    )
    assert baseline["record_id"] != next_year[0]["record_id"]


def test_rate_limiter_stays_strictly_below_45_requests_per_minute():
    clock = FakeClock()
    limiter = PacedRateLimiter(
        requests_per_minute=42,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    for _ in range(45):
        limiter.acquire()

    # Forty-five calls require 44 evenly-spaced intervals. At 42/minute the
    # 45th call is safely beyond the first rolling minute.
    assert clock.now == pytest.approx(44 * 60 / 42)
    assert len(clock.sleeps) == 44
    with pytest.raises(CacheError):
        PacedRateLimiter(requests_per_minute=45)


def test_smoke_cache_is_atomic_resumable_read_only_and_contains_no_key(tmp_path):
    event_list = [
        {
            "archived_preds": "yes",
            "calendar_year": 2024,
            "event_id": 14,
            "event_name": "Masters Tournament",
            "matchups": "yes",
            "outrights": "yes",
        }
    ]
    payloads = {
        "historical-odds/event-list": event_list,
        "historical-odds/matchups": {"14": _matchup_event()},
        "historical-odds/outrights": {"14": _outright_event()},
        "preds/pre-tournament-archive": _prediction_event(),
    }
    session = FakeSession(payloads)
    clock = FakeClock()
    secret = "never-write-this-secret"
    client = DataGolfClient(
        secret,
        session=session,
        limiter=PacedRateLimiter(monotonic=clock.monotonic, sleep=clock.sleep),
    )

    first = download_historical_cache(
        tmp_path,
        start_date=START,
        end_date=END,
        books=("betonline",),
        outright_markets=("win",),
        smoke=True,
        client=client,
    )
    assert len(session.calls) == 4
    assert first["downloaded_shards"] == 4
    assert first["row_counts"] == {
        "event_list": 1,
        "matchups": 1,
        "outrights": 1,
        "predictions": 2,
    }
    assert "bookmaker" in first["known_unsupported_books"]

    second = download_historical_cache(
        tmp_path,
        start_date=START,
        end_date=END,
        books=("betonline",),
        outright_markets=("win",),
        smoke=True,
        client=client,
    )
    assert len(session.calls) == 4
    assert second["downloaded_shards"] == 0
    assert second["resumed_shards"] == 4

    reader = HistoricalOddsCacheReader(tmp_path)
    assert reader.matchups()[0]["p1_outcome_text"] == "win"
    assert reader.outrights(market="win")[0]["bet_outcome_numeric"] == 1
    assert len(reader.predictions(model="baseline_history_fit")) == 1

    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")
    assert all(call["params"]["key"] == secret for call in session.calls)


def test_download_error_does_not_echo_api_key():
    secret = "do-not-echo"
    session = FakeSession({}, status_code=500)
    client = DataGolfClient(secret, session=session, limiter=PacedRateLimiter())
    plan = RequestPlan(
        family="matchups",
        endpoint="historical-odds/matchups",
        year=2024,
        event_id="all",
        book="betonline",
    )

    with pytest.raises(DataGolfDownloadError) as caught:
        client.fetch_json(plan)

    assert secret not in str(caught.value)
    assert secret not in repr(client)


def test_reader_rejects_tampered_shard(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "schema_version": 1,
                "active_shards": [
                    {
                        "family": "matchups",
                        "data_path": "matchups/tampered.jsonl",
                        "sha256": "not-the-file-hash",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    shard = tmp_path / "matchups" / "tampered.jsonl"
    shard.parent.mkdir()
    shard.write_text("{}\n", encoding="utf-8")

    with pytest.raises(CacheError, match="Checksum mismatch"):
        HistoricalOddsCacheReader(tmp_path).matchups()

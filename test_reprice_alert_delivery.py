"""Offline tests for the cache-free reprice alert-before-storage contract."""

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import reprice
import reprice_core as rc


def _rows():
    return pd.DataFrame([
        {
            "Player 1": "alpha & one",
            "Player 2": "bravo",
            "Bookmaker": "pinnacle",
            "bet_on": "alpha & one",
            "edge_on": 8.2,
            "P1 Odds": 120,
            "P2 Odds": -140,
            "Fair_p1": -105,
            "Fair_p2": -115,
        },
        {
            "Player 1": "charlie",
            "Player 2": "delta",
            "Bookmaker": "fanduel",
            "bet_on": "charlie",
            "edge_on": 9.1,
            "P1 Odds": 110,
            "P2 Odds": -130,
            "Fair_p1": -110,
            "Fair_p2": -110,
        },
    ])


class _Response:
    def __init__(self, status_code=200, payload=None, json_error=False):
        self.status_code = status_code
        self.payload = {"ok": True, "result": {"message_id": 1}} if payload is None else payload
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise ValueError("not json")
        return self.payload


class TelegramTransportTests(unittest.TestCase):
    def test_required_delivery_rejects_missing_credentials_without_network(self):
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "",
        }), patch.object(rc.requests, "post") as post:
            with self.assertRaises(rc.TelegramDeliveryError):
                rc.send_telegram("bet", required=True)
        post.assert_not_called()

    def test_optional_diagnostic_returns_false_when_credentials_missing(self):
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "",
        }), patch.object(rc.requests, "post") as post:
            self.assertFalse(rc.send_telegram("diagnostic"))
        post.assert_not_called()

    def test_required_delivery_checks_http_status(self):
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "secret-token",
            "TELEGRAM_CHAT_ID": "123",
        }), patch.object(rc.requests, "post", return_value=_Response(status_code=503)):
            with self.assertRaisesRegex(rc.TelegramDeliveryError, "HTTP 503"):
                rc.send_telegram("bet", required=True)

    def test_required_delivery_checks_api_ok(self):
        response = _Response(payload={"ok": False, "description": "Bad Request"})
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "secret-token",
            "TELEGRAM_CHAT_ID": "123",
        }), patch.object(rc.requests, "post", return_value=response):
            with self.assertRaisesRegex(rc.TelegramDeliveryError, "ok=false"):
                rc.send_telegram("bet", required=True)

    def test_required_delivery_checks_valid_json(self):
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "secret-token",
            "TELEGRAM_CHAT_ID": "123",
        }), patch.object(
            rc.requests, "post", return_value=_Response(json_error=True)
        ):
            with self.assertRaisesRegex(rc.TelegramDeliveryError, "invalid JSON"):
                rc.send_telegram("bet", required=True)

    def test_request_failure_does_not_expose_bot_token(self):
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "never-print-this-token",
            "TELEGRAM_CHAT_ID": "123",
        }), patch.object(rc.requests, "post", side_effect=RuntimeError("request URL")):
            with self.assertRaises(rc.TelegramDeliveryError) as caught:
                rc.send_telegram("bet", required=True)
        self.assertNotIn("never-print-this-token", str(caught.exception))

    def test_matchup_alert_reports_success_only_after_api_acceptance(self):
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "secret-token",
            "TELEGRAM_CHAT_ID": "123",
            "TELEGRAM_ROUND_BETS_CHAT_ID": "456",
        }), patch.object(
            rc.requests, "post", return_value=_Response()
        ) as post:
            delivered = rc.send_matchup_alert(_rows().iloc[[0]], 4, "test_event")

        self.assertEqual(delivered, 1)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_id"], "456")
        self.assertIn("alpha &amp; one", payload["text"])
        self.assertNotIn("alpha & one", payload["text"])


class AlertPartitionAndRetryTests(unittest.TestCase):
    def test_partition_requires_only_fresh_sharp_rows(self):
        rows = _rows()
        must_alert, no_alert = rc.partition_matchup_alert_rows(rows, set())
        self.assertEqual(must_alert["Bookmaker"].tolist(), ["pinnacle"])
        self.assertEqual(no_alert["Bookmaker"].tolist(), ["fanduel"])

        seen = {rc.alerted_key("alpha & one", "bravo", "alpha & one")}
        must_alert, no_alert = rc.partition_matchup_alert_rows(rows, seen)
        self.assertTrue(must_alert.empty)
        self.assertEqual(len(no_alert), 2)

    def test_soft_stored_row_cannot_suppress_sharp_retry(self):
        records = [{
            "event_id": "77",
            "round": 4,
            "player_1": "alpha & one",
            "player_2": "bravo",
            "bookmaker": "fanduel",
            "p1_odds": 120,
            "p2_odds": -140,
            "bet_on": "alpha & one",
        }]
        ws = Mock()
        ws.get_all_records.return_value = records
        fake_storage = types.ModuleType("sheets_storage")
        fake_storage.TAB_ROUND_MU = "Round Matchups"
        fake_storage.ROUND_MU_HEADERS = []
        fake_storage._get_or_create_tab = lambda *_args, **_kwargs: ws

        sharp = _rows().iloc[[0]].copy()
        with patch.dict(sys.modules, {"sheets_storage": fake_storage}):
            new_rows, seen = rc.dedup_round_matchups(sharp, object(), "77", 4)

        self.assertEqual(len(new_rows), 1)
        self.assertEqual(seen, set())
        retry_alert, _ = rc.partition_matchup_alert_rows(new_rows, seen)
        self.assertEqual(len(retry_alert), 1)

    def test_existing_sharp_row_does_suppress_repeat_alert(self):
        records = [{
            "event_id": "77",
            "round": 4,
            "player_1": "alpha & one",
            "player_2": "bravo",
            "bookmaker": "pinnacle",
            "p1_odds": 115,
            "p2_odds": -135,
            "bet_on": "alpha & one",
        }]
        ws = Mock()
        ws.get_all_records.return_value = records
        fake_storage = types.ModuleType("sheets_storage")
        fake_storage.TAB_ROUND_MU = "Round Matchups"
        fake_storage.ROUND_MU_HEADERS = []
        fake_storage._get_or_create_tab = lambda *_args, **_kwargs: ws

        sharp_price_move = _rows().iloc[[0]].copy()
        with patch.dict(sys.modules, {"sheets_storage": fake_storage}):
            new_rows, seen = rc.dedup_round_matchups(sharp_price_move, object(), "77", 4)

        self.assertEqual(len(new_rows), 1)
        retry_alert, no_alert = rc.partition_matchup_alert_rows(new_rows, seen)
        self.assertTrue(retry_alert.empty)
        self.assertEqual(len(no_alert), 1)


class SideEffectOrderingTests(unittest.TestCase):
    def _kwargs(self, rc_module, store_func):
        return dict(
            sim_round=4,
            tourney="test_event",
            event_id="77",
            dg_id_lookup={},
            spreadsheet=object(),
            rc_module=rc_module,
            store_func=store_func,
        )

    def test_successful_sharp_delivery_precedes_storage(self):
        calls = []
        fake_rc = types.SimpleNamespace(
            TelegramDeliveryError=rc.TelegramDeliveryError,
            partition_matchup_alert_rows=rc.partition_matchup_alert_rows,
            send_matchup_alert=lambda *args, **kwargs: calls.append("alert") or 1,
        )

        def store(frame, *_args, **_kwargs):
            calls.append(("store", frame["Bookmaker"].tolist()))

        stored, alerted = reprice._deliver_then_store_matchups(
            _rows(), set(), **self._kwargs(fake_rc, store)
        )
        self.assertEqual(calls, ["alert", ("store", ["pinnacle", "fanduel"])])
        self.assertEqual((stored, alerted), (2, 1))

    def test_failed_delivery_stores_only_non_alert_rows_and_raises(self):
        stored_books = []

        def fail_alert(*_args, **_kwargs):
            raise rc.TelegramDeliveryError("no delivery")

        fake_rc = types.SimpleNamespace(
            TelegramDeliveryError=rc.TelegramDeliveryError,
            partition_matchup_alert_rows=rc.partition_matchup_alert_rows,
            send_matchup_alert=fail_alert,
        )

        def store(frame, *_args, **_kwargs):
            stored_books.extend(frame["Bookmaker"].tolist())

        with self.assertRaises(rc.TelegramDeliveryError):
            reprice._deliver_then_store_matchups(
                _rows(), set(), **self._kwargs(fake_rc, store)
            )
        self.assertEqual(stored_books, ["fanduel"])

    def test_soft_only_batch_stores_without_telegram_credentials(self):
        calls = []
        fake_rc = types.SimpleNamespace(
            TelegramDeliveryError=rc.TelegramDeliveryError,
            partition_matchup_alert_rows=rc.partition_matchup_alert_rows,
            send_matchup_alert=lambda *_args, **_kwargs: calls.append("unexpected alert"),
        )

        def store(frame, *_args, **_kwargs):
            calls.append(("store", frame["Bookmaker"].tolist()))

        stored, alerted = reprice._deliver_then_store_matchups(
            _rows().iloc[[1]], set(), **self._kwargs(fake_rc, store)
        )
        self.assertEqual(calls, [("store", ["fanduel"])])
        self.assertEqual((stored, alerted), (1, 0))


if __name__ == "__main__":
    unittest.main()

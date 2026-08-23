"""Offline tests for the cache-free reprice alert-before-storage contract."""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import reprice
import reprice_core as rc
import round_sim
import sheets_storage


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


def _score_rows(side="Under"):
    return pd.DataFrame([{
        "Player": "alpha & one",
        "Line": 68.5,
        "Book": "fanduel",
        "Best_Side": side,
        "Mkt_Under": -110,
        "Mkt_Over": -110,
        "Best_Edge": 7.0,
    }])


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
        fake_storage.is_excluded_or_invalid_result = lambda value: False

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
        fake_storage.is_excluded_or_invalid_result = lambda value: False

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

    def test_health_is_rechecked_at_delivery_and_storage_boundaries(self):
        calls = []
        fake_rc = types.SimpleNamespace(
            TelegramDeliveryError=rc.TelegramDeliveryError,
            partition_matchup_alert_rows=rc.partition_matchup_alert_rows,
            send_matchup_alert=lambda *_args, **_kwargs: calls.append("alert") or 1,
        )

        def store(*_args, **_kwargs):
            calls.append("store")

        reprice._deliver_then_store_matchups(
            _rows().iloc[[0]],
            set(),
            health_check=lambda: calls.append("health"),
            **self._kwargs(fake_rc, store),
        )

        self.assertEqual(calls, ["health", "alert", "health", "store"])

    def test_failed_health_gate_stores_neither_sharp_nor_soft_rows(self):
        calls = []
        fake_rc = types.SimpleNamespace(
            TelegramDeliveryError=rc.TelegramDeliveryError,
            partition_matchup_alert_rows=rc.partition_matchup_alert_rows,
            send_matchup_alert=lambda *_args, **_kwargs: calls.append("alert") or 1,
        )

        def fail_health():
            calls.append("health")
            raise RuntimeError("unhealthy artifact")

        with self.assertRaisesRegex(RuntimeError, "unhealthy artifact"):
            reprice._deliver_then_store_matchups(
                _rows(),
                set(),
                health_check=fail_health,
                **self._kwargs(
                    fake_rc,
                    lambda *_args, **_kwargs: calls.append("store"),
                ),
            )

        self.assertEqual(calls, ["health"])


class LegacyScoreDedupTests(unittest.TestCase):
    @staticmethod
    def _worksheet(record):
        ws = Mock()
        ws.get_all_records.return_value = [record]
        return ws

    def test_score_side_flip_is_a_new_alertable_row(self):
        existing = {
            "event_id": "77", "round": 4, "player": "alpha & one",
            "line": 68.5, "book": "fanduel", "best_side": "Under",
            "mkt_under": -110, "mkt_over": -110, "result": "",
        }
        with patch.object(
            sheets_storage,
            "_get_or_create_tab",
            return_value=self._worksheet(existing),
        ):
            fresh = round_sim._dedup_score_edges(
                _score_rows("Over"), object(), "77", 4
            )

        self.assertEqual(fresh["Best_Side"].tolist(), ["Over"])

    def test_excluded_score_row_does_not_suppress_corrected_retry(self):
        existing = {
            "event_id": "77", "round": 4, "player": "alpha & one",
            "line": 68.5, "book": "fanduel", "best_side": "Under",
            "mkt_under": -110, "mkt_over": -110,
            "result": "excluded_invalid_model",
        }
        with patch.object(
            sheets_storage,
            "_get_or_create_tab",
            return_value=self._worksheet(existing),
        ):
            fresh = round_sim._dedup_score_edges(
                _score_rows("Under"), object(), "77", 4
            )

        self.assertEqual(len(fresh), 1)


class LegacyRoundSimRepriceTests(unittest.TestCase):
    def _patched_dependencies(self, *, alert_side_effect, calls):
        sharp = _rows().iloc[[0]].copy()
        empty = pd.DataFrame()
        patches = [
            patch.object(round_sim, "_dedup_round_matchups", return_value=(sharp, set())),
            patch.object(round_sim, "_dedup_score_edges", return_value=empty),
            patch.object(round_sim, "_dedup_finish_positions", return_value=empty),
            patch.object(
                round_sim,
                "_send_reprice_alert",
                side_effect=alert_side_effect,
            ),
            patch.object(sheets_storage, "get_spreadsheet", return_value=object()),
            patch.object(sheets_storage, "load_dg_id_lookup", return_value={}),
            patch.object(
                sheets_storage,
                "store_round_matchups",
                side_effect=lambda *_args, **_kwargs: calls.append("store_matchup"),
            ),
            patch.object(
                sheets_storage,
                "store_score_edges",
                side_effect=lambda *_args, **_kwargs: calls.append("store_score"),
            ),
            patch.object(
                sheets_storage,
                "store_finish_positions",
                side_effect=lambda *_args, **_kwargs: calls.append("store_outright"),
            ),
        ]
        return patches

    def test_legacy_delivery_failure_propagates_before_any_storage(self):
        calls = []

        def fail_delivery(*_args, **_kwargs):
            calls.append("alert")
            raise rc.TelegramDeliveryError("not accepted")

        patches = self._patched_dependencies(
            alert_side_effect=fail_delivery, calls=calls
        )
        for active in patches:
            active.start()
            self.addCleanup(active.stop)

        with self.assertRaises(rc.TelegramDeliveryError):
            round_sim._reprice_store_and_alert(
                _rows(), pd.DataFrame(), 4, "test_event", "77",
                health_check=lambda: calls.append("health"),
            )

        self.assertEqual(calls, ["health", "alert"])

    def test_legacy_successful_delivery_precedes_storage(self):
        calls = []
        patches = self._patched_dependencies(
            alert_side_effect=lambda *_args, **_kwargs: calls.append("alert") or 1,
            calls=calls,
        )
        for active in patches:
            active.start()
            self.addCleanup(active.stop)

        round_sim._reprice_store_and_alert(
            _rows(), pd.DataFrame(), 4, "test_event", "77",
            health_check=lambda: calls.append("health"),
        )

        self.assertEqual(
            calls, ["health", "alert", "health", "store_matchup"]
        )

    def test_legacy_sender_uses_required_telegram_transport(self):
        with patch.object(
            rc,
            "send_telegram",
            side_effect=rc.TelegramDeliveryError("rejected"),
        ) as send:
            with self.assertRaises(rc.TelegramDeliveryError):
                round_sim._send_reprice_alert(
                    _rows().iloc[[0]], pd.DataFrame(), 4, "test_event"
                )
        self.assertTrue(send.call_args.kwargs["required"])

    def test_legacy_entrypoint_does_not_swallow_reprice_failure(self):
        source = (Path(__file__).resolve().parent / "round_sim.py").read_text(
            encoding="utf-8"
        )
        marker = source.index("# ── Step 6a: --reprice exits here")
        block = source[marker:source.index("# ── Step 6: Email", marker)]
        call = block.index("_reprice_store_and_alert(")
        assert block.index("if args.dry_run:") < call
        assert "except Exception" not in block[:call + len("_reprice_store_and_alert(")]


if __name__ == "__main__":
    unittest.main()

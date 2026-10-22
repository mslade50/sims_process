"""--live fails closed unless DataGolf confirms the configured event.

Synthetic orders only. The fake client never opens a socket and has no
order-placement method that the pull is allowed to call.
"""
import os

os.environ.setdefault("COEFFS_FROM_CACHE", "1")


class FakeOrdersClient:
    """Scripted portfolio client. Records GET and DELETE. POST raises."""

    def __init__(self, orders, delete_status=200, drop_on_success=True,
                 list_error=None, delete_error=None, verify_error=None):
        self.orders = [dict(order) for order in orders]
        self.calls = []
        self.delete_status = delete_status
        self.drop_on_success = drop_on_success
        self.list_error = list_error
        self.delete_error = delete_error
        self.verify_error = verify_error
        self.lists = 0
        self.deletes = 0

    def request(self, method, path):
        self.calls.append((method, path))
        if method == "POST":
            raise AssertionError("quote pull must not place orders")
        if method == "GET":
            self.lists += 1
            if self.lists == 1 and self.list_error is not None:
                raise self.list_error
            if self.lists > 1 and self.verify_error is not None:
                raise self.verify_error
            return [dict(order) for order in self.orders]
        if method == "DELETE":
            self.deletes += 1
            if self.delete_error is not None:
                raise self.delete_error
            order_id = path.rsplit("/", 1)[-1]
            if 200 <= self.delete_status < 300 and self.drop_on_success:
                self.orders = [order for order in self.orders
                               if str(order.get("order_id")) != order_id]
            return self.delete_status
        raise AssertionError(f"unexpected method {method}")


def _script(order_id, ticker="KXPGATOUR-BUBC26-P001"):
    return {
        "order_id": order_id,
        "ticker": ticker,
        "client_order_id": f"smk_{order_id}",
    }


def _manual(order_id="manual-1"):
    return {
        "order_id": order_id,
        "ticker": "KXPGATOUR-BUBC26-P002",
        "client_order_id": "",
    }


def _bind(client):
    import kalshi_maker as maker

    def list_resting():
        return client.request("GET", "/trade-api/v2/portfolio/orders?status=resting")

    def cancel(order_id, ticker, reason=""):
        assert str(ticker).startswith("KXPGA")
        status = client.request(
            "DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
        return 200 <= status < 300

    return maker, list_resting, maker._is_script_order, cancel


def _pull(client):
    import maker_guard
    _maker, list_resting, is_script, cancel = _bind(client)
    return maker_guard.pull_script_quotes(list_resting, is_script, cancel)


def test_pull_retries_a_cancel_exception_then_verifies_clean():
    client = FakeOrdersClient([_script("a"), _manual()], delete_error=RuntimeError("down"))
    calls = {"n": 0}

    def cancel(order_id, ticker, reason=""):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cancel transport down")
        status = client.request("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
        return 200 <= status < 300

    import maker_guard
    maker, list_resting, is_script, _cancel = _bind(client)
    # The first DELETE is the raised attempt; the client's own request is the retry.
    client.delete_error = None
    cancelled, error = maker_guard.pull_script_quotes(list_resting, is_script, cancel)
    assert error is None
    assert cancelled == 1
    assert calls["n"] == 2
    assert not any(method == "POST" for method, _path in client.calls)
    assert all(path.endswith("/a") for method, path in client.calls if method == "DELETE")
    assert maker._is_script_order(_manual()) is False


def test_pull_cancel_exception_exits_nonzero_and_alerts():
    client = FakeOrdersClient(
        [_script("a"), _manual()], delete_error=RuntimeError("cancel transport down"))
    maker, list_resting, is_script, cancel = _bind(client)
    alerts = []
    code = maker.halt_live_run(
        "LIVE EVENT UNCONFIRMED: no DATAGOLF_API_KEY; refusing to quote",
        confirmation_failed=True, list_resting=list_resting, is_script=is_script,
        cancel_order=cancel, alert=alerts.append)
    assert code == 1
    assert alerts
    assert "quote pull failed" in alerts[0]
    assert "still resting after cancel: a" in alerts[0]
    assert client.deletes == 2  # first try and the retry
    assert not any(method == "POST" for method, _path in client.calls)
    assert not any(path.endswith("/manual-1") for _method, path in client.calls)


def test_pull_non_success_cancel_exits_nonzero():
    client = FakeOrdersClient([_script("a"), _manual()], delete_status=500)
    maker, list_resting, is_script, cancel = _bind(client)
    alerts = []
    code = maker.halt_live_run(
        "NOT READY YET: sim_inputs is event 527, live event is 528 (name); advance sim_inputs",
        confirmation_failed=True, list_resting=list_resting, is_script=is_script,
        cancel_order=cancel, alert=alerts.append)
    assert code == 1
    assert "still resting after cancel: a" in alerts[0]
    assert client.deletes == 2
    assert any(order["order_id"] == "a" for order in client.orders)
    assert not any(method == "POST" for method, _path in client.calls)


def test_pull_list_error_does_not_cancel_and_exits_nonzero():
    client = FakeOrdersClient(
        [_script("a")], list_error=RuntimeError("orders list down"))
    maker, list_resting, is_script, cancel = _bind(client)
    alerts = []
    code = maker.halt_live_run(
        "LIVE EVENT UNCONFIRMED: schedule failed (RuntimeError); refusing to quote",
        confirmation_failed=True, list_resting=list_resting, is_script=is_script,
        cancel_order=cancel, alert=alerts.append)
    assert code == 1
    assert "list resting orders failed (RuntimeError)" in alerts[0]
    assert client.deletes == 0
    assert not any(method == "DELETE" for method, _path in client.calls)
    assert not any(method == "POST" for method, _path in client.calls)


def test_pull_verify_finds_leftover_smk_orders():
    client = FakeOrdersClient([_script("a"), _manual()], drop_on_success=False)
    maker, list_resting, is_script, cancel = _bind(client)
    alerts = []
    code = maker.halt_live_run(
        "kill switch", confirmation_failed=False, list_resting=list_resting,
        is_script=is_script, cancel_order=cancel, alert=alerts.append)
    assert code == 1
    assert "still resting after cancel: a" in alerts[0]
    assert "manual-1" not in alerts[0]
    assert not any(method == "POST" for method, _path in client.calls)


def test_confirmed_halt_with_a_clean_pull_still_exits_nonzero():
    """A positive confirmation failure pages even after smk_ orders are gone."""
    client = FakeOrdersClient([_script("a"), _manual()])
    maker, list_resting, is_script, cancel = _bind(client)
    alerts = []
    code = maker.halt_live_run(
        "NOT READY YET: sim_inputs is event 527, live event is 528 (Bermuda); advance sim_inputs",
        confirmation_failed=True, list_resting=list_resting, is_script=is_script,
        cancel_order=cancel, alert=alerts.append)
    assert code == 1
    assert "quote pull failed" not in alerts[0]
    assert "advance sim_inputs" in alerts[0]
    assert client.deletes == 1
    assert not any(str(order.get("client_order_id", "")).startswith("smk_")
                   for order in client.orders)
    assert any(order["order_id"] == "manual-1" for order in client.orders)


def test_ordinary_halt_with_a_clean_pull_exits_zero():
    client = FakeOrdersClient([])
    maker, list_resting, is_script, cancel = _bind(client)
    code = maker.halt_live_run(
        "kill switch test", confirmation_failed=False, list_resting=list_resting,
        is_script=is_script, cancel_order=cancel, alert=lambda _text: None)
    assert code == 0


def _expect_confirmation_halt(monkeypatch, event_ids, lookup):
    import kalshi_maker as maker
    import sim_inputs
    monkeypatch.setattr(sim_inputs, "event_ids", event_ids)
    monkeypatch.setattr(maker, "_live_event_for_orders", lambda: lookup)
    called = {"should_trade": 0, "post": 0}

    def should_trade(*_a, **_k):
        called["should_trade"] += 1
        return True, ""

    monkeypatch.setattr(maker.maker_guard, "should_trade", should_trade)
    monkeypatch.setattr(maker, "reconcile_and_post", lambda _cands: called.__setitem__("post", 1))
    ok, reason, confirmation_failed = maker._check_live_preconditions()
    assert ok is False
    assert confirmation_failed is True
    assert called == {"should_trade": 0, "post": 0}
    return reason


def test_missing_datagolf_key_halts(monkeypatch):
    import kalshi_maker as maker
    import sim_inputs
    monkeypatch.delenv("DATAGOLF_API_KEY", raising=False)
    monkeypatch.setattr(sim_inputs, "event_ids", [528])
    called = {"should_trade": 0}
    monkeypatch.setattr(
        maker.maker_guard, "should_trade",
        lambda *_a, **_k: called.__setitem__("should_trade", 1) or (True, ""))
    ok, reason, confirmation_failed = maker._check_live_preconditions()
    assert ok is False
    assert confirmation_failed is True
    assert reason == "LIVE EVENT UNCONFIRMED: no DATAGOLF_API_KEY; refusing to quote"
    assert called["should_trade"] == 0


def test_datagolf_import_error_halts(monkeypatch):
    import kalshi_maker as maker
    import sim_inputs
    monkeypatch.setenv("DATAGOLF_API_KEY", "synthetic")
    monkeypatch.setattr(sim_inputs, "event_ids", [528])

    def boom():
        raise ImportError("api_utils missing")

    monkeypatch.setattr(maker, "_load_datagolf_lookups", boom)
    called = {"should_trade": 0}
    monkeypatch.setattr(
        maker.maker_guard, "should_trade",
        lambda *_a, **_k: called.__setitem__("should_trade", 1) or (True, ""))
    ok, reason, confirmation_failed = maker._check_live_preconditions()
    assert ok is False
    assert confirmation_failed is True
    assert "import failed (ImportError)" in reason
    assert "refusing to quote" in reason
    assert called["should_trade"] == 0


def test_both_datagolf_lookups_failing_halts(monkeypatch):
    import kalshi_maker as maker
    import sim_inputs
    monkeypatch.setenv("DATAGOLF_API_KEY", "synthetic")
    monkeypatch.setattr(sim_inputs, "event_ids", [528])

    def fail_field(*_a, **_k):
        raise RuntimeError("field down")

    def fail_week(*_a, **_k):
        raise RuntimeError("schedule down")

    monkeypatch.setattr(
        maker, "_load_datagolf_lookups", lambda: (fail_field, fail_week))
    called = {"should_trade": 0, "post": 0}
    monkeypatch.setattr(
        maker.maker_guard, "should_trade",
        lambda *_a, **_k: called.__setitem__("should_trade", 1) or (True, ""))
    monkeypatch.setattr(
        maker, "reconcile_and_post",
        lambda *_a, **_k: called.__setitem__("post", 1))
    ok, reason, confirmation_failed = maker._check_live_preconditions()
    assert ok is False
    assert confirmation_failed is True
    assert "field-updates failed (RuntimeError)" in reason
    assert "schedule failed (RuntimeError)" in reason
    assert "refusing to quote" in reason
    assert called == {"should_trade": 0, "post": 0}


def test_empty_event_ids_halts():
    import kalshi_preflight as kp
    ok, reason = kp.inputs_ready_to_quote(
        [], 528, "Butterfield Bermuda Championship",
        week_events=[{"event_id": 528, "event_name": "Butterfield Bermuda Championship"}])
    assert ok is False
    assert reason == ("LIVE EVENT UNCONFIRMED: sim_inputs.event_ids is empty; "
                      "refusing to quote")


def test_empty_event_ids_halts_the_maker(monkeypatch):
    reason = _expect_confirmation_halt(monkeypatch, [], (
        528, "Butterfield Bermuda Championship",
        [{"event_id": 528, "event_name": "Butterfield Bermuda Championship"}],
        None,
    ))
    assert "event_ids is empty" in reason


def test_field_and_schedule_disagree_and_config_matches_neither(monkeypatch):
    reason = _expect_confirmation_halt(monkeypatch, [527], (
        528, "Butterfield Bermuda Championship",
        [{"event_id": 529, "event_name": "Other Championship"}],
        None,
    ))
    assert "live event is 528" in reason
    assert "field-updates wins over schedule event 529" in reason
    assert "527" in reason


def test_multi_event_week_with_no_matching_config_halts(monkeypatch):
    reason = _expect_confirmation_halt(monkeypatch, [526], (
        None, None,
        [{"event_id": 527, "event_name": "Baycurrent Classic"},
         {"event_id": 528, "event_name": "Butterfield Bermuda Championship"}],
        None,
    ))
    assert "schedule lists events 527, 528" in reason
    assert "field id is missing" in reason


def test_monday_rollover_field_wins_over_the_old_schedule_id(monkeypatch):
    """Field-updates has rolled to 528. The schedule still names 527.

    The field id is the event DataGolf is publishing now, so a config that
    only contains the old schedule id halts. A config that contains 528
    is confirmed and falls through to the later gates.
    """
    import kalshi_maker as maker
    import sim_inputs
    lookup = (
        528, "Butterfield Bermuda Championship",
        [{"event_id": 527, "event_name": "Baycurrent Classic"}],
        None,
    )
    reason = _expect_confirmation_halt(monkeypatch, [527], lookup)
    assert reason == (
        "NOT READY YET: sim_inputs is event 527, live event is 528 "
        "(Butterfield Bermuda Championship); field-updates wins over "
        "schedule event 527; advance sim_inputs")

    monkeypatch.setattr(sim_inputs, "event_ids", [528])
    monkeypatch.setattr(maker, "_live_event_for_orders", lambda: lookup)
    monkeypatch.setattr(
        maker.maker_guard, "should_trade", lambda *_a, **_k: (False, "kill switch test"))
    ok, later, confirmation_failed = maker._check_live_preconditions()
    assert confirmation_failed is False
    assert ok is False
    assert later == "kill switch test"

"""Fail-closed delivery and simulation-engine guard regressions."""

import importlib.util
from pathlib import Path
import sys
import types

import pandas as pd
import pytest


@pytest.fixture()
def round_module(monkeypatch):
    sheet_config = types.ModuleType("sheet_config")
    sheet_config.load_config = lambda: {
        "tourney": "guard_test",
        "std_dev": 2.8,
        "course_pars": [72],
        "cut_line": 65,
        "use_10_shot_rule": False,
        "simulations": 100,
        "event_id": 999,
        "course_cat_mults": {},
        "course_cat_skew": {},
    }
    sim_inputs = types.ModuleType("sim_inputs")
    sim_inputs.name_replacements = {}
    for name in (
        "coefficients_r1_high", "coefficients_r1_midh",
        "coefficients_r1_midl", "coefficients_r1_low",
        "coefficients_r2", "coefficients_r2_6_30", "coefficients_r2_30_up",
        "coefficients_r3", "coefficients_r3_mid", "coefficients_r3_high",
    ):
        setattr(sim_inputs, name, {})

    monkeypatch.setitem(sys.modules, "sheet_config", sheet_config)
    monkeypatch.setitem(sys.modules, "sim_inputs", sim_inputs)
    spec = importlib.util.spec_from_file_location(
        "round_sim_delivery_guard_test", Path(__file__).with_name("round_sim.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_email_rejects_missing_configuration(round_module, monkeypatch):
    monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
    round_module.EMAIL_FROM = None
    round_module.EMAIL_TO = []

    with pytest.raises(round_module.EmailDeliveryError):
        round_module.send_round_sim_email(
            pd.DataFrame(), 4, {}, required=True
        )


def test_required_email_returns_receipt_after_smtp_acceptance(round_module, monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, sender, password):
            assert sender == "sender@example.com"
            assert password == "app-password"

        def sendmail(self, sender, recipients, message):
            sent.append((sender, recipients, message))

    monkeypatch.setenv("EMAIL_PASSWORD", "app-password")
    round_module.EMAIL_FROM = "sender@example.com"
    round_module.EMAIL_TO = ["receiver@example.com"]
    monkeypatch.setattr(round_module, "build_matchup_email_html", lambda *a, **k: "<p>ok</p>")
    monkeypatch.setattr(round_module.smtplib, "SMTP_SSL", FakeSMTP)

    accepted = round_module.send_round_sim_email(
        pd.DataFrame(), 4, {}, required=True
    )

    assert accepted is True
    assert len(sent) == 1
    assert sent[0][1] == ["receiver@example.com"]


def test_required_kernel_refuses_implicit_python_fallback(round_module, monkeypatch):
    monkeypatch.setenv("REQUIRE_SIMS_KERNEL", "1")

    with pytest.raises(RuntimeError, match="refusing to silently switch"):
        round_module._handle_rust_kernel_failure("run_single_round", ValueError("boom"))


def test_interactive_kernel_may_use_explicitly_logged_fallback(round_module, monkeypatch):
    monkeypatch.delenv("REQUIRE_SIMS_KERNEL", raising=False)

    assert round_module._handle_rust_kernel_failure(
        "run_single_round", ValueError("boom")
    ) is None


def test_required_report_rejects_missing_sharp_book_coverage(round_module):
    with pytest.raises(round_module.SimulationHealthError, match="betonline=2/5"):
        round_module.require_pricing_pipeline_healthy(
            matchup_book_counts={"betcris": 8, "betonline": 2},
            require_complete_email=True,
        )


def test_zero_qualifying_edges_are_valid_when_inputs_completed(round_module):
    # The gate cares about source lines successfully priced, not how many rows
    # survived the model's edge thresholds.
    round_module.require_pricing_pipeline_healthy(
        matchup_book_counts={"betcris": 5, "betonline": 5},
        require_complete_email=True,
        finish_probs=pd.DataFrame({"player_name": ["a"]}),
        require_live_tournament=True,
    )


@pytest.mark.parametrize(
    "kwargs, pattern",
    [
        ({"matchup_error": ValueError("bad feed")}, "matchup pricing"),
        (
            {
                "matchup_book_counts": {"betcris": 5, "betonline": 5},
                "require_complete_email": True,
                "threeball_error": ValueError("bad triples"),
            },
            "3-ball pricing",
        ),
        (
            {
                "matchup_book_counts": {"betcris": 5, "betonline": 5},
                "require_complete_email": True,
                "score_line_error": ValueError("bad props"),
            },
            "score-line pricing",
        ),
        (
            {
                "require_live_tournament": True,
                "tournament_error": ValueError("bad tape"),
            },
            "tournament/finish pricing",
        ),
        (
            {"require_live_tournament": True, "finish_probs": pd.DataFrame()},
            "outputs are missing",
        ),
    ],
)
def test_incomplete_pricing_pipeline_fails_closed(round_module, kwargs, pattern):
    with pytest.raises(round_module.SimulationHealthError, match=pattern):
        round_module.require_pricing_pipeline_healthy(**kwargs)

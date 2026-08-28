"""Fail-closed delivery and simulation-engine guard regressions."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
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


def test_kernel_always_refuses_implicit_python_fallback(round_module, monkeypatch):
    monkeypatch.delenv("REQUIRE_SIMS_KERNEL", raising=False)
    with pytest.raises(RuntimeError, match="refusing to silently switch"):
        round_module._handle_rust_kernel_failure("run_single_round", ValueError("boom"))


def test_python_engine_requires_explicit_cli_flag(round_module):
    source = Path(round_module.__file__).read_text(encoding="utf-8")
    assert "if not _USE_PYTHON" in source
    assert 'add_argument("--use-python"' in source


def test_category_first_inputs_never_fall_back_to_legacy(round_module, monkeypatch, tmp_path):
    missing = tmp_path / "missing_dists.csv"
    monkeypatch.setattr(round_module, "DISTS_FILE_V2", str(missing))
    with pytest.raises(FileNotFoundError, match="Required category-first"):
        round_module._load_catfirst_dists(["alpha"])

    malformed = tmp_path / "malformed_dists.csv"
    pd.DataFrame({"player_name": ["alpha"]}).to_csv(malformed, index=False)
    monkeypatch.setattr(round_module, "DISTS_FILE_V2", str(malformed))
    with pytest.raises(ValueError, match="required category-first columns"):
        round_module._load_catfirst_dists(["alpha"])


def _category_dists(players):
    return pd.DataFrame([
        {
            "player_name": player,
            "category_clean": category,
            "mean": 0.1,
            "std": 1.0,
            "skew": 0.0,
            "n_eff": 25.0,
        }
        for player in players
        for category in ("sg_ott", "sg_app", "sg_arg", "sg_putt")
    ])


def test_round_category_dists_require_complete_finite_active_field(
    round_module, monkeypatch, tmp_path
):
    dists_path = tmp_path / "dists.csv"
    monkeypatch.setattr(round_module, "DISTS_FILE_V2", str(dists_path))
    monkeypatch.setattr(
        round_module,
        "apply_shot_dispersion_overlay",
        lambda stds, *_args, **_kwargs: stds,
    )
    monkeypatch.setattr(round_module, "load_corr_matrix", lambda _cats: np.eye(4))

    _category_dists(["unrelated"]).to_csv(dists_path, index=False)
    with pytest.raises(ValueError, match="missing active-field category coverage"):
        round_module._load_catfirst_dists(["alpha"])

    incomplete = _category_dists(["alpha"])
    incomplete = incomplete[incomplete["category_clean"] != "sg_putt"]
    incomplete.to_csv(dists_path, index=False)
    with pytest.raises(ValueError, match="alpha/sg_putt"):
        round_module._load_catfirst_dists(["alpha"])

    non_finite = _category_dists(["alpha"])
    non_finite.loc[
        non_finite["category_clean"] == "sg_app", "mean"
    ] = np.inf
    non_finite.to_csv(dists_path, index=False)
    with pytest.raises(ValueError, match="non-finite active-field values"):
        round_module._load_catfirst_dists(["alpha"])

    _category_dists(["alpha"]).to_csv(dists_path, index=False)
    params, skew, correlation = round_module._load_catfirst_dists(["alpha"])
    assert len(params) == 1
    assert np.isfinite(params[0][0]).all()
    assert np.isfinite(params[0][1]).all()
    assert np.isfinite(skew).all()
    np.testing.assert_array_equal(correlation, np.eye(4))


def test_live_round_category_dists_allow_only_explicit_active_subset(
    round_module, monkeypatch, tmp_path
):
    dists_path = tmp_path / "dists.csv"
    _category_dists(["alpha"]).to_csv(dists_path, index=False)
    monkeypatch.setattr(round_module, "DISTS_FILE_V2", str(dists_path))
    captured = {}

    def overlay(stds, *_args, **kwargs):
        captured.update(kwargs)
        return stds

    monkeypatch.setattr(round_module, "apply_shot_dispersion_overlay", overlay)
    monkeypatch.setattr(round_module, "load_corr_matrix", lambda _cats: np.eye(4))

    round_module._load_catfirst_dists(["alpha"], allow_player_subset=True)

    assert captured["allow_active_subset"] is True


def test_production_correlation_matrix_never_falls_back(round_module, monkeypatch, tmp_path):
    missing = tmp_path / "preferred_missing.csv"
    fallback = tmp_path / "different_fallback.csv"
    pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]], index=["a", "b"], columns=["a", "b"]
    ).to_csv(fallback)
    monkeypatch.setattr(round_module, "CORR_PREFS", [str(missing), str(fallback)])

    with pytest.raises(FileNotFoundError, match="production category correlation"):
        round_module.load_corr_matrix(["a", "b"])


def test_required_report_rejects_missing_sharp_book_coverage(round_module):
    with pytest.raises(round_module.SimulationHealthError, match="betonline=2/5"):
        round_module.require_pricing_pipeline_healthy(
            matchup_book_counts={"betcris": 8, "betonline": 2},
            require_complete_email=True,
        )


def test_required_report_accepts_manual_pinnacle_pair(round_module):
    round_module.require_pricing_pipeline_healthy(
        matchup_book_counts={"betonline": 5, "pinnacle": 5},
        require_complete_email=True,
        required_matchup_books=("betonline", "pinnacle"),
    )


def test_required_report_rejects_partial_manual_pinnacle_pair(round_module):
    with pytest.raises(round_module.SimulationHealthError, match="pinnacle=4/5"):
        round_module.require_pricing_pipeline_healthy(
            matchup_book_counts={"betonline": 5, "pinnacle": 4},
            require_complete_email=True,
            required_matchup_books=("betonline", "pinnacle"),
        )


def test_round_delivery_rejects_any_unjoined_matchup_name(round_module):
    with pytest.raises(round_module.SimulationHealthError, match="did not join"):
        round_module.require_pricing_pipeline_healthy(
            matchup_name_mismatches={"misspelled player": {"betcris"}},
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

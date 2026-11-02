"""The bet-sizing Kelly multiplier is kelly_config.KELLY_FRACTION (0.4).

Quote floors (maker_quotes.KELLY_MIN_BY_TYPE) and stake caps are different
numbers. This file checks the sizing haircut, and that those other numbers
were left alone.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

import kelly_config
import kelly_solver
from correlated_kelly import optimize_correlated_kelly
from kalshi_ancillary import (
    KALSHI_BANKROLL,
    KALSHI_KELLY,
    KALSHI_MAX_RISK_FRAC,
    KALSHI_UNIT,
    kelly_contracts,
)
from kelly_config import KELLY_FRACTION
from sheets_storage import _FINISH_BANKROLL, _FINISH_KELLY_FRACTION, _size_exchange_finish_rows

ROOT = Path(__file__).resolve().parent
FRACTION_NAMES = {"KELLY_FRACTION", "_FINISH_KELLY_FRACTION", "KALSHI_KELLY"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "target"}


def _production_py_files():
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "kelly_config.py":
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        yield path


def _is_shared_ref(node):
    if isinstance(node, ast.Name) and node.id == "KELLY_FRACTION":
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "KELLY_FRACTION"
        and isinstance(node.value, ast.Name)
        and node.value.id == "kelly_config"
    )


def _is_number(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def _function_defaults(fn):
    args = fn.args
    positional = args.args
    offset = len(positional) - len(args.defaults)
    for arg, default in zip(positional[offset:], args.defaults):
        yield arg, default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            yield arg, default


def _fraction_literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in FRACTION_NAMES:
                    if not _is_shared_ref(node.value):
                        problems.append(f"{path.name}:{node.lineno} assigns {target.id} from a local value")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in FRACTION_NAMES and node.value is not None and not _is_shared_ref(node.value):
                problems.append(f"{path.name}:{node.lineno} assigns {node.target.id} from a local value")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.name in FRACTION_NAMES or bound in FRACTION_NAMES:
                    if node.module != "kelly_config" or alias.name != "KELLY_FRACTION":
                        problems.append(f"{path.name}:{node.lineno} imports a Kelly fraction from {node.module}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg, default in _function_defaults(node):
                if arg.arg == "kelly_fraction" and not _is_shared_ref(default):
                    problems.append(f"{path.name}:{node.lineno} {node.name} defaults kelly_fraction locally")
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "kelly_fraction" and _is_number(kw.value):
                    problems.append(f"{path.name}:{kw.value.lineno} passes a numeric kelly_fraction")
    return problems


def _assigned_number(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name and _is_number(node.value):
                found.append(node.value.value)
    return found


def test_shared_constant_is_point_four():
    assert kelly_config.KELLY_FRACTION == 0.4
    assert KELLY_FRACTION == 0.4


def test_no_production_file_sets_its_own_kelly_fraction():
    problems = []
    for path in _production_py_files():
        problems.extend(_fraction_literals(path))
    assert problems == []


def test_sizers_import_the_shared_constant():
    required = [
        "correlated_kelly.py",
        "kelly_solver.py",
        "kalshi_trader.py",
        "kalshi_ancillary.py",
        "sheets_storage.py",
        "round_sim.py",
        "new_sim.py",
        "archive/new_sim_v1.py",
    ]
    missing = []
    for rel in required:
        text = (ROOT / rel).read_text(encoding="utf-8")
        if "from kelly_config import KELLY_FRACTION" not in text:
            missing.append(rel)
    assert missing == []


def test_callable_defaults_and_aliases_use_the_shared_constant():
    corr = inspect.signature(optimize_correlated_kelly)
    player = inspect.signature(kelly_solver.optimize_player_kelly)
    outright = inspect.signature(kelly_solver.compute_optimal_outright_sizing)
    assert corr.parameters["kelly_fraction"].default == KELLY_FRACTION
    assert player.parameters["kelly_fraction"].default == KELLY_FRACTION
    assert outright.parameters["kelly_fraction"].default == KELLY_FRACTION
    assert KALSHI_KELLY == KELLY_FRACTION
    assert _FINISH_KELLY_FRACTION == KELLY_FRACTION

    import kalshi_trader
    assert kalshi_trader.KELLY_FRACTION == KELLY_FRACTION


def test_caps_and_floors_stay_put():
    corr = inspect.signature(optimize_correlated_kelly)
    player = inspect.signature(kelly_solver.optimize_player_kelly)
    outright = inspect.signature(kelly_solver.compute_optimal_outright_sizing)
    assert corr.parameters["bankroll"].default == 30_000.0
    assert corr.parameters["golfer_cap"].default == 0.15
    assert corr.parameters["portfolio_cap"].default == 0.35
    assert player.parameters["max_total_fraction"].default == 0.25
    assert outright.parameters["max_total_fraction_per_player"].default == 0.25
    assert _FINISH_BANKROLL == 10_000.0
    assert KALSHI_BANKROLL == 50_000.0
    assert KALSHI_UNIT == 500.0
    assert KALSHI_MAX_RISK_FRAC == 0.05
    assert _assigned_number(ROOT / "round_sim.py", "BANKROLL") == [10_000.0]
    assert _assigned_number(ROOT / "new_sim.py", "BANKROLL") == [10_000.0]

    import kalshi_trader
    assert kalshi_trader.MAX_CONTRACTS == 50
    assert kalshi_trader.MAX_RISK_PCT == 0.40

    maker = (ROOT / "kalshi_maker.py").read_text(encoding="utf-8")
    assert "max_total_fraction_per_player=0.25" in maker
    assert "kelly_fraction=" not in maker

    floors = {}
    tree = ast.parse((ROOT / "maker_quotes.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "_envf"):
            continue
        floors[node.args[0].value] = node.args[1].value
    assert floors == {
        "MAKER_KELLY_MIN_WINNER": 0.25,
        "MAKER_KELLY_MIN_TOP5": 0.25,
        "MAKER_KELLY_MIN_TOP10": 0.25,
        "MAKER_KELLY_MIN_TOP20": 0.15,
        "MAKER_KELLY_MIN_H2H": 0.05,
    }


def test_correlated_kelly_default_scales_a_single_bet_by_point_four():
    bet = {
        "player_name": "sample, player",
        "market_type": "win",
        "decimal_odds": 2.0,
        "sim_prob": 0.55,
        "bookmaker": "pinnacle",
    }
    scores = np.zeros((1, 4))
    default = optimize_correlated_kelly([bet], [], scores, ["other, golfer"])
    full = optimize_correlated_kelly(
        [bet], [], scores, ["other, golfer"], kelly_fraction=1.0,
    )
    assert default["summary"]["kelly_fraction"] == 0.4
    assert full["summary"]["kelly_fraction"] == 1.0
    assert default["bets"][0]["corr_f"] == pytest.approx(0.04)
    assert default["bets"][0]["dollars"] == pytest.approx(1_200.0)
    assert full["bets"][0]["corr_f"] == pytest.approx(0.10)
    assert default["bets"][0]["corr_f"] == pytest.approx(full["bets"][0]["corr_f"] * 0.4)


def test_player_kelly_default_is_point_four_of_the_full_solve():
    candidates = [{"market": "winner", "side": "yes", "post_price": 0.40}]
    buckets = np.array([0.50, 0.10, 0.10, 0.10, 0.20])
    full = kelly_solver.optimize_player_kelly(candidates, buckets, kelly_fraction=1.0)
    default = kelly_solver.optimize_player_kelly(candidates, buckets)
    assert full[0] > 0
    assert full[0] <= 0.25
    assert default[0] == pytest.approx(full[0] * 0.4)


def test_exchange_finish_and_kalshi_contracts_use_point_four():
    import pandas as pd

    frame = pd.DataFrame([{
        "bookmaker": "kalshi",
        "sim_prob": 0.55,
        "implied_prob": 0.50,
        "market_type": "win",
        "side": "yes",
    }])
    sized = _size_exchange_finish_rows(frame)
    # f* = 0.10 on a $10k bankroll. 0.4 Kelly is $400. The 5% risk cap does not apply here.
    assert sized.loc[0, "kelly_stake"] == pytest.approx(400.0)

    contracts, dollars, units = kelly_contracts(0.55, 0.50)
    # Same f* on the $50k Kalshi bankroll: 0.4 Kelly is $2,000, under the $2,500 risk cap.
    assert dollars == pytest.approx(2_000.0)
    assert contracts == 4_000
    assert units == pytest.approx(4.0)

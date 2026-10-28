import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import archive.all_round_joint_market_eval as joint_eval

from archive.all_round_joint_market_eval import (
    DIAGNOSTIC_FALLBACK_RULE,
    DIAGNOSTIC_SELECTION_MODE,
    DIAGNOSTIC_SELECTION_OUTCOME,
    MARKETS,
    PRIMARY_SELECTION_MODE,
    PRIMARY_SELECTION_OUTCOME,
    SELECTION_STATUS_FIELDS,
    _sha256_file,
    _normalize_quotes,
    build_parser,
    evaluate_frozen_joint_candidate,
    evaluate_joint_markets,
    joint_selection_summary,
    main,
    select_and_settle_best_close,
    select_joint_candidate_2023,
    validate_evaluation_prediction_authority,
    validate_selection_freeze,
    write_selection_freeze,
)


PLAYERS = ("p0", "p1", "p2", "p3")
PAIRS = (("p0", "p3"), ("p1", "p2"))
BOOKS = ("betonline", "betcris", "pinnacle")


def _synthetic_joint_inputs():
    event_rows = []
    for index in range(18):
        event_rows.append((f"s{index:02d}", pd.Timestamp("2023-04-06") + pd.Timedelta(days=14 * index), index))
    for index in range(6):
        event_rows.append((f"e{index:02d}", pd.Timestamp("2024-01-04") + pd.Timedelta(days=7 * index), index + 20))

    base_predictions = []
    round_predictions = []
    tournament_quotes = []
    round_quotes = []
    for event_key, event_date, event_index in event_rows:
        tournament_values = {}
        for pair_index, (left, right) in enumerate(PAIRS):
            p1_wins = (event_index + pair_index) % 3 != 0
            sign = 1.0 if p1_wins else -1.0
            tournament_values[left] = sign
            tournament_values[right] = -sign
            for book_index, book in enumerate(BOOKS):
                p1_id, p2_id = left, right
                p1_outcome, p2_outcome = (1.0, 0.0) if p1_wins else (0.0, 1.0)
                p1_close, p2_close = (-110.0 + 5 * book_index, -110.0 - 5 * book_index)
                # Exercise cross-book side canonicalization.
                if book == "pinnacle" and pair_index == 0:
                    p1_id, p2_id = p2_id, p1_id
                    p1_outcome, p2_outcome = p2_outcome, p1_outcome
                    p1_close, p2_close = p2_close, p1_close
                tournament_quotes.append(
                    {
                        "event_key": event_key,
                        "event_start_date": event_date,
                        "book": book,
                        "p1_id": p1_id,
                        "p2_id": p2_id,
                        "p1_close": p1_close,
                        "p2_close": p2_close,
                        "p1_outcome": p1_outcome,
                        "p2_outcome": p2_outcome,
                        "tie_rule": "ties push",
                    }
                )
        for candidate, multiplier in (("baseline", -1.0), ("challenger", 1.0)):
            for player in PLAYERS:
                base_predictions.append(
                    {
                        "candidate": candidate,
                        "event_key": event_key,
                        "player_id": player,
                        "prediction": multiplier * tournament_values[player],
                    }
                )

        for round_num in range(1, 5):
            round_values = {}
            for pair_index, (left, right) in enumerate(PAIRS):
                is_tie = (event_index + pair_index + round_num) % 7 == 0
                p1_wins = (event_index + pair_index + round_num) % 3 != 0
                sign = 0.0 if is_tie else (1.0 if p1_wins else -1.0)
                round_values[left] = sign
                round_values[right] = -sign
                for book_index, book in enumerate(BOOKS):
                    if is_tie:
                        p1_outcome = p2_outcome = 0.5
                    else:
                        p1_outcome, p2_outcome = (
                            (1.0, 0.0) if p1_wins else (0.0, 1.0)
                        )
                    round_quotes.append(
                        {
                            "event_key": event_key,
                            "event_start_date": event_date,
                            "round_num": round_num,
                            "book": book,
                            "p1_id": left,
                            "p2_id": right,
                            "p1_close": -110.0 + 5 * book_index,
                            "p2_close": -110.0 - 5 * book_index,
                            "p1_outcome": p1_outcome,
                            "p2_outcome": p2_outcome,
                            "tie_rule": "ties push",
                        }
                    )
            for candidate, multiplier in (("baseline", -1.0), ("challenger", 1.0)):
                for player in PLAYERS:
                    round_predictions.append(
                        {
                            "candidate": candidate,
                            "event_key": event_key,
                            "player_id": player,
                            "round_num": round_num,
                            "prediction": multiplier * round_values[player],
                        }
                    )
    return tuple(
        pd.DataFrame(rows)
        for rows in (
            base_predictions,
            round_predictions,
            tournament_quotes,
            round_quotes,
        )
    )


def test_joint_evaluation_uses_apr_jun_maps_then_freezes_winner_for_2024():
    base, rounds, tournament_quotes, round_quotes = _synthetic_joint_inputs()
    # One challenger hole must remove the corresponding 72h opportunity from
    # every candidate, rather than giving the challenger a self-selected menu.
    base = base.loc[
        ~(
            base["candidate"].eq("challenger")
            & base["event_key"].eq("e00")
            & base["player_id"].eq("p3")
        )
    ].copy()
    result = evaluate_joint_markets(
        base,
        rounds,
        tournament_quotes,
        round_quotes,
        None,
        "baseline",
        "2023-04-01",
        "2023-12-31",
        "2024-01-01",
        "2024-12-31",
        3,
        100,
        bootstrap_seed=17,
        allow_unfrozen_test_only=True,
    )

    assert result.selected_candidate == "challenger"
    assert set(result.selection_summary["market"]) == set(MARKETS)
    assert set(result.evaluation_summary["market"]) == set(MARKETS)
    assert result.mapper_final.shape[0] == 2 * len(MARKETS)
    assert result.scored_selection["event_start_date"].dt.year.eq(2023).all()
    assert result.scored_evaluation["event_start_date"].dt.year.eq(2024).all()

    assert pd.to_datetime(result.mapper_crossfit["mapper_training_end"]).dt.date.eq(
        pd.Timestamp("2023-06-30").date()
    ).all()
    assert result.scored_selection["event_start_date"].ge(pd.Timestamp("2023-07-01")).all()
    assert set(result.scored_evaluation["candidate"]) == {"baseline", "challenger"}

    audit = result.common_menu_audit.loc[
        result.common_menu_audit["sample"].eq("evaluation")
        & result.common_menu_audit["market"].eq("72h")
    ]
    assert audit["quoted_opportunities"].eq(12).all()
    assert audit["exact_common_opportunities"].eq(11).all()
    common_counts = (
        result.scored_evaluation.loc[result.scored_evaluation["market"].eq("72h")]
        [["candidate", "event_key", "pair_key"]]
        .drop_duplicates()
        .groupby("candidate")
        .size()
    )
    assert common_counts.nunique() == 1
    assert common_counts.iloc[0] == 11
    assert {"event_equal_three_way_log_loss_delta", "pnl_delta"}.issubset(
        set(result.bootstrap["metric"])
    )


def test_one_shot_joint_helper_requires_explicit_test_only_opt_in():
    with pytest.raises(ValueError, match="non-authoritative and test-only"):
        evaluate_joint_markets(
            pd.DataFrame(),
            None,
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "baseline",
        )


def test_best_close_uses_only_three_books_and_supports_both_stakes():
    rows = []
    for pair_key, prices, probability in (
        ("a|b", (-130.0, -120.0, -125.0), 0.90),
        ("c|d", (-120.0, -110.0, 150.0), 0.75),
    ):
        left, right = pair_key.split("|")
        for book, price in zip(BOOKS, prices):
            rows.append(
                {
                    "candidate": "candidate",
                    "sample": "evaluation",
                    "event_key": "event",
                    "event_start_date": pd.Timestamp("2024-01-01"),
                    "market": "72h",
                    "book": book,
                    "pair_key": pair_key,
                    "p1_id": left,
                    "p2_id": right,
                    "p1_close": price,
                    "p2_close": 130.0,
                    "tie_rule": "ties push",
                    "physical_p1_outcome": 1.0,
                    "skill_difference": 1.0,
                    "sigma": 1.0,
                    "tie_threshold": 0.01,
                    "p1_probability": probability,
                    "p2_probability": 1.0 - probability,
                    "push_probability": 0.0,
                }
            )
    bets = select_and_settle_best_close(pd.DataFrame(rows), min_ev=0.03)
    flat = bets.loc[bets["sizing"].eq("flat_1u")].set_index("pair_key")
    risk = bets.loc[bets["sizing"].eq("risk_to_win_1u")].set_index("pair_key")

    assert flat.loc["a|b", "book"] == "betcris"
    assert flat.loc["a|b", "close_odds"] == -120.0
    assert flat.loc["c|d", "book"] == "pinnacle"
    assert flat.loc["c|d", "close_odds"] == 150.0
    assert risk.loc["a|b", "stake"] == 1.2
    assert np.isclose(risk.loc["a|b", "pnl"], 1.0)
    assert np.isclose(risk.loc["c|d", "stake"], 2.0 / 3.0)
    assert np.isclose(risk.loc["c|d", "pnl"], 1.0)


def test_ties_lose_is_priced_and_settled_as_a_loss():
    scored = pd.DataFrame(
        {
            "candidate": ["candidate"],
            "sample": ["evaluation"],
            "event_key": ["event"],
            "event_start_date": [pd.Timestamp("2024-01-01")],
            "market": ["r1"],
            "book": ["pinnacle"],
            "pair_key": ["a|b"],
            "p1_id": ["a"],
            "p2_id": ["b"],
            "p1_close": [200.0],
            "p2_close": [-200.0],
            "tie_rule": ["ties lose"],
            "physical_p1_outcome": [0.5],
            "skill_difference": [0.0],
            "sigma": [1.0],
            "tie_threshold": [0.2],
            "p1_probability": [0.45],
            "p2_probability": [0.45],
            "push_probability": [0.10],
        }
    )
    bets = select_and_settle_best_close(scored, min_ev=0.0, sizings=("flat_1u",))
    assert len(bets) == 1
    assert np.isclose(bets.iloc[0]["ev_per_unit"], 0.35)
    assert bets.iloc[0]["settlement_status"] == "tie_loss"
    assert bets.iloc[0]["pnl"] == -1.0


def test_administrative_void_is_counted_as_a_refunded_bet_not_a_physical_tie():
    scored = pd.DataFrame(
        {
            "candidate": ["candidate"],
            "sample": ["evaluation"],
            "event_key": ["event"],
            "event_start_date": [pd.Timestamp("2024-01-01")],
            "market": ["72h"],
            "book": ["pinnacle"],
            "pair_key": ["a|b"],
            "p1_id": ["a"],
            "p2_id": ["b"],
            "p1_close": [200.0],
            "p2_close": [-200.0],
            "tie_rule": ["Bets are void in event of tie"],
            "physical_p1_outcome": [np.nan],
            "settlement_p1_outcome": [0.5],
            "skill_difference": [0.0],
            "sigma": [1.0],
            "tie_threshold": [0.2],
            "p1_probability": [0.45],
            "p2_probability": [0.45],
            "push_probability": [0.10],
        }
    )

    bets = select_and_settle_best_close(scored, min_ev=0.0, sizings=("flat_1u",))

    assert len(bets) == 1
    assert bets.iloc[0]["settlement_status"] == "push"
    assert bets.iloc[0]["stake"] == 1.0
    assert bets.iloc[0]["pnl"] == 0.0


def test_datagolf_round_numeric_tie_with_void_text_is_a_physical_refunded_tie():
    quote = pd.DataFrame(
        {
            "event_key": ["event"],
            "event_start_date": ["2024-01-01"],
            "round_num": [1],
            "book": ["pinnacle"],
            "p1_id": ["a"],
            "p2_id": ["b"],
            "p1_close": [-110],
            "p2_close": [-110],
            "p1_outcome": [0.5],
            "p2_outcome": [0.5],
            "p1_outcome_text": ["void"],
            "p2_outcome_text": ["void"],
            "tie_rule": ["Bets are void in event of tie"],
        }
    )

    normalized = _normalize_quotes(quote, rounds=True)

    assert normalized.iloc[0]["physical_p1_outcome"] == 0.5
    assert normalized.iloc[0]["settlement_p1_outcome"] == 0.5


def test_tournament_numeric_void_needs_score_authority_for_physical_tie():
    quote = pd.DataFrame(
        {
            "event_key": ["event", "event2"],
            "event_start_date": ["2024-01-01", "2024-01-08"],
            "market": ["72h", "72h"],
            "book": ["pinnacle", "pinnacle"],
            "p1_id": ["a", "c"],
            "p2_id": ["b", "d"],
            "p1_close": [-110, -110],
            "p2_close": [-110, -110],
            "p1_outcome": [0.5, 0.5],
            "p2_outcome": [0.5, 0.5],
            "p1_outcome_text": ["void", "void"],
            "p2_outcome_text": ["void", "void"],
            "tie_rule": [
                "Bets are void in event of tie",
                "Bets are void in event of tie",
            ],
            "physical_p1_outcome": [np.nan, 0.5],
            "other_void_with_scores": [True, False],
        }
    )

    normalized = _normalize_quotes(quote, rounds=False)

    assert np.isnan(normalized.iloc[0]["physical_p1_outcome"])
    assert normalized.iloc[0]["settlement_p1_outcome"] == 0.5
    assert normalized.iloc[1]["physical_p1_outcome"] == 0.5
    assert normalized.iloc[1]["settlement_p1_outcome"] == 0.5


def test_explicit_no_score_void_is_not_reclassified_as_numeric_tie():
    row = pd.Series(
        {
            "p1_outcome": 0.5,
            "p2_outcome": 0.5,
            "p1_outcome_text": "void",
            "no_score_void": True,
            "tie_rule": "Bets are void in event of tie",
        }
    )

    assert np.isnan(joint_eval._physical_p1_outcome(row))


def test_tournament_normalizer_filters_r1_rows_from_mixed_cache():
    quote = pd.DataFrame(
        {
            "event_key": ["event", "event"],
            "event_start_date": ["2023-07-01", "2023-07-01"],
            "market": ["72h", "r1"],
            "book": ["pinnacle", "pinnacle"],
            "p1_id": ["a", "a"],
            "p2_id": ["b", "b"],
            "p1_close": [-110, -110],
            "p2_close": [-110, -110],
            "p1_outcome": [1.0, 1.0],
            "p2_outcome": [0.0, 0.0],
            "tie_rule": ["ties push", "ties push"],
        }
    )
    normalized = _normalize_quotes(quote, rounds=False)
    assert len(normalized) == 1
    assert normalized.iloc[0]["market"] == "72h"


def test_selection_cli_predicate_pushes_2023_before_outcome_materialization(
    tmp_path, monkeypatch
):
    base, rounds, tournament_quotes, round_quotes = _synthetic_joint_inputs()
    tournament_quotes = tournament_quotes.assign(market="72h", forbidden_label="sentinel")
    round_quotes = round_quotes.assign(forbidden_label="sentinel")
    paths = {
        "base": tmp_path / "base.parquet",
        "rounds": tmp_path / "rounds.parquet",
        "tournament": tmp_path / "tournament.parquet",
        "round_quotes": tmp_path / "round_quotes.parquet",
        "catalog": tmp_path / "catalog.csv",
        "output": tmp_path / "selection",
    }
    base.to_parquet(paths["base"], index=False)
    rounds.to_parquet(paths["rounds"], index=False)
    tournament_quotes.to_parquet(paths["tournament"], index=False)
    round_quotes.to_parquet(paths["round_quotes"], index=False)
    pd.DataFrame(
        {
            "candidate": ["baseline", "challenger"],
            "course_fit_lambda": [1.0, 0.5],
            "live_update_lambda": [1.0, 0.5],
        }
    ).to_csv(paths["catalog"], index=False)

    original_read_parquet = pd.read_parquet
    quote_paths = {paths["tournament"].resolve(), paths["round_quotes"].resolve()}
    observed = []

    def read_spy(path, *args, **kwargs):
        frame = original_read_parquet(path, *args, **kwargs)
        resolved = Path(path).resolve()
        if resolved in quote_paths:
            observed.append(
                {
                    "path": str(resolved),
                    "filters": kwargs.get("filters"),
                    "columns": kwargs.get("columns"),
                    "max_date": pd.to_datetime(frame["event_start_date"]).max(),
                }
            )
        return frame

    monkeypatch.setattr(pd, "read_parquet", read_spy)
    main(
        [
            "--phase",
            "select",
            "--base-predictions",
            str(paths["base"]),
            "--round-predictions",
            str(paths["rounds"]),
            "--tournament-quotes",
            str(paths["tournament"]),
            "--round-quotes",
            str(paths["round_quotes"]),
            "--candidate-catalog",
            str(paths["catalog"]),
            "--output-dir",
            str(paths["output"]),
            "--baseline",
            "baseline",
            "--folds",
            "3",
        ]
    )
    assert len(observed) == 2
    assert all(item["filters"] for item in observed)
    assert all(item["max_date"] <= pd.Timestamp("2023-12-31") for item in observed)
    assert all("forbidden_label" not in item["columns"] for item in observed)
    freeze = paths["output"] / "selection_freeze.json"
    payload = validate_selection_freeze(
        freeze,
        baseline="baseline",
        min_ev=0.03,
        selection_start="2023-04-01",
        selection_end="2023-12-31",
        candidate_catalog_path=paths["catalog"],
    )
    assert payload["contract_sha256"].startswith("4091a019")
    assert payload["selected_spec"]["live_update_lambda"] in {0.0, 0.5, 1.0}
    assert payload["frozen_spec"]["evaluation_start"] == "2024-01-01"
    assert payload["frozen_spec"]["evaluation_end"] == "2026-12-31"
    assert payload["frozen_spec"]["bootstrap_draws"] == 10_000

    evaluate_args = [
        "--phase",
        "evaluate",
        "--base-predictions",
        str(paths["base"]),
        "--round-predictions",
        str(paths["rounds"]),
        "--tournament-quotes",
        str(paths["tournament"]),
        "--round-quotes",
        str(paths["round_quotes"]),
        "--candidate-catalog",
        str(paths["catalog"]),
        "--selection-freeze",
        str(freeze),
        "--output-dir",
        str(tmp_path / "evaluation"),
        "--baseline",
        "baseline",
    ]
    for override, message in (
        (["--eval-start", "2024-02-01"], "eval_start"),
        (["--eval-end", "2026-08-23"], "eval_end"),
        (["--bootstrap-draws", "9999"], "bootstrap_draws"),
        (["--bootstrap-seed", "17"], "bootstrap_seed"),
    ):
        prior_reads = len(observed)
        with pytest.raises(ValueError, match=message):
            main(evaluate_args + override)
        assert len(observed) == prior_reads

    # Tampering must fail before another quote read occurs.
    tampered = json.loads(freeze.read_text(encoding="utf-8"))
    tampered["baseline"] = "not-the-baseline"
    freeze.write_text(json.dumps(tampered), encoding="utf-8")
    prior_reads = len(observed)
    with pytest.raises(ValueError, match="freeze file hash"):
        main(evaluate_args)
    assert len(observed) == prior_reads


def test_evaluate_rejects_prediction_authority_tamper_before_any_data_read(monkeypatch):
    reads = []

    monkeypatch.setattr(
        joint_eval,
        "validate_selection_freeze",
        lambda *args, **kwargs: {
            "selected_candidate": "challenger",
            "baseline": "baseline",
            "contract_sha256": joint_eval.EXPECTED_CONTRACT_SHA256,
            "frozen_spec": {
                "course_fit_lambda": 0.5,
                "live_update_lambda": 0.5,
            },
        },
    )

    def reject_authority(*args, **kwargs):
        raise ValueError("prediction output combined_base_predictions hash mismatch")

    monkeypatch.setattr(joint_eval, "validate_prediction_authority", reject_authority)
    monkeypatch.setattr(
        joint_eval,
        "_read_prediction_frame_window",
        lambda *args, **kwargs: reads.append("prediction"),
    )
    monkeypatch.setattr(
        joint_eval,
        "_read_quote_frame_window",
        lambda *args, **kwargs: reads.append("quote"),
    )
    with pytest.raises(ValueError, match="prediction output"):
        joint_eval.main(
            [
                "--phase",
                "evaluate",
                "--base-predictions",
                "combined_base.parquet",
                "--round-predictions",
                "combined_round.parquet",
                "--tournament-quotes",
                "test_tournament_quotes.parquet",
                "--round-quotes",
                "test_round_quotes.parquet",
                "--candidate-catalog",
                "market_authority.csv",
                "--selection-freeze",
                "selection_freeze.json",
                "--prediction-authority",
                "prediction_authority.json",
                "--output-dir",
                "unused",
                "--baseline",
                "baseline",
            ]
        )
    assert reads == []


def test_full_prediction_authority_checks_freeze_status_years_menu_and_lambdas(
    tmp_path, monkeypatch
):
    base = tmp_path / "combined_base_predictions.parquet"
    rounds = tmp_path / "combined_round_predictions.parquet"
    freeze = tmp_path / "selection_freeze.json"
    base.write_bytes(b"sealed base")
    rounds.write_bytes(b"sealed rounds")
    freeze.write_bytes(b"sealed freeze")
    frozen_payload = {
        "contract_sha256": joint_eval.EXPECTED_CONTRACT_SHA256,
        "selected_candidate": "challenger",
        "baseline": "baseline",
        "selected_spec": {
            "candidate": "challenger",
            "course_fit_lambda": 0.5,
            "live_update_lambda": 0.5,
        },
        "primary_selection_outcome": PRIMARY_SELECTION_OUTCOME,
        "promotion_eligible": True,
        "selection_mode": PRIMARY_SELECTION_MODE,
        "diagnostic_fallback_rule": "not_applicable",
    }
    authority = {
        "schema_version": 1,
        "research_only": True,
        "production_files_written": False,
        "phase": "sealed_joint_market_prediction_panels",
        "contract_sha256": joint_eval.EXPECTED_CONTRACT_SHA256,
        "training_catalog_sha256": "a" * 64,
        "selection_freeze_sha256": _sha256_file(freeze),
        "selected_candidate": "challenger",
        "baseline": "baseline",
        "downstream_lambda": 0.5,
        "live_update_lambda": 0.5,
        "promotion_eligible": True,
        "diagnostic_fallback": False,
        "selection_mode": PRIMARY_SELECTION_MODE,
        "candidate_closure": ["baseline", "challenger"],
        "selection_years": [2023],
        "evaluation_years": [2024, 2025, 2026],
        "evaluation_outcome_quote_or_label_columns_read": [],
        "outputs": {
            "combined_base_predictions": {
                "path": str(base.resolve()),
                "sha256": _sha256_file(base),
                "rows": 10,
            },
            "combined_round_predictions": {
                "path": str(rounds.resolve()),
                "sha256": _sha256_file(rounds),
                "rows": 40,
            },
        },
    }

    monkeypatch.setattr(
        joint_eval,
        "validate_prediction_authority",
        lambda *args, **kwargs: copy.deepcopy(authority),
    )
    validated = validate_evaluation_prediction_authority(
        tmp_path / "prediction_authority.json",
        base_predictions_path=base,
        round_predictions_path=rounds,
        selection_freeze_path=freeze,
        frozen_payload=frozen_payload,
    )
    assert validated["candidate_closure"] == ["baseline", "challenger"]

    cases = []
    for message, mutation in (
        ("candidate closure", lambda value: value.update(candidate_closure=["challenger"])),
        ("selection years", lambda value: value.update(selection_years=[2022, 2023])),
        ("evaluation years", lambda value: value.update(evaluation_years=[2024, 2025])),
        ("selection mode", lambda value: value.update(selection_mode=DIAGNOSTIC_SELECTION_MODE)),
        ("promotion eligibility", lambda value: value.update(promotion_eligible=False)),
        ("diagnostic-fallback", lambda value: value.update(diagnostic_fallback=True)),
        ("downstream lambda", lambda value: value.update(downstream_lambda=1.0)),
        ("live lambda", lambda value: value.update(live_update_lambda=1.0)),
        ("selection-freeze hash", lambda value: value.update(selection_freeze_sha256="0" * 64)),
    ):
        bad = copy.deepcopy(authority)
        mutation(bad)
        cases.append((message, bad))
    bad_menu = copy.deepcopy(authority)
    bad_menu["outputs"]["unexpected_panel"] = dict(
        bad_menu["outputs"]["combined_base_predictions"]
    )
    cases.append(("output menu", bad_menu))
    bad_path = copy.deepcopy(authority)
    bad_path["outputs"]["combined_base_predictions"]["path"] = str(
        (tmp_path / "other.parquet").resolve()
    )
    cases.append(("path differs", bad_path))

    for message, bad in cases:
        monkeypatch.setattr(
            joint_eval,
            "validate_prediction_authority",
            lambda *args, _bad=bad, **kwargs: copy.deepcopy(_bad),
        )
        with pytest.raises(ValueError, match=message):
            validate_evaluation_prediction_authority(
                tmp_path / "prediction_authority.json",
                base_predictions_path=base,
                round_predictions_path=rounds,
                selection_freeze_path=freeze,
                frozen_payload=frozen_payload,
            )


def test_select_rejects_prediction_authority_before_any_data_read(monkeypatch):
    reads = []
    monkeypatch.setattr(
        joint_eval,
        "_read_prediction_frame_window",
        lambda *args, **kwargs: reads.append("prediction"),
    )
    monkeypatch.setattr(
        joint_eval,
        "_read_quote_frame_window",
        lambda *args, **kwargs: reads.append("quote"),
    )
    with pytest.raises(ValueError, match="evaluate-phase only"):
        joint_eval.main(
            [
                "--phase",
                "select",
                "--base-predictions",
                "base.parquet",
                "--round-predictions",
                "round.parquet",
                "--tournament-quotes",
                "tournament.parquet",
                "--round-quotes",
                "round_quotes.parquet",
                "--candidate-catalog",
                "catalog.csv",
                "--prediction-authority",
                "prediction_authority.json",
                "--output-dir",
                "unused",
                "--baseline",
                "baseline",
            ]
        )
    assert reads == []


def test_selection_rank_uses_flat_pnl_only_not_roi():
    summary_rows = []
    for market in MARKETS:
        for candidate, loss in (("baseline", 0.60), ("challenger", 0.61)):
            summary_rows.append(
                {
                    "candidate": candidate,
                    "market": market,
                    "events": 10,
                    "graded_pairs": 100,
                    "event_equal_three_way_log_loss": loss,
                    "event_equal_three_way_brier": loss / 2,
                    "event_equal_log_loss": loss,
                    "event_equal_brier": loss / 2,
                }
            )
    bet_rows = []
    for market in MARKETS:
        bet_rows.append(
            {
                "candidate": "baseline",
                "round_market": market,
                "sizing": "flat_1u",
                "stake": 1.0,
                "pnl": 0.10,
            }
        )
        for _ in range(10):
            bet_rows.append(
                {
                    "candidate": "challenger",
                    "round_market": market,
                    "sizing": "flat_1u",
                    "stake": 1.0,
                    "pnl": 0.02,
                }
            )
    joint, selected = joint_selection_summary(
        pd.DataFrame(summary_rows), "baseline", pd.DataFrame(bet_rows)
    )
    assert selected == "challenger"
    assert "roi_rank" not in joint.columns
    assert joint.loc[joint["candidate"].eq("challenger"), "aggregate_selection_pnl"].iloc[0] > joint.loc[
        joint["candidate"].eq("baseline"), "aggregate_selection_pnl"
    ].iloc[0]


def test_baseline_is_comparator_not_selectable_and_no_challenger_fails_closed():
    summary_rows = []
    for market in MARKETS:
        for candidate, loss in (("baseline", 0.50), ("challenger", 0.60)):
            summary_rows.append(
                {
                    "candidate": candidate,
                    "market": market,
                    "events": 10,
                    "graded_pairs": 100,
                    "event_equal_three_way_log_loss": loss,
                    "event_equal_three_way_brier": loss / 2,
                    "event_equal_log_loss": loss,
                    "event_equal_brier": loss / 2,
                }
            )

    # The baseline wins the proper-score rank, but the challenger wins P&L and
    # survives dominance.  A baseline/challenger rank tie must select challenger.
    opposing_bets = []
    for market in MARKETS:
        opposing_bets.extend(
            [
                {
                    "candidate": "baseline",
                    "round_market": market,
                    "sizing": "flat_1u",
                    "stake": 1.0,
                    "pnl": 0.0,
                },
                {
                    "candidate": "challenger",
                    "round_market": market,
                    "sizing": "flat_1u",
                    "stake": 1.0,
                    "pnl": 1.0,
                },
            ]
        )
    joint, selected = joint_selection_summary(
        pd.DataFrame(summary_rows), "baseline", pd.DataFrame(opposing_bets)
    )
    assert selected == "challenger"
    assert not joint.loc[
        joint["candidate"].eq("baseline"), "eligible_for_challenger_selection"
    ].iloc[0]

    dominated_bets = pd.DataFrame(opposing_bets).assign(
        pnl=lambda frame: np.where(frame["candidate"].eq("baseline"), 1.0, -1.0)
    )
    with pytest.raises(ValueError, match="no non-baseline challenger"):
        joint_selection_summary(
            pd.DataFrame(summary_rows), "baseline", dominated_bets
        )

    fallback_joint, fallback_selected = joint_selection_summary(
        pd.DataFrame(summary_rows),
        "baseline",
        dominated_bets,
        diagnostic_fallback_on_no_eligible=True,
    )
    assert fallback_selected == "challenger"
    assert fallback_joint["primary_selection_outcome"].eq(
        DIAGNOSTIC_SELECTION_OUTCOME
    ).all()
    assert fallback_joint["selection_mode"].eq(DIAGNOSTIC_SELECTION_MODE).all()
    assert fallback_joint["promotion_eligible"].eq(False).all()
    assert fallback_joint["qualified_winner"].eq(False).all()
    assert fallback_joint["diagnostic_fallback_rule"].eq(
        DIAGNOSTIC_FALLBACK_RULE
    ).all()


def test_diagnostic_fallback_freeze_and_evaluation_propagate_status(tmp_path):
    base, rounds, tournament_quotes, round_quotes = _synthetic_joint_inputs()
    # Make the exact control the strong arm and the sole challenger dominated.
    swap = {"baseline": "challenger", "challenger": "baseline"}
    base["candidate"] = base["candidate"].map(swap)
    rounds["candidate"] = rounds["candidate"].map(swap)
    failed_output = tmp_path / "failed_primary"
    with pytest.raises(ValueError, match="no non-baseline challenger"):
        select_joint_candidate_2023(
            base,
            rounds,
            tournament_quotes,
            round_quotes,
            failed_output,
            "baseline",
        )
    assert not failed_output.exists()

    output = tmp_path / "diagnostic_selection"
    selection = select_joint_candidate_2023(
        base,
        rounds,
        tournament_quotes,
        round_quotes,
        output,
        "baseline",
        diagnostic_fallback_on_no_eligible=True,
    )
    assert selection.selected_candidate == "challenger"
    assert selection.selection_mode == DIAGNOSTIC_SELECTION_MODE
    assert selection.primary_selection_outcome == DIAGNOSTIC_SELECTION_OUTCOME
    assert selection.promotion_eligible is False
    selection_frames = (
        selection.selection_summary,
        selection.selection_joint_summary,
        selection.mapper_crossfit,
        selection.common_menu_audit,
        selection.scored_selection,
        selection.selection_bets,
        selection.selection_betting_summary,
    )
    for frame in selection_frames:
        assert set(SELECTION_STATUS_FIELDS).issubset(frame.columns)
        assert frame["selection_mode"].eq(DIAGNOSTIC_SELECTION_MODE).all()
        assert frame["promotion_eligible"].eq(False).all()

    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "candidate": ["baseline", "challenger"],
            "course_fit_lambda": [1.0, 0.5],
            "live_update_lambda": [1.0, 0.5],
        }
    ).to_csv(catalog, index=False)
    freeze = write_selection_freeze(
        output,
        selection,
        baseline="baseline",
        min_ev=0.03,
        candidate_catalog_path=catalog,
    )
    payload = validate_selection_freeze(
        freeze,
        baseline="baseline",
        min_ev=0.03,
        selection_start="2023-04-01",
        selection_end="2023-12-31",
        candidate_catalog_path=catalog,
    )
    assert payload["selection_mode"] == DIAGNOSTIC_SELECTION_MODE
    assert payload["primary_selection_outcome"] == DIAGNOSTIC_SELECTION_OUTCOME
    assert payload["promotion_eligible"] is False
    assert payload["eligible_challenger_count"] == 0
    assert payload["selected_survived_dominance"] is False
    assert payload["frozen_spec"]["diagnostic_fallback_rule"] == DIAGNOSTIC_FALLBACK_RULE

    evaluation = evaluate_frozen_joint_candidate(
        base,
        rounds,
        tournament_quotes,
        round_quotes,
        tmp_path / "diagnostic_evaluation",
        "baseline",
        selection.selected_candidate,
        eval_end="2024-12-31",
        bootstrap_draws=20,
        selection_status={field: payload[field] for field in SELECTION_STATUS_FIELDS},
    )
    assert evaluation.selection_mode == DIAGNOSTIC_SELECTION_MODE
    assert evaluation.promotion_eligible is False
    evaluation_frames = (
        evaluation.evaluation_summary,
        evaluation.mapper_final,
        evaluation.common_menu_audit,
        evaluation.scored_evaluation,
        evaluation.bets,
        evaluation.betting_summary,
        evaluation.open_bets,
        evaluation.open_betting_summary,
        evaluation.edge_bucket_summary,
        evaluation.betting_breakdowns,
        evaluation.bootstrap,
    )
    for frame in evaluation_frames:
        assert set(SELECTION_STATUS_FIELDS).issubset(frame.columns)
        assert frame["selection_mode"].eq(DIAGNOSTIC_SELECTION_MODE).all()
        assert frame["promotion_eligible"].eq(False).all()
    assert evaluation.evaluation_summary["qualified_winner"].eq(False).all()


@pytest.mark.parametrize("phase", ["select", "evaluate"])
def test_authoritative_cli_requires_round_predictions(phase):
    arguments = [
        "--phase",
        phase,
        "--base-predictions",
        "base.parquet",
        "--tournament-quotes",
        "tournament.parquet",
        "--round-quotes",
        "round.parquet",
        "--output-dir",
        "output",
        "--baseline",
        "baseline",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)

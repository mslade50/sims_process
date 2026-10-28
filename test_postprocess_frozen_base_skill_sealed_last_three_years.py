from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from archive import market_backtest_scoring as scoring
from archive import postprocess_frozen_base_skill_sealed_last_three_years as post


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dependency_source_pins_match_and_fail_on_altered_pin() -> None:
    audit = post.verify_dependency_pins()
    assert set(audit) == set(post.DEPENDENCY_PINS)
    relative = next(iter(post.DEPENDENCY_PINS))
    broken = dict(post.DEPENDENCY_PINS)
    broken[relative] = (broken[relative][0], "0" * 64)
    with pytest.raises(ValueError, match="source pin mismatch"):
        post.verify_dependency_pins(broken)


def _inventory_fixture(tmp_path: Path) -> tuple[Path, dict]:
    output = tmp_path / "parent"
    bet = output / "bet_partitions" / "synthetic.parquet"
    bet.parent.mkdir(parents=True)
    bet.write_bytes(b"synthetic-bet-bytes")
    data = output / "data.bin"
    data.write_bytes(b"bound-data")
    records = [
        {"path": "bet_partitions/synthetic.parquet", "bytes": bet.stat().st_size, "sha256": _hash(bet)},
        {"path": "data.bin", "bytes": data.stat().st_size, "sha256": _hash(data)},
    ]
    return output, {"artifacts": records}


def test_artifact_inventory_rejects_tamper_extra_and_path_escape(tmp_path: Path) -> None:
    output, manifest = _inventory_fixture(tmp_path)
    paths = post.validate_artifact_inventory(output, manifest, required=())
    assert set(paths) == {"bet_partitions/synthetic.parquet", "data.bin"}

    (output / "data.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes differ"):
        post.validate_artifact_inventory(output, manifest, required=())

    output, manifest = _inventory_fixture(tmp_path / "extra")
    (output / "untracked.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs"):
        post.validate_artifact_inventory(output, manifest, required=())

    output, manifest = _inventory_fixture(tmp_path / "escape")
    manifest["artifacts"][0]["path"] = "../escape.parquet"
    with pytest.raises(ValueError, match="unsafe|escapes"):
        post.validate_artifact_inventory(output, manifest, required=())


def test_completed_source_requires_external_parent_manifest_anchor(tmp_path: Path) -> None:
    output = tmp_path / "parent-anchor"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="externally supplied hash"):
        post.validate_completed_source_output(output, "0" * 64)
    with pytest.raises(ValueError, match="64 hex"):
        post.validate_completed_source_output(output, "not-a-hash")


def _logical_boundaries() -> pd.DataFrame:
    rows = []
    dates = [
        post.SLICE_START - pd.Timedelta(days=1),
        post.SLICE_START,
        post.SLICE_END,
    ]
    for index, date in enumerate(dates):
        for candidate, probability in ((post.BASELINE, 0.5), ("challenger", 0.6)):
            rows.append(
                {
                    "candidate": candidate,
                    "mapped_event_key": f"event-{index}",
                    "event_start_date": date,
                    "market": "72h",
                    "pair_key": "1|2",
                    "canonical_outcome": 1.0,
                    "canonical_probability": probability,
                    "canonical_push_probability": 0.0,
                    "sigma": 1.5,
                    "tie_threshold": 0.1,
                    "probability_fit_end": pd.Timestamp("2022-12-31"),
                }
            )
    return pd.DataFrame(rows)


def test_logical_filter_uses_exact_inclusive_boundaries_and_fails_closed() -> None:
    logical = _logical_boundaries()
    sliced = post.validate_and_slice_logical(logical)
    assert sorted(sliced["event_start_date"].unique()) == [
        post.SLICE_START.to_datetime64(),
        post.SLICE_END.to_datetime64(),
    ]

    after_parent = logical.iloc[[0]].copy()
    after_parent["event_start_date"] = post.SLICE_END + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="exceed the full source period"):
        post.validate_and_slice_logical(after_parent)

    duplicate = pd.concat([logical, logical.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        post.validate_and_slice_logical(duplicate)

    disagreement = logical.copy()
    row = disagreement[
        disagreement["candidate"].eq("challenger")
        & disagreement["mapped_event_key"].eq("event-1")
    ].index[0]
    disagreement.loc[row, "canonical_outcome"] = 0.0
    with pytest.raises(ValueError, match="outcomes disagree"):
        post.validate_and_slice_logical(disagreement)

    future_fit = logical.copy()
    future_fit["probability_fit_end"] = pd.Timestamp("2023-01-01")
    with pytest.raises(ValueError, match="mapper fit after discovery"):
        post.validate_and_slice_logical(future_fit)

    malformed = logical.copy()
    malformed["canonical_outcome"] = malformed["canonical_outcome"].astype(object)
    malformed.loc[0, "canonical_outcome"] = "garbage"
    with pytest.raises(ValueError, match="nonnumeric or boolean"):
        post.validate_and_slice_logical(malformed)

    boolean_outcome = logical.copy()
    boolean_outcome["canonical_outcome"] = boolean_outcome[
        "canonical_outcome"
    ].astype(object)
    boolean_outcome.loc[0, "canonical_outcome"] = True
    with pytest.raises(ValueError, match="nonnumeric or boolean"):
        post.validate_and_slice_logical(boolean_outcome)


def test_proper_scores_are_event_equal_and_condition_on_push() -> None:
    rows = []
    event_specs = [
        ("event-a", 1, 0.72, 0.20),
        ("event-b", 3, 0.10, 0.00),
    ]
    for event, pairs, probability, push in event_specs:
        for pair in range(pairs):
            rows.append(
                {
                    "candidate": "challenger",
                    "mapped_event_key": event,
                    "event_start_date": post.SLICE_START,
                    "market": "72h",
                    "pair_key": f"{pair}|{pair+10}",
                    "canonical_outcome": 1.0,
                    "canonical_probability": probability,
                    "canonical_push_probability": push,
                    "sigma": 1.5,
                    "tie_threshold": 0.1,
                }
            )
    logical = pd.DataFrame(rows)
    benchmark = logical[["mapped_event_key", "market", "pair_key", "canonical_outcome"]].rename(
        columns={"canonical_outcome": "quote_canonical_outcome"}
    )
    benchmark["canonical_left_close"] = -110.0
    benchmark["canonical_right_close"] = -110.0
    result = post.proper_scores_from_logical(logical, benchmark).iloc[0]
    expected = (-np.log(0.9) - np.log(0.1)) / 2.0
    pooled = (-np.log(0.9) - 3.0 * np.log(0.1)) / 4.0
    assert np.isclose(result["event_equal_log_loss"], expected)
    assert not np.isclose(result["event_equal_log_loss"], pooled)
    assert result["events"] == 2


def _valid_bet(
    *,
    candidate: str = "challenger",
    event: str = "event-a",
    date: pd.Timestamp = post.SLICE_START,
    outcome: float = 1.0,
    comparison: str = "challenger",
    opportunity: str = "all_available",
) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "candidate": [candidate],
            "event_id": [event],
            "event_key": [event],
            "event_start_date": [date],
            "round_market": ["72h"],
            "market": ["tournament_matchup"],
            "pair_key": ["1|2"],
            "selection_key": [f"{candidate}|quote|p1"],
            "selection_id": ["1"],
            "selection_name": ["one"],
            "opponent_id": ["2"],
            "book": ["pinnacle"],
            "close_odds": [100.0],
            "probability": [0.70],
            "push_probability": [0.0],
            "tie_rule": ["ties push"],
            "outcome_numeric": [outcome],
            "outcome": ["win" if outcome == 1.0 else "loss"],
            "pair_reliability_ok": [True],
            "stake": [1.0],
        }
    )
    settled = scoring.settle_bets(raw)
    decimal = float(scoring.american_decimal_odds(100.0))
    settled["ev_per_unit"] = 0.70 * (decimal - 1.0) - (1.0 - 0.70)
    settled["price_basis"] = "close"
    settled["line_policy"] = "pinnacle"
    settled["sizing"] = "flat_1u"
    settled["reliability_policy"] = "ungated"
    settled["comparison_candidate"] = comparison
    settled["opportunity_mode"] = opportunity
    settled["open_to_close_clv_pp"] = 0.0
    return settled


def test_materialized_bet_reconstructs_and_rejects_forgery() -> None:
    valid = _valid_bet()
    checked = post.validate_materialized_bets(valid, "synthetic.parquet")
    assert checked.iloc[0]["pnl"] == 1.0

    forged = valid.copy()
    forged["pnl"] += 1e-6
    with pytest.raises(ValueError, match="P&L does not reconstruct"):
        post.validate_materialized_bets(forged, "synthetic.parquet")

    string_bool = valid.copy()
    string_bool["pair_reliability_ok"] = "false"
    with pytest.raises(ValueError, match="exact booleans"):
        post.validate_materialized_bets(string_bool, "synthetic.parquet")

    ungraded = valid.copy()
    ungraded["settlement_status"] = "ungraded"
    with pytest.raises(ValueError, match="ungraded"):
        post.validate_materialized_bets(ungraded, "synthetic.parquet")

    near_win = valid.copy()
    near_win["outcome_numeric"] = 1.000001
    with pytest.raises(ValueError, match="not exactly"):
        post.validate_materialized_bets(near_win, "synthetic.parquet")

    unsuppressed = valid.copy()
    unsuppressed["reliability_policy"] = "suppress_unreliable_ge10pct"
    unsuppressed["pair_reliability_ok"] = False
    with pytest.raises(ValueError, match="reliability suppression"):
        post.validate_materialized_bets(unsuppressed, "synthetic.parquet")

    wrong_menu = valid.copy()
    wrong_menu["comparison_candidate"] = "someone-else"
    with pytest.raises(ValueError, match="different comparison"):
        post.validate_materialized_bets(wrong_menu, "synthetic.parquet")

    forged_push = _valid_bet(outcome=0.5)
    forged_push["pnl"] = 1e-11
    with pytest.raises(ValueError, match="P&L does not reconstruct|zero-P&L"):
        post.validate_materialized_bets(forged_push, "synthetic.parquet")

    boolean_outcome = valid.copy()
    boolean_outcome["outcome_numeric"] = True
    with pytest.raises(ValueError, match="nonnumeric or boolean"):
        post.validate_materialized_bets(boolean_outcome, "synthetic.parquet")


def test_materialized_bet_is_bound_to_quote_identity_price_and_probability() -> None:
    bets = _valid_bet()
    quotes = pd.DataFrame(
        {
            "quote_id": ["quote"],
            "mapped_event_key": ["event-a"],
            "event_start_date": [post.SLICE_START],
            "market": ["72h"],
            "book": ["pinnacle"],
            "pair_key": ["1|2"],
            "p1_id": ["1"],
            "p2_id": ["2"],
            "p1_open": [95.0],
            "p2_open": [-105.0],
            "p1_close": [100.0],
            "p2_close": [-110.0],
            "p1_outcome": [1.0],
            "p2_outcome": [0.0],
            "p1_outcome_text": ["win"],
            "p2_outcome_text": ["loss"],
            "tie_rule": ["ties push"],
        }
    )
    logical = pd.DataFrame(
        {
            "candidate": ["challenger"],
            "mapped_event_key": ["event-a"],
            "market": ["72h"],
            "pair_key": ["1|2"],
            "canonical_probability": [0.70],
            "canonical_push_probability": [0.0],
        }
    )
    audit = post.validate_bets_against_authorities(bets, quotes, logical)
    assert audit["bet_rows_bound_to_quote_identity"] == 1

    forged_identity = bets.copy()
    forged_identity["selection_id"] = "999"
    with pytest.raises(ValueError, match="selection/opponent IDs"):
        post.validate_bets_against_authorities(
            forged_identity, quotes, logical
        )
    forged_price = bets.copy()
    forged_price["close_odds"] = 101.0
    with pytest.raises(ValueError, match="execution odds"):
        post.validate_bets_against_authorities(forged_price, quotes, logical)
    forged_probability = bets.copy()
    forged_probability["probability"] = 0.71
    with pytest.raises(ValueError, match="probability differs"):
        post.validate_bets_against_authorities(
            forged_probability, quotes, logical
        )

    extra = pd.concat(
        [
            quotes,
            quotes.assign(
                quote_id="quote-betcris",
                book="betcris",
                p1_close=105.0,
                p2_close=-115.0,
            ),
            quotes.assign(
                quote_id="quote-betonline",
                book="betonline",
                p1_close=105.0,
                p2_close=-115.0,
            ),
        ],
        ignore_index=True,
    )
    best = bets.copy()
    best["selection_key"] = "challenger|quote-betcris|p1"
    best["book"] = "betcris"
    best["close_odds"] = 105.0
    best["line_policy"] = "best_of_three"
    best = scoring.settle_bets(best)
    best_decimal = float(scoring.american_decimal_odds(105.0))
    best["ev_per_unit"] = 0.70 * (best_decimal - 1.0) - (1.0 - 0.70)
    post.validate_materialized_bets(best, "synthetic-best.parquet")
    best_audit = post.validate_bets_against_authorities(best, extra, logical)
    assert best_audit["best_of_three_rows_reconstructed"] == 1
    wrong_tie_book = best.copy()
    wrong_tie_book["selection_key"] = "challenger|quote-betonline|p1"
    wrong_tie_book["book"] = "betonline"
    with pytest.raises(ValueError, match="pinned best valid price"):
        post.validate_bets_against_authorities(wrong_tie_book, extra, logical)


def test_run_rejects_output_nested_inside_immutable_parent_before_write(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "sealed-parent"
    parent.mkdir()
    nested = parent / "postprocess"
    args = SimpleNamespace(
        bootstrap=10_000,
        block_sizes=(2, 3, 4),
        sealed_output=parent,
        output_dir=nested,
    )
    with pytest.raises(ValueError, match="must not equal or be nested"):
        post.run(args)
    assert not nested.exists()


def test_summary_preserves_challenger_specific_pairwise_baselines_and_zero_grid() -> None:
    c1 = "challenger-1"
    c2 = "challenger-2"
    bets = pd.concat(
        [
            _valid_bet(candidate=post.BASELINE, comparison=post.BASELINE),
            _valid_bet(candidate=c1, comparison=c1),
            _valid_bet(candidate=c2, comparison=c2),
            _valid_bet(
                candidate=post.BASELINE,
                comparison=c1,
                opportunity="pairwise_common_with_strict_pt_baseline",
                outcome=1.0,
            ),
            _valid_bet(
                candidate=c1,
                comparison=c1,
                opportunity="pairwise_common_with_strict_pt_baseline",
                outcome=0.0,
            ),
            _valid_bet(
                candidate=post.BASELINE,
                comparison=c2,
                opportunity="pairwise_common_with_strict_pt_baseline",
                outcome=0.0,
            ),
            _valid_bet(
                candidate=c2,
                comparison=c2,
                opportunity="pairwise_common_with_strict_pt_baseline",
                outcome=1.0,
            ),
        ],
        ignore_index=True,
    )
    performance, events = post.summarize_bets(
        bets, [post.BASELINE, c1, c2]
    )
    key = (
        performance["opportunity_mode"].eq(
            "pairwise_common_with_strict_pt_baseline"
        )
        & performance["candidate"].eq(post.BASELINE)
        & performance["market"].eq("72h")
        & performance["price_basis"].eq("close")
        & performance["line_policy"].eq("pinnacle")
        & performance["sizing"].eq("flat_1u")
        & performance["reliability_policy"].eq("ungated")
    )
    rows = performance.loc[key].set_index("comparison_candidate")
    assert rows.loc[c1, "pnl"] == 1.0
    assert rows.loc[c2, "pnl"] == -1.0
    empty = performance.loc[
        performance["market"].eq("r1")
        & performance["price_basis"].eq("open")
    ]
    assert (empty["bets"] == 0).all()
    assert not events.empty


def test_money_event_padding_distinguishes_zero_pnl_from_true_no_bet() -> None:
    challenger = "challenger"
    events = pd.DataFrame(
        {
            "mapped_event_key": ["event-a", "event-b", "event-a", "event-b"],
            "event_start_date": [post.SLICE_START] * 4,
            "market": ["72h", "72h", "r1", "r1"],
        }
    )
    event_performance = pd.DataFrame(
        [
            {
                "event_key": "event-a",
                "event_start_date": post.SLICE_START,
                "pnl": 0.0,
                "stake": 1.0,
                "bets": 1,
                "candidate": candidate,
                "comparison_candidate": comparison,
                "market": "72h",
                "opportunity_mode": opportunity,
                "price_basis": "close",
                "line_policy": "pinnacle",
                "sizing": "flat_1u",
                "reliability_policy": "ungated",
            }
            for opportunity, comparison, candidate in (
                ("all_available", post.BASELINE, post.BASELINE),
                ("all_available", challenger, challenger),
                (
                    "pairwise_common_with_strict_pt_baseline",
                    challenger,
                    post.BASELINE,
                ),
                (
                    "pairwise_common_with_strict_pt_baseline",
                    challenger,
                    challenger,
                ),
            )
        ]
    )
    padded = post.build_money_event_deltas_exact(
        event_performance, events, [post.BASELINE, challenger], post.BASELINE
    )
    target = padded.loc[
        padded["candidate"].eq(challenger)
        & padded["opportunity_mode"].eq("all_available")
        & padded["market"].eq("72h")
        & padded["price_basis"].eq("close")
        & padded["line_policy"].eq("pinnacle")
        & padded["sizing"].eq("flat_1u")
        & padded["reliability_policy"].eq("ungated")
    ].set_index("event_key")
    assert target.loc["event-a", "zero_bet_event"]
    assert not target.loc["event-a", "true_no_bet_event"]
    assert target.loc["event-b", "zero_bet_event"]
    assert target.loc["event-b", "true_no_bet_event"]


def test_parent_delta_comparison_rejects_one_cell_tamper() -> None:
    frame = pd.DataFrame(
        {
            "candidate": ["a"],
            "event_key": ["e"],
            "event_delta": [1.0],
        }
    )
    post.assert_filtered_parent_deltas_match(
        frame,
        frame.copy(),
        key_columns=["candidate", "event_key"],
        value_columns=["event_delta"],
        label="synthetic",
    )
    forged = frame.copy()
    forged["event_delta"] += 1e-6
    with pytest.raises(ValueError, match="differs in event_delta"):
        post.assert_filtered_parent_deltas_match(
            frame,
            forged,
            key_columns=["candidate", "event_key"],
            value_columns=["event_delta"],
            label="synthetic",
        )

    boolean = frame.assign(zero_bet_event=True)
    forged_bool = boolean.copy()
    forged_bool["zero_bet_event"] = "False"
    with pytest.raises(ValueError, match="exact booleans"):
        post.assert_filtered_parent_deltas_match(
            boolean,
            forged_bool,
            key_columns=["candidate", "event_key"],
            value_columns=["zero_bet_event"],
            label="synthetic bool",
        )


def test_bootstrap_and_block_max_stat_are_deterministic_on_zero_padded_panel() -> None:
    rows = []
    for candidate, scale in (("a", 1.0), ("b", -0.5)):
        for index in range(8):
            rows.append(
                {
                    "event_key": f"event-{index}",
                    "event_start_date": post.SLICE_START + pd.Timedelta(days=7 * index),
                    "event_delta": scale * (index - 2.5),
                    "candidate": candidate,
                    "baseline": post.BASELINE,
                    "opportunity_mode": "all_available",
                    "market": "72h",
                    "price_basis": "close",
                    "line_policy": "best_of_three",
                    "sizing": "flat_1u",
                    "reliability_policy": "ungated",
                    "metric": "pnl_delta",
                    "zero_bet_event": index == 0,
                }
            )
    fixture = pd.DataFrame(rows)
    first = post.sealed.event_cluster_bootstrap(
        fixture, draws=10_000, seed=17, better_when="positive"
    )
    second = post.sealed.event_cluster_bootstrap(
        fixture, draws=10_000, seed=17, better_when="positive"
    )
    pd.testing.assert_frame_equal(first, second)
    reality_first = post.sealed.block_max_stat_reality_check(
        fixture, draws=10_000, seed=19, block_sizes=(2, 3, 4)
    )
    reality_second = post.sealed.block_max_stat_reality_check(
        fixture, draws=10_000, seed=19, block_sizes=(2, 3, 4)
    )
    pd.testing.assert_frame_equal(reality_first, reality_second)
    assert set(reality_first["block_size_events"]) == {2, 3, 4}

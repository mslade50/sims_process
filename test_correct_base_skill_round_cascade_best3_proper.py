from __future__ import annotations

import pandas as pd
import pytest

from archive import correct_base_skill_round_cascade_best3_proper as correction


def test_best_price_selection_is_isolated_by_model() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": "m0",
                "event_key": "event",
                "round_market": "r1",
                "pair_key": "a|b",
                "book": "betonline",
                "p1_close": -105,
                "close_time": "2026-01-01T12:00:00Z",
            },
            {
                "model": "m0",
                "event_key": "event",
                "round_market": "r1",
                "pair_key": "a|b",
                "book": "pinnacle",
                "p1_close": 100,
                "close_time": "2026-01-01T12:01:00Z",
            },
            {
                "model": "m1",
                "event_key": "event",
                "round_market": "r1",
                "pair_key": "a|b",
                "book": "betcris",
                "p1_close": 110,
                "close_time": "2026-01-01T12:02:00Z",
            },
            {
                "model": "m1",
                "event_key": "event",
                "round_market": "r1",
                "pair_key": "a|b",
                "book": "pinnacle",
                "p1_close": 105,
                "close_time": "2026-01-01T12:03:00Z",
            },
        ]
    )
    selected = correction.model_scoped_best_price_market_rows(frame, "close")
    assert len(selected) == 2
    assert dict(zip(selected["model"], selected["book"])) == {
        "m0": "pinnacle",
        "m1": "betcris",
    }


def test_best_price_selection_requires_model() -> None:
    with pytest.raises(ValueError, match="lacks model"):
        correction.model_scoped_best_price_market_rows(
            pd.DataFrame(
                {
                    "event_key": ["event"],
                    "round_market": ["r1"],
                    "pair_key": ["a|b"],
                    "book": ["pinnacle"],
                    "p1_close": [100],
                }
            ),
            "close",
        )


def _closed_summary() -> pd.DataFrame:
    models = [f"m{index}" for index in range(correction.EXPECTED_MODELS)]
    rows = []
    for sample in ("discovery", "sealed", "comparable_3y"):
        for model in models:
            for round_market in ("r1", "r2", "r3", "r4", "pooled"):
                for line_policy in (
                    *correction.cascade.BOOKS,
                    "best_of_three",
                ):
                    for price_point in correction.cascade.PRICE_POINTS:
                        rows.append(
                            {
                                "sample": sample,
                                "model": model,
                                "candidate": f"candidate-{model}",
                                "variant": "variant",
                                "round_market": round_market,
                                "line_policy": line_policy,
                                "price_point": price_point,
                            }
                        )
    return pd.DataFrame(rows)


def test_summary_closure_requires_every_model_policy_round_and_sample() -> None:
    frame = _closed_summary()
    correction.validate_summary_closure(
        frame,
        models=frame["model"].unique(),
        label="synthetic",
    )
    with pytest.raises(ValueError, match="logical closure differs"):
        correction.validate_summary_closure(
            frame.iloc[:-1],
            models=frame["model"].unique(),
            label="synthetic",
        )

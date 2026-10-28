"""Cheap workload audit for ``evaluate_shot_quality_tilt.py``.

This does not reproduce or score the model.  It counts the two quantities that
dominate the target-scoring kernel: retained shot rows and player/state cells.
The latter is a close upper bound for the evaluator's exact floating-ability
cells because category skill is constant within player/event.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from archive.evaluate_shot_quality_tilt import (
    ALL_CATEGORIES,
    DEFAULT_DB,
    DEFAULT_EXPORT,
    _distance_state,
    _lie,
    available_event_keys,
    load_calendar,
    select_calendar,
)


def main() -> None:
    calendar = select_calendar(
        load_calendar(DEFAULT_DB, 2020, 2026),
        available_event_keys(DEFAULT_EXPORT, range(2020, 2027)),
        None,
    )
    warmup_date = calendar["event_start"].min()
    columns = [
        "event_key",
        "source_player_id",
        "sg_category",
        "start_lie",
        "start_distance_yd",
        "sg_raw",
        "sg_valid",
        "complete_hole",
    ]
    rows: list[dict[str, int]] = []
    for season, events in calendar.groupby("season", sort=True):
        frame = pd.read_parquet(
            DEFAULT_EXPORT / f"pga_{int(season)}_shot_sg.parquet",
            columns=columns,
            filters=[("event_key", "in", events["event_key"].astype(str).tolist())],
        )
        frame = frame[
            frame["sg_category"].isin(ALL_CATEGORIES)
            & frame["sg_valid"].eq(1)
            & frame["complete_hole"].eq(1)
            & frame["sg_raw"].notna()
            & frame["start_distance_yd"].notna()
        ].copy()
        frame = frame.merge(
            events[["event_key", "event_start"]],
            on="event_key",
            how="inner",
            validate="many_to_one",
        )
        # The first date trains the one frozen green curve and is not scored.
        frame = frame[frame["event_start"].gt(warmup_date)].copy()
        frame["lie_state"] = frame["start_lie"].map(_lie)
        frame["distance_state"] = ""
        for category in ALL_CATEGORIES:
            mask = frame["sg_category"].eq(category)
            frame.loc[mask, "distance_state"] = _distance_state(
                category,
                frame.loc[mask, "start_distance_yd"].to_numpy(float),
            )
        frame["exact_state"] = (
            frame["sg_category"].astype(str)
            + "|"
            + frame["lie_state"].astype(str)
            + "|"
            + frame["distance_state"].astype(str)
        )
        grouped = frame.groupby(
            ["event_key", "sg_category", "exact_state"], observed=True
        )
        cells = grouped["source_player_id"].nunique()
        rows.append(
            {
                "season": int(season),
                "events": int(frame["event_key"].nunique()),
                "scored_rows_proxy": int(len(frame)),
                "state_groups": int(len(cells)),
                "player_state_cells_proxy": int(cells.sum()),
            }
        )
    result = pd.DataFrame(rows)
    result["row_share"] = result["scored_rows_proxy"] / result["scored_rows_proxy"].sum()
    result["cell_share"] = (
        result["player_state_cells_proxy"]
        / result["player_state_cells_proxy"].sum()
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

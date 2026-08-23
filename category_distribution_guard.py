"""Validation for production category-first distribution inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


BASE_NUMERIC_COLUMNS = ("mean", "std", "skew", "n_eff")


def _normalise_names(
    values: Sequence[Any],
    replacements: Mapping[Any, Any] | None,
) -> list[str]:
    names = pd.Series(list(values), dtype="object").map(
        lambda value: "" if pd.isna(value) else str(value).casefold().strip()
    )
    if replacements:
        names = names.replace(dict(replacements))
    return names.map(
        lambda value: "" if pd.isna(value) else str(value).casefold().strip()
    ).tolist()


def _preview_pairs(pairs: Sequence[tuple[Any, Any]], *, limit: int = 8) -> str:
    rendered = [f"{player}/{category}" for player, category in pairs[:limit]]
    if len(pairs) > limit:
        rendered.append(f"... +{len(pairs) - limit} more")
    return ", ".join(rendered)


def require_complete_category_distributions(
    distributions: pd.DataFrame,
    active_player_names: Sequence[Any],
    category_order: Sequence[str],
    *,
    name_replacements: Mapping[Any, Any] | None = None,
    source_label: str = "category distribution input",
    extra_numeric_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Return normalized inputs only when every active player is fully covered.

    Category-first production draws must never synthesize an active player's
    mean, variance, or skew inputs from field-wide values.  The returned frame
    has normalized names/categories and numeric value columns; callers can
    safely pivot it and use direct indexed lookups for the active field.
    """
    numeric_columns = tuple(dict.fromkeys(
        (*BASE_NUMERIC_COLUMNS, *extra_numeric_columns)
    ))
    required_columns = {"player_name", "category_clean", *numeric_columns}
    missing_columns = required_columns - set(distributions.columns)
    if missing_columns:
        raise ValueError(
            f"{source_label} missing required category-first columns: "
            f"{sorted(missing_columns)}"
    )

    categories = [str(category).casefold().strip() for category in category_order]
    if (
        not categories
        or len(categories) != len(set(categories))
        or any(not category for category in categories)
    ):
        raise ValueError("Production category order is empty, duplicated, or invalid")

    players = _normalise_names(active_player_names, name_replacements)
    if not players:
        raise ValueError("Category-first simulation has no active players")
    if any(not player for player in players):
        raise ValueError("Category-first active player names contain blank values")
    if len(players) != len(set(players)):
        raise ValueError(
            "Category-first active player names are not unique after normalization"
        )

    frame = distributions.copy()
    frame["player_name"] = _normalise_names(
        frame["player_name"].tolist(), name_replacements
    )
    frame["category_clean"] = (
        frame["category_clean"].astype("string").fillna("").str.casefold().str.strip()
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    required_rows = frame[frame["category_clean"].isin(categories)]
    duplicate_mask = required_rows.duplicated(
        ["player_name", "category_clean"], keep=False
    )
    if duplicate_mask.any():
        pairs = list(dict.fromkeys(
            required_rows.loc[
                duplicate_mask, ["player_name", "category_clean"]
            ].itertuples(index=False, name=None)
        ))
        raise ValueError(
            f"{source_label} has duplicate player/category rows: "
            f"{_preview_pairs(pairs)}"
        )

    active_rows = required_rows[required_rows["player_name"].isin(players)]
    actual_index = pd.MultiIndex.from_frame(
        active_rows[["player_name", "category_clean"]]
    )
    expected_index = pd.MultiIndex.from_product(
        [players, categories], names=["player_name", "category_clean"]
    )
    missing_index = expected_index.difference(actual_index, sort=False)
    if len(missing_index):
        raise ValueError(
            f"{source_label} is missing active-field category coverage: "
            f"{_preview_pairs(list(missing_index))}"
        )

    aligned = active_rows.set_index(["player_name", "category_clean"]).reindex(
        expected_index
    )
    values = aligned.loc[:, numeric_columns].to_numpy(dtype=float)
    bad_rows, bad_columns = np.where(~np.isfinite(values))
    if len(bad_rows):
        cells = [
            (
                f"{expected_index[row][0]}/{expected_index[row][1]}:"
                f"{numeric_columns[column]}"
            )
            for row, column in zip(bad_rows[:8], bad_columns[:8])
        ]
        if len(bad_rows) > 8:
            cells.append(f"... +{len(bad_rows) - 8} more")
        raise ValueError(
            f"{source_label} has non-finite active-field values: {', '.join(cells)}"
        )

    invalid_std = aligned["std"].to_numpy(dtype=float) <= 0.0
    if invalid_std.any():
        pairs = [expected_index[index] for index in np.flatnonzero(invalid_std)]
        raise ValueError(
            f"{source_label} has non-positive active-field standard deviations: "
            f"{_preview_pairs(pairs)}"
        )
    invalid_neff = aligned["n_eff"].to_numpy(dtype=float) < 0.0
    if invalid_neff.any():
        pairs = [expected_index[index] for index in np.flatnonzero(invalid_neff)]
        raise ValueError(
            f"{source_label} has negative active-field effective sample sizes: "
            f"{_preview_pairs(pairs)}"
        )

    return frame, players

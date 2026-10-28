"""Build score-audited physical-tie authority for 72-hole matchup quotes.

DataGolf's numeric 0.5/0.5 plus text ``void`` is a settlement encoding, not by
itself proof of a physical 72-hole tie.  This research helper verifies those
rows against the last available cumulative score for both golfers.  It adds
only three evaluator fields to the original normalized quote cache:

``physical_p1_outcome``
    0.5 only when both golfers completed the same number of rounds at the same
    cumulative score.  Other quote outcomes remain null and are interpreted
    from the original archive by the evaluator.
``no_score_void``
    A numeric 0.5/0.5 settlement for which either final score is unavailable.
``other_void_with_scores``
    A numeric 0.5/0.5 settlement with scores that do not prove a physical tie.

The helper is research-only and never edits its source files or production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from archive.all_round_joint_market_eval import _stable_id
except ModuleNotFoundError:  # direct ``python archive/...py`` execution
    from all_round_joint_market_eval import _stable_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_tournament_score_tie_authority(
    quotes: pd.DataFrame, player_rounds: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the quote cache with score-verified 0.5/0.5 classifications."""

    output = quotes.copy()
    if "market" in output:
        source_market = (
            output["market"]
            .astype(str)
            .str.casefold()
            .str.replace(r"[^a-z0-9]", "", regex=True)
        )
        tournament_mask = source_market.isin(
            {
                "72h",
                "72hole",
                "72holematchup",
                "72holematchups",
                "tournamentmatchup",
            }
        )
        output = output.loc[tournament_mask].copy()
        if output.empty:
            raise ValueError("quote source contains no 72-hole matchup rows")
    event_column = "mapped_event_key" if "mapped_event_key" in output else "event_key"
    p1_column = "p1_id" if "p1_id" in output else "p1_dg_id"
    p2_column = "p2_id" if "p2_id" in output else "p2_dg_id"
    required_quotes = {
        event_column,
        p1_column,
        p2_column,
        "p1_outcome",
        "p2_outcome",
    }
    missing_quotes = required_quotes.difference(output.columns)
    if missing_quotes:
        raise ValueError(f"tournament quotes missing columns: {sorted(missing_quotes)}")
    required_rounds = {
        "event_key",
        "dg_id",
        "round_num",
        "round_date",
        "cumulative_score",
    }
    missing_rounds = required_rounds.difference(player_rounds.columns)
    if missing_rounds:
        raise ValueError(f"player round panel missing columns: {sorted(missing_rounds)}")

    work = pd.DataFrame(index=output.index)
    work["event_key"] = output[event_column].map(_stable_id)
    work["p1_id"] = output[p1_column].map(_stable_id)
    work["p2_id"] = output[p2_column].map(_stable_id)
    p1_outcome = pd.to_numeric(output["p1_outcome"], errors="coerce")
    p2_outcome = pd.to_numeric(output["p2_outcome"], errors="coerce")
    work["numeric_half_half"] = np.isclose(p1_outcome, 0.5) & np.isclose(
        p2_outcome, 0.5
    )

    rounds = player_rounds[list(required_rounds)].copy()
    rounds["event_key"] = rounds["event_key"].map(_stable_id)
    rounds["dg_id"] = rounds["dg_id"].map(_stable_id)
    rounds["round_num"] = pd.to_numeric(rounds["round_num"], errors="coerce")
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce")
    rounds["cumulative_score"] = pd.to_numeric(
        rounds["cumulative_score"], errors="coerce"
    )
    rounds = (
        rounds.sort_values(
            ["event_key", "dg_id", "round_num", "round_date"], kind="mergesort"
        )
        .drop_duplicates(["event_key", "dg_id"], keep="last")
        .reset_index(drop=True)
    )

    p1 = rounds.rename(
        columns={
            "dg_id": "p1_id",
            "round_num": "p1_final_round",
            "cumulative_score": "p1_final_score",
        }
    )[["event_key", "p1_id", "p1_final_round", "p1_final_score"]]
    p2 = rounds.rename(
        columns={
            "dg_id": "p2_id",
            "round_num": "p2_final_round",
            "cumulative_score": "p2_final_score",
        }
    )[["event_key", "p2_id", "p2_final_round", "p2_final_score"]]
    audited = work.merge(p1, on=["event_key", "p1_id"], how="left", validate="many_to_one")
    audited = audited.merge(
        p2, on=["event_key", "p2_id"], how="left", validate="many_to_one"
    )
    both_scores = audited["p1_final_score"].notna() & audited["p2_final_score"].notna()
    same_completion = audited["p1_final_round"].eq(audited["p2_final_round"])
    equal_score = np.isclose(
        audited["p1_final_score"], audited["p2_final_score"], equal_nan=False
    )
    half_half = audited["numeric_half_half"].astype(bool)
    verified_tie = half_half & both_scores & same_completion & equal_score
    no_score_void = half_half & ~both_scores
    other_void = half_half & both_scores & ~verified_tie

    output["physical_p1_outcome"] = np.where(verified_tie, 0.5, np.nan)
    output["no_score_void"] = no_score_void.to_numpy(bool)
    output["other_void_with_scores"] = other_void.to_numpy(bool)

    if "pair_key" in output:
        logical_pair = output["pair_key"].astype(str)
    else:
        logical_pair = pd.Series(
            ["|".join(sorted(pair)) for pair in zip(work["p1_id"], work["p2_id"])],
            index=output.index,
        )
    logical_rows = pd.DataFrame(
        {
            "event_key": work["event_key"],
            "pair_key": logical_pair,
            "half_half": half_half.to_numpy(bool),
            "verified_tie": verified_tie.to_numpy(bool),
            "no_score_void": no_score_void.to_numpy(bool),
            "other_void": other_void.to_numpy(bool),
        }
    )
    logical = (
        logical_rows.loc[logical_rows["half_half"]]
        .groupby(["event_key", "pair_key"], as_index=False, observed=True)
        .agg(
            half_half=("half_half", "max"),
            verified_tie=("verified_tie", "max"),
            no_score_void=("no_score_void", "max"),
            other_void=("other_void", "max"),
        )
    )
    audit: dict[str, object] = {
        "quote_rows": int(len(output)),
        "numeric_half_half_book_rows": int(half_half.sum()),
        "verified_physical_tie_book_rows": int(verified_tie.sum()),
        "no_score_void_book_rows": int(no_score_void.sum()),
        "other_void_with_scores_book_rows": int(other_void.sum()),
        "numeric_half_half_logical_pairs": int(logical["half_half"].sum()),
        "verified_physical_tie_logical_pairs": int(logical["verified_tie"].sum()),
        "no_score_void_logical_pairs": int(logical["no_score_void"].sum()),
        "other_void_with_scores_logical_pairs": int(logical["other_void"].sum()),
        "classification": (
            "physical tie iff numeric 0.5/0.5, both final cumulative scores "
            "exist, completion rounds match, and cumulative scores match"
        ),
    }
    if (
        audit["numeric_half_half_book_rows"]
        != audit["verified_physical_tie_book_rows"]
        + audit["no_score_void_book_rows"]
        + audit["other_void_with_scores_book_rows"]
    ):
        raise AssertionError("0.5/0.5 tournament classifications do not close")
    return output, audit


def build_authority(
    quote_path: str | Path, round_path: str | Path, output_dir: str | Path
) -> dict[str, object]:
    quote_source = Path(quote_path).resolve()
    round_source = Path(round_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    quotes = pd.read_parquet(quote_source)
    rounds = pd.read_parquet(
        round_source,
        columns=["event_key", "dg_id", "round_num", "round_date", "cumulative_score"],
    )
    enriched, audit = add_tournament_score_tie_authority(quotes, rounds)
    quote_output = output / "normalized_matchup_quotes_score_authority.parquet"
    enriched.to_parquet(quote_output, index=False)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "research_only": True,
        "production_files_written": False,
        "source_quotes": {"path": str(quote_source), "sha256": sha256_file(quote_source)},
        "source_player_round_panel": {
            "path": str(round_source),
            "sha256": sha256_file(round_source),
            "columns_read": [
                "event_key",
                "dg_id",
                "round_num",
                "round_date",
                "cumulative_score",
            ],
        },
        "output_quotes": {
            "path": str(quote_output),
            "sha256": sha256_file(quote_output),
            "rows": int(len(enriched)),
        },
        "audit": audit,
    }
    manifest_path = output / "tournament_outcome_authority.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", type=Path, required=True)
    parser.add_argument("--player-round-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = build_authority(args.quotes, args.player_round_panel, args.output_dir)
    print(json.dumps(manifest["audit"], indent=2))


if __name__ == "__main__":
    main()

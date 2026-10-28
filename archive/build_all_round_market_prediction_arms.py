"""Build research-only prediction arms for the 2023 all-round rebuild.

This adapter has two deliberately separate phases.

``build_2023_selection_arms`` consumes chronological 2023 out-of-fold base
skill predictions.  For each challenger it chooses a pre-event downstream
(course/history) carry-through scalar and one *common* R2-R4 completed-round
update scalar from ``{0, .5, 1}``.  The loss averages squared error across
players within event-round, rounds within event, and finally events, so neither
a large field nor the larger pre-cut rounds can dominate selection.  Every
observed R1-R4 target is used.  Numerical ties prefer 1.0 for each scalar.

``apply_frozen_arms`` is the sealed 2024-26 path.  It accepts the selected
challenger and its already-frozen scalar row, and emits exactly the current
control plus that challenger.  Its state projection contains only identifiers,
dates, and the existing cascade R1-R4 predictions; it never accesses outcome
or target columns.  ``apply_frozen_arms_from_paths`` also projects those exact
columns at parquet read time, providing a testable no-label-read boundary.

The current control is exact: its 72-hole/R1 value is the cached current
pre-event prediction and its R1-R4 values are the cached current cascade
predictions.  Challenger construction follows the audited bridge identity::

    pre_event = new_core + downstream_lambda * downstream_preserved_total
    round 1   = pre_event
    round 2-4 = pre_event + live_lambda * (current_round - current_r1)

Nothing here imports or writes a production input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LAMBDA_GRID = (0.0, 0.5, 1.0)
SELECTION_YEAR = 2023
EVALUATION_YEARS = (2024, 2025, 2026)
ARCHITECTURE_SELECTION_START = pd.Timestamp("2023-07-01")
ARCHITECTURE_SELECTION_END = pd.Timestamp("2023-12-31")
EXPECTED_CONTRACT_SHA256 = (
    "4091a0192a959640a9dc88ff6b348759cf3052ed7aed3a116b5989019111374c"
)
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "ALL_ROUND_BASE_SKILL_REBUILD_CONTRACT_2026_08.md"
CONTROL_CANDIDATE = "baseline_current_prod"
ROUND_NUMBERS = (1, 2, 3, 4)
ROUND_TARGET_COLUMNS = {
    1: "label_dg_sg_adj_r1",
    2: "label_dg_sg_adj_r2",
    3: "label_dg_sg_adj_r3",
    4: "label_dg_sg_adj_r4",
}
ROUND_STATE_COLUMNS = {
    1: "cascade_prediction_r1",
    2: "cascade_prediction_r2",
    3: "cascade_prediction_r3",
    4: "cascade_prediction_r4",
}
BASE_OUTPUT_COLUMNS = ["candidate", "event_key", "dg_id_key", "prediction"]
ROUND_OUTPUT_COLUMNS = [
    "candidate",
    "event_key",
    "dg_id_key",
    "round_num",
    "prediction",
]
TIE_TOLERANCE = 1e-12
BRIDGE_IDENTITY_TOLERANCE = 1e-10
RAW_APPLY_PREDICTION_STAGE = "raw_base_core_before_downstream_and_live"
APPLY_STAGE_COLUMNS = (
    "prediction_stage",
    "downstream_lambda_applied",
    "live_update_lambda_applied",
)


@dataclass
class SelectionArmBuild:
    """Artifacts created without using any post-2023 target."""

    base_predictions: pd.DataFrame
    round_predictions: pd.DataFrame
    lambda_tuning: pd.DataFrame
    frozen_lambdas: pd.DataFrame
    market_candidate_authority: pd.DataFrame
    audit: dict[str, Any]


@dataclass
class FrozenArmBuild:
    """Two-candidate artifacts created by the label-blind apply phase."""

    base_predictions: pd.DataFrame
    round_predictions: pd.DataFrame
    audit: dict[str, Any]


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def assert_locked_contract(path: Path = CONTRACT_PATH) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest.casefold() != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(
            "all-round contract hash changed before prediction-arm execution: "
            f"expected {EXPECTED_CONTRACT_SHA256}, observed {digest}"
        )
    return digest


def _stable_id(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0].lstrip("+")
    return text.casefold()


def _event_key(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().casefold()


def _date_from_event_key(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    dashed = text.str.extract(r"(20\d{2}-\d{2}-\d{2})(?!.*20\d{2}-\d{2}-\d{2})")[0]
    compact = text.str.extract(r"(20\d{6})(?!.*20\d{6})")[0]
    parsed = pd.to_datetime(dashed, errors="coerce")
    fallback = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    return parsed.fillna(fallback).dt.normalize()


def _normalize_base_predictions(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    output = frame.copy()
    aliases: dict[str, str] = {}
    if "event_key" not in output and "mapped_event_key" in output:
        aliases["mapped_event_key"] = "event_key"
    if "dg_id_key" not in output:
        if "player_id" in output:
            aliases["player_id"] = "dg_id_key"
        elif "dg_id" in output:
            aliases["dg_id"] = "dg_id_key"
    if "candidate" not in output and "model" in output:
        aliases["model"] = "candidate"
    output = output.rename(columns=aliases)
    _require(output, {"candidate", "event_key", "dg_id_key", "prediction"}, label)
    output["candidate"] = output["candidate"].astype(str)
    output["event_key"] = output["event_key"].map(_event_key)
    output["dg_id_key"] = output["dg_id_key"].map(_stable_id)
    output["prediction"] = pd.to_numeric(output["prediction"], errors="coerce")
    if "event_start_date" in output:
        output["_base_event_date"] = pd.to_datetime(
            output["event_start_date"], errors="coerce"
        ).dt.normalize()
    elif "event_start" in output:
        output["_base_event_date"] = pd.to_datetime(
            output["event_start"], errors="coerce"
        ).dt.normalize()
    else:
        output["_base_event_date"] = _date_from_event_key(output["event_key"])
    if output["event_key"].eq("").any() or output["dg_id_key"].eq("").any():
        raise ValueError(f"{label} contains blank event/player identities")
    if ~np.isfinite(output["prediction"]).all():
        raise ValueError(f"{label} contains a non-finite prediction")
    keys = ["candidate", "event_key", "dg_id_key"]
    if output.duplicated(keys).any():
        raise ValueError(f"{label} is ambiguous on {keys}")
    return output.reset_index(drop=True)


def _assert_raw_apply_predictions(frame: pd.DataFrame) -> None:
    """Fail closed unless every input row is an unapplied raw base core."""

    _require(frame, APPLY_STAGE_COLUMNS, "selected final base predictions")
    stage = frame["prediction_stage"]
    if stage.isna().any() or not stage.astype(str).eq(RAW_APPLY_PREDICTION_STAGE).all():
        observed = sorted(stage.dropna().astype(str).unique().tolist())
        raise ValueError(
            "apply requires prediction_stage="
            f"{RAW_APPLY_PREDICTION_STAGE!r}; observed {observed}"
        )
    for column in ("downstream_lambda_applied", "live_update_lambda_applied"):
        values = frame[column]
        # Deliberately do not use astype(bool): the string "False" is truthy.
        false_flag = values.map(
            lambda value: isinstance(value, (bool, np.bool_)) and not bool(value)
        )
        if values.isna().any() or not false_flag.all():
            observed = sorted(values.dropna().astype(str).unique().tolist())
            raise ValueError(f"apply requires {column}=False on every row; observed {observed}")


def _normalize_bridge(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "dg_id_key" not in output:
        source = "dg_id" if "dg_id" in output else "player_id"
        if source in output:
            output["dg_id_key"] = output[source]
    date_source = "event_start_date" if "event_start_date" in output else "event_start"
    _require(
        output,
        {
            date_source,
            "dg_id_key",
            "baseline_r1_prediction",
            "downstream_preserved_total",
        },
        "legacy core bridge",
    )
    output["event_start_date"] = pd.to_datetime(
        output[date_source], errors="coerce"
    ).dt.normalize()
    output["dg_id_key"] = output["dg_id_key"].map(_stable_id)
    numeric = ["baseline_r1_prediction", "downstream_preserved_total"]
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    output = output[
        ["event_start_date", "dg_id_key", *numeric]
    ].copy()
    if output[["event_start_date", "dg_id_key"]].isna().any().any():
        raise ValueError("legacy core bridge contains a blank date/player key")
    if output.duplicated(["event_start_date", "dg_id_key"]).any():
        raise ValueError("legacy core bridge is ambiguous by event date/player")
    if not np.isfinite(output[numeric].to_numpy(float)).all():
        raise ValueError("legacy core bridge contains a non-finite reconstruction field")
    return output


def _normalize_state(
    frame: pd.DataFrame,
    *,
    include_targets: bool,
) -> pd.DataFrame:
    """Normalize cached cascade state; apply mode never selects target columns."""

    id_source = "dg_id_key"
    if id_source not in frame:
        id_source = "dg_id" if "dg_id" in frame else "player_id"
    _require(
        frame,
        {"event_key", "event_start_date", id_source, *ROUND_STATE_COLUMNS.values()},
        "player event cascade state",
    )
    selected = [
        "event_key",
        "event_start_date",
        id_source,
        *ROUND_STATE_COLUMNS.values(),
    ]
    if include_targets:
        _require(frame, ROUND_TARGET_COLUMNS.values(), "2023 player event outcomes")
        selected.extend(ROUND_TARGET_COLUMNS.values())
    # This projection is intentionally before every other operation.  In apply
    # mode even a caller-supplied frame carrying labels cannot expose them to
    # downstream code.
    output = frame[selected].copy()
    if id_source != "dg_id_key":
        output = output.rename(columns={id_source: "dg_id_key"})
    output["event_key"] = output["event_key"].map(_event_key)
    output["event_start_date"] = pd.to_datetime(
        output["event_start_date"], errors="coerce"
    ).dt.normalize()
    output["dg_id_key"] = output["dg_id_key"].map(_stable_id)
    numeric = list(ROUND_STATE_COLUMNS.values())
    if include_targets:
        numeric.extend(ROUND_TARGET_COLUMNS.values())
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    if output[["event_key", "event_start_date", "dg_id_key"]].isna().any().any():
        raise ValueError("player event cascade state contains a blank key")
    if output["event_key"].eq("").any() or output["dg_id_key"].eq("").any():
        raise ValueError("player event cascade state contains a blank identity")
    if output.duplicated(["event_key", "dg_id_key"]).any():
        raise ValueError("player event cascade state is ambiguous by event/player")
    return output


def _attach_state(
    base: pd.DataFrame,
    state: pd.DataFrame,
    bridge: pd.DataFrame,
) -> pd.DataFrame:
    """Attach canonical state, falling back to date/player for legacy keys."""

    state_columns = [column for column in state if column not in {"event_start_date"}]
    exact = base.merge(
        state[state_columns + ["event_start_date"]],
        on=["event_key", "dg_id_key"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_state"),
        indicator="_state_exact",
    )
    exact["_event_date"] = exact["event_start_date"]
    missing = exact["_state_exact"].eq("left_only")
    exact["_state_matched"] = ~missing
    if missing.any():
        fallback_source = state.rename(columns={"event_key": "_canonical_event_key"})
        fallback_left = base.loc[missing].copy()
        fallback_left["_base_row_index"] = fallback_left.index
        fallback_left = fallback_left.drop(
            columns=["event_start_date", "event_start"], errors="ignore"
        )
        fallback = fallback_left.merge(
            fallback_source,
            left_on=["_base_event_date", "dg_id_key"],
            right_on=["event_start_date", "dg_id_key"],
            how="left",
            validate="many_to_one",
        )
        # Assign by the stable base row index rather than merge order.
        fallback = fallback.set_index("_base_row_index")
        for column in state.columns:
            if column == "event_key":
                exact.loc[missing, "event_key"] = fallback["_canonical_event_key"]
            elif column == "event_start_date":
                exact.loc[missing, "_event_date"] = fallback[column]
            else:
                exact.loc[missing, column] = fallback[column]
        exact.loc[missing, "_state_matched"] = fallback[
            "_canonical_event_key"
        ].notna()
    if not exact["_state_matched"].all():
        raise ValueError(
            "base predictions have "
            f"{int((~exact['_state_matched']).sum())} rows without a player-event state row"
        )
    exact["event_start_date"] = exact["_event_date"]
    exact = exact.drop(
        columns=["_state_exact", "_state_matched", "_event_date"], errors="ignore"
    )
    exact = exact.merge(
        bridge,
        on=["event_start_date", "dg_id_key"],
        how="left",
        validate="many_to_one",
    )
    missing_bridge = (
        exact["downstream_preserved_total"].isna()
        | exact["baseline_r1_prediction"].isna()
        | exact[ROUND_STATE_COLUMNS[1]].isna()
    )
    # The historical bridge does not cover Q-school and has sparse individual
    # gaps.  Those rows cannot support a challenger reconstruction, so exclude
    # them explicitly; the exact control remains available and the joint
    # evaluator later enforces the common quote intersection.
    exact = exact.loc[~missing_bridge].copy()
    if exact.empty:
        raise ValueError("base predictions have no rows with legacy bridge context")
    identity_error = (
        exact["baseline_r1_prediction"] - exact[ROUND_STATE_COLUMNS[1]]
    ).abs()
    if identity_error.max() > BRIDGE_IDENTITY_TOLERANCE:
        raise ValueError("bridge baseline does not equal the exact cached R1 control")
    return exact


def _validate_chronology(frame: pd.DataFrame, *, phase: str) -> None:
    event_date = pd.to_datetime(frame["event_start_date"], errors="coerce")
    if phase == "selection":
        invalid_year = ~event_date.dt.year.eq(SELECTION_YEAR)
        if invalid_year.any():
            years = sorted(event_date.loc[invalid_year].dt.year.dropna().astype(int).unique())
            raise ValueError(f"selection predictions must be 2023-only; found {years}")
    else:
        invalid_year = ~event_date.dt.year.isin(EVALUATION_YEARS)
        if invalid_year.any():
            years = sorted(event_date.loc[invalid_year].dt.year.dropna().astype(int).unique())
            raise ValueError(f"apply predictions must be 2024-26-only; found {years}")

    if "chronology_ok" in frame and not frame["chronology_ok"].fillna(False).astype(bool).all():
        raise ValueError(f"{phase} base predictions contain chronology_ok=False")
    if "prediction_asof" in frame:
        asof = pd.to_datetime(frame["prediction_asof"], errors="coerce")
        if (asof.notna() & asof.gt(event_date)).any():
            raise ValueError(f"{phase} base predictions contain a future prediction_asof")
    if "training_max_event_end" in frame:
        training_end = pd.to_datetime(frame["training_max_event_end"], errors="coerce")
        if (training_end.notna() & training_end.ge(event_date)).any():
            raise ValueError(f"{phase} base predictions train through their target event")


def _outcomes_long(attached: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate", "event_key", "event_start_date", "dg_id_key", "prediction"]
    pieces: list[pd.DataFrame] = []
    for round_num in ROUND_NUMBERS:
        piece = attached[keys].copy()
        piece["round_num"] = round_num
        piece["target"] = pd.to_numeric(
            attached[ROUND_TARGET_COLUMNS[round_num]], errors="coerce"
        )
        piece["current_round_prediction"] = pd.to_numeric(
            attached[ROUND_STATE_COLUMNS[round_num]], errors="coerce"
        )
        piece["current_r1_prediction"] = pd.to_numeric(
            attached[ROUND_STATE_COLUMNS[1]], errors="coerce"
        )
        piece["downstream_preserved_total"] = attached["downstream_preserved_total"]
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _event_equal_mse(rows: pd.DataFrame) -> tuple[float, int, int]:
    valid = rows.loc[
        np.isfinite(rows["target"]) & np.isfinite(rows["arm_prediction"])
    ].copy()
    if valid.empty:
        return np.nan, 0, 0
    valid["squared_error"] = (valid["target"] - valid["arm_prediction"]) ** 2
    # First give each observed round equal mass inside its event (so the larger
    # pre-cut field cannot swamp R3/R4), then give every event equal mass.
    event_round_loss = valid.groupby(
        ["event_key", "round_num"], observed=True
    )["squared_error"].mean()
    event_loss = event_round_loss.groupby(level="event_key").mean()
    return float(event_loss.mean()), int(len(valid)), int(len(event_loss))


def _choose_lambda_row(group: pd.DataFrame) -> pd.Series:
    eligible = group.loc[np.isfinite(group["event_equal_squared_error"])].copy()
    if eligible.empty:
        raise ValueError(f"no observed 2023 R1-R4 targets for {group['candidate'].iloc[0]}")
    best = float(eligible["event_equal_squared_error"].min())
    tied = eligible.loc[
        eligible["event_equal_squared_error"].le(best + TIE_TOLERANCE)
    ].sort_values(
        ["downstream_lambda", "live_lambda"], ascending=[False, False], kind="mergesort"
    )
    return tied.iloc[0]


def _make_prediction_panels(
    attached: pd.DataFrame,
    frozen_lambdas: pd.DataFrame,
    state: pd.DataFrame,
    *,
    control_candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lambda_lookup = frozen_lambdas.set_index("candidate")[
        ["downstream_lambda", "live_lambda"]
    ]
    challenger = attached.join(lambda_lookup, on="candidate", validate="many_to_one")
    if challenger[["downstream_lambda", "live_lambda"]].isna().any().any():
        raise ValueError("one or more challenger candidates lack frozen lambdas")
    challenger["pre_event_prediction"] = (
        challenger["prediction"]
        + challenger["downstream_lambda"] * challenger["downstream_preserved_total"]
    )
    base_challenger = challenger.rename(
        columns={"pre_event_prediction": "_output_prediction"}
    )
    base_challenger = base_challenger[
        ["candidate", "event_key", "dg_id_key", "_output_prediction"]
    ].rename(columns={"_output_prediction": "prediction"})

    round_pieces: list[pd.DataFrame] = []
    for round_num in ROUND_NUMBERS:
        piece = challenger[["candidate", "event_key", "dg_id_key"]].copy()
        piece["round_num"] = round_num
        stage = (
            challenger[ROUND_STATE_COLUMNS[round_num]]
            - challenger[ROUND_STATE_COLUMNS[1]]
        )
        scalar = 0.0 if round_num == 1 else challenger["live_lambda"]
        piece["prediction"] = challenger["pre_event_prediction"] + scalar * stage
        round_pieces.append(piece.loc[np.isfinite(piece["prediction"])])
    round_challenger = pd.concat(round_pieces, ignore_index=True)

    control_base = state.loc[np.isfinite(state[ROUND_STATE_COLUMNS[1]])].copy()
    control_base["candidate"] = control_candidate
    control_base["prediction"] = control_base[ROUND_STATE_COLUMNS[1]]
    control_base = control_base[BASE_OUTPUT_COLUMNS]
    control_round_pieces: list[pd.DataFrame] = []
    for round_num in ROUND_NUMBERS:
        piece = state.loc[np.isfinite(state[ROUND_STATE_COLUMNS[round_num]])].copy()
        piece["candidate"] = control_candidate
        piece["round_num"] = round_num
        piece["prediction"] = piece[ROUND_STATE_COLUMNS[round_num]]
        control_round_pieces.append(piece[ROUND_OUTPUT_COLUMNS])
    control_round = pd.concat(control_round_pieces, ignore_index=True)

    base_output = pd.concat([control_base, base_challenger], ignore_index=True)
    round_output = pd.concat([control_round, round_challenger], ignore_index=True)
    if base_output.duplicated(BASE_OUTPUT_COLUMNS[:-1]).any():
        raise ValueError("base prediction arms contain duplicate keys")
    if round_output.duplicated(ROUND_OUTPUT_COLUMNS[:-1]).any():
        raise ValueError("round prediction arms contain duplicate keys")
    return (
        base_output[BASE_OUTPUT_COLUMNS].sort_values(
            BASE_OUTPUT_COLUMNS[:-1], kind="mergesort"
        ).reset_index(drop=True),
        round_output[ROUND_OUTPUT_COLUMNS].sort_values(
            ROUND_OUTPUT_COLUMNS[:-1], kind="mergesort"
        ).reset_index(drop=True),
    )


def build_market_candidate_authority(
    frozen_lambdas: pd.DataFrame,
    base_predictions: pd.DataFrame,
    round_predictions: pd.DataFrame,
    *,
    control_candidate: str = CONTROL_CANDIDATE,
) -> pd.DataFrame:
    """Bind every and only emitted scored arm to its frozen lambda pair."""

    _require(
        frozen_lambdas,
        {"candidate", "downstream_lambda", "live_lambda"},
        "frozen challenger lambdas",
    )
    _require(base_predictions, {"candidate"}, "emitted base predictions")
    _require(round_predictions, {"candidate"}, "emitted round predictions")
    if frozen_lambdas.duplicated("candidate").any():
        raise ValueError("frozen challenger lambdas contain duplicate candidates")
    base_candidates = set(base_predictions["candidate"].astype(str))
    round_candidates = set(round_predictions["candidate"].astype(str))
    if base_candidates != round_candidates:
        raise ValueError(
            "base/round emitted candidate closure differs: "
            f"base_only={sorted(base_candidates - round_candidates)}, "
            f"round_only={sorted(round_candidates - base_candidates)}"
        )
    challengers = set(frozen_lambdas["candidate"].astype(str))
    expected = challengers | {str(control_candidate)}
    if base_candidates != expected:
        raise ValueError(
            "market authority does not exactly close over emitted candidates: "
            f"unexpected_emitted={sorted(base_candidates - expected)}, "
            f"expected_not_emitted={sorted(expected - base_candidates)}"
        )

    rows: list[dict[str, Any]] = [
        {
            "candidate": str(control_candidate),
            "course_fit_lambda": 1.0,
            "live_update_lambda": 1.0,
            "authority_role": "exact_cached_control",
            "exact_cached_control": True,
            "lambda_source": "not_applied_exact_cached_control",
            "prediction_source": "exact_cached_current_pre_event_and_round_cascade",
            "selection_year": SELECTION_YEAR,
        }
    ]
    for row in frozen_lambdas.sort_values("candidate", kind="mergesort").itertuples(
        index=False
    ):
        rows.append(
            {
                "candidate": str(row.candidate),
                "course_fit_lambda": float(row.downstream_lambda),
                "live_update_lambda": float(row.live_lambda),
                "authority_role": "challenger_2023_tuned",
                "exact_cached_control": False,
                "lambda_source": "2023_jul_dec_event_round_equal_squared_error",
                "prediction_source": "raw_core_plus_frozen_downstream_and_live_scalars",
                "selection_year": SELECTION_YEAR,
            }
        )
    authority = pd.DataFrame(rows).sort_values("candidate", kind="mergesort").reset_index(
        drop=True
    )
    if set(authority["candidate"]) != base_candidates or len(authority) != len(
        base_candidates
    ):
        raise AssertionError("serialized market candidate authority is not exact closure")
    return authority


def build_2023_selection_arms(
    oof_base_predictions: pd.DataFrame,
    legacy_core_bridge: pd.DataFrame,
    player_event_outcomes: pd.DataFrame,
    *,
    control_candidate: str = CONTROL_CANDIDATE,
    lambda_grid: Sequence[float] = LAMBDA_GRID,
) -> SelectionArmBuild:
    """Tune both scalars on 2023 OOF all-round outcomes and build market arms."""

    grid = tuple(float(value) for value in lambda_grid)
    if grid != LAMBDA_GRID:
        raise ValueError(f"the frozen lambda grid must be exactly {LAMBDA_GRID}")
    base = _normalize_base_predictions(oof_base_predictions, "2023 OOF base predictions")
    if control_candidate in set(base["candidate"]):
        raise ValueError("control candidate name collides with a challenger")
    known_base_dates = base["_base_event_date"].dropna()
    if not known_base_dates.dt.year.eq(SELECTION_YEAR).all():
        years = sorted(known_base_dates.dt.year.astype(int).unique())
        raise ValueError(f"selection predictions must be 2023-only; found {years}")
    bridge = _normalize_bridge(legacy_core_bridge)
    state_all = _normalize_state(player_event_outcomes, include_targets=True)
    state = state_all.loc[state_all["event_start_date"].dt.year.eq(SELECTION_YEAR)].copy()
    attached = _attach_state(base, state, bridge)
    _validate_chronology(attached, phase="selection")
    long_all = _outcomes_long(attached)
    long = long_all.loc[
        long_all["event_start_date"].between(
            ARCHITECTURE_SELECTION_START, ARCHITECTURE_SELECTION_END
        )
    ].copy()

    tuning_rows: list[dict[str, Any]] = []
    for candidate, candidate_rows in long.groupby("candidate", observed=True, sort=True):
        for downstream_lambda in grid:
            pre_event = (
                candidate_rows["prediction"]
                + downstream_lambda * candidate_rows["downstream_preserved_total"]
            )
            stage = (
                candidate_rows["current_round_prediction"]
                - candidate_rows["current_r1_prediction"]
            )
            for live_lambda in grid:
                scored = candidate_rows.copy()
                applied_live = np.where(
                    scored["round_num"].eq(1), 0.0, live_lambda
                )
                scored["arm_prediction"] = pre_event + applied_live * stage
                loss, observations, events = _event_equal_mse(scored)
                tuning_rows.append(
                    {
                        "candidate": str(candidate),
                        "downstream_lambda": float(downstream_lambda),
                        "live_lambda": float(live_lambda),
                        "event_equal_squared_error": loss,
                        "observed_player_rounds": observations,
                        "events": events,
                        "selection_year": SELECTION_YEAR,
                        "uses_rounds": "r1_r2_r3_r4_all_observed",
                        "live_lambda_scope": "one_common_scalar_r2_r4",
                    }
                )
    tuning = pd.DataFrame(tuning_rows)
    if tuning.empty:
        raise ValueError("no challenger could be tuned")
    chosen_rows = []
    for _, group in tuning.groupby("candidate", observed=True, sort=True):
        chosen_rows.append(_choose_lambda_row(group))
    frozen = pd.DataFrame(chosen_rows).reset_index(drop=True)
    frozen["selected"] = True
    frozen["live_update_lambda"] = frozen["live_lambda"]
    tuning = tuning.merge(
        frozen[["candidate", "downstream_lambda", "live_lambda"]].assign(selected=True),
        on=["candidate", "downstream_lambda", "live_lambda"],
        how="left",
    )
    tuning["selected"] = tuning["selected"].eq(True)
    base_output, round_output = _make_prediction_panels(
        attached, frozen, state, control_candidate=control_candidate
    )
    authority = build_market_candidate_authority(
        frozen,
        base_output,
        round_output,
        control_candidate=control_candidate,
    )
    audit = {
        "contract": "all_round_market_prediction_arms_v1",
        "phase": "2023_chronological_oof_selection",
        "selection_year": SELECTION_YEAR,
        "architecture_selection_start": str(ARCHITECTURE_SELECTION_START.date()),
        "architecture_selection_end": str(ARCHITECTURE_SELECTION_END.date()),
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "lambda_grid": list(LAMBDA_GRID),
        "target_rounds": list(ROUND_NUMBERS),
        "target_rule": "all_observed_cached_field_adjusted_sg_rounds",
        "loss_weighting": (
            "mean_players_within_event_round_then_mean_rounds_within_event_then_mean_events"
        ),
        "tie_break": "prefer_1.0_downstream_then_1.0_live",
        "live_lambda_scope": "one_common_scalar_for_r2_r4_r1_forced_zero",
        "challenger_candidates": sorted(base["candidate"].unique().tolist()),
        "challenger_input_rows": int(len(base)),
        "challenger_rows_with_bridge": int(len(attached)),
        "challenger_rows_without_bridge": int(len(base) - len(attached)),
        "challenger_rows_without_reconstruction_context": int(len(base) - len(attached)),
        "control_candidate": control_candidate,
        "control_source": "exact_cached_current_cascade_predictions",
        "base_rows": int(len(base_output)),
        "round_rows": int(len(round_output)),
        "market_candidate_authority_rows": int(len(authority)),
        "market_candidate_authority_exact_closure": True,
        "target_columns_read": list(ROUND_TARGET_COLUMNS.values()),
        "evaluation_target_columns_read": [],
    }
    return SelectionArmBuild(
        base_predictions=base_output,
        round_predictions=round_output,
        lambda_tuning=tuning,
        frozen_lambdas=frozen,
        market_candidate_authority=authority,
        audit=audit,
    )


def _normalize_frozen_lambdas(
    value: pd.DataFrame | Mapping[str, Any], selected_candidate: str
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        rows = value.get("lambdas", value.get("frozen_lambdas", []))
        if not rows and {"candidate", "downstream_lambda"}.issubset(value):
            rows = [value]
        frame = pd.DataFrame(rows)
    if "live_lambda" not in frame and "live_update_lambda" in frame:
        frame = frame.rename(columns={"live_update_lambda": "live_lambda"})
    _require(
        frame,
        {"candidate", "downstream_lambda", "live_lambda"},
        "frozen lambdas",
    )
    frame = frame.loc[frame["candidate"].astype(str).eq(str(selected_candidate))].copy()
    if len(frame) != 1:
        raise ValueError("frozen lambdas must have exactly one row for the selected candidate")
    for column in ("downstream_lambda", "live_lambda"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame[column].isin(LAMBDA_GRID).all():
            raise ValueError(f"frozen {column} is outside {LAMBDA_GRID}")
    if "selection_year" in frame and not frame["selection_year"].eq(SELECTION_YEAR).all():
        raise ValueError("frozen lambdas were not selected on 2023")
    return frame[[column for column in frame if column in {
        "candidate", "downstream_lambda", "live_lambda", "selection_year",
        "event_equal_squared_error", "observed_player_rounds", "events", "selected"
    }]].reset_index(drop=True)


def apply_frozen_arms(
    selected_base_predictions: pd.DataFrame,
    frozen_lambdas: pd.DataFrame | Mapping[str, Any],
    legacy_core_bridge: pd.DataFrame,
    cascade_state: pd.DataFrame,
    *,
    selected_candidate: str,
    control_candidate: str = CONTROL_CANDIDATE,
) -> FrozenArmBuild:
    """Apply frozen 2023 choices to 2024-26 without accessing outcome labels."""

    base_all = _normalize_base_predictions(
        selected_base_predictions, "selected final base predictions"
    )
    # This check precedes candidate selection, bridge attachment, and all
    # arithmetic.  A composed pre-event artifact can therefore never be
    # silently treated as a raw core and have lambdas applied twice.
    _assert_raw_apply_predictions(base_all)
    base = base_all.loc[base_all["candidate"].astype(str).eq(str(selected_candidate))].copy()
    if base.empty:
        raise ValueError("selected candidate is absent from final base predictions")
    if control_candidate == selected_candidate:
        raise ValueError("selected candidate collides with the exact control name")
    bridge = _normalize_bridge(legacy_core_bridge)
    # Critical sealed boundary: no target/label column is selected here.
    state_all = _normalize_state(cascade_state, include_targets=False)
    state = state_all.loc[
        state_all["event_start_date"].dt.year.isin(EVALUATION_YEARS)
    ].copy()
    # It is safe for a multi-year prediction file to include 2023 rows; make the
    # final application scope explicit before attaching any state.
    event_dates = base["_base_event_date"]
    if event_dates.isna().any():
        date_lookup = state[["event_key", "event_start_date"]].drop_duplicates()
        base = base.merge(date_lookup, on="event_key", how="left", validate="many_to_one")
        base["_base_event_date"] = base["_base_event_date"].fillna(base["event_start_date"])
        base = base.drop(columns=["event_start_date"], errors="ignore")
    base = base.loc[base["_base_event_date"].dt.year.isin(EVALUATION_YEARS)].copy()
    if base.empty:
        raise ValueError("selected final base predictions contain no 2024-26 rows")
    attached = _attach_state(base, state, bridge)
    _validate_chronology(attached, phase="apply")
    frozen = _normalize_frozen_lambdas(frozen_lambdas, selected_candidate)
    base_output, round_output = _make_prediction_panels(
        attached, frozen, state, control_candidate=control_candidate
    )
    expected = {control_candidate, str(selected_candidate)}
    if set(base_output["candidate"]) != expected or set(round_output["candidate"]) != expected:
        raise AssertionError("sealed apply emitted a candidate other than baseline + selected")
    audit = {
        "contract": "all_round_market_prediction_arms_v1",
        "phase": "2024_2026_frozen_label_blind_apply",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "evaluation_years": list(EVALUATION_YEARS),
        "selected_candidate": str(selected_candidate),
        "control_candidate": control_candidate,
        "emitted_candidates": sorted(expected),
        "downstream_lambda": float(frozen.iloc[0]["downstream_lambda"]),
        "live_lambda": float(frozen.iloc[0]["live_lambda"]),
        "state_columns_read": [
            "event_key", "event_start_date", "dg_id_key", *ROUND_STATE_COLUMNS.values()
        ],
        "input_prediction_stage": RAW_APPLY_PREDICTION_STAGE,
        "input_downstream_lambda_applied": False,
        "input_live_update_lambda_applied": False,
        "target_columns_read": [],
        "base_rows": int(len(base_output)),
        "round_rows": int(len(round_output)),
    }
    return FrozenArmBuild(base_output, round_output, audit)


def _read_parquet_projected(
    path: Path,
    required: Sequence[str],
    optional: Sequence[str] = (),
    *,
    filters: Sequence[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    names = set(pq.ParquetFile(path).schema.names)
    missing = sorted(set(required).difference(names))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    columns = list(required) + [column for column in optional if column in names]
    return pd.read_parquet(
        path, columns=list(dict.fromkeys(columns)), filters=filters
    )


def apply_frozen_arms_from_paths(
    selected_base_predictions_path: Path,
    frozen_lambdas_path: Path,
    legacy_core_bridge_path: Path,
    player_event_outcomes_path: Path,
    *,
    selected_candidate: str,
    control_candidate: str = CONTROL_CANDIDATE,
) -> FrozenArmBuild:
    """Path boundary whose parquet projection makes test-label reads impossible."""

    base = _read_parquet_projected(
        Path(selected_base_predictions_path),
        [
            "candidate", "event_key", "dg_id_key", "prediction",
            *APPLY_STAGE_COLUMNS,
        ],
        [
            "event_start_date", "event_start", "prediction_asof",
            "training_max_event_end", "chronology_ok",
        ],
    )
    bridge = _read_parquet_projected(
        Path(legacy_core_bridge_path),
        [
            "event_start", "dg_id", "baseline_r1_prediction",
            "downstream_preserved_total",
        ],
    )
    # Do not add any label/target column here.  This is the sealed read contract.
    state = _read_parquet_projected(
        Path(player_event_outcomes_path),
        ["event_key", "event_start_date", "dg_id", *ROUND_STATE_COLUMNS.values()],
    )
    frozen_payload = json.loads(Path(frozen_lambdas_path).read_text(encoding="utf-8"))
    if str(frozen_payload.get("contract_sha256", "")).casefold() != EXPECTED_CONTRACT_SHA256:
        raise ValueError("frozen lambdas were not written under the locked all-round contract")
    return apply_frozen_arms(
        base,
        frozen_payload,
        bridge,
        state,
        selected_candidate=selected_candidate,
        control_candidate=control_candidate,
    )


def write_selection_artifacts(result: SelectionArmBuild, output_dir: Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_path = output / "selection_base_predictions.parquet"
    round_path = output / "selection_round_predictions.parquet"
    result.base_predictions.to_parquet(base_path, index=False)
    result.round_predictions.to_parquet(round_path, index=False)
    base_sha256 = hashlib.sha256(base_path.read_bytes()).hexdigest()
    round_sha256 = hashlib.sha256(round_path.read_bytes()).hexdigest()
    result.lambda_tuning.to_csv(output / "lambda_tuning_2023.csv", index=False)
    authority_path = output / "market_candidate_authority.csv"
    result.market_candidate_authority.to_csv(
        authority_path,
        index=False,
        lineterminator="\n",
        encoding="utf-8",
    )
    authority_sha256 = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    payload = {
        "contract": result.audit["contract"],
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "selection_year": SELECTION_YEAR,
        "lambda_grid": list(LAMBDA_GRID),
        "market_candidate_authority_path": authority_path.name,
        "market_candidate_authority_sha256": authority_sha256,
        "market_candidate_authority_rows": int(len(result.market_candidate_authority)),
        "candidate_catalog_path": authority_path.name,
        "candidate_catalog_sha256": authority_sha256,
        "selection_base_predictions_path": base_path.name,
        "selection_base_predictions_sha256": base_sha256,
        "selection_base_predictions_rows": int(len(result.base_predictions)),
        "selection_round_predictions_path": round_path.name,
        "selection_round_predictions_sha256": round_sha256,
        "selection_round_predictions_rows": int(len(result.round_predictions)),
        "lambdas": result.frozen_lambdas.to_dict(orient="records"),
    }
    frozen_path = output / "frozen_lambdas_2023.json"
    frozen_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    audit = {
        **result.audit,
        "market_candidate_authority_path": authority_path.name,
        "market_candidate_authority_sha256": authority_sha256,
        "market_candidate_authority_rows": int(len(result.market_candidate_authority)),
        "candidate_catalog_path": authority_path.name,
        "candidate_catalog_sha256": authority_sha256,
        "selection_base_predictions_path": base_path.name,
        "selection_base_predictions_sha256": base_sha256,
        "selection_base_predictions_rows": int(len(result.base_predictions)),
        "selection_round_predictions_path": round_path.name,
        "selection_round_predictions_sha256": round_sha256,
        "selection_round_predictions_rows": int(len(result.round_predictions)),
    }
    result.audit.update(
        {
            "market_candidate_authority_path": authority_path.name,
            "market_candidate_authority_sha256": authority_sha256,
            "candidate_catalog_path": authority_path.name,
            "candidate_catalog_sha256": authority_sha256,
            "selection_base_predictions_path": base_path.name,
            "selection_base_predictions_sha256": base_sha256,
            "selection_round_predictions_path": round_path.name,
            "selection_round_predictions_sha256": round_sha256,
        }
    )
    (output / "selection_arm_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "research_only": True,
        "production_files_written": False,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "market_candidate_authority_path": authority_path.name,
        "market_candidate_authority_sha256": authority_sha256,
        "market_candidate_authority_rows": int(len(result.market_candidate_authority)),
        "candidate_catalog_path": authority_path.name,
        "candidate_catalog_sha256": authority_sha256,
        "frozen_lambdas_path": frozen_path.name,
        "frozen_lambdas_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "selection_base_predictions_path": base_path.name,
        "selection_base_predictions_sha256": base_sha256,
        "selection_base_predictions_rows": int(len(result.base_predictions)),
        "selection_round_predictions_path": round_path.name,
        "selection_round_predictions_sha256": round_sha256,
        "selection_round_predictions_rows": int(len(result.round_predictions)),
        "candidate_closure": sorted(result.market_candidate_authority["candidate"].tolist()),
    }
    (output / "selection_arm_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_apply_artifacts(
    result: FrozenArmBuild,
    output_dir: Path,
    *,
    source_paths: Mapping[str, Path] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_path = output / "evaluation_base_predictions.parquet"
    round_path = output / "evaluation_round_predictions.parquet"
    audit_path = output / "evaluation_arm_audit.json"
    manifest_path = output / "evaluation_arm_manifest.json"
    result.base_predictions.to_parquet(base_path, index=False)
    result.round_predictions.to_parquet(round_path, index=False)
    base_sha256 = hashlib.sha256(base_path.read_bytes()).hexdigest()
    round_sha256 = hashlib.sha256(round_path.read_bytes()).hexdigest()
    sources: dict[str, dict[str, Any]] = {}
    for name, raw_path in (source_paths or {}).items():
        source = Path(raw_path).resolve()
        record: dict[str, Any] = {"path": str(source)}
        if name == "cascade_state":
            record.update(
                {
                    "whole_file_sha256": None,
                    "whole_file_hash_omitted": (
                        "source may contain outcome columns; apply projected only the "
                        "label-blind state columns recorded in the audit"
                    ),
                    "projected_columns": result.audit.get("state_columns_read", []),
                }
            )
        else:
            record["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        sources[name] = record
    audit = {
        **result.audit,
        "research_only": True,
        "production_files_written": False,
        "evaluation_base_predictions_path": base_path.name,
        "evaluation_base_predictions_sha256": base_sha256,
        "evaluation_round_predictions_path": round_path.name,
        "evaluation_round_predictions_sha256": round_sha256,
        "source_artifacts": sources,
    }
    audit["audit_payload_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "evaluation_arm_audit.sha256").write_text(
        hashlib.sha256(audit_path.read_bytes()).hexdigest() + f"  {audit_path.name}\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "research_only": True,
        "production_files_written": False,
        "phase": result.audit.get("phase"),
        "contract_sha256": result.audit.get("contract_sha256"),
        "selected_candidate": result.audit.get("selected_candidate"),
        "control_candidate": result.audit.get("control_candidate"),
        "downstream_lambda": result.audit.get("downstream_lambda"),
        "live_lambda": result.audit.get("live_lambda"),
        "input_prediction_stage": result.audit.get("input_prediction_stage"),
        "input_downstream_lambda_applied": result.audit.get(
            "input_downstream_lambda_applied"
        ),
        "input_live_update_lambda_applied": result.audit.get(
            "input_live_update_lambda_applied"
        ),
        "target_columns_read": result.audit.get("target_columns_read", []),
        "source_artifacts": sources,
        "evaluation_arm_audit_path": audit_path.name,
        "evaluation_arm_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "outputs": {
            "evaluation_base_predictions": {
                "path": base_path.name,
                "sha256": base_sha256,
                "rows": int(len(result.base_predictions)),
            },
            "evaluation_round_predictions": {
                "path": round_path.name,
                "sha256": round_sha256,
                "rows": int(len(result.round_predictions)),
            },
        },
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "evaluation_arm_manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        + f"  {manifest_path.name}\n",
        encoding="utf-8",
    )


def _selection_from_paths(args: argparse.Namespace) -> SelectionArmBuild:
    base = _read_parquet_projected(
        args.base_predictions,
        ["candidate", "event_key", "dg_id_key", "prediction", "event_start_date"],
        ["prediction_asof", "training_max_event_end", "chronology_ok"],
    )
    bridge = _read_parquet_projected(
        args.bridge,
        [
            "event_start", "dg_id", "baseline_r1_prediction",
            "downstream_preserved_total", "year",
        ],
        filters=[("year", "=", SELECTION_YEAR)],
    )
    outcomes = _read_parquet_projected(
        args.player_event_outcomes,
        [
            "event_key", "event_start_date", "dg_id", "year",
            *ROUND_STATE_COLUMNS.values(), *ROUND_TARGET_COLUMNS.values(),
        ],
        filters=[("year", "=", SELECTION_YEAR)],
    )
    return build_2023_selection_arms(
        base, bridge, outcomes, control_candidate=args.control_candidate
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    select = sub.add_parser("select", help="tune arms on 2023 chronological OOF predictions")
    select.add_argument("--base-predictions", type=Path, required=True)
    select.add_argument("--bridge", type=Path, required=True)
    select.add_argument("--player-event-outcomes", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--control-candidate", default=CONTROL_CANDIDATE)
    apply = sub.add_parser("apply", help="apply frozen arms to 2024-26 without labels")
    apply.add_argument("--base-predictions", type=Path, required=True)
    apply.add_argument("--frozen-lambdas", type=Path, required=True)
    apply.add_argument("--bridge", type=Path, required=True)
    apply.add_argument("--player-event-outcomes", type=Path, required=True)
    apply.add_argument("--selected-candidate", required=True)
    apply.add_argument("--output-dir", type=Path, required=True)
    apply.add_argument("--control-candidate", default=CONTROL_CANDIDATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    assert_locked_contract()
    if args.phase == "select":
        result = _selection_from_paths(args)
        write_selection_artifacts(result, args.output_dir)
    else:
        result = apply_frozen_arms_from_paths(
            args.base_predictions,
            args.frozen_lambdas,
            args.bridge,
            args.player_event_outcomes,
            selected_candidate=args.selected_candidate,
            control_candidate=args.control_candidate,
        )
        write_apply_artifacts(
            result,
            args.output_dir,
            source_paths={
                "runner_raw_core_predictions": args.base_predictions,
                "frozen_lambdas": args.frozen_lambdas,
                "legacy_core_bridge": args.bridge,
                "cascade_state": args.player_event_outcomes,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_OUTPUT_COLUMNS",
    "CONTROL_CANDIDATE",
    "EVALUATION_YEARS",
    "FrozenArmBuild",
    "LAMBDA_GRID",
    "ROUND_OUTPUT_COLUMNS",
    "SELECTION_YEAR",
    "SelectionArmBuild",
    "apply_frozen_arms",
    "apply_frozen_arms_from_paths",
    "assert_locked_contract",
    "build_market_candidate_authority",
    "build_2023_selection_arms",
    "write_apply_artifacts",
    "write_selection_artifacts",
]

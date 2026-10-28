"""Build a lossless, fixed-reference ledger for bounded real golf holes.

The normalized-chain cache is intentionally compact and cannot support a
category mean anchor: it no longer contains causal donor identities, linked
penalties, or a frozen reward ruler.  This research-only module exercises the
smallest honest seam using completed historical holes before the same contract
is emitted by the streaming engine.

Each row is one complete causal transition.  Immediately linked PENALTY/DROP
rows are collapsed into the preceding causal shot, so::

    stroke_cost = causal_strokes + penalty_strokes
    fixed_reference_reward = V(start) - stroke_cost - V(end)

The end state of every nonterminal row must exactly equal the next row's start
state.  Consequently a completed hole telescopes to ``V(first) - strokes``.
Historical target-event outcomes in this ledger are mechanics fixtures only;
they are never eligible as pre-event donor or production-target inputs.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive import hole_sim_fixed_reference_value as reference  # noqa: E402


CONTRACT = "historical_complete_transition_ledger_v2"
STATE_TABLE_CONTRACT = "materialized_fixed_reference_states_v1"
HOLE_AUDIT_CONTRACT = "complete_hole_telescoping_audit_v1"
EVENT_KEY = "pga:R2026034"
PLAYER_ID = "33204"
FIXTURE_HOLES = ((1, 6), (2, 13))
DEFAULT_EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")
DEFAULT_REFERENCE = REPO / "permanent_data" / "hole_sim_fixed_reference_travelers_v2"
DEFAULT_OUTPUT = REPO / "permanent_data" / "hole_sim_real_transition_ledger_lowry_v2"
LEDGER_NAME = "complete_transition_ledger.parquet"
STATE_NAME = "materialized_reference_states.parquet"
AUDIT_NAME = "hole_telescoping_audit.parquet"
REPORT_NAME = "REPORT.md"
MANIFEST_NAME = "manifest.json"
TOLERANCE = 1e-12
CATEGORIES = ("ott", "app", "arg", "putt")


class TransitionLedgerUnavailable(ValueError):
    """Raised when source rows cannot form a complete physical ledger."""


REQUIRED_LEDGER_COLUMNS = (
    "event_key",
    "source_player_id",
    "player_name",
    "round_no",
    "hole_no",
    "transition_no",
    "category",
    "atom_id",
    "physical_atom_sha256",
    "causal_shot_key",
    "raw_source_order",
    "linked_penalty_row_keys",
    "linked_drop_row_keys",
    "resolution_shot_key",
    "source_row_keys",
    "source_stroke_no",
    "start_lie",
    "start_distance_yd",
    "causal_observed_end_lie",
    "causal_observed_end_distance_yd",
    "causal_observed_end_state_key",
    "end_lie",
    "end_distance_yd",
    "start_state_key",
    "end_state_key",
    "terminal",
    "holed",
    "complete_atom",
    "atom_kind",
    "causal_strokes",
    "penalty_strokes",
    "stroke_cost",
    "hazard_resolution_status",
    "start_reference_value",
    "end_reference_value",
    "fixed_reference_reward",
    "reward_reference_descriptor",
    "reward_reference_version",
    "reward_reference_function_sha256",
    "materialized_state_table_sha256",
    "source_shot_sg_sha256",
    "source_raw_shots_sha256",
    "historical_truth_fixture_only",
    "pre_event_donor_eligible",
    "production_target_eligible",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise TransitionLedgerUnavailable("Cannot hash a non-finite number")
        if number == 0.0:
            number = 0.0
        return format(number, ".17g")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _stable_sha(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_canonical_value,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise TransitionLedgerUnavailable(f"Frame hash misses columns={sorted(missing)}")
    rows = [
        {column: _canonical_value(getattr(row, column)) for column in columns}
        for row in frame.loc[:, list(columns)].itertuples(index=False)
    ]
    return _stable_sha({"columns": list(columns), "rows": rows})


def _json_keys(values: Iterable[object]) -> str:
    return json.dumps([str(value) for value in values], separators=(",", ":"))


def _parse_keys(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise TransitionLedgerUnavailable("Linked row keys are not valid JSON") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise TransitionLedgerUnavailable("Linked row keys must be a JSON string list")
    return parsed


def _physical_payload(row: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "event_key",
        "source_player_id",
        "round_no",
        "hole_no",
        "transition_no",
        "category",
        "causal_shot_key",
        "raw_source_order",
        "linked_penalty_row_keys",
        "linked_drop_row_keys",
        "resolution_shot_key",
        "source_stroke_no",
        "start_lie",
        "start_distance_yd",
        "causal_observed_end_lie",
        "causal_observed_end_distance_yd",
        "causal_observed_end_state_key",
        "end_lie",
        "end_distance_yd",
        "terminal",
        "holed",
        "causal_strokes",
        "penalty_strokes",
        "stroke_cost",
        "hazard_resolution_status",
    )
    return {field: _canonical_value(row[field]) for field in fields}


def _source_paths(export_root: Path, season: int = 2026) -> tuple[Path, Path]:
    return (
        export_root / f"pga_{season}_shot_sg.parquet",
        export_root / f"pga_{season}_shots.parquet",
    )


def load_fixture_sources(
    export_root: Path,
    *,
    event_key: str = EVENT_KEY,
    source_player_id: str = PLAYER_ID,
    fixture_holes: Sequence[tuple[int, int]] = FIXTURE_HOLES,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Load only the requested event/player before selecting bounded holes."""

    sg_path, raw_path = _source_paths(export_root)
    if not sg_path.is_file() or not raw_path.is_file():
        raise TransitionLedgerUnavailable("ShotLink source Parquet is unavailable")
    filters = [
        ("event_key", "==", str(event_key)),
        ("source_player_id", "==", str(source_player_id)),
    ]
    sg = pd.read_parquet(sg_path, filters=filters)
    raw = pd.read_parquet(raw_path, filters=filters)
    # Parquet filtering preserves source row order inside this single file.
    # Capture that order before selecting the bounded holes because PENALTY and
    # DROP administrative rows can share the same stroke number.
    raw["raw_source_order"] = np.arange(len(raw), dtype=np.int64)
    requested = {(int(round_no), int(hole_no)) for round_no, hole_no in fixture_holes}
    sg = sg.loc[
        [(int(r), int(h)) in requested for r, h in sg[["round_no", "hole_no"]].itertuples(index=False, name=None)]
    ].copy()
    raw = raw.loc[
        [(int(r), int(h)) in requested for r, h in raw[["round_no", "hole_no"]].itertuples(index=False, name=None)]
    ].copy()
    observed = set(
        (int(r), int(h))
        for r, h in sg[["round_no", "hole_no"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed != requested:
        raise TransitionLedgerUnavailable(
            f"Requested holes differ from SG sources: requested={sorted(requested)}, observed={sorted(observed)}"
        )
    return sg, raw, sg_path, raw_path


def _build_hole_rows(
    sg_hole: pd.DataFrame,
    raw_hole: pd.DataFrame,
    model: reference.ReferenceValueModel,
    *,
    sg_sha256: str,
    raw_sha256: str,
) -> list[dict[str, object]]:
    if sg_hole.empty or raw_hole.empty:
        raise TransitionLedgerUnavailable("A fixture hole has no source rows")
    if not sg_hole["complete_hole"].eq(1).all() or not sg_hole["sg_valid"].eq(1).all():
        raise TransitionLedgerUnavailable("Fixture SG rows must be valid and complete")
    if raw_hole["shot_key"].duplicated().any():
        raise TransitionLedgerUnavailable("Raw shot keys are not unique within the fixture hole")
    if sg_hole["shot_key"].duplicated().any():
        raise TransitionLedgerUnavailable("SG shot keys are not unique within the fixture hole")
    sg_by_key = sg_hole.set_index("shot_key", drop=False)
    ordered_raw = raw_hole.sort_values("raw_source_order", kind="mergesort")
    result: list[dict[str, object]] = []
    consumed_sg_keys: set[str] = set()
    pending_penalty_index: int | None = None

    for raw_row in ordered_raw.itertuples(index=False):
        shot_key = str(raw_row.shot_key)
        event_type = str(raw_row.event_type).strip().upper()
        if event_type in {"STROKE", "PROVISIONAL"}:
            if shot_key not in sg_by_key.index:
                raise TransitionLedgerUnavailable(f"Played raw row has no SG row: {shot_key}")
            sg_row = sg_by_key.loc[shot_key]
            consumed_sg_keys.add(shot_key)
            inferred = str(sg_row.state_inferred or "").casefold()
            if "penalty" in inferred:
                raise TransitionLedgerUnavailable("A played row is marked as an inferred penalty")
            category = str(sg_row.sg_category).strip().casefold()
            if category not in CATEGORIES:
                raise TransitionLedgerUnavailable(f"Unsupported SG category={category!r}")
            start_lie = reference.canonical_lie(sg_row.start_lie)
            causal_end_lie = reference.canonical_lie(sg_row.end_lie)
            start_distance = float(sg_row.start_distance_yd)
            causal_end_distance = float(sg_row.end_distance_yd)
            terminal = bool(raw_row.ball_holed)
            if terminal != (causal_end_lie == "hole"):
                raise TransitionLedgerUnavailable("Raw holed flag and SG terminal state differ")
            if terminal:
                causal_end_lie = "hole"
                causal_end_distance = 0.0
            result.append(
            {
                "event_key": str(sg_row.event_key),
                "source_player_id": str(sg_row.source_player_id),
                "player_name": str(sg_row.player_name),
                "round_no": int(sg_row.round_no),
                "hole_no": int(sg_row.hole_no),
                "transition_no": len(result) + 1,
                "category": category,
                "causal_shot_key": shot_key,
                "raw_source_order": int(raw_row.raw_source_order),
                "linked_penalty_row_keys": _json_keys([]),
                "linked_drop_row_keys": _json_keys([]),
                "resolution_shot_key": "",
                "source_row_keys": _json_keys([shot_key]),
                "source_stroke_no": int(sg_row.stroke_no),
                "start_lie": start_lie,
                "start_distance_yd": start_distance,
                "causal_observed_end_lie": causal_end_lie,
                "causal_observed_end_distance_yd": causal_end_distance,
                "causal_observed_end_state_key": model.state_key(
                    causal_end_lie, causal_end_distance
                ),
                "end_lie": causal_end_lie,
                "end_distance_yd": causal_end_distance,
                "start_state_key": model.state_key(start_lie, start_distance),
                "end_state_key": model.state_key(causal_end_lie, causal_end_distance),
                "terminal": terminal,
                "holed": terminal,
                "complete_atom": True,
                "atom_kind": "terminal_holed" if terminal else "causal_physical_shot",
                "causal_strokes": 1,
                "penalty_strokes": 0,
                "stroke_cost": 1,
                "hazard_resolution_status": "no_linked_penalty",
                "reward_reference_descriptor": model.descriptor,
                "reward_reference_version": reference.VERSION,
                "reward_reference_function_sha256": model.model_sha256,
                "materialized_state_table_sha256": "pending",
                "source_shot_sg_sha256": sg_sha256,
                "source_raw_shots_sha256": raw_sha256,
                "historical_truth_fixture_only": True,
                "pre_event_donor_eligible": False,
                "production_target_eligible": False,
                "_penalty_sg_rows": [],
            }
            )
            pending_penalty_index = None
            continue

        if event_type == "PENALTY":
            if not result or result[-1]["terminal"]:
                raise TransitionLedgerUnavailable("Penalty has no nonterminal causal shot")
            if shot_key not in sg_by_key.index:
                raise TransitionLedgerUnavailable("Penalty raw row has no inferred SG row")
            penalty_sg = sg_by_key.loc[shot_key]
            consumed_sg_keys.add(shot_key)
            inferred = str(penalty_sg.state_inferred or "").casefold()
            if "penalty" not in inferred:
                raise TransitionLedgerUnavailable("Penalty SG row lacks inferred linkage status")
            causal = result[-1]
            if str(penalty_sg.sg_category).strip().casefold() != causal["category"]:
                raise TransitionLedgerUnavailable("Penalty category differs from causal category")
            if model.state_key(penalty_sg.start_lie, penalty_sg.start_distance_yd) != causal[
                "causal_observed_end_state_key"
            ]:
                raise TransitionLedgerUnavailable("Penalty does not begin at causal observed end")
            penalty_keys = _parse_keys(causal["linked_penalty_row_keys"])
            penalty_keys.append(shot_key)
            causal["linked_penalty_row_keys"] = _json_keys(penalty_keys)
            causal["_penalty_sg_rows"].append(penalty_sg)
            causal["source_row_keys"] = _json_keys(
                [*(_parse_keys(causal["source_row_keys"])), shot_key]
            )
            pending_penalty_index = len(result) - 1
            continue

        if event_type == "DROP":
            if pending_penalty_index is None:
                raise TransitionLedgerUnavailable("DROP has no preceding linked penalty")
            causal = result[pending_penalty_index]
            drop_keys = _parse_keys(causal["linked_drop_row_keys"])
            drop_keys.append(shot_key)
            causal["linked_drop_row_keys"] = _json_keys(drop_keys)
            causal["source_row_keys"] = _json_keys(
                [*(_parse_keys(causal["source_row_keys"])), shot_key]
            )
            continue

        raise TransitionLedgerUnavailable(f"Unsupported raw event_type={event_type!r}")

    if consumed_sg_keys != set(sg_hole["shot_key"].astype(str)):
        raise TransitionLedgerUnavailable("Not every SG row was consumed exactly once")
    if int(raw_hole["ball_holed"].fillna(0).eq(1).sum()) != 1:
        raise TransitionLedgerUnavailable("Fixture hole must contain one raw holed row")
    if not result or not result[-1]["terminal"]:
        raise TransitionLedgerUnavailable("Fixture hole did not terminate on its final causal row")

    for index, row in enumerate(result):
        penalty_rows = row.pop("_penalty_sg_rows")
        penalty_keys = _parse_keys(row["linked_penalty_row_keys"])
        drop_keys = _parse_keys(row["linked_drop_row_keys"])
        if penalty_rows:
            if index + 1 >= len(result) or row["terminal"]:
                raise TransitionLedgerUnavailable("Linked penalty has no subsequent physical resolution")
            resolution = result[index + 1]
            if not drop_keys:
                raise TransitionLedgerUnavailable("Linked penalty has no DROP provenance")
            expected_penalties = (
                int(resolution["source_stroke_no"]) - int(row["source_stroke_no"]) - 1
            )
            if expected_penalties != len(penalty_rows) or expected_penalties < 1:
                raise TransitionLedgerUnavailable("Penalty count disagrees with physical stroke sequence")
            resolution_state = str(resolution["start_state_key"])
            for penalty_sg in penalty_rows:
                if model.state_key(penalty_sg.end_lie, penalty_sg.end_distance_yd) != resolution_state:
                    raise TransitionLedgerUnavailable("Penalty endpoint differs from resolution start")
            row["resolution_shot_key"] = str(resolution["causal_shot_key"])
            row["end_lie"] = str(resolution["start_lie"])
            row["end_distance_yd"] = float(resolution["start_distance_yd"])
            row["end_state_key"] = resolution_state
            row["penalty_strokes"] = len(penalty_rows)
            row["stroke_cost"] = 1 + len(penalty_rows)
            row["atom_kind"] = "causal_shot_with_linked_penalty_drop"
            row["hazard_resolution_status"] = "linked_penalty_drop_collapsed"
        elif drop_keys:
            raise TransitionLedgerUnavailable("DROP provenance exists without a penalty")

    all_drop_keys = set(
        raw_hole.loc[
            raw_hole["event_type"].astype(str).str.upper().eq("DROP"), "shot_key"
        ].astype(str)
    )
    linked_drop_keys = {
        item for row in result for item in _parse_keys(row["linked_drop_row_keys"])
    }
    if all_drop_keys != linked_drop_keys:
        raise TransitionLedgerUnavailable("DROP rows were not linked exactly once")

    for row in result:
        row["start_reference_value"] = model.value(row["start_lie"], row["start_distance_yd"])
        row["end_reference_value"] = model.value(row["end_lie"], row["end_distance_yd"])
        row["fixed_reference_reward"] = model.transition_reward(
            start_lie=row["start_lie"],
            start_distance_yd=row["start_distance_yd"],
            end_lie=row["end_lie"],
            end_distance_yd=row["end_distance_yd"],
            stroke_cost=int(row["stroke_cost"]),
        )
        physical_sha = _stable_sha(_physical_payload(row))
        row["physical_atom_sha256"] = physical_sha
        row["atom_id"] = physical_sha
    return result


def materialize_reference_states(
    ledger: pd.DataFrame,
    model: reference.ReferenceValueModel,
) -> tuple[pd.DataFrame, str]:
    starts = ledger.rename(
        columns={
            "start_state_key": "state_key",
            "start_lie": "lie",
            "start_distance_yd": "distance_yd",
            "start_reference_value": "reference_value",
        }
    )[["state_key", "lie", "distance_yd", "reference_value"]]
    ends = ledger.rename(
        columns={
            "end_state_key": "state_key",
            "end_lie": "lie",
            "end_distance_yd": "distance_yd",
            "end_reference_value": "reference_value",
        }
    )[["state_key", "lie", "distance_yd", "reference_value"]]
    states = (
        pd.concat([starts, ends], ignore_index=True)
        .sort_values(["state_key", "lie", "distance_yd"], kind="mergesort")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if states["state_key"].duplicated().any():
        raise TransitionLedgerUnavailable("A state key maps to multiple physical states")
    for row in states.itertuples(index=False):
        if model.state_key(row.lie, row.distance_yd) != row.state_key:
            raise TransitionLedgerUnavailable("Materialized state key changed")
        expected = model.value(row.lie, row.distance_yd)
        if not np.isclose(expected, float(row.reference_value), atol=TOLERANCE, rtol=0.0):
            raise TransitionLedgerUnavailable("Materialized reference value changed")
    columns = ("state_key", "lie", "distance_yd", "reference_value")
    digest = frame_sha256(states, columns)
    states["reference_state_table_sha256"] = digest
    states["reward_reference_function_sha256"] = model.model_sha256
    states["contract"] = STATE_TABLE_CONTRACT
    return states, digest


def validate_complete_transition_ledger(
    ledger: pd.DataFrame,
    model: reference.ReferenceValueModel,
    *,
    expected_source_shot_sg_sha256: str,
    expected_source_raw_shots_sha256: str,
    expected_state_table_sha256: str | None = None,
) -> pd.DataFrame:
    missing = set(REQUIRED_LEDGER_COLUMNS).difference(ledger.columns)
    if missing:
        raise TransitionLedgerUnavailable(f"Ledger misses columns={sorted(missing)}")
    if ledger.empty:
        raise TransitionLedgerUnavailable("Transition ledger is empty")
    if ledger["atom_id"].duplicated().any():
        raise TransitionLedgerUnavailable("Transition atom IDs are not unique")
    if not ledger["category"].isin(CATEGORIES).all():
        raise TransitionLedgerUnavailable("Ledger contains an unsupported category")
    if not ledger["complete_atom"].eq(True).all():
        raise TransitionLedgerUnavailable("Ledger contains incomplete atoms")
    if not ledger["historical_truth_fixture_only"].eq(True).all():
        raise TransitionLedgerUnavailable("Historical fixture scope was lost")
    if ledger["pre_event_donor_eligible"].any() or ledger["production_target_eligible"].any():
        raise TransitionLedgerUnavailable("Historical fixture was made anchor-eligible")
    for column, expected in (
        ("source_shot_sg_sha256", expected_source_shot_sg_sha256),
        ("source_raw_shots_sha256", expected_source_raw_shots_sha256),
    ):
        values = ledger[column].astype(str)
        if not values.map(_is_sha256).all() or values.nunique(dropna=False) != 1:
            raise TransitionLedgerUnavailable(f"{column} must be one lowercase SHA-256")
        if not _is_sha256(expected) or not values.eq(str(expected)).all():
            raise TransitionLedgerUnavailable(f"{column} differs from the expected source")
    if not ledger["causal_strokes"].eq(1).all():
        raise TransitionLedgerUnavailable("Every complete atom must have one causal stroke")
    penalties = pd.to_numeric(ledger["penalty_strokes"], errors="coerce")
    costs = pd.to_numeric(ledger["stroke_cost"], errors="coerce")
    if penalties.isna().any() or costs.isna().any():
        raise TransitionLedgerUnavailable("Stroke accounting is nonnumeric")
    if (penalties < 0).any() or not np.equal(penalties, np.floor(penalties)).all():
        raise TransitionLedgerUnavailable("Penalty strokes must be nonnegative integers")
    if not np.equal(costs, 1 + penalties).all():
        raise TransitionLedgerUnavailable("Stroke cost does not equal causal plus penalties")
    if not ledger["terminal"].eq(ledger["holed"]).all():
        raise TransitionLedgerUnavailable("Terminal and holed flags differ")
    if (ledger["terminal"] & penalties.gt(0)).any():
        raise TransitionLedgerUnavailable("A terminal atom cannot carry a linked penalty")
    if not ledger["reward_reference_function_sha256"].eq(model.model_sha256).all():
        raise TransitionLedgerUnavailable("Reward reference function hash changed")
    if not ledger["reward_reference_descriptor"].eq(model.descriptor).all():
        raise TransitionLedgerUnavailable("Reward reference descriptor changed")
    if not ledger["reward_reference_version"].eq(reference.VERSION).all():
        raise TransitionLedgerUnavailable("Reward reference version changed")
    _, derived_state_table_sha256 = materialize_reference_states(ledger, model)
    if not ledger["materialized_state_table_sha256"].eq(derived_state_table_sha256).all():
        raise TransitionLedgerUnavailable("Materialized state table hash changed")
    if (
        expected_state_table_sha256 is not None
        and derived_state_table_sha256 != str(expected_state_table_sha256)
    ):
        raise TransitionLedgerUnavailable("Materialized state table differs from the expected table")

    audits: list[dict[str, object]] = []
    seen_ledger_source_keys: set[str] = set()
    hole_keys = ["event_key", "source_player_id", "round_no", "hole_no"]
    for keys, group in ledger.groupby(hole_keys, observed=True, sort=True):
        group = group.sort_values("transition_no", kind="mergesort")
        expected_numbers = np.arange(1, len(group) + 1)
        if not np.array_equal(group["transition_no"].to_numpy(int), expected_numbers):
            raise TransitionLedgerUnavailable(f"Transition order is not contiguous for hole={keys}")
        if not np.all(np.diff(group["raw_source_order"].to_numpy(int)) > 0):
            raise TransitionLedgerUnavailable(f"Causal source order is not increasing for hole={keys}")
        stroke_numbers = pd.to_numeric(group["source_stroke_no"], errors="coerce")
        if (
            stroke_numbers.isna().any()
            or not np.equal(stroke_numbers, np.floor(stroke_numbers)).all()
            or (stroke_numbers < 1).any()
        ):
            raise TransitionLedgerUnavailable(f"Hole={keys} has invalid source stroke numbers")
        stroke_numbers_array = stroke_numbers.to_numpy(int)
        group_costs = pd.to_numeric(group["stroke_cost"], errors="coerce").to_numpy(int)
        if stroke_numbers_array[0] != 1 or not np.array_equal(
            stroke_numbers_array[1:], stroke_numbers_array[:-1] + group_costs[:-1]
        ):
            raise TransitionLedgerUnavailable(f"Hole={keys} source stroke sequence is discontinuous")
        if not group.iloc[-1]["terminal"] or int(group["terminal"].sum()) != 1:
            raise TransitionLedgerUnavailable(f"Hole={keys} has no unique terminal transition")
        if group.iloc[:-1]["terminal"].any():
            raise TransitionLedgerUnavailable(f"Hole={keys} terminates before its last transition")
        if reference.canonical_lie(group.iloc[0]["start_lie"]) != "tee":
            raise TransitionLedgerUnavailable(f"Hole={keys} does not begin from the tee")
        end_keys = group["end_state_key"].iloc[:-1].to_numpy(str)
        next_keys = group["start_state_key"].iloc[1:].to_numpy(str)
        if not np.array_equal(end_keys, next_keys):
            raise TransitionLedgerUnavailable(f"Hole={keys} physical states are discontinuous")
        end_lies = group["end_lie"].iloc[:-1].to_numpy(str)
        next_lies = group["start_lie"].iloc[1:].to_numpy(str)
        end_distances = group["end_distance_yd"].iloc[:-1].to_numpy(float)
        next_distances = group["start_distance_yd"].iloc[1:].to_numpy(float)
        if not np.array_equal(end_lies, next_lies) or not np.array_equal(
            end_distances, next_distances
        ):
            raise TransitionLedgerUnavailable(f"Hole={keys} physical coordinates are discontinuous")

        maximum_atom_residual = 0.0
        seen_source_keys: set[str] = set()
        for row in group.to_dict("records"):
            for lie_column in ("start_lie", "causal_observed_end_lie", "end_lie"):
                if reference.canonical_lie(row[lie_column]) != str(row[lie_column]):
                    raise TransitionLedgerUnavailable("Ledger stores a noncanonical physical lie")
            if model.state_key(row["start_lie"], row["start_distance_yd"]) != row["start_state_key"]:
                raise TransitionLedgerUnavailable("Stored start state key changed")
            if model.state_key(row["end_lie"], row["end_distance_yd"]) != row["end_state_key"]:
                raise TransitionLedgerUnavailable("Stored end state key changed")
            if model.state_key(
                row["causal_observed_end_lie"], row["causal_observed_end_distance_yd"]
            ) != row["causal_observed_end_state_key"]:
                raise TransitionLedgerUnavailable("Stored causal observed end state changed")
            start_value = model.value(row["start_lie"], row["start_distance_yd"])
            end_value = model.value(row["end_lie"], row["end_distance_yd"])
            reward = start_value - float(row["stroke_cost"]) - end_value
            residuals = (
                abs(start_value - float(row["start_reference_value"])),
                abs(end_value - float(row["end_reference_value"])),
                abs(reward - float(row["fixed_reference_reward"])),
            )
            maximum_atom_residual = max(maximum_atom_residual, *residuals)
            if maximum_atom_residual > TOLERANCE:
                raise TransitionLedgerUnavailable("Stored fixed-reference reward changed")
            if _stable_sha(_physical_payload(row)) != row["physical_atom_sha256"]:
                raise TransitionLedgerUnavailable("Physical atom content hash changed")
            if row["atom_id"] != row["physical_atom_sha256"]:
                raise TransitionLedgerUnavailable("Atom ID differs from physical content hash")
            penalty_keys = _parse_keys(row["linked_penalty_row_keys"])
            drop_keys = _parse_keys(row["linked_drop_row_keys"])
            source_keys = _parse_keys(row["source_row_keys"])
            causal_key = str(row["causal_shot_key"])
            if not _is_sha256(causal_key) or any(
                not _is_sha256(value) for value in (*penalty_keys, *drop_keys, *source_keys)
            ):
                raise TransitionLedgerUnavailable("Physical source-row keys must be lowercase SHA-256")
            if (
                len(set(penalty_keys)) != len(penalty_keys)
                or len(set(drop_keys)) != len(drop_keys)
                or set(penalty_keys).intersection(drop_keys)
            ):
                raise TransitionLedgerUnavailable("Penalty/drop row provenance is duplicated")
            if int(row["penalty_strokes"]) != len(penalty_keys):
                raise TransitionLedgerUnavailable("Penalty count differs from linked rows")
            if len(penalty_keys) != len(drop_keys):
                raise TransitionLedgerUnavailable("Penalty/drop linkage is incomplete")
            expected_sources = [causal_key, *penalty_keys, *drop_keys]
            if source_keys != expected_sources:
                raise TransitionLedgerUnavailable("Physical source-row provenance changed")
            if (
                len(set(source_keys)) != len(source_keys)
                or seen_source_keys.intersection(source_keys)
                or seen_ledger_source_keys.intersection(source_keys)
            ):
                raise TransitionLedgerUnavailable("Physical source rows are not unique within the ledger")
            seen_source_keys.update(source_keys)
            seen_ledger_source_keys.update(source_keys)
            terminal = bool(row["terminal"])
            has_penalty = bool(penalty_keys)
            expected_atom_kind = (
                "terminal_holed"
                if terminal
                else (
                    "causal_shot_with_linked_penalty_drop"
                    if has_penalty
                    else "causal_physical_shot"
                )
            )
            expected_hazard_status = (
                "linked_penalty_drop_collapsed" if has_penalty else "no_linked_penalty"
            )
            if str(row["atom_kind"]) != expected_atom_kind:
                raise TransitionLedgerUnavailable("Atom kind differs from its physical semantics")
            if str(row["hazard_resolution_status"]) != expected_hazard_status:
                raise TransitionLedgerUnavailable("Hazard status differs from its penalty provenance")
            if terminal:
                if (
                    str(row["end_lie"]) != "hole"
                    or float(row["end_distance_yd"]) != 0.0
                    or str(row["end_state_key"]) != "terminal"
                ):
                    raise TransitionLedgerUnavailable("Holed terminal atom has a nonterminal endpoint")
            elif str(row["end_lie"]) == "hole" or str(row["start_lie"]) == "hole":
                raise TransitionLedgerUnavailable("A nonterminal atom contains a terminal physical state")
            if penalty_keys:
                if not str(row["resolution_shot_key"]):
                    raise TransitionLedgerUnavailable("Penalty has no physical resolution shot")
                if not _is_sha256(row["resolution_shot_key"]):
                    raise TransitionLedgerUnavailable("Penalty resolution key must be lowercase SHA-256")
                next_rows = group.loc[group["transition_no"].eq(int(row["transition_no"]) + 1)]
                if len(next_rows) != 1 or str(next_rows.iloc[0]["causal_shot_key"]) != str(
                    row["resolution_shot_key"]
                ):
                    raise TransitionLedgerUnavailable("Penalty resolution shot differs from next atom")
            elif str(row["resolution_shot_key"]):
                raise TransitionLedgerUnavailable("Ordinary atom declares a penalty resolution")
            elif (
                row["causal_observed_end_state_key"] != row["end_state_key"]
                or str(row["causal_observed_end_lie"]) != str(row["end_lie"])
                or float(row["causal_observed_end_distance_yd"]) != float(row["end_distance_yd"])
            ):
                raise TransitionLedgerUnavailable("Ordinary causal endpoint was rewritten")

        start_value = float(group.iloc[0]["start_reference_value"])
        reward_sum = float(group["fixed_reference_reward"].sum())
        total_strokes = int(group["stroke_cost"].sum())
        telescoping_residual = reward_sum - (start_value - total_strokes)
        if abs(telescoping_residual) > TOLERANCE:
            raise TransitionLedgerUnavailable(f"Hole={keys} does not telescope")
        audits.append(
            {
                "contract": HOLE_AUDIT_CONTRACT,
                "event_key": keys[0],
                "source_player_id": keys[1],
                "round_no": int(keys[2]),
                "hole_no": int(keys[3]),
                "transitions": int(len(group)),
                "causal_strokes": int(group["causal_strokes"].sum()),
                "penalty_strokes": int(group["penalty_strokes"].sum()),
                "total_strokes": total_strokes,
                "starting_reference_value": start_value,
                "fixed_reference_reward_sum": reward_sum,
                "telescoping_residual": telescoping_residual,
                "maximum_atom_residual": maximum_atom_residual,
                "categories": _json_keys(group["category"]),
                "terminal": True,
            }
        )
    return pd.DataFrame(audits).sort_values(["round_no", "hole_no"]).reset_index(drop=True)


def build_ledger(
    sg: pd.DataFrame,
    raw: pd.DataFrame,
    model: reference.ReferenceValueModel,
    *,
    sg_sha256: str,
    raw_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    hole_keys = ["event_key", "source_player_id", "round_no", "hole_no"]
    raw_groups = {
        tuple(keys): group.copy()
        for keys, group in raw.groupby(hole_keys, observed=True, sort=False)
    }
    for keys, sg_hole in sg.groupby(hole_keys, observed=True, sort=True):
        key = tuple(keys)
        if key not in raw_groups:
            raise TransitionLedgerUnavailable(f"SG hole has no raw source={key}")
        rows.extend(
            _build_hole_rows(
                sg_hole,
                raw_groups[key],
                model,
                sg_sha256=sg_sha256,
                raw_sha256=raw_sha256,
            )
        )
    ledger = pd.DataFrame(rows).sort_values(
        ["round_no", "hole_no", "transition_no"], kind="mergesort"
    ).reset_index(drop=True)
    states, state_sha = materialize_reference_states(ledger, model)
    ledger["materialized_state_table_sha256"] = state_sha
    ledger = ledger.loc[:, list(REQUIRED_LEDGER_COLUMNS)]
    audit = validate_complete_transition_ledger(
        ledger,
        model,
        expected_state_table_sha256=state_sha,
        expected_source_shot_sg_sha256=sg_sha256,
        expected_source_raw_shots_sha256=raw_sha256,
    )
    observed_scores = (
        raw.groupby(hole_keys, observed=True)["stroke_no"].max().reset_index(name="observed_hole_score")
    )
    audit = audit.merge(observed_scores, on=hole_keys, how="left", validate="one_to_one")
    if not audit["observed_hole_score"].eq(audit["total_strokes"]).all():
        raise TransitionLedgerUnavailable("Ledger stroke totals differ from raw completed-hole score")
    return ledger, states, audit


def build_artifact(
    *,
    export_root: Path,
    reference_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TransitionLedgerUnavailable(f"Output directory is not empty: {output_dir}")
    source_start_sha = _sha256(Path(__file__))
    sg, raw, sg_path, raw_path = load_fixture_sources(export_root)
    model = reference.ReferenceValueModel.load(reference_dir)
    input_sha = {
        "shot_sg": _sha256(sg_path),
        "raw_shots": _sha256(raw_path),
        "reference_manifest": _sha256(reference_dir / "manifest.json"),
        "reference_knots": _sha256(reference_dir / "reference_value_knots.parquet"),
    }
    ledger, states, audit = build_ledger(
        sg,
        raw,
        model,
        sg_sha256=input_sha["shot_sg"],
        raw_sha256=input_sha["raw_shots"],
    )
    source_end_sha = _sha256(Path(__file__))
    if source_start_sha != source_end_sha:
        raise TransitionLedgerUnavailable("Ledger source changed during artifact construction")

    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / LEDGER_NAME
    state_path = output_dir / STATE_NAME
    audit_path = output_dir / AUDIT_NAME
    report_path = output_dir / REPORT_NAME
    ledger.to_parquet(ledger_path, index=False)
    states.to_parquet(state_path, index=False)
    audit.to_parquet(audit_path, index=False)
    r1 = audit.loc[(audit["round_no"] == 1) & (audit["hole_no"] == 6)].iloc[0]
    r2 = audit.loc[(audit["round_no"] == 2) & (audit["hole_no"] == 13)].iloc[0]
    report = "\n".join(
        [
            "# Real transition ledger smoke",
            "",
            "Historical truth fixture only; not a donor pool or production target.",
            "",
            f"- Complete causal atoms: {len(ledger)}",
            f"- Materialized fixed-reference states: {len(states)}",
            f"- R1 H6 strokes/reward/residual: {r1.total_strokes} / {r1.fixed_reference_reward_sum:.15g} / {r1.telescoping_residual:.3g}",
            f"- R2 H13 strokes/reward/residual: {r2.total_strokes} / {r2.fixed_reference_reward_sum:.15g} / {r2.telescoping_residual:.3g}",
            f"- Linked penalty atoms: {int(ledger['penalty_strokes'].gt(0).sum())}",
            f"- Reward reference function SHA-256: `{model.model_sha256}`",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")
    artifact_sha = {
        path.name: _sha256(path)
        for path in (ledger_path, state_path, audit_path, report_path)
    }
    manifest = {
        "contract": CONTRACT,
        "research_only": True,
        "historical_truth_fixture_only": True,
        "pre_event_donor_eligible": False,
        "production_target_eligible": False,
        "event_key": EVENT_KEY,
        "source_player_id": PLAYER_ID,
        "fixture_holes": [list(value) for value in FIXTURE_HOLES],
        "complete_atoms": int(len(ledger)),
        "linked_penalty_atoms": int(ledger["penalty_strokes"].gt(0).sum()),
        "materialized_states": int(len(states)),
        "reward_reference_function_sha256": model.model_sha256,
        "reward_reference_descriptor": model.descriptor,
        "materialized_state_table_sha256": str(ledger["materialized_state_table_sha256"].iloc[0]),
        "ledger_logical_sha256": frame_sha256(ledger, REQUIRED_LEDGER_COLUMNS),
        "state_logical_sha256": frame_sha256(
            states,
            (
                "state_key",
                "lie",
                "distance_yd",
                "reference_value",
                "reference_state_table_sha256",
                "reward_reference_function_sha256",
                "contract",
            ),
        ),
        "audit_logical_sha256": frame_sha256(audit, tuple(audit.columns)),
        "maximum_atom_residual": float(audit["maximum_atom_residual"].max()),
        "maximum_hole_telescoping_residual": float(audit["telescoping_residual"].abs().max()),
        "input_sha256": input_sha,
        "source_sha256_start": source_start_sha,
        "source_sha256_end": source_end_sha,
        "artifact_sha256": artifact_sha,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_artifact(
        export_root=args.export_root,
        reference_dir=args.reference_dir,
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

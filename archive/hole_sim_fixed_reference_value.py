"""Build and consume one frozen Tour expected-strokes reference function.

The shot simulator must calculate SG from physical state transitions, not from
the donor row's historical SG.  This archive-only module freezes that ruler:

* non-green lie/distance values come from the immutable Broadie Table B.1
  knots used by the ShotLink SG exporter;
* green values use the exact prior-only monotone curve already validated by
  ``evaluate_shot_quality_tilt.py``;
* terminal ``hole`` has value zero;
* non-green interpolation/extrapolation exactly matches that exporter.

Every transition reward is then ``V(start) - stroke_cost - V(end)``.  Linked
penalties live in ``stroke_cost`` and the reward telescopes over a complete
physical hole path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from archive import evaluate_shot_quality_tilt as quality_eval  # noqa: E402


CONTRACT = "frozen_tour_expected_strokes_reference_v1"
VERSION = "broadie_2020_2025_plus_2020_prior_putting_v1"
NON_GREEN_LIES = ("tee", "fairway", "rough", "sand", "recovery")
ALL_LIES = (*NON_GREEN_LIES, "green", "hole")
DEFAULT_EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")
DEFAULT_BROADIE_SOURCE = (
    DEFAULT_EXPORT.parent.parent / "shot_collector" / "strokes_gained.py"
)
DEFAULT_OUTPUT = REPO / "permanent_data" / "hole_sim_fixed_reference_travelers_v2"
DEFAULT_SEASONS = tuple(range(2020, 2026))
PUTTING_WARMUP_EVENT = "pga:R2020016"
PUTTING_WARMUP_START = pd.Timestamp("2020-01-02")
PUTTING_AVAILABLE_AT = pd.Timestamp("2020-01-09")
EXPECTED_PUTTING_SHA256 = "aace8326debb5c2af07b16e319730601a56d9ddd978d31ce4a0d3ee1d82f7c93"
DISTANCE_DECIMALS = 6
VALUE_DECIMALS = 12


class ReferenceValueUnavailable(ValueError):
    """Raised when a physical state has no frozen reference value."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def canonical_lie(value: object) -> str:
    text = str(value).strip().casefold()
    if text in ALL_LIES:
        return text
    if "bunker" in text:
        return "sand"
    if text in {"holed", "terminal"}:
        return "hole"
    raise ReferenceValueUnavailable(f"Unsupported reference lie: {value!r}")


def _canonical_number(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ReferenceValueUnavailable("Reference model contains non-finite values")
    if number == 0.0:
        number = 0.0
    return format(number, ".17g")


def _table_payload(table: pd.DataFrame, putting_sha256: str) -> dict[str, object]:
    ordered = table.sort_values(["lie", "distance_yd"], kind="mergesort")
    rows = [
        {
            "lie": str(row.lie),
            "distance_yd": _canonical_number(row.distance_yd),
            "expected_strokes": _canonical_number(row.expected_strokes),
            "source_kind": str(row.source_kind),
        }
        for row in ordered.itertuples(index=False)
    ]
    return {
        "contract": CONTRACT,
        "version": VERSION,
        "distance_rounding_decimals": DISTANCE_DECIMALS,
        "value_rounding_decimals": VALUE_DECIMALS,
        "non_green_interpolation": "linear_extrapolation_exact_broadie_exporter",
        "green_interpolation": "linear_constant_endpoint_exact_quality_eval_curve",
        "terminal_value": 0.0,
        "putting_curve_sha256": putting_sha256,
        "rows": rows,
    }


def reference_model_sha256(table: pd.DataFrame, putting_sha256: str) -> str:
    return _stable_sha(_table_payload(table, putting_sha256))


def collect_non_green_knots(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Collect one consistent Broadie value for every observed lie/distance."""

    pieces: list[pd.DataFrame] = []
    required = {
        "start_lie",
        "start_distance_yd",
        "expected_strokes_start",
        "end_lie",
        "end_distance_yd",
        "expected_strokes_end",
        "sg_valid",
    }
    for frame in frames:
        missing = required.difference(frame.columns)
        if missing:
            raise ReferenceValueUnavailable(f"Shot SG source misses fields={sorted(missing)}")
        valid = frame.loc[frame["sg_valid"].eq(1)]
        start = valid.rename(
            columns={
                "start_lie": "lie",
                "start_distance_yd": "distance_yd",
                "expected_strokes_start": "expected_strokes",
            }
        )[["lie", "distance_yd", "expected_strokes"]]
        end = valid.rename(
            columns={
                "end_lie": "lie",
                "end_distance_yd": "distance_yd",
                "expected_strokes_end": "expected_strokes",
            }
        )[["lie", "distance_yd", "expected_strokes"]]
        pieces.extend((start, end))
    if not pieces:
        raise ReferenceValueUnavailable("No reference observations were supplied")
    values = pd.concat(pieces, ignore_index=True)
    values["lie"] = values["lie"].astype(str).str.strip().str.casefold()
    values = values.loc[values["lie"].isin((*NON_GREEN_LIES, "hole"))].copy()
    values["distance_yd"] = pd.to_numeric(values["distance_yd"], errors="coerce")
    values["expected_strokes"] = pd.to_numeric(values["expected_strokes"], errors="coerce")
    values = values.dropna(subset=["distance_yd", "expected_strokes"])
    values["distance_yd"] = values["distance_yd"].round(DISTANCE_DECIMALS)
    values["expected_strokes"] = values["expected_strokes"].round(VALUE_DECIMALS)
    consistency = values.groupby(["lie", "distance_yd"], observed=True)["expected_strokes"].agg(
        ["min", "max", "nunique"]
    )
    conflicts = consistency.loc[
        (consistency["nunique"] > 1)
        & ((consistency["max"] - consistency["min"]).abs() > 10.0 ** (-VALUE_DECIMALS))
    ]
    if len(conflicts):
        raise ReferenceValueUnavailable(
            f"Broadie reference changes at {len(conflicts)} lie/distance knots"
        )
    table = (
        values.groupby(["lie", "distance_yd"], observed=True, as_index=False)["expected_strokes"]
        .first()
        .sort_values(["lie", "distance_yd"], kind="mergesort")
        .reset_index(drop=True)
    )
    table["source_kind"] = np.where(
        table["lie"].eq("hole"), "terminal_zero", "fixed_broadie_export"
    )
    terminal = table.loc[table["lie"].eq("hole")]
    if len(terminal) != 1 or not np.allclose(
        terminal[["distance_yd", "expected_strokes"]], 0.0, atol=0.0, rtol=0.0
    ):
        raise ReferenceValueUnavailable("Terminal reference must be one exact zero knot")
    for lie in NON_GREEN_LIES:
        if not table["lie"].eq(lie).any():
            raise ReferenceValueUnavailable(f"Reference table has no {lie} knots")
    return table


def load_broadie_knots(source_path: Path = DEFAULT_BROADIE_SOURCE) -> pd.DataFrame:
    """Load the canonical Broadie knots from the exporter source itself."""

    golf_root = source_path.resolve().parents[1]
    if str(golf_root) not in sys.path:
        sys.path.insert(0, str(golf_root))
    from shot_collector import strokes_gained as strokes_gained  # type: ignore

    if Path(strokes_gained.__file__).resolve() != source_path.resolve():
        raise ReferenceValueUnavailable("Imported Broadie source path is not the requested file")
    rows: list[dict[str, object]] = []
    for lie in NON_GREEN_LIES:
        distances = (
            strokes_gained._TEE_DISTANCES
            if lie == "tee"
            else strokes_gained._DISTANCES
        )
        values = strokes_gained._BROADIE_VALUES[lie]
        if len(distances) != len(values):
            raise ReferenceValueUnavailable(f"Broadie {lie} knots and values differ in length")
        rows.extend(
            {
                "lie": lie,
                "distance_yd": float(distance),
                "expected_strokes": float(value),
                "source_kind": "broadie_2012_table_b1",
            }
            for distance, value in zip(distances, values)
        )
    rows.append(
        {
            "lie": "hole",
            "distance_yd": 0.0,
            "expected_strokes": 0.0,
            "source_kind": "terminal_zero",
        }
    )
    return pd.DataFrame(rows).sort_values(["lie", "distance_yd"]).reset_index(drop=True)


def reconstruct_frozen_putting_curve(
    shot_sg_path: Path,
    raw_shots_path: Path,
    *,
    event_key: str = PUTTING_WARMUP_EVENT,
    event_start: object = PUTTING_WARMUP_START,
    available_at: object = PUTTING_AVAILABLE_AT,
    minimum_samples: int = 500,
    expected_sha256: str | None = EXPECTED_PUTTING_SHA256,
) -> quality_eval.CurveSnapshot:
    """Reproduce the exact completed-prior-hole monotone putting curve."""

    sg_columns = [
        "event_key",
        "source_player_id",
        "round_no",
        "hole_no",
        "stroke_no",
        "start_lie",
        "start_distance_yd",
        "shot_key",
    ]
    raw_columns = [
        "event_key",
        "source_player_id",
        "round_no",
        "hole_no",
        "stroke_no",
        "event_type",
        "ball_holed",
        "shot_key",
    ]
    sg = pd.read_parquet(
        shot_sg_path,
        columns=sg_columns,
        filters=[("event_key", "==", event_key)],
    )
    raw = pd.read_parquet(
        raw_shots_path,
        columns=raw_columns,
        filters=[("event_key", "==", event_key)],
    )
    if sg.empty or raw.empty:
        raise ReferenceValueUnavailable("Putting warmup event is absent from source exports")
    context = raw.drop_duplicates("shot_key")[["shot_key", "event_type", "ball_holed"]]
    sg = sg.merge(context, on="shot_key", how="left", validate="many_to_one")
    hole_keys = ["event_key", "source_player_id", "round_no", "hole_no"]
    terminal = (
        raw.groupby(hole_keys, observed=True)
        .agg(raw_hole_complete=("ball_holed", "max"), final_stroke_no=("stroke_no", "max"))
        .reset_index()
    )
    sg = sg.merge(terminal, on=hole_keys, how="left", validate="many_to_one")
    observations = sg.loc[
        sg["start_lie"].eq("green")
        & sg["raw_hole_complete"].fillna(False).astype(bool)
        & sg["event_type"].eq("STROKE")
        & sg["start_distance_yd"].notna()
    ].copy()
    observations["green_start_distance_yd"] = observations["start_distance_yd"]
    observations["green_remaining_strokes"] = (
        pd.to_numeric(observations["final_stroke_no"], errors="coerce")
        - pd.to_numeric(observations["stroke_no"], errors="coerce")
        + 1.0
    )
    accumulator = quality_eval.PriorPuttingCurve()
    accumulator.update(observations, event_start=event_start)
    curve = accumulator.snapshot(minimum_samples, as_of=available_at)
    if curve is None:
        raise ReferenceValueUnavailable("Putting warmup sample did not meet the declared gate")
    if expected_sha256 is not None and curve.sha256 != str(expected_sha256):
        raise ReferenceValueUnavailable(
            f"Putting curve changed: expected {expected_sha256}, observed {curve.sha256}"
        )
    return curve


def build_reference_table(
    non_green_table: pd.DataFrame,
    putting_curve: quality_eval.CurveSnapshot,
) -> pd.DataFrame:
    table = non_green_table.copy()
    green = pd.DataFrame(
        {
            "lie": "green",
            "distance_yd": np.asarray(putting_curve.distances_ft, float) / 3.0,
            "expected_strokes": np.asarray(putting_curve.expected, float),
            "source_kind": "frozen_prior_putting_curve",
        }
    )
    if green["distance_yd"].duplicated().any():
        raise ReferenceValueUnavailable("Putting reference has duplicate distances")
    return (
        pd.concat([table, green], ignore_index=True)
        .sort_values(["lie", "distance_yd"], kind="mergesort")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class ReferenceValueModel:
    table: pd.DataFrame
    putting_curve_sha256: str
    model_sha256: str

    def __post_init__(self) -> None:
        observed = reference_model_sha256(self.table, self.putting_curve_sha256)
        if observed != self.model_sha256:
            raise ReferenceValueUnavailable("Reference model content hash changed")
        if set(self.table["lie"].astype(str)) != set(ALL_LIES):
            raise ReferenceValueUnavailable("Reference model does not cover all canonical lies")

    @classmethod
    def load(cls, directory: Path) -> "ReferenceValueModel":
        manifest_path = directory / "manifest.json"
        table_path = directory / "reference_value_knots.parquet"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != CONTRACT or manifest.get("version") != VERSION:
            raise ReferenceValueUnavailable("Reference artifact contract/version changed")
        if _sha256(table_path) != manifest["artifact_sha256"][table_path.name]:
            raise ReferenceValueUnavailable("Reference Parquet byte hash changed")
        table = pd.read_parquet(table_path)
        return cls(
            table=table,
            putting_curve_sha256=str(manifest["putting_curve"]["sha256"]),
            model_sha256=str(manifest["reference_model_sha256"]),
        )

    @property
    def descriptor(self) -> str:
        return f"{CONTRACT}:{self.model_sha256}"

    def value(self, lie: object, distance_yd: object) -> float:
        state_lie = canonical_lie(lie)
        distance = float(distance_yd)
        if not np.isfinite(distance):
            raise ReferenceValueUnavailable("Reference distance must be finite")
        distance = max(0.0, distance)
        if state_lie == "hole":
            if not np.isclose(distance, 0.0, atol=1e-9, rtol=0.0):
                raise ReferenceValueUnavailable("Terminal state must have zero distance")
            return 0.0
        knots = self.table.loc[self.table["lie"].eq(state_lie)].sort_values("distance_yd")
        x = knots["distance_yd"].to_numpy(float)
        y = knots["expected_strokes"].to_numpy(float)
        if state_lie == "green":
            return float(np.interp(distance, x, y, left=float(y[0]), right=float(y[-1])))
        if distance <= x[0]:
            left = 0
        elif distance >= x[-1]:
            left = len(x) - 2
        else:
            left = int(np.searchsorted(x, distance, side="left") - 1)
            left = max(0, min(left, len(x) - 2))
        return float(y[left] + (distance - x[left]) * (y[left + 1] - y[left]) / (x[left + 1] - x[left]))

    def state_key(self, lie: object, distance_yd: object) -> str:
        state_lie = canonical_lie(lie)
        if state_lie == "hole":
            self.value(state_lie, distance_yd)
            return "terminal"
        distance = float(distance_yd)
        self.value(state_lie, distance)
        return f"{state_lie}|{distance:.6f}yd"

    def transition_reward(
        self,
        *,
        start_lie: object,
        start_distance_yd: object,
        end_lie: object,
        end_distance_yd: object,
        stroke_cost: int,
    ) -> float:
        cost = int(stroke_cost)
        if cost < 1 or float(stroke_cost) != cost:
            raise ReferenceValueUnavailable("Transition stroke cost must be a positive integer")
        return (
            self.value(start_lie, start_distance_yd)
            - float(cost)
            - self.value(end_lie, end_distance_yd)
        )


def audit_export_reference_values(
    model: ReferenceValueModel,
    source_paths: Sequence[Path],
    *,
    tolerance: float = 5.1e-7,
) -> dict[str, object]:
    """Verify historical exported non-green values against the frozen function."""

    columns = [
        "start_lie",
        "start_distance_yd",
        "expected_strokes_start",
        "end_lie",
        "end_distance_yd",
        "expected_strokes_end",
        "sg_valid",
    ]
    path_rows: list[dict[str, object]] = []
    global_max = 0.0
    global_rows = 0
    for path in source_paths:
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["sg_valid"].eq(1)]
        start = frame.rename(
            columns={
                "start_lie": "lie",
                "start_distance_yd": "distance_yd",
                "expected_strokes_start": "expected_strokes",
            }
        )[["lie", "distance_yd", "expected_strokes"]]
        end = frame.rename(
            columns={
                "end_lie": "lie",
                "end_distance_yd": "distance_yd",
                "expected_strokes_end": "expected_strokes",
            }
        )[["lie", "distance_yd", "expected_strokes"]]
        states = pd.concat([start, end], ignore_index=True)
        states["lie"] = states["lie"].astype(str).str.strip().str.casefold()
        states = states.loc[states["lie"].isin((*NON_GREEN_LIES, "hole"))].copy()
        states["distance_yd"] = pd.to_numeric(states["distance_yd"], errors="coerce")
        states["expected_strokes"] = pd.to_numeric(states["expected_strokes"], errors="coerce")
        states = states.dropna().drop_duplicates(["lie", "distance_yd", "expected_strokes"])
        observed = np.asarray(
            [model.value(lie, distance) for lie, distance in states[["lie", "distance_yd"]].itertuples(index=False, name=None)],
            dtype=float,
        )
        residual = np.abs(observed - states["expected_strokes"].to_numpy(float))
        maximum = float(np.max(residual, initial=0.0))
        if maximum > tolerance:
            raise ReferenceValueUnavailable(
                f"{path.name} differs from frozen Broadie function by {maximum:.9g}"
            )
        path_rows.append(
            {
                "path": str(path.resolve()),
                "unique_states": int(len(states)),
                "maximum_abs_residual": maximum,
            }
        )
        global_rows += int(len(states))
        global_max = max(global_max, maximum)
    return {
        "contract": "historical_export_reference_reproduction_v1",
        "unique_state_rows": global_rows,
        "maximum_abs_residual": global_max,
        "tolerance": tolerance,
        "files": path_rows,
    }


def build_artifact(
    *,
    export_root: Path,
    seasons: Sequence[int],
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    source_paths = [export_root / f"pga_{int(season)}_shot_sg.parquet" for season in seasons]
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    raw_2020 = export_root / "pga_2020_shots.parquet"
    putting = reconstruct_frozen_putting_curve(source_paths[0], raw_2020)
    broadie_source = export_root.parent.parent / "shot_collector" / "strokes_gained.py"
    table = build_reference_table(load_broadie_knots(broadie_source), putting)
    model_sha = reference_model_sha256(table, putting.sha256)
    model = ReferenceValueModel(table, putting.sha256, model_sha)
    export_audit = audit_export_reference_values(model, source_paths)

    output.mkdir(parents=True)
    table_path = output / "reference_value_knots.parquet"
    table.to_parquet(table_path, index=False)
    sources = {
        str(path.resolve()): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in [*source_paths, raw_2020]
    }
    manifest: dict[str, object] = {
        "contract": CONTRACT,
        "version": VERSION,
        "research_only": True,
        "production_enabled": False,
        "reference_model_sha256": model.model_sha256,
        "descriptor": model.descriptor,
        "seasons": list(map(int, seasons)),
        "knots": int(len(table)),
        "knots_by_lie": {str(k): int(v) for k, v in table["lie"].value_counts().sort_index().items()},
        "domains_yd": {
            str(lie): [float(group["distance_yd"].min()), float(group["distance_yd"].max())]
            for lie, group in table.groupby("lie", observed=True)
        },
        "putting_curve": {
            "sha256": putting.sha256,
            "samples": int(putting.samples),
            "latest_source_event_start": str(putting.latest_source_event_start),
            "available_at": str(PUTTING_AVAILABLE_AT),
        },
        "historical_export_audit": export_audit,
        "exact_test_vectors": {
            "tee_582": model.value("tee", 582.0),
            "fairway_288": model.value("fairway", 288.0),
            "fairway_17_888889": model.value("fairway", 17.888889),
            "green_0_722222": model.value("green", 0.722222),
            "terminal": model.value("hole", 0.0),
        },
        "interpolation": {
            "non_green": "linear_extrapolation_exact_broadie_exporter",
            "green": "linear_constant_endpoint_exact_quality_eval_curve",
            "terminal": "exact_zero",
        },
        "sources": sources,
        "broadie_source": {
            "path": str(broadie_source.resolve()),
            "sha256": _sha256(broadie_source),
            "size_bytes": broadie_source.stat().st_size,
        },
        "source_sha256": _sha256(Path(__file__)),
        "quality_evaluator_source_sha256": _sha256(Path(quality_eval.__file__)),
        "artifact_sha256": {table_path.name: _sha256(table_path)},
    }
    report = (
        "# Frozen Tour expected-strokes reference\n\n"
        "Research only. Non-green values use the invariant Broadie curve encoded "
        "in 2020-2025 exports. Green values use the exact prior-only curve from "
        f"4,163 completed warmup observations (SHA `{putting.sha256}`).\n\n"
        f"Reference model SHA: `{model.model_sha256}`.\n"
        f"Knots: {len(table)}. Non-green extrapolation matches the frozen exporter.\n"
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    manifest["artifact_sha256"]["REPORT.md"] = _sha256(output / "REPORT.md")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    # Round-trip validation before returning.
    loaded = ReferenceValueModel.load(output)
    if loaded.model_sha256 != model.model_sha256:
        raise AssertionError("Reference artifact round trip changed model identity")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--seasons", default="2020:2025")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pieces = [int(value) for value in args.seasons.split(":")]
    if len(pieces) != 2 or pieces[1] < pieces[0]:
        raise ValueError("--seasons must be start:end")
    seasons = tuple(range(pieces[0], pieces[1] + 1))
    manifest = build_artifact(export_root=args.export_root, seasons=seasons, output=args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

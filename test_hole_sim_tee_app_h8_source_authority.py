from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_tee_app_h8_source_authority as authority


def _fake_resolver(provider: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        donors=provider,
        setup_prior=12.0,
        hole_prior=40.0,
        spatial_bandwidth_yd=24.0,
        max_nearest_distance_yd=120.0,
        same_hole_fallback_max_distance_yd=None,
        minimum_category_support=12,
        physical_course_id="geometry:test",
        coordinate_reference_year=2025,
        coordinate_reference_event_key="pga:R2025034",
    )


class TrackingFrame(pd.DataFrame):
    _metadata = ["events"]

    @property
    def _constructor(self):  # type: ignore[no-untyped-def]
        return TrackingFrame

    def copy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append("copy")
        copied = super().copy(*args, **kwargs)
        copied.events = self.events
        return copied


def _tracking_provider(categories: list[str], events: list[str]) -> TrackingFrame:
    frame = TrackingFrame(
        {
            "donor_key": [f"donor-{index}" for index in range(len(categories))],
            "category": categories,
        }
    )
    frame.events = events
    return frame


def _non_authorizing_candidate() -> dict[str, object]:
    return {
        "authority_id": None,
        "authority_ready": False,
        "production_enabled": False,
        "admission_status": authority.ADMISSION_STATUS,
        "review_lock_sha256": None,
        "candidate_digest_is_not_authority": True,
    }


def test_frozen_real_scope_and_caps_are_exact() -> None:
    assert authority.EXPECTED_SEAM_SHA256 == (
        "70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83"
    )
    assert authority.EXPECTED_PROVIDER_TOTAL == 1_046
    assert authority.EXPECTED_PROVIDER_BY_CATEGORY == {"app": 364, "ott": 682}
    assert authority.ABSOLUTE_HAZARD_PROVIDER_CEILING == 8_192
    assert authority.MAX_HAZARD_DONORS_PER_SURFACE == 8_192
    assert authority.MAX_BATCH_ROWS == 8_192
    assert authority.MAX_COMPLETE_ATOMS_PER_PROPOSAL == 16_384
    assert authority.EXPECTED_QUERY_CONTEXT_SHA256 == (
        "3737e0d82354591d3395204b35fd434299f43e6ced66d817fa1e8032ef276fbd"
    )


def test_cli_exposes_only_fresh_output_directory() -> None:
    parser = authority.build_parser()
    parsed = parser.parse_args(["--output-dir", "candidate-v1"])
    assert vars(parsed) == {"output_dir": Path("candidate-v1")}
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--output-dir", "candidate-v1", "--hazard-digest", "a" * 64]
        )
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_provider_preflight_precedes_hash_and_copy() -> None:
    events: list[str] = []
    provider = _tracking_provider(["ott", "ott", "app", "app"], events)

    def hash_fn(resolver, *, max_donor_count):  # type: ignore[no-untyped-def]
        assert max_donor_count == authority.ABSOLUTE_HAZARD_PROVIDER_CEILING
        events.append("hash")
        return "f" * 64

    snapshot, _, digest, audit = authority._capture_hazard_snapshot(
        _fake_resolver(provider),
        expected_total=4,
        expected_by_category={"app": 2, "ott": 2},
        hash_fn=hash_fn,
    )
    assert events[:2] == ["hash", "copy"]
    assert digest == "f" * 64
    assert audit["provider_donors"] == 4
    assert len(snapshot) == 4
    assert snapshot["hazard_provider_row_order"].tolist() == [0, 1, 2, 3]


def test_provider_category_mutation_rejects_before_hash_or_copy() -> None:
    events: list[str] = []
    provider = _tracking_provider(["ott", "ott", "ott", "app"], events)

    def forbidden_hash(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("hash")
        return "f" * 64

    with pytest.raises(authority.SourceAuthorityError, match="categories changed"):
        authority._capture_hazard_snapshot(
            _fake_resolver(provider),
            expected_total=4,
            expected_by_category={"app": 2, "ott": 2},
            hash_fn=forbidden_hash,
        )
    assert events == []


def test_provider_8193_rejects_before_distribution_hash_or_copy() -> None:
    events: list[str] = []
    provider = _tracking_provider(["ott"] * 8_193, events)

    def forbidden_hash(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("hash")
        return "f" * 64

    with pytest.raises(authority.SourceAuthorityError, match="absolute ceiling"):
        authority._capture_hazard_snapshot(
            _fake_resolver(provider),
            expected_total=8_193,
            expected_by_category={"ott": 8_193},
            hash_fn=forbidden_hash,
        )
    assert events == []


def test_closed_output_index_accepts_exact_synthetic_tree(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    (root / "nested").mkdir(parents=True)
    (root / "one.txt").write_text("one", encoding="utf-8")
    (root / "nested" / "two.txt").write_text("two", encoding="utf-8")
    authority._write_output_index(root)
    entries = authority._validate_closed_output_index(root)
    assert set(entries) == {"one.txt", "nested/two.txt"}


def test_closed_output_index_rejects_mutation(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("sealed", encoding="utf-8")
    authority._write_output_index(root)
    payload.write_text("mutated", encoding="utf-8")
    with pytest.raises(authority.SourceAuthorityError, match="checksum mismatch"):
        authority._validate_closed_output_index(root)


def test_closed_output_index_rejects_added_file(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "payload.txt").write_text("sealed", encoding="utf-8")
    authority._write_output_index(root)
    (root / "unindexed.txt").write_text("added", encoding="utf-8")
    with pytest.raises(authority.SourceAuthorityError, match="not closed"):
        authority._validate_closed_output_index(root)


def test_start_end_closure_is_exact() -> None:
    fixed = {"builder": "a" * 64, "dependencies": {"seam": "b" * 64}}
    authority._require_equal_maps(fixed, dict(fixed), "synthetic")
    with pytest.raises(authority.SourceAuthorityError, match="start-to-end"):
        authority._require_equal_maps(
            fixed,
            {"builder": "c" * 64, "dependencies": {"seam": "b" * 64}},
            "synthetic",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority_ready", True),
        ("production_enabled", True),
        ("authority_id", "candidate-derived"),
        ("review_lock_sha256", "d" * 64),
        ("candidate_digest_is_not_authority", False),
        ("admission_status", "admitted"),
    ],
)
def test_candidate_readiness_mutations_fail_closed(field: str, value: object) -> None:
    candidate = _non_authorizing_candidate()
    authority._assert_candidate_non_authorizing(candidate)
    candidate[field] = value
    with pytest.raises(authority.SourceAuthorityError, match="fail-closed"):
        authority._assert_candidate_non_authorizing(candidate)


def test_ordered_inventory_hash_detects_record_mutation() -> None:
    frame = pd.DataFrame(
        {
            "inventory_record_order": np.arange(3, dtype=np.int64),
            "value": ["alpha", "beta", "gamma"],
        }
    )
    original = authority._inventory_frame_hash(frame, "synthetic-v1", ("value",))
    mutated = frame.copy()
    mutated.loc[1, "value"] = "changed"
    assert authority._inventory_frame_hash(
        mutated, "synthetic-v1", ("value",)
    ) != original
    reordered = frame.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(authority.SourceAuthorityError, match="not contiguous"):
        authority._inventory_frame_hash(reordered, "synthetic-v1", ("value",))


def test_physical_numeric_ids_use_fixed_u64_and_proposal_order() -> None:
    proposal = pd.DataFrame(
        {
            "proposal_row_order": [2, 0, 1],
            "atom_id": ["terminal", "physical-a", "physical-b"],
            "terminal": [True, False, False],
        }
    )
    atom_ids = authority._physical_atom_ids(proposal)
    assert atom_ids.dtype == np.dtype("<U64")
    assert atom_ids.tolist() == ["physical-a", "physical-b"]


def test_builder_source_has_no_seam_admission_mutation() -> None:
    source_path = Path(authority.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_attributes = {
        "_SOURCE_PINNED_AUTHORITY_RECORDS",
        "_SOURCE_PINNED_AUTHORITY_LOCK_SHA256",
        "source_inventory_authority_records",
    }
    referenced = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert referenced.isdisjoint(forbidden_attributes)
    true_ready_literals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value in {"authority_ready", "production_enabled"}
                and isinstance(value, ast.Constant)
                and value.value is True
            ):
                true_ready_literals.append((key.value, value.value))
    assert true_ready_literals == []


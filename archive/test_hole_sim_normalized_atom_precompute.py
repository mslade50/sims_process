from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_normalized_atom_precompute as module
import hole_sim_tail_policy as tail_policy


def test_precompute_uses_shared_unbounded_score_tail_policy() -> None:
    assert module.REFERENCE_PRECOMPUTE_VERSION == "normalized_atom_precompute_v1"
    assert module.PRECOMPUTE_VERSION != module.REFERENCE_PRECOMPUTE_VERSION
    assert module.tail_policy.manifest_policy() == tail_policy.manifest_policy()
    signature = inspect.signature(module._run_terminal_parity)
    assert "max_transitions" not in signature.parameters
    assert "max_hole_strokes" not in signature.parameters
    source = inspect.getsource(module._run_terminal_parity)
    assert "max_transitions=" not in source
    assert "max_hole_strokes=" not in source


def test_precompute_cli_has_no_tail_override_controls() -> None:
    with mock.patch.object(sys, "argv", ["hole_sim_normalized_atom_precompute.py"]):
        args = module.parse_args()
    assert not hasattr(args, "max_transitions")
    assert not hasattr(args, "max_hole_strokes")


def test_v2_default_cannot_target_immutable_v1_reference() -> None:
    immutable_v1 = (
        module.REPO
        / "permanent_data"
        / "hole_sim_normalized_atom_precompute_travelers_smoke"
    )
    assert module.DEFAULT_OUTPUT != immutable_v1
    assert "v2_unbounded_score_tail" in module.DEFAULT_OUTPUT.name


def test_precompute_refuses_nonempty_output_before_loading_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "already-used"
    output.mkdir()
    (output / "sentinel.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing non-empty"):
        module.run_smoke(SimpleNamespace(output_dir=output))
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "preserve"


def test_support_arrays_are_contiguous_and_use_right_closed_cdf() -> None:
    support = module.SupportArrays.create(
        key="fixture",
        atom_ids=np.asarray(["a", "b", "c"]),
        points=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        weights=np.asarray([0.2, 0.3, 0.5]),
        source="unit",
        n_setup=1,
        n_hole=2,
        n_broad=3,
    )
    assert support.points.flags.c_contiguous
    assert support.weights.flags.c_contiguous
    assert support.cdf.flags.c_contiguous
    index, atom_ids = support.choose(np.asarray([0.0, 0.2, 0.499, 0.5, 0.999]))
    assert index.tolist() == [0, 1, 1, 2, 2]
    assert atom_ids.tolist() == ["a", "b", "b", "c", "c"]


def test_raw_app_cdf_mirrors_legacy_second_normalization() -> None:
    unconditional = np.asarray([0.07, 0.13, 0.29, 0.51], dtype=float)
    conditional = unconditional[[0, 2, 3]].copy()
    conditional /= conditional.sum()
    support = module.SupportArrays.create(
        key="raw-app-double-normalization",
        atom_ids=np.asarray(["a", "c", "d"]),
        points=np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        weights=conditional,
        source="unit",
        n_setup=1,
        n_hole=2,
        n_broad=3,
    )
    legacy_choice_mass = conditional / conditional.sum()
    legacy_cdf = np.cumsum(legacy_choice_mass)
    legacy_cdf[-1] = 1.0
    assert np.array_equal(support.weights, legacy_choice_mass)
    assert np.array_equal(support.cdf, legacy_cdf)


def test_final_probability_is_base_mass_times_model_weight() -> None:
    base_mass = np.asarray([0.15, 0.25, 0.60])
    model_weight = np.asarray([0.5, 1.0, 1.5])
    support = module.SupportArrays.create(
        key="mass-times-model-weight",
        atom_ids=np.asarray(["a", "b", "c"]),
        points=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        weights=base_mass * model_weight,
        source="unit",
        n_setup=1,
        n_hole=2,
        n_broad=3,
    )
    expected = base_mass * model_weight
    expected /= expected.sum()
    assert np.array_equal(support.weights, expected)


def test_tee_distribution_rows_reproduces_existing_mass_exactly() -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "shot_key": f"s{index}",
                "round_no": 1 + index % 2,
                "hole_no": 1 + index % 3,
                "par": 4 + index % 2,
                "long_yd": 250.0 + index,
                "lat_yd": float(index % 7 - 3),
            }
        )
    frame = pd.DataFrame(rows)
    support_rows, weights, reference = module._tee_distribution_rows(
        frame, round_no=1, hole_no=1, par=4
    )
    assert support_rows["shot_key"].is_unique
    assert np.array_equal(
        support_rows[["long_yd", "lat_yd"]].to_numpy(float), reference.points
    )
    assert np.array_equal(weights, reference.weights)


def test_deferred_classifier_never_fabricates_physical_hazard() -> None:
    frame = pd.DataFrame({"model_long_yd": [1.0], "model_lat_yd": [2.0]})
    classified = module._deferred_surface_classifier(frame)
    assert classified.loc[0, "model_end_lie"] == "deferred_runtime_surface_probability"
    assert pd.isna(classified.loc[0, "model_end_hazard"])

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scoring_shadow import (
    ShadowUnavailable,
    classify_cut_format,
    compute_setup_yardage_signal,
    compute_shadow_forecast,
    load_setup_yardage_calibration,
    load_shadow_calibration,
    unavailable_setup_yardage_signal,
)

TOURCHAMP_COMPLETED = [
    {
        "round": 1,
        "score": 66.65517241379311,
        "weather_effect": 0.509,
        "structural_residual": -1.04,
        "coverage": 1.0,
    },
    {
        "round": 2,
        "score": 68.10344827586206,
        "weather_effect": 0.298,
        "structural_residual": 0.40,
        "coverage": 1.0,
    },
    {
        "round": 3,
        "score": 67.27586206896552,
        "weather_effect": 0.775,
        "structural_residual": -1.10,
        "coverage": 1.0,
    },
]

EAST_LAKE_2026 = {
    1: [
        (4, 503),
        (3, 204),
        (4, 411),
        (4, 467),
        (4, 436),
        (5, 522),
        (4, 501),
        (4, 397),
        (3, 216),
        (4, 418),
        (3, 209),
        (4, 377),
        (4, 452),
        (4, 520),
        (3, 156),
        (4, 457),
        (4, 434),
        (5, 590),
    ],
    2: [
        (4, 514),
        (3, 183),
        (4, 416),
        (4, 471),
        (4, 456),
        (5, 517),
        (4, 488),
        (4, 384),
        (3, 229),
        (4, 412),
        (3, 226),
        (4, 375),
        (4, 433),
        (4, 534),
        (3, 155),
        (4, 434),
        (4, 451),
        (5, 572),
    ],
    3: [
        (4, 500),
        (3, 191),
        (4, 431),
        (4, 452),
        (4, 439),
        (5, 536),
        (4, 493),
        (4, 320),
        (3, 270),
        (4, 426),
        (3, 191),
        (4, 392),
        (4, 456),
        (4, 534),
        (3, 147),
        (4, 468),
        (4, 431),
        (5, 583),
    ],
    4: [
        (4, 522),
        (3, 211),
        (4, 404),
        (4, 460),
        (4, 458),
        (5, 531),
        (4, 506),
        (4, 383),
        (3, 208),
        (4, 431),
        (3, 211),
        (4, 394),
        (4, 437),
        (4, 548),
        (3, 218),
        (4, 438),
        (4, 423),
        (5, 586),
    ],
}


def _east_lake_geometry():
    return [
        {
            "geometry_key": f"r{rnd}h{hole}",
            "event_key": "pga:R2026060",
            # PGA fallback layout hashes legitimately change with daily setup.
            "course_id": f"layout:r{rnd}",
            "round_no": rnd,
            "hole_no": hole,
            "par": par,
            "yardage": yardage,
        }
        for rnd, holes in EAST_LAKE_2026.items()
        for hole, (par, yardage) in enumerate(holes, start=1)
    ]


def _synthetic_v3_setup_calibration(selected_mode="broadie"):
    """Small, internally consistent v3 artifact independent of the fixture."""
    training_event_keys = [
        "pga:R2023001",
        "pga:R2024001",
        "pga:R2025001",
    ]
    training_hash = hashlib.sha256(
        json.dumps(training_event_keys, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "calibration_version": "synthetic-setup-v3",
        "schema_version": "live-scoring-setup-yardage/v3",
        "status": "shadow_only",
        "frozen": True,
        "method": "parallel_broadie_and_empirical_yardage_arms",
        "selected_adjustment_mode": selected_mode,
        "response_weight": 1.0,
        "response_weight_provenance": {"selection": "synthetic test"},
        "max_abs_adjustment": 0.5,
        "max_abs_round_yardage_delta": 300.0,
        "max_abs_actual_official_hole_delta": 100.0,
        "required_holes": 18,
        "round_reference_mean": {"2": 0.0, "3": 0.0, "4": 0.0},
        "round_reference_mode": {
            "2": "global",
            "3": "global",
            "4": "global",
        },
        "round_course_eb_pseudocount": {
            "2": None,
            "3": None,
            "4": None,
        },
        "course_references": {},
        "yardage_coefficient_model": {
            "schema_version": "course-yardage-coefficient/v1",
            "model_version": "synthetic-course-yardage-v1",
            "units": "strokes_per_10_yards",
            "design": {
                "observation": "synthetic event-round-hole field mean",
                "formula": "score ~ yardage + event_hole + event_round",
                "fixed_effects": ["event_hole", "event_round"],
                "within_transform": "synthetic two-way demeaning",
                "standard_errors": "synthetic clustered SE",
                "course_identity": "exact DataGolf course_num",
                "informative_edition": "synthetic complete edition",
                "heterogeneity": "synthetic REML",
                "shrinkage": "synthetic normal-normal EB",
            },
            "training_cutoff_calendar_year": 2025,
            "training_event_keys": training_event_keys,
            "training_event_keys_sha256": training_hash,
            "eligibility": {
                "min_informative_years": 3,
                "min_varying_holes_per_edition": 8,
            },
            "global": {
                "beta": 0.028,
                "cluster_se": 0.002,
                "n_events": 3,
                "n_hole_rounds": 216,
            },
            "max_abs_adjustment": 0.35,
            "heterogeneity": {
                "method": "fixed_center_marginal_ml",
                "mu": 0.028,
                "tau2": 0.0001,
                "tau": 0.01,
            },
            "courses": {
                "688": {
                    "course_names": ["Synthetic East Lake"],
                    "informative_years": [2023, 2024, 2025],
                    "n_informative_years": 3,
                    "n_events": 3,
                    "event_keys": training_event_keys,
                    "raw_beta": 0.018,
                    "cluster_se": 0.005,
                    "reliability": 0.8,
                    "shrunk_beta": 0.02,
                }
            },
            "fallback": "global",
        },
        "yardage_delta_references": {
            "definition": ("Target round total yards minus strictly-prior-round mean"),
            "round_reference_mean": {
                "2": 0.0,
                "3": 0.0,
                "4": 9.0,
            },
            "round_reference_mode": {
                "2": "global",
                "3": "global",
                "4": "course_eb",
            },
            "round_course_eb_pseudocount": {
                "2": None,
                "3": None,
                "4": 3.0,
            },
            "course_references": {
                "688": {
                    "n": 3,
                    "course_names": ["Synthetic East Lake"],
                    "rounds": {
                        "2": {
                            "sum_delta": 0.0,
                            "mean_delta": 0.0,
                            "eb_reference": None,
                        },
                        "3": {
                            "sum_delta": 0.0,
                            "mean_delta": 0.0,
                            "eb_reference": None,
                        },
                        "4": {
                            "sum_delta": 3.0,
                            "mean_delta": 1.0,
                            "eb_reference": 5.0,
                        },
                    },
                }
            },
        },
        "guards": {
            "par_3_yardage": [70.0, 350.0],
            "par_4_yardage": [180.0, 600.0],
            "par_5_yardage": [350.0, 750.0],
            "par_6_yardage": [500.0, 900.0],
        },
    }


def _tourchamp(calibration, **overrides):
    inputs = {
        "target_round": 4,
        "course_id": 688,
        "cut_format": "no_cut",
        "active_players": 29,
        "cohort_members": [f"player-{idx:02d}" for idx in range(29)],
        "completed_rounds": copy.deepcopy(TOURCHAMP_COMPLETED),
        "target_baseline": 68.98,
        "target_field_skill": 1.2084487401680728,
        "target_weather_effect": 1.0671749417624523,
        "production_candidate": 68.4277,
        "sheet_before": 67.5,
        "published_after": 68.4,
    }
    inputs.update(overrides)
    return compute_shadow_forecast(calibration, **inputs)


class ScoringShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_shadow_calibration()
        cls.setup_calibration = load_setup_yardage_calibration()

    def test_tourchamp_r4_active_cohort_fixture(self):
        result = _tourchamp(self.calibration)

        self.assertEqual(classify_cut_format(100, 29), "no_cut")
        self.assertEqual(result["transition_source"], "course_eb:688")
        self.assertAlmostEqual(result["transition_used"], 0.04987647500056093)
        self.assertAlmostEqual(result["weather_delta"], 0.5398416084291189)
        self.assertAlmostEqual(result["weather_paired"], 67.93454566963659)
        self.assertAlmostEqual(result["robust_structural"], 68.11072620159438)
        self.assertAlmostEqual(result["shadow_unrounded"], 67.98299531592459)
        self.assertEqual(result["shadow_display"], 68.0)
        self.assertRegex(result["calibration_hash"], r"^[0-9a-f]{64}$")

    def test_tourchamp_daily_setup_moves_only_the_shadow(self):
        setup = compute_setup_yardage_signal(
            self.setup_calibration,
            _east_lake_geometry(),
            4,
            course_id=688,
            event_key="pga:R2026060",
        )
        result = _tourchamp(self.calibration, setup_yardage=setup)

        self.assertEqual(
            setup["round_yardages"],
            {
                "1": 7270.0,
                "2": 7250.0,
                "3": 7260.0,
                "4": 7369.0,
            },
        )
        self.assertAlmostEqual(setup["prior_yardage_average"], 7260.0)
        self.assertAlmostEqual(setup["yardage_delta"], 109.0)
        self.assertAlmostEqual(setup["raw_expected_strokes_delta"], 0.3461666666666652)
        self.assertAlmostEqual(
            setup["historical_round_reference_global"],
            0.021309523809523685,
        )
        self.assertAlmostEqual(
            setup["historical_round_reference_course_mean"],
            -0.0567333333333437,
        )
        self.assertAlmostEqual(
            setup["historical_round_reference"], -0.008706959706963769
        )
        self.assertEqual(setup["reference_source"], "course_eb:688")
        self.assertEqual(setup["reference_course_id"], 688)
        self.assertEqual(setup["reference_course_n"], 5)
        self.assertEqual(setup["reference_pseudocount"], 8.0)
        self.assertAlmostEqual(
            setup["centered_expected_strokes_delta"],
            0.3548736263736254,
        )
        self.assertFalse(setup["was_capped"])
        self.assertAlmostEqual(result["shadow_before_setup"], 67.98299531592459)
        self.assertAlmostEqual(result["setup_adjustment"], 0.3548736263736254)
        self.assertAlmostEqual(result["shadow_unrounded"], 68.33786894229822)
        self.assertEqual(result["shadow_display"], 68.3)
        self.assertEqual(result["production_candidate"], 68.4277)
        self.assertEqual(result["sheet_before"], 67.5)

    def test_setup_unknown_course_uses_global_reference(self):
        setup = compute_setup_yardage_signal(
            self.setup_calibration,
            _east_lake_geometry(),
            4,
            course_id=999999,
            event_key="pga:R2026060",
        )

        self.assertEqual(setup["reference_source"], "global")
        self.assertEqual(setup["reference_course_id"], 999999)
        self.assertEqual(setup["reference_course_n"], 0)
        self.assertEqual(setup["reference_pseudocount"], 8.0)
        self.assertIsNone(setup["historical_round_reference_course_mean"])
        self.assertAlmostEqual(
            setup["historical_round_reference"], 0.021309523809523685
        )
        self.assertAlmostEqual(setup["adjustment"], 0.32485714285713796)

    def test_setup_never_uses_geometry_layout_id_as_physical_course(self):
        misleading_geometry = _east_lake_geometry()
        for row in misleading_geometry:
            row["course_id"] = "688"

        setup = compute_setup_yardage_signal(
            self.setup_calibration,
            misleading_geometry,
            4,
            course_id=999999,
            event_key="pga:R2026060",
        )

        self.assertEqual(setup["reference_source"], "global")
        self.assertEqual(setup["reference_course_id"], 999999)

    def test_setup_r2_cv_selection_ignores_known_course_entry(self):
        known = compute_setup_yardage_signal(
            self.setup_calibration,
            _east_lake_geometry(),
            2,
            course_id=688,
            event_key="pga:R2026060",
        )
        unknown = compute_setup_yardage_signal(
            self.setup_calibration,
            _east_lake_geometry(),
            2,
            course_id=999999,
            event_key="pga:R2026060",
        )

        for setup in (known, unknown):
            self.assertEqual(setup["reference_source"], "global")
            self.assertEqual(setup["reference_course_n"], 0)
            self.assertIsNone(setup["reference_pseudocount"])
            self.assertIsNone(setup["historical_round_reference_course_mean"])
            self.assertAlmostEqual(
                setup["historical_round_reference"],
                -0.0543551020408156,
            )
        self.assertAlmostEqual(known["adjustment"], unknown["adjustment"])

    def test_setup_rejects_incomplete_and_corrupt_daily_geometry(self):
        with self.assertRaisesRegex(ShadowUnavailable, "18 unique holes"):
            compute_setup_yardage_signal(
                self.setup_calibration,
                _east_lake_geometry()[:-1],
                4,
                event_key="pga:R2026060",
            )

        corrupt = _east_lake_geometry()
        par_five = next(
            row for row in corrupt if row["round_no"] == 4 and row["hole_no"] == 18
        )
        par_five.update(actual_yardage=186, official_yardage=601)
        with self.assertRaisesRegex(
            ShadowUnavailable, "actual/official yardage differs"
        ):
            compute_setup_yardage_signal(
                self.setup_calibration,
                corrupt,
                4,
                event_key="pga:R2026060",
            )

    def test_setup_allows_daily_layout_ids_but_rejects_round_duplicates(self):
        geometry = _east_lake_geometry()
        setup = compute_setup_yardage_signal(
            self.setup_calibration,
            geometry,
            4,
            event_key="pga:R2026060",
        )
        self.assertEqual(setup["status"], "ok")

        duplicate = copy.deepcopy(geometry)
        extra = copy.deepcopy(duplicate[18])
        extra["course_id"] = "layout:second-r2-course"
        duplicate.append(extra)
        with self.assertRaisesRegex(ShadowUnavailable, "duplicate geometry"):
            compute_setup_yardage_signal(
                self.setup_calibration,
                duplicate,
                4,
                event_key="pga:R2026060",
            )

    def test_setup_v3_parallel_arms_default_to_broadie(self):
        calibration = _synthetic_v3_setup_calibration()
        setup = compute_setup_yardage_signal(
            calibration,
            _east_lake_geometry(),
            4,
            course_id=688,
            event_key="pga:R2026060",
        )

        self.assertEqual(setup["selected_adjustment_mode"], "broadie")
        self.assertAlmostEqual(setup["yardage_delta"], 109.0)
        self.assertAlmostEqual(
            setup["broadie_adjustment"],
            setup["raw_expected_strokes_delta"],
        )
        self.assertAlmostEqual(setup["empirical_global_adjustment"], 0.28)
        self.assertAlmostEqual(setup["empirical_course_adjustment"], 0.208)
        self.assertAlmostEqual(setup["adjustment"], setup["broadie_adjustment"])
        self.assertEqual(setup["empirical_course_coefficient_source"], "course_eb:688")
        self.assertAlmostEqual(setup["empirical_course_coefficient"], 0.02)
        self.assertAlmostEqual(setup["empirical_course_cluster_se"], 0.005)
        self.assertEqual(setup["empirical_course_n_events"], 3)
        self.assertAlmostEqual(setup["yardage_delta_reference_global"], 9.0)
        self.assertAlmostEqual(setup["yardage_delta_reference_course"], 5.0)
        self.assertEqual(setup["yardage_delta_reference_source"], "course_eb:688")
        self.assertTrue(setup["training_event_overlap_checked"])
        self.assertEqual(
            setup["adjustment_arms"]["empirical_course_eb"]["coefficient_provenance"][
                "n_informative_years"
            ],
            3,
        )

    def test_setup_v3_selected_modes_and_caps_are_independent(self):
        global_calibration = _synthetic_v3_setup_calibration("empirical_global")
        global_setup = compute_setup_yardage_signal(
            global_calibration,
            _east_lake_geometry(),
            4,
            course_id=688,
            event_key="pga:R2026060",
        )
        self.assertAlmostEqual(global_setup["adjustment"], 0.28)
        self.assertEqual(global_setup["max_abs_adjustment"], 0.35)

        capped = _synthetic_v3_setup_calibration("empirical_course_eb")
        capped["max_abs_adjustment"] = 0.3
        capped["yardage_coefficient_model"]["max_abs_adjustment"] = 0.25
        capped_setup = compute_setup_yardage_signal(
            capped,
            _east_lake_geometry(),
            4,
            course_id=688,
            event_key="pga:R2026060",
        )
        self.assertAlmostEqual(capped_setup["broadie_adjustment"], 0.3)
        self.assertTrue(capped_setup["broadie_was_capped"])
        self.assertAlmostEqual(capped_setup["empirical_global_adjustment"], 0.25)
        self.assertTrue(capped_setup["empirical_global_was_capped"])
        self.assertAlmostEqual(capped_setup["empirical_course_adjustment"], 0.208)
        self.assertFalse(capped_setup["empirical_course_was_capped"])
        self.assertAlmostEqual(capped_setup["adjustment"], 0.208)
        self.assertFalse(capped_setup["was_capped"])
        self.assertEqual(capped_setup["max_abs_adjustment"], 0.25)
        self.assertEqual(capped_setup["broadie_max_abs_adjustment"], 0.3)
        self.assertEqual(capped_setup["empirical_max_abs_adjustment"], 0.25)

    def test_setup_v3_course_arm_requires_exact_eligible_course_id(self):
        calibration = _synthetic_v3_setup_calibration("empirical_course_eb")
        canonical_string = compute_setup_yardage_signal(
            calibration,
            _east_lake_geometry(),
            4,
            course_id="688",
            event_key="pga:R2026060",
        )
        self.assertEqual(
            canonical_string["empirical_course_coefficient_source"],
            "course_eb:688",
        )

        for course_id in (999999, "layout:east-lake", 688.0, "0688"):
            with self.subTest(course_id=course_id):
                setup = compute_setup_yardage_signal(
                    calibration,
                    _east_lake_geometry(),
                    4,
                    course_id=course_id,
                    event_key="pga:R2026060",
                )
                self.assertEqual(setup["empirical_course_coefficient_source"], "global")
                self.assertAlmostEqual(setup["empirical_course_coefficient"], 0.028)
                self.assertAlmostEqual(setup["yardage_delta_reference_course"], 9.0)
                self.assertAlmostEqual(setup["adjustment"], 0.28)

        no_eligible_course = copy.deepcopy(calibration)
        no_eligible_course["yardage_coefficient_model"]["courses"] = {}
        fallback = compute_setup_yardage_signal(
            no_eligible_course,
            _east_lake_geometry(),
            4,
            course_id=688,
            event_key="pga:R2026060",
        )
        self.assertEqual(
            fallback["empirical_course_coefficient_fallback_reason"],
            "course_not_calibrated",
        )
        self.assertEqual(fallback["yardage_delta_reference_source"], "global")

    def test_setup_v3_training_overlap_guard_is_fail_closed(self):
        calibration = _synthetic_v3_setup_calibration()
        with self.assertRaisesRegex(ShadowUnavailable, "requires current event_key"):
            compute_setup_yardage_signal(
                calibration,
                _east_lake_geometry(),
                4,
                course_id=688,
            )
        with self.assertRaisesRegex(ShadowUnavailable, "overlaps.*training panel"):
            compute_setup_yardage_signal(
                calibration,
                _east_lake_geometry(),
                4,
                course_id=688,
                event_key="pga:R2025001",
            )

        unavailable = unavailable_setup_yardage_signal(calibration, "training overlap")
        self.assertEqual(unavailable["adjustment"], 0.0)
        self.assertEqual(unavailable["selected_adjustment"], 0.0)
        self.assertEqual(unavailable["broadie_adjustment"], 0.0)
        self.assertEqual(unavailable["empirical_global_adjustment"], 0.0)
        self.assertEqual(unavailable["empirical_course_adjustment"], 0.0)

    def test_setup_v3_loader_validates_hash_and_model_arithmetic(self):
        valid = _synthetic_v3_setup_calibration()
        zero_variance = copy.deepcopy(valid)
        model = zero_variance["yardage_coefficient_model"]
        model["heterogeneity"].update(tau2=0.0, tau=0.0)
        model["courses"]["688"].update(
            cluster_se=0.0,
            reliability=0.0,
            shrunk_beta=model["global"]["beta"],
        )

        invalid_cases = []
        bad_hash = copy.deepcopy(valid)
        bad_hash["yardage_coefficient_model"]["training_event_keys_sha256"] = "0" * 64
        invalid_cases.append(("training_event_keys_sha256", bad_hash))

        bad_mu = copy.deepcopy(valid)
        bad_mu["yardage_coefficient_model"]["heterogeneity"]["mu"] = 0.1
        invalid_cases.append(("mean must equal", bad_mu))

        bad_method = copy.deepcopy(valid)
        bad_method["yardage_coefficient_model"]["heterogeneity"]["method"] = "reml"
        invalid_cases.append(("must be fixed_center_marginal_ml", bad_method))

        bad_global_events = copy.deepcopy(valid)
        bad_global_events["yardage_coefficient_model"]["global"]["n_events"] = 4
        invalid_cases.append(("Global.*event count is inconsistent", bad_global_events))

        bad_events = copy.deepcopy(valid)
        bad_events["yardage_coefficient_model"]["courses"]["688"]["n_events"] = 4
        invalid_cases.append(("event count is inconsistent", bad_events))

        bad_years = copy.deepcopy(valid)
        course = bad_years["yardage_coefficient_model"]["courses"]["688"]
        course["informative_years"] = [2024, 2025]
        course["n_informative_years"] = 2
        invalid_cases.append(("minimum informative-year", bad_years))

        bad_reliability = copy.deepcopy(valid)
        bad_reliability["yardage_coefficient_model"]["courses"]["688"][
            "reliability"
        ] = 0.7
        invalid_cases.append(("reliability is inconsistent", bad_reliability))

        bad_shrinkage = copy.deepcopy(valid)
        bad_shrinkage["yardage_coefficient_model"]["courses"]["688"]["shrunk_beta"] = (
            0.1
        )
        invalid_cases.append(("shrunk beta is inconsistent", bad_shrinkage))

        bad_cap = copy.deepcopy(valid)
        bad_cap["yardage_coefficient_model"]["max_abs_adjustment"] = 0.0
        invalid_cases.append(("must be positive", bad_cap))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            def load(payload, filename):
                path = temp_dir / filename
                path.write_text(json.dumps(payload), encoding="utf-8")
                return load_setup_yardage_calibration(path)

            self.assertEqual(
                load(valid, "valid.json")["selected_adjustment_mode"],
                "broadie",
            )
            self.assertEqual(
                load(zero_variance, "zero-variance.json")["yardage_coefficient_model"][
                    "courses"
                ]["688"]["reliability"],
                0.0,
            )
            for index, (message, payload) in enumerate(invalid_cases):
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(ShadowUnavailable, message),
                ):
                    load(payload, f"invalid-{index}.json")

    def test_authoritative_value_is_diagnostic_only(self):
        original = _tourchamp(self.calibration, sheet_before=67.5)
        changed = _tourchamp(self.calibration, sheet_before=-999)

        self.assertEqual(original["model_input_hash"], changed["model_input_hash"])
        self.assertNotEqual(original["input_hash"], changed["input_hash"])
        self.assertEqual(original["shadow_unrounded"], changed["shadow_unrounded"])

    def test_cut_classifier_uses_opening_field(self):
        cases = [
            ((65, 150), "cut"),
            ((100, 29), "no_cut"),
            ((29, 29), "no_cut"),
            ((0, 150), "no_cut"),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(classify_cut_format(*args), expected)

    def test_unknown_and_unaliased_course_use_global_prior(self):
        unknown = _tourchamp(self.calibration, course_id=999999)
        anomalous_2024_id = _tourchamp(self.calibration, course_id=933)

        for result in (unknown, anomalous_2024_id):
            self.assertEqual(result["transition_source"], "global")
            self.assertAlmostEqual(result["transition_used"], -0.07399259299906512)

    def test_requires_exact_prior_rounds_and_minimum_coverage(self):
        with self.assertRaisesRegex(ShadowUnavailable, "needs completed rounds"):
            _tourchamp(
                self.calibration,
                completed_rounds=TOURCHAMP_COMPLETED[:2],
            )
        low = copy.deepcopy(TOURCHAMP_COMPLETED)
        low[1]["coverage"] = 0.7999
        with self.assertRaisesRegex(ShadowUnavailable, "below 80.0%"):
            _tourchamp(self.calibration, completed_rounds=low)
        passing = copy.deepcopy(TOURCHAMP_COMPLETED)
        passing[1]["coverage"] = 0.8
        self.assertEqual(
            _tourchamp(self.calibration, completed_rounds=passing)["status"],
            "ok",
        )

    def test_rejects_score_to_par_and_nonfinite_weather(self):
        relative = copy.deepcopy(TOURCHAMP_COMPLETED)
        relative[0]["score"] = -3.2
        with self.assertRaisesRegex(ShadowUnavailable, "absolute field score"):
            _tourchamp(self.calibration, completed_rounds=relative)
        with self.assertRaisesRegex(ShadowUnavailable, "not finite"):
            _tourchamp(self.calibration, target_weather_effect=math.nan)

    def test_cohort_membership_and_hash_are_deterministic(self):
        members = [f"player-{idx:02d}" for idx in range(29)]
        forward = _tourchamp(self.calibration, cohort_members=members)
        reverse = _tourchamp(self.calibration, cohort_members=list(reversed(members)))
        self.assertEqual(forward["input_hash"], reverse["input_hash"])
        self.assertEqual(forward["cohort_hash"], reverse["cohort_hash"])
        with self.assertRaisesRegex(ShadowUnavailable, "membership"):
            _tourchamp(self.calibration, cohort_members=members[:-1])

    def test_calibration_content_changes_model_hash(self):
        original = _tourchamp(self.calibration)
        changed_calibration = copy.deepcopy(self.calibration)
        changed_calibration["rounds"]["4"]["paired_weight_beta"] = 0.5
        changed = _tourchamp(changed_calibration)

        self.assertNotEqual(original["calibration_hash"], changed["calibration_hash"])
        self.assertNotEqual(original["model_input_hash"], changed["model_input_hash"])

    def test_frozen_artifact_course_eb_recomputes(self):
        calibration = self.calibration
        self.assertEqual(
            calibration["schema_version"],
            "live-scoring-paired-calibration/v1",
        )
        self.assertEqual(calibration["status"], "shadow_only")
        self.assertTrue(calibration["frozen"])
        self.assertEqual(calibration["training_panel"]["year_max"], 2025)
        self.assertEqual(calibration["training_panel"]["editions"], 369)
        canonical = copy.deepcopy(calibration)
        artifact_hash = canonical.pop("_calibration_sha256")
        expected_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(artifact_hash, expected_hash)
        self.assertEqual(
            artifact_hash,
            "bd277fef0d66c9f514aaae2fc67c2dab8d8a436b4ef25626024196164a64fbc3",
        )

        for rnd, formats in calibration["course_transitions"].items():
            round_cfg = calibration["rounds"][rnd]
            pseudo = round_cfg["course_transition_eb_pseudocount"]
            for cut_format, courses in formats.items():
                global_mean = round_cfg["global_transition_prior"][cut_format]
                for course_id, entry in courses.items():
                    with self.subTest(
                        rnd=rnd, cut_format=cut_format, course_id=course_id
                    ):
                        self.assertGreaterEqual(entry["n"], 3)
                        expected = (entry["sum_delta"] + pseudo * global_mean) / (
                            entry["n"] + pseudo
                        )
                        self.assertAlmostEqual(entry["eb_prior"], expected, places=8)

    def test_frozen_setup_artifact_course_eb_recomputes(self):
        calibration = self.setup_calibration
        self.assertIn(
            calibration["schema_version"],
            {
                "live-scoring-setup-yardage/v2",
                "live-scoring-setup-yardage/v3",
            },
        )
        panel = calibration["training_panel"]
        self.assertEqual(panel["eligible_single_course_four_round_editions"], 245)
        self.assertEqual(panel["exact_datagolf_course_mapped_editions"], 244)
        self.assertEqual(
            panel["geometry_rejection_events"]["yardage_outside_par_guard"],
            ["pga:R2021535", "pga:R2022026"],
        )
        rejection_details = panel["geometry_rejection_details"]
        self.assertEqual(
            rejection_details["pga:R2021535"]["violations"][0]["yardage"],
            602.0,
        )
        self.assertEqual(
            {
                violation["yardage"]
                for violation in rejection_details["pga:R2022026"]["violations"]
            },
            {173.0},
        )
        self.assertEqual(
            [row["event_key"] for row in panel["unmapped_course_editions"]],
            ["pga:R2024018"],
        )
        self.assertEqual(
            panel["mapped_editions"]["pga:R2020060"],
            [2020, 688, "event_id_exact_date"],
        )
        self.assertEqual(
            panel["mapped_editions"]["pga:R2024060"],
            [2024, 933, "event_id_exact_date"],
        )
        self.assertEqual(
            panel["mapped_editions"]["pga:R2025060"],
            [2025, 688, "event_id_exact_date"],
        )
        self.assertIn(
            "No course-name or TOURCAST layout-ID aliases", panel["mapping_definition"]
        )
        self.assertEqual(
            calibration["round_course_eb_pseudocount"],
            {"2": None, "3": 8.0, "4": 8.0},
        )
        self.assertEqual(
            calibration["round_reference_mode"],
            {"2": "global", "3": "course_eb", "4": "course_eb"},
        )
        self.assertEqual(calibration["course_references"]["688"]["n"], 5)
        self.assertEqual(calibration["course_references"]["933"]["n"], 1)
        response = calibration["response_weight_provenance"]
        self.assertEqual(calibration["response_weight"], 1.0)
        self.assertIn("not a fitted", response["selection"])
        r2_fit = response["historical_score_outcome_audit"]["per_round"]["2"]
        self.assertEqual(r2_fit["n"], 241)
        self.assertAlmostEqual(r2_fit["slope"], 0.3777293278068579)
        self.assertAlmostEqual(r2_fit["slope_standard_error"], 0.35998539043681743)
        self.assertLess(r2_fit["slope_95pct_normal_approx"][0], 0.0)
        self.assertGreater(r2_fit["slope_95pct_normal_approx"][1], 0.0)

        for course_id, course in calibration["course_references"].items():
            for rnd, entry in course["rounds"].items():
                with self.subTest(course_id=course_id, rnd=rnd):
                    n = course["n"]
                    pseudo = calibration["round_course_eb_pseudocount"][rnd]
                    global_mean = calibration["round_reference_mean"][rnd]
                    self.assertAlmostEqual(entry["mean_delta"], entry["sum_delta"] / n)
                    if calibration["round_reference_mode"][rnd] == "global":
                        self.assertIsNone(entry["eb_reference"])
                    else:
                        self.assertAlmostEqual(
                            entry["eb_reference"],
                            (entry["sum_delta"] + pseudo * global_mean) / (n + pseudo),
                        )


if __name__ == "__main__":
    unittest.main()

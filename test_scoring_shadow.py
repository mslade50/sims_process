import copy
import hashlib
import json
import math
import unittest

from scoring_shadow import (
    ShadowUnavailable,
    classify_cut_format,
    compute_shadow_forecast,
    load_shadow_calibration,
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

    def test_authoritative_value_is_diagnostic_only(self):
        original = _tourchamp(self.calibration, sheet_before=67.5)
        changed = _tourchamp(self.calibration, sheet_before=-999)

        self.assertEqual(
            original["model_input_hash"], changed["model_input_hash"]
        )
        self.assertNotEqual(original["input_hash"], changed["input_hash"])
        self.assertEqual(
            original["shadow_unrounded"], changed["shadow_unrounded"]
        )

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
            self.assertAlmostEqual(
                result["transition_used"], -0.07399259299906512
            )

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
        reverse = _tourchamp(
            self.calibration, cohort_members=list(reversed(members))
        )
        self.assertEqual(forward["input_hash"], reverse["input_hash"])
        self.assertEqual(forward["cohort_hash"], reverse["cohort_hash"])
        with self.assertRaisesRegex(ShadowUnavailable, "membership"):
            _tourchamp(self.calibration, cohort_members=members[:-1])

    def test_calibration_content_changes_model_hash(self):
        original = _tourchamp(self.calibration)
        changed_calibration = copy.deepcopy(self.calibration)
        changed_calibration["rounds"]["4"]["paired_weight_beta"] = 0.5
        changed = _tourchamp(changed_calibration)

        self.assertNotEqual(
            original["calibration_hash"], changed["calibration_hash"]
        )
        self.assertNotEqual(
            original["model_input_hash"], changed["model_input_hash"]
        )

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
            json.dumps(
                canonical, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
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
                        expected = (
                            entry["sum_delta"] + pseudo * global_mean
                        ) / (entry["n"] + pseudo)
                        self.assertAlmostEqual(
                            entry["eb_prior"], expected, places=8
                        )


if __name__ == "__main__":
    unittest.main()

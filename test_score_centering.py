import unittest

import pandas as pd

from score_centering import (
    CENTERING_VERSION,
    center_player_advantages,
    expected_field_score,
    validate_centered_scores,
    validate_field_relative_predictions,
)


class ScoreCenteringTests(unittest.TestCase):
    def test_single_field_centers_skill_and_weather(self):
        frame = pd.DataFrame({
            "player": ["a", "b", "c"],
            "skill": [1.0, 0.0, -0.5],
            "wind": [0.8, 1.2, 1.0],
            "dew": [-0.1, 0.2, -0.1],
        })

        centered = center_player_advantages(
            frame,
            skill_col="skill",
            score_col="scores_r4",
            weather_col="weather_sg_r4",
            wind_cost_col="wind",
            dew_cost_col="dew",
        )

        self.assertAlmostEqual(centered["scores_r4"].mean(), 0.0, places=12)
        self.assertAlmostEqual(centered["weather_sg_r4"].mean(), 0.0, places=12)
        self.assertAlmostEqual(centered["field_skill_mean"].iloc[0], 1 / 6)
        self.assertTrue(
            (centered["centering_version"] == CENTERING_VERSION).all()
        )

        field_average = 69.9
        expected_scores = field_average - centered["scores_r4"]
        self.assertAlmostEqual(expected_scores.mean(), field_average, places=12)

    def test_multicourse_centers_each_course(self):
        frame = pd.DataFrame({
            "course": ["north", "north", "south", "south"],
            "skill": [1.0, 0.0, 0.5, -0.5],
            "wind": [0.5, 1.5, 2.0, 1.0],
            "dew": [0.1, -0.1, 0.2, -0.2],
        })

        centered = center_player_advantages(
            frame,
            skill_col="skill",
            score_col="scores_r2",
            weather_col="weather_sg_r2",
            wind_cost_col="wind",
            dew_cost_col="dew",
            group_col="course",
        )

        means = centered.groupby("course")["scores_r2"].mean()
        self.assertTrue((means.abs() < 1e-12).all())

    def test_expected_field_score_uses_active_field_skill(self):
        result = expected_field_score(
            base_score=69.32,
            field_mean_skill=0.207692691780822,
            wind_effect=1.3183369716895,
            dew_effect=-0.131,
            difficulty_feedback=-0.399,
        )
        self.assertEqual(round(result, 1), 69.9)

    def test_validation_rejects_uncentered_scores(self):
        frame = pd.DataFrame({"scores_r4": [0.4, 0.2, -0.1]})
        with self.assertRaisesRegex(ValueError, "not field-centered"):
            validate_centered_scores(frame, "scores_r4")

    def test_validation_rejects_broken_skill_weather_formula(self):
        frame = pd.DataFrame({
            "skill": [0.4, 0.0],
            "field_skill_mean": [0.2, 0.2],
            "weather": [0.1, -0.1],
            "score": [0.4, -0.4],
        })
        with self.assertRaisesRegex(ValueError, "centered skill plus weather"):
            validate_field_relative_predictions(
                frame,
                skill_col="skill",
                score_col="score",
                weather_col="weather",
            )


if __name__ == "__main__":
    unittest.main()

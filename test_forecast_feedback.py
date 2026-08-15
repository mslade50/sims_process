import unittest

from forecast_feedback import (
    forecast_feedback,
    round_scoring_result,
    single_published_forecast,
)


class ForecastFeedbackTests(unittest.TestCase):
    def test_single_published_forecast_requires_one_absolute_score(self):
        self.assertEqual(single_published_forecast([69.3]), 69.3)
        self.assertEqual(single_published_forecast(69.1), 69.1)
        self.assertIsNone(single_published_forecast([69.1, 70.2]))
        self.assertIsNone(single_published_forecast(1.2))

    def test_round_result_uses_published_forecast_for_miss(self):
        result = round_scoring_result(
            published_forecast=[69.3],
            actual_score=69.54,
            base_score=68.69,
            field_adjustment=-0.882,
            wind_impact=0.931,
            dew_impact=-0.080,
        )

        self.assertAlmostEqual(result["forecast_miss"], 0.24)
        self.assertAlmostEqual(result["structural_baseline"], 68.659)
        self.assertAlmostEqual(result["structural_residual"], 0.881)

    def test_feedback_uses_published_forecast_misses(self):
        feedback, weight = forecast_feedback([0.45, 0.24])

        self.assertEqual(weight, 0.6)
        self.assertEqual(feedback, 0.207)


if __name__ == "__main__":
    unittest.main()

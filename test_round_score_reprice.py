import numpy as np

from round_sim import build_round_score_probs


def test_shifted_draws_publish_integer_settlement_pmf():
    probs = build_round_score_probs(
        {"player": np.array([68.0, 68.5, 68.7, 69.49, 69.5])},
        68.4,
    )

    actual = dict(zip(probs["score"], probs["prob"]))
    assert actual == {68: 0.2, 69: 0.6, 70: 0.2}
    assert probs["expected_avg"].unique().tolist() == [68.4]


def test_settlement_pmf_preserves_half_stroke_under_probabilities():
    shifted = np.array([67.7, 68.7, 69.7, 70.7])
    probs = build_round_score_probs({"player": shifted}, 68.4)

    for line in (67.5, 68.5, 69.5, 70.5, 71.5):
        direct = float(np.mean(shifted < line))
        published = float(probs.loc[probs["score"] < line, "prob"].sum())
        assert published == direct

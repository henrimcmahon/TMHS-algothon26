import numpy as np

from bayesian.hypotheses import (
    MeanReversionHypothesis,
)


def test_mean_reversion_predicts_opposite_direction() -> None:
    hypothesis = MeanReversionHypothesis(
        equilibrium=0.0,
        reversion_speed=0.50,
    )

    positive_history = np.asarray(
        [0.001, 0.002, 0.010],
        dtype=np.float64,
    )

    positive_prediction = (
        hypothesis.predict_distribution(
            positive_history
        )
    )

    assert positive_prediction.mean < 0

    negative_history = np.asarray(
        [-0.001, -0.002, -0.010],
        dtype=np.float64,
    )

    negative_prediction = (
        hypothesis.predict_distribution(
            negative_history
        )
    )

    assert negative_prediction.mean > 0
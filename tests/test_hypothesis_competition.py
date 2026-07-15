import numpy as np

from bayesian import BeliefState, BayesianUpdater
from bayesian.hypotheses import (
    MeanReversionHypothesis,
    MomentumHypothesis,
    NoiseHypothesis,
)


def test_updater_compares_three_hypotheses() -> None:
    updater = BayesianUpdater(
        hypotheses=[
            MomentumHypothesis(
                trend_learning_rate=0.20,
            ),
            MeanReversionHypothesis(
                reversion_speed=0.50,
            ),
            NoiseHypothesis(),
        ],
        belief_state=BeliefState.uniform(
            number_of_hypotheses=3,
            forgetting_rate=0.02,
        ),
    )

    observations = np.asarray(
        [
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
        ],
        dtype=np.float64,
    )

    history = observations[:2].copy()

    for observation in observations[2:]:
        prediction = updater.predict(history)

        updater.observe(
            observation=float(observation),
            prediction=prediction,
        )

        history = np.append(
            history,
            observation,
        )

    probabilities = updater.probabilities

    assert np.isclose(
        np.sum(probabilities),
        1.0,
    )

    assert np.all(
        probabilities >= 0
    )
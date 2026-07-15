import numpy as np

from bayesian import BayesianUpdater
from bayesian.hypotheses import NoiseHypothesis


def test_updater_predicts_and_observes() -> None:
    hypotheses = [
        NoiseHypothesis(
            variance=1e-4,
            label="low_variance_noise",
        ),
        NoiseHypothesis(
            variance=1e-2,
            label="high_variance_noise",
        ),
    ]

    updater = BayesianUpdater(hypotheses)

    history = np.asarray(
        [0.001, -0.001, 0.002, -0.002],
        dtype=np.float64,
    )

    prediction = updater.predict(history)

    assert np.isfinite(
        prediction.combined_distribution.mean
    )

    assert (
        prediction.combined_distribution.variance
        > 0
    )

    posterior = updater.observe(
        observation=0.001,
        prediction=prediction,
    )

    assert np.isclose(
        np.sum(posterior),
        1.0,
    )
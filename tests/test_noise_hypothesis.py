import numpy as np

from bayesian.hypotheses import NoiseHypothesis


def test_noise_hypothesis_updates_variance() -> None:
    history = np.asarray(
        [0.002, -0.001, 0.003, -0.002],
        dtype=np.float64,
    )

    hypothesis = NoiseHypothesis(
        variance=1e-4,
        learning_rate=0.05,
    )

    prediction = hypothesis.predict_distribution(history)

    assert prediction.mean == 0.0
    assert prediction.variance > 0

    old_variance = hypothesis.variance

    observation = 0.0015
    log_probability = prediction.log_probability(observation)

    hypothesis.update(observation)

    assert np.isfinite(log_probability)
    assert hypothesis.variance > 0
    assert hypothesis.variance != old_variance
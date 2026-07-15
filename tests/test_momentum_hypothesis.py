import numpy as np

from bayesian.hypotheses import MomentumHypothesis


def test_momentum_updates_towards_observation() -> None:
    hypothesis = MomentumHypothesis(
        trend=0.0,
        variance=1e-4,
        trend_learning_rate=0.20,
    )

    history = np.asarray(
        [0.001, 0.002, 0.003],
        dtype=np.float64,
    )

    old_trend = hypothesis.trend

    hypothesis.update(0.01)

    assert hypothesis.trend > old_trend
    assert hypothesis.variance > 0

    prediction = hypothesis.predict_distribution(
        history
    )

    assert prediction.mean == hypothesis.trend
    assert prediction.variance > 0
import numpy as np

from bayesian import (
    BeliefState,
    BayesianUpdater,
)
from bayesian.hypotheses import (
    MomentumHypothesis,
    NoiseHypothesis,
)


def test_diagnostics_are_finite() -> None:
    updater = BayesianUpdater(
        hypotheses=[
            MomentumHypothesis(),
            NoiseHypothesis(),
        ],
        belief_state=BeliefState.uniform(
            2,
            forgetting_rate=0.01,
        ),
    )

    history = np.asarray(
        [0.001, 0.002, 0.003, 0.004],
        dtype=np.float64,
    )

    prediction = updater.predict(history)

    diagnostics = (
        updater.observe_with_diagnostics(
            observation=0.005,
            prediction=prediction,
        )
    )

    assert np.isfinite(
        diagnostics.negative_log_likelihood
    )

    assert np.isfinite(
        diagnostics.information_gain
    )

    assert diagnostics.information_gain >= 0

    assert diagnostics.predictive_variance > 0

    assert np.isclose(
        np.sum(
            diagnostics
            .posterior_probabilities
        ),
        1.0,
    )


def test_information_gain_is_zero_when_beliefs_do_not_change() -> None:
    updater = BayesianUpdater(
        hypotheses=[
            NoiseHypothesis(
                variance=1e-4,
                label="noise_1",
            ),
            NoiseHypothesis(
                variance=1e-4,
                label="noise_2",
            ),
        ],
        belief_state=BeliefState.uniform(2),
    )

    history = np.asarray(
        [0.001, -0.001, 0.002],
        dtype=np.float64,
    )

    prediction = updater.predict(history)

    diagnostics = (
        updater.observe_with_diagnostics(
            observation=0.001,
            prediction=prediction,
        )
    )

    assert np.isclose(
        diagnostics.information_gain,
        0.0,
        atol=1e-12,
    )


def test_surprising_observation_has_higher_negative_log_likelihood() -> None:
    def calculate_negative_log_likelihood(
        observation: float,
    ) -> float:
        updater = BayesianUpdater(
            hypotheses=[
                NoiseHypothesis(
                    variance=1e-4,
                )
            ]
        )

        history = np.asarray(
            [0.0, 0.001, -0.001],
            dtype=np.float64,
        )

        prediction = updater.predict(history)

        diagnostics = (
            updater.observe_with_diagnostics(
                observation=observation,
                prediction=prediction,
            )
        )

        return diagnostics.negative_log_likelihood

    expected_negative_log_likelihood = calculate_negative_log_likelihood(
        0.001
    )

    extreme_negative_log_likelihood = calculate_negative_log_likelihood(
        0.10
    )

    assert extreme_negative_log_likelihood > expected_negative_log_likelihood
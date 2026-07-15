from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bayesian import BeliefState, BayesianUpdater
from bayesian.hypotheses import (
    MeanReversionHypothesis,
    MomentumHypothesis,
    NoiseHypothesis,
)


FloatArray = NDArray[np.float64]


def create_updater() -> BayesianUpdater:
    return BayesianUpdater(
        hypotheses=[
            MomentumHypothesis(
                trend_learning_rate=0.15,
                variance_learning_rate=0.10,
                variance=1e-4,
            ),
            MeanReversionHypothesis(
                reversion_speed=0.70,
                equilibrium_learning_rate=0.02,
                variance_learning_rate=0.10,
                variance=1e-4,
            ),
            NoiseHypothesis(
                variance=1e-4,
                learning_rate=0.10,
            ),
        ],
        belief_state=BeliefState.uniform(
            number_of_hypotheses=3,
            forgetting_rate=0.01,
        ),
    )


def run_inference(
    observations: FloatArray,
) -> dict[str, FloatArray]:
    updater = create_updater()

    history = observations[:2].copy()

    probability_history: list[FloatArray] = [
        updater.probabilities
    ]

    predictive_means: list[float] = []
    predictive_variances: list[float] = []

    prior_entropies: list[float] = []
    posterior_entropies: list[float] = []
    negative_log_likelihoods: list[float] = []
    information_gains: list[float] = []

    for observation in observations[2:]:
        prediction = updater.predict(history)

        diagnostics = (
            updater.observe_with_diagnostics(
                observation=float(observation),
                prediction=prediction,
            )
        )

        predictive_means.append(
            diagnostics.predictive_mean
        )

        predictive_variances.append(
            diagnostics.predictive_variance
        )

        prior_entropies.append(
            diagnostics.prior_entropy
        )

        posterior_entropies.append(
            diagnostics.posterior_entropy
        )

        negative_log_likelihoods.append(
            diagnostics.negative_log_likelihood
        )

        information_gains.append(
            diagnostics.information_gain
        )

        probability_history.append(
            diagnostics
            .posterior_probabilities
            .copy()
        )

        history = np.append(
            history,
            observation,
        )

    return {
        "probabilities": np.vstack(
            probability_history
        ),
        "predictive_mean": np.asarray(
            predictive_means,
            dtype=np.float64,
        ),
        "predictive_variance": np.asarray(
            predictive_variances,
            dtype=np.float64,
        ),
        "prior_entropy": np.asarray(
            prior_entropies,
            dtype=np.float64,
        ),
        "posterior_entropy": np.asarray(
            posterior_entropies,
            dtype=np.float64,
        ),
        "negative_log_likelihood": np.asarray(
            negative_log_likelihoods,
            dtype=np.float64,
        ),
        "information_gain": np.asarray(
            information_gains,
            dtype=np.float64,
        ),
    }


def generate_momentum_data(
    number_of_observations: int,
    seed: int = 1,
) -> FloatArray:
    rng = np.random.default_rng(seed)

    return rng.normal(
        loc=0.004,
        scale=0.002,
        size=number_of_observations,
    )


def generate_noise_data(
    number_of_observations: int,
    seed: int = 2,
) -> FloatArray:
    rng = np.random.default_rng(seed)

    return rng.normal(
        loc=0.0,
        scale=0.01,
        size=number_of_observations,
    )


def generate_mean_reverting_data(
    number_of_observations: int,
    phi: float = -0.75,
    noise_scale: float = 0.002,
    seed: int = 3,
) -> FloatArray:
    """
    Generate negatively autocorrelated residual returns:

        x_t = phi * x_{t-1} + epsilon_t

    Negative phi creates alternating, mean-reverting behaviour.
    """
    rng = np.random.default_rng(seed)

    values = np.zeros(
        number_of_observations,
        dtype=np.float64,
    )

    values[0] = 0.01

    for index in range(
        1,
        number_of_observations,
    ):
        values[index] = (
            phi * values[index - 1]
            + rng.normal(
                loc=0.0,
                scale=noise_scale,
            )
        )

    return values


def test_identifies_momentum_regime() -> None:
    observations = generate_momentum_data(250)

    results = run_inference(
        observations
    )

    probability_history = results[
        "probabilities"
    ]

    final_probabilities = probability_history[-1]

    momentum_probability = final_probabilities[0]

    assert momentum_probability == np.max(
        final_probabilities
    )


def test_identifies_mean_reverting_regime() -> None:
    observations = generate_mean_reverting_data(250)

    results = run_inference(
        observations
    )

    probability_history = results[
        "probabilities"
    ]

    final_probabilities = probability_history[-1]

    mean_reversion_probability = (
        final_probabilities[1]
    )

    assert mean_reversion_probability == np.max(
        final_probabilities
    )


def test_identifies_noise_regime() -> None:
    observations = generate_noise_data(250)

    results = run_inference(
        observations
    )

    probability_history = results[
        "probabilities"
    ]

    final_probabilities = probability_history[-1]

    noise_probability = final_probabilities[2]

    assert noise_probability == np.max(
        final_probabilities
    )
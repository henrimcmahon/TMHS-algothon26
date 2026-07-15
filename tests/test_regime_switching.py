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


MOMENTUM_INDEX = 0
MEAN_REVERSION_INDEX = 1
NOISE_INDEX = 2


def create_updater(
    forgetting_rate: float = 0.03,
) -> BayesianUpdater:
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
            forgetting_rate=forgetting_rate,
        ),
    )


def generate_momentum_data(
    number_of_observations: int,
    seed: int = 1,
) -> FloatArray:
    """
    Positive-drift residual returns.

    This matches the current MomentumHypothesis, which estimates
    a persistent conditional mean.
    """
    rng = np.random.default_rng(seed)

    return np.asarray(
        rng.normal(
            loc=0.004,
            scale=0.002,
            size=number_of_observations,
        ),
        dtype=np.float64,
    )


def generate_mean_reverting_data(
    number_of_observations: int,
    phi: float = -0.80,
    noise_scale: float = 0.002,
    seed: int = 2,
) -> FloatArray:
    """
    Negatively autocorrelated process:

        x_t = phi * x_{t-1} + epsilon_t

    A negative phi makes positive observations tend to be followed by
    negative observations, and vice versa.
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


def generate_noise_data(
    number_of_observations: int,
    seed: int = 3,
) -> FloatArray:
    """
    Zero-mean independent residual returns.
    """
    rng = np.random.default_rng(seed)

    return np.asarray(
        rng.normal(
            loc=0.0,
            scale=0.01,
            size=number_of_observations,
        ),
        dtype=np.float64,
    )


def generate_switching_process(
    regime_length: int = 150,
) -> tuple[FloatArray, dict[str, tuple[int, int]]]:
    momentum = generate_momentum_data(
        regime_length,
        seed=1,
    )

    mean_reversion = generate_mean_reverting_data(
        regime_length,
        seed=2,
    )

    noise = generate_noise_data(
        regime_length,
        seed=3,
    )

    observations = np.concatenate(
        (
            momentum,
            mean_reversion,
            noise,
        )
    )

    boundaries = {
        "momentum": (
            0,
            regime_length,
        ),
        "mean_reversion": (
            regime_length,
            2 * regime_length,
        ),
        "noise": (
            2 * regime_length,
            3 * regime_length,
        ),
    }

    return observations, boundaries


def run_switching_inference(
    observations: FloatArray,
) -> dict[str, FloatArray]:
    updater = create_updater()

    minimum_history = max(
        hypothesis.minimum_history
        for hypothesis in updater.hypotheses
    )

    history = observations[
        :minimum_history
    ].copy()

    probability_history: list[FloatArray] = []
    entropy_history: list[float] = []
    negative_log_likelihoods: list[float] = []
    information_gains: list[float] = []
    predictive_means: list[float] = []
    predictive_variances: list[float] = []
    observation_indices: list[int] = []

    for index in range(
        minimum_history,
        len(observations),
    ):
        observation = float(
            observations[index]
        )

        prediction = updater.predict(history)

        diagnostics = (
            updater.observe_with_diagnostics(
                observation=observation,
                prediction=prediction,
            )
        )

        probability_history.append(
            diagnostics
            .posterior_probabilities
            .copy()
        )

        entropy_history.append(
            diagnostics.posterior_entropy
        )

        negative_log_likelihoods.append(
            diagnostics.negative_log_likelihood
        )

        information_gains.append(
            diagnostics.information_gain
        )

        predictive_means.append(
            diagnostics.predictive_mean
        )

        predictive_variances.append(
            diagnostics.predictive_variance
        )

        observation_indices.append(index)

        history = np.append(
            history,
            observation,
        )

    return {
        "days": np.asarray(
            observation_indices,
            dtype=np.int64,
        ),
        "probabilities": np.vstack(
            probability_history
        ),
        "entropy": np.asarray(
            entropy_history,
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
        "predictive_mean": np.asarray(
            predictive_means,
            dtype=np.float64,
        ),
        "predictive_variance": np.asarray(
            predictive_variances,
            dtype=np.float64,
        ),
    }


def regime_average_probabilities(
    days: NDArray[np.int64],
    probabilities: FloatArray,
    start: int,
    end: int,
    burn_in: int = 50,
) -> FloatArray:
    """
    Average posterior probabilities after allowing the updater time
    to adapt following a regime switch.
    """
    evaluation_start = start + burn_in

    mask = (
        (days >= evaluation_start)
        & (days < end)
    )

    if not np.any(mask):
        raise ValueError(
            "No observations remain after burn-in"
        )

    return np.mean(
        probabilities[mask],
        axis=0,
    )


def adaptation_delay(
    days: NDArray[np.int64],
    probabilities: FloatArray,
    regime_start: int,
    regime_end: int,
    target_index: int,
    threshold: float = 0.50,
    consecutive_days: int = 5,
) -> int | None:
    """
    Number of observations after a regime switch until the target
    hypothesis exceeds `threshold` for `consecutive_days`.

    Returns None when the model never adapts within the regime.
    """
    regime_mask = (
        (days >= regime_start)
        & (days < regime_end)
    )

    regime_days = days[regime_mask]
    target_probabilities = probabilities[
        regime_mask,
        target_index,
    ]

    if len(regime_days) < consecutive_days:
        return None

    above_threshold = (
        target_probabilities >= threshold
    )

    for index in range(
        0,
        len(above_threshold)
        - consecutive_days
        + 1,
    ):
        window = above_threshold[
            index:
            index + consecutive_days
        ]

        if np.all(window):
            first_dominant_day = int(
                regime_days[index]
            )

            return (
                first_dominant_day
                - regime_start
            )

    return None


def test_switching_process_identifies_each_regime() -> None:
    observations, boundaries = (
        generate_switching_process()
    )

    results = run_switching_inference(
        observations
    )

    days = results["days"]
    probabilities = results[
        "probabilities"
    ]

    momentum_average = (
        regime_average_probabilities(
            days=days,
            probabilities=probabilities,
            start=boundaries["momentum"][0],
            end=boundaries["momentum"][1],
        )
    )

    mean_reversion_average = (
        regime_average_probabilities(
            days=days,
            probabilities=probabilities,
            start=(
                boundaries["mean_reversion"][0]
            ),
            end=(
                boundaries["mean_reversion"][1]
            ),
        )
    )

    noise_average = (
        regime_average_probabilities(
            days=days,
            probabilities=probabilities,
            start=boundaries["noise"][0],
            end=boundaries["noise"][1],
        )
    )

    assert (
        int(np.argmax(momentum_average))
        == MOMENTUM_INDEX
    )

    assert (
        int(np.argmax(mean_reversion_average))
        == MEAN_REVERSION_INDEX
    )

    assert (
        int(np.argmax(noise_average))
        == NOISE_INDEX
    )


def test_switching_process_adapts_within_each_regime() -> None:
    observations, boundaries = (
        generate_switching_process()
    )

    results = run_switching_inference(
        observations
    )

    days = results["days"]
    probabilities = results[
        "probabilities"
    ]

    momentum_delay = adaptation_delay(
        days=days,
        probabilities=probabilities,
        regime_start=(
            boundaries["momentum"][0]
        ),
        regime_end=(
            boundaries["momentum"][1]
        ),
        target_index=MOMENTUM_INDEX,
    )

    mean_reversion_delay = adaptation_delay(
        days=days,
        probabilities=probabilities,
        regime_start=(
            boundaries["mean_reversion"][0]
        ),
        regime_end=(
            boundaries["mean_reversion"][1]
        ),
        target_index=MEAN_REVERSION_INDEX,
    )

    noise_delay = adaptation_delay(
        days=days,
        probabilities=probabilities,
        regime_start=(
            boundaries["noise"][0]
        ),
        regime_end=(
            boundaries["noise"][1]
        ),
        target_index=NOISE_INDEX,
    )

    assert momentum_delay is not None
    assert mean_reversion_delay is not None
    assert noise_delay is not None

    # Initial conservative bound. Tighten this after inspecting results.
    assert momentum_delay < 100
    assert mean_reversion_delay < 100
    assert noise_delay < 100


def test_switches_create_information_or_uncertainty_spikes() -> None:
    observations, boundaries = (
        generate_switching_process()
    )

    results = run_switching_inference(
        observations
    )

    days = results["days"]
    entropy = results["entropy"]
    information_gain = results[
        "information_gain"
    ]

    for regime_name in (
        "mean_reversion",
        "noise",
    ):
        switch_day = boundaries[
            regime_name
        ][0]

        before_mask = (
            (days >= switch_day - 30)
            & (days < switch_day)
        )

        after_mask = (
            (days >= switch_day)
            & (days < switch_day + 30)
        )

        before_entropy = float(
            np.mean(entropy[before_mask])
        )

        after_entropy_peak = float(
            np.max(entropy[after_mask])
        )

        before_information = float(
            np.mean(
                information_gain[
                    before_mask
                ]
            )
        )

        after_information_peak = float(
            np.max(
                information_gain[
                    after_mask
                ]
            )
        )

        assert (
            after_entropy_peak
            > before_entropy
            or after_information_peak
            > before_information
        )
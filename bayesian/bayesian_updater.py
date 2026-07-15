from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from bayesian.belief_state import BeliefState
from bayesian.distributions import (
    GaussianDistribution,
    PredictiveDistribution,
)
from bayesian.hypotheses import PredictiveHypothesis


FloatArray = NDArray[np.float64]


def _logsumexp(
    values: FloatArray,
) -> float:
    maximum = float(np.max(values))

    if np.isneginf(maximum):
        return -np.inf

    return float(
        maximum
        + np.log(
            np.sum(
                np.exp(values - maximum)
            )
        )
    )


def _entropy(
    probabilities: FloatArray,
) -> float:
    positive = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            positive * np.log(positive)
        )
    )


def _kl_divergence(
    posterior: FloatArray,
    prior: FloatArray,
) -> float:
    positive = posterior > 0

    safe_prior = np.maximum(
        prior[positive],
        np.finfo(np.float64).tiny,
    )

    return float(
        np.sum(
            posterior[positive]
            * (
                np.log(posterior[positive])
                - np.log(safe_prior)
            )
        )
    )

@dataclass(frozen=True, slots=True)
class BayesianPrediction:
    """
    Result of one prediction step.

    component_distributions:
        One predictive distribution per hypothesis.

    combined_distribution:
        Gaussian moment-matched approximation to the Bayesian mixture.

    probabilities:
        Hypothesis probabilities used to combine the predictions.
    """

    component_distributions: tuple[
        PredictiveDistribution,
        ...
    ]
    combined_distribution: GaussianDistribution
    probabilities: FloatArray

@dataclass(frozen=True, slots=True)
class BayesianUpdateDiagnostics:
    """
    Diagnostics produced by one Bayesian observation update.

    prior_probabilities:
        Beliefs before incorporating the observation.

    posterior_probabilities:
        Beliefs after incorporating the observation.

    prior_entropy:
        Uncertainty over hypotheses before the update.

    posterior_entropy:
        Uncertainty over hypotheses after the update.

    negative_log_likelihood:
        Negative log probability of the observation under the
        Bayesian model-averaged predictive distribution.

    information_gain:
        KL divergence from posterior beliefs to prior beliefs.

    predictive_mean:
        Mean of the combined predictive distribution.

    predictive_variance:
        Variance of the combined predictive distribution.
    """

    observation: float

    prior_probabilities: FloatArray
    posterior_probabilities: FloatArray

    prior_entropy: float
    posterior_entropy: float

    negative_log_likelihood: float
    information_gain: float

    predictive_mean: float
    predictive_variance: float

    component_log_likelihoods: FloatArray

class BayesianUpdater:
    """
    Orchestrates prediction and Bayesian updating across hypotheses.

    Daily lifecycle:

        prediction = updater.predict(history)
        updater.observe(actual_return, prediction)
    """

    def __init__(
        self,
        hypotheses: Sequence[PredictiveHypothesis],
        belief_state: BeliefState | None = None,
    ) -> None:
        if len(hypotheses) == 0:
            raise ValueError(
                "at least one hypothesis is required"
            )

        names = [
            hypothesis.name
            for hypothesis in hypotheses
        ]

        if len(set(names)) != len(names):
            raise ValueError(
                "hypothesis names must be unique"
            )

        self.hypotheses = tuple(hypotheses)

        self._latest_diagnostics: (
            BayesianUpdateDiagnostics | None
        ) = None

        if belief_state is None:
            self.belief_state = BeliefState.uniform(
                len(self.hypotheses)
            )
        else:
            if (
                len(belief_state.probabilities)
                != len(self.hypotheses)
            ):
                raise ValueError(
                    "belief count must match hypothesis count"
                )

            self.belief_state = belief_state

        self._latest_prediction: (
            BayesianPrediction | None
        ) = None

    @property
    def hypothesis_names(self) -> tuple[str, ...]:
        return tuple(
            hypothesis.name
            for hypothesis in self.hypotheses
        )

    @property
    def probabilities(self) -> FloatArray:
        return self.belief_state.probabilities.copy()
    
    @property
    def latest_diagnostics(
        self,
    ) -> BayesianUpdateDiagnostics | None:
        return self._latest_diagnostics

    def predict(
        self,
        residual_history: FloatArray,
    ) -> BayesianPrediction:
        """
        Ask every hypothesis for a predictive distribution, then combine
        them through Bayesian model averaging.
        """
        history = np.asarray(
            residual_history,
            dtype=np.float64,
        )

        if history.ndim != 1:
            raise ValueError(
                "residual_history must be one-dimensional"
            )

        if np.any(~np.isfinite(history)):
            raise ValueError(
                "residual_history must be finite"
            )

        required_history = max(
            hypothesis.minimum_history
            for hypothesis in self.hypotheses
        )

        if len(history) < required_history:
            raise ValueError(
                f"at least {required_history} observations are required"
            )

        component_distributions = tuple(
            hypothesis.predict_distribution(
                history
            )
            for hypothesis in self.hypotheses
        )

        probabilities = (
            self.belief_state
            .probabilities
            .copy()
        )

        means = np.asarray(
            [
                distribution.mean
                for distribution
                in component_distributions
            ],
            dtype=np.float64,
        )

        variances = np.asarray(
            [
                distribution.variance
                for distribution
                in component_distributions
            ],
            dtype=np.float64,
        )

        combined_mean = float(
            np.dot(
                probabilities,
                means,
            )
        )

        # Law of total variance:
        #
        # Var(X) = E[Var(X | H)] + Var(E[X | H])
        combined_second_moment = float(
            np.dot(
                probabilities,
                variances + means**2,
            )
        )

        combined_variance = (
            combined_second_moment
            - combined_mean**2
        )

        combined_variance = max(
            combined_variance,
            np.finfo(np.float64).eps,
        )

        prediction = BayesianPrediction(
            component_distributions=(
                component_distributions
            ),
            combined_distribution=(
                GaussianDistribution(
                    location=combined_mean,
                    scale_squared=combined_variance,
                )
            ),
            probabilities=probabilities,
        )

        self._latest_prediction = prediction

        return prediction

    def observe(
        self,
        observation: float,
        prediction: BayesianPrediction | None = None,
    ) -> FloatArray:
        """
        Observe the realised residual return and update model beliefs. 

        Diagnostics from the update are stored in 'latest_diagnostics'.
        """
        diagnostics = self.observe_with_diagnostics(
            observation=observation,
            prediction=prediction,
        )

        return (
            diagnostics
            .posterior_probabilities
            .copy()
        )
    
    def observe_with_diagnostics(
        self,
        observation: float,
        prediction: BayesianPrediction | None = None,
    ) -> BayesianUpdateDiagnostics:
        """
        Perform one Bayesian update and return inference diagnostics.
        """
        if not np.isfinite(observation):
            raise ValueError(
                "observation must be finite"
            )

        active_prediction = (
            prediction
            if prediction is not None
            else self._latest_prediction
        )

        if active_prediction is None:
            raise RuntimeError(
                "predict() must be called before observe()"
            )

        if (
            len(active_prediction.component_distributions)
            != len(self.hypotheses)
        ):
            raise ValueError(
                "prediction does not match the updater's hypotheses"
            )

        prior_probabilities = (
            self.belief_state
            .probabilities
            .copy()
        )

        prior_entropy = _entropy(
            prior_probabilities
        )

        component_log_likelihoods = np.asarray(
            [
                distribution.log_probability(
                    observation
                )
                for distribution
                in active_prediction
                .component_distributions
            ],
            dtype=np.float64,
        )

        # Bayesian model-averaged predictive probability:
        #
        # p(x) = sum_k p(H_k) p(x | H_k)
        log_prior = np.log(
            np.maximum(
                prior_probabilities,
                np.finfo(np.float64).tiny,
            )
        )

        log_predictive_probability = _logsumexp(
            log_prior
            + component_log_likelihoods
        )

        negative_log_likelihood = (
            -log_predictive_probability
            if np.isfinite(
                log_predictive_probability
            )
            else np.inf
        )

        self.belief_state.update(
            component_log_likelihoods
        )

        posterior_probabilities = (
            self.belief_state
            .probabilities
            .copy()
        )

        posterior_entropy = _entropy(
            posterior_probabilities
        )

        information_gain = _kl_divergence(
            posterior=posterior_probabilities,
            prior=prior_probabilities,
        )

        for hypothesis in self.hypotheses:
            hypothesis.update(observation)

        diagnostics = BayesianUpdateDiagnostics(
            observation=float(observation),

            prior_probabilities=(
                prior_probabilities
            ),
            posterior_probabilities=(
                posterior_probabilities
            ),

            prior_entropy=prior_entropy,
            posterior_entropy=(
                posterior_entropy
            ),

            negative_log_likelihood=float(negative_log_likelihood),
            information_gain=(
                information_gain
            ),

            predictive_mean=float(
                active_prediction
                .combined_distribution
                .mean
            ),
            predictive_variance=float(
                active_prediction
                .combined_distribution
                .variance
            ),

            component_log_likelihoods=(
                component_log_likelihoods
            ),
        )

        self._latest_prediction = None
        self._latest_diagnostics = diagnostics

        return diagnostics

    def step(
        self,
        residual_history: FloatArray,
        observation: float,
    ) -> tuple[
        BayesianPrediction,
        BayesianUpdateDiagnostics,
    ]:
        prediction = self.predict(
            residual_history
        )

        diagnostics = self.observe_with_diagnostics(
            observation=observation,
            prediction=prediction,
        )

        return prediction, diagnostics

    def belief_table(
        self,
    ) -> dict[str, float]:
        return {
            name: float(probability)
            for name, probability in zip(
                self.hypothesis_names,
                self.belief_state.probabilities,
            )
        }
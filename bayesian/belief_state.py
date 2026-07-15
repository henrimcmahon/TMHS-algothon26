from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class BeliefState:
    """
    Probability distribution over competing hypotheses.

    The probabilities are updated using Bayes' rule in log space.
    """

    probabilities: FloatArray
    forgetting_rate: float = 0.0

    def __post_init__(self) -> None:
        probabilities = np.asarray(
            self.probabilities,
            dtype=np.float64,
        )

        if probabilities.ndim != 1:
            raise ValueError(
                "probabilities must be one-dimensional"
            )

        if len(probabilities) == 0:
            raise ValueError(
                "at least one hypothesis is required"
            )

        if np.any(~np.isfinite(probabilities)):
            raise ValueError(
                "probabilities must be finite"
            )

        if np.any(probabilities < 0):
            raise ValueError(
                "probabilities cannot be negative"
            )

        total = float(np.sum(probabilities))

        if total <= 0:
            raise ValueError(
                "probabilities must contain positive mass"
            )

        if not 0.0 <= self.forgetting_rate <= 1.0:
            raise ValueError(
                "forgetting_rate must be between 0 and 1"
            )

        self.probabilities = probabilities / total

    @classmethod
    def uniform(
        cls,
        number_of_hypotheses: int,
        forgetting_rate: float = 0.0,
    ) -> BeliefState:
        """
        Construct equal prior probabilities.
        """
        if number_of_hypotheses < 1:
            raise ValueError(
                "number_of_hypotheses must be positive"
            )

        return cls(
            probabilities=np.full(
                number_of_hypotheses,
                1.0 / number_of_hypotheses,
                dtype=np.float64,
            ),
            forgetting_rate=forgetting_rate,
        )

    def diffuse(self) -> None:
        """
        Move the current beliefs slightly towards a uniform prior.

        This prevents one hypothesis from becoming permanently dominant
        in a non-stationary market.
        """
        if self.forgetting_rate == 0:
            return

        uniform = np.full_like(
            self.probabilities,
            1.0 / len(self.probabilities),
        )

        self.probabilities = (
            (1.0 - self.forgetting_rate)
            * self.probabilities
            + self.forgetting_rate
            * uniform
        )

    def update(
        self,
        log_likelihoods: FloatArray,
    ) -> None:
        """
        Update posterior probabilities from one log-likelihood per
        hypothesis.
        """
        values = np.asarray(
            log_likelihoods,
            dtype=np.float64,
        )

        if values.shape != self.probabilities.shape:
            raise ValueError(
                "one log-likelihood is required for each hypothesis"
            )

        if np.any(np.isnan(values)):
            raise ValueError(
                "log_likelihoods cannot contain NaN"
            )

        self.diffuse()

        minimum_probability = np.finfo(
            np.float64
        ).tiny

        log_prior = np.log(
            np.maximum(
                self.probabilities,
                minimum_probability,
            )
        )

        log_posterior_unnormalised = (
            log_prior + values
        )

        maximum = float(
            np.max(log_posterior_unnormalised)
        )

        if np.isneginf(maximum):
            # Every model assigned zero probability to the observation.
            # Fall back to the current prior rather than producing NaNs.
            return

        posterior_weights = np.exp(
            log_posterior_unnormalised
            - maximum
        )

        normalising_constant = float(
            np.sum(posterior_weights)
        )

        if (
            not np.isfinite(normalising_constant)
            or normalising_constant <= 0
        ):
            raise FloatingPointError(
                "could not normalise posterior probabilities"
            )

        self.probabilities = (
            posterior_weights
            / normalising_constant
        )

    def probability(self, index: int) -> float:
        return float(self.probabilities[index])

    @property
    def entropy(self) -> float:
        """
        Shannon entropy of the current hypothesis beliefs.
        """
        positive = self.probabilities[
            self.probabilities > 0
        ]

        return float(
            -np.sum(
                positive * np.log(positive)
            )
        )

    def copy(self) -> BeliefState:
        return BeliefState(
            probabilities=self.probabilities.copy(),
            forgetting_rate=self.forgetting_rate,
        )
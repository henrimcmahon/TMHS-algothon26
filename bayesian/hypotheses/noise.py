from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bayesian.distributions import (
    GaussianDistribution,
    PredictiveDistribution,
)
from bayesian.hypotheses.hypothesis import PredictiveHypothesis


@dataclass(slots=True)
class NoiseHypothesis(PredictiveHypothesis):
    """
    Assumes residual returns have zero mean and time-varying variance.

    Variance is updated using an exponentially weighted moving average.
    """

    variance: float = 1e-4
    learning_rate: float = 0.05
    variance_floor: float = 1e-8
    label: str = "noise"

    def __post_init__(self) -> None:
        if self.variance <= 0:
            raise ValueError("variance must be positive")

        if not 0 < self.learning_rate <= 1:
            raise ValueError(
                "learning_rate must be in (0, 1]"
            )

        if self.variance_floor <= 0:
            raise ValueError(
                "variance_floor must be positive"
            )

    @property
    def name(self) -> str:
        return self.label

    def predict_distribution(
        self,
        residual_history: NDArray[np.float64],
    ) -> PredictiveDistribution:
        if len(residual_history) < self.minimum_history:
            raise ValueError(
                f"{self.name} requires at least "
                f"{self.minimum_history} observations"
            )

        return GaussianDistribution(
            location=0.0,
            scale_squared=max(
                self.variance,
                self.variance_floor,
            ),
        )

    def update(
        self,
        observation: float,
    ) -> None:
        squared_error = observation**2

        self.variance = (
            (1.0 - self.learning_rate)
            * self.variance
            + self.learning_rate
            * squared_error
        )

        self.variance = max(
            self.variance,
            self.variance_floor,
        )
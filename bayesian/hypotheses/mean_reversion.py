from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bayesian.distributions import (
    GaussianDistribution,
    PredictiveDistribution,
)
from bayesian.hypotheses.hypothesis import PredictiveHypothesis


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class MeanReversionHypothesis(PredictiveHypothesis):
    """
    Predicts movement back toward a dynamically estimated equilibrium.

    Predicted return:

        reversion_speed * (equilibrium - latest_residual)

    This is a discrete approximation to an Ornstein-Uhlenbeck process.
    """

    equilibrium: float = 0.0
    variance: float = 1e-4
    reversion_speed: float = 0.25
    equilibrium_learning_rate: float = 0.02
    variance_learning_rate: float = 0.05
    variance_floor: float = 1e-8
    label: str = "mean_reversion"

    def __post_init__(self) -> None:
        if not np.isfinite(self.equilibrium):
            raise ValueError(
                "equilibrium must be finite"
            )

        if not np.isfinite(self.variance) or self.variance <= 0:
            raise ValueError(
                "variance must be finite and positive"
            )

        if not 0 < self.reversion_speed <= 1:
            raise ValueError(
                "reversion_speed must be in (0, 1]"
            )

        if not 0 < self.equilibrium_learning_rate <= 1:
            raise ValueError(
                "equilibrium_learning_rate must be in (0, 1]"
            )

        if not 0 < self.variance_learning_rate <= 1:
            raise ValueError(
                "variance_learning_rate must be in (0, 1]"
            )

        if self.variance_floor <= 0:
            raise ValueError(
                "variance_floor must be positive"
            )

    @property
    def name(self) -> str:
        return self.label

    @property
    def minimum_history(self) -> int:
        return 2

    def predict_distribution(
        self,
        residual_history: FloatArray,
    ) -> PredictiveDistribution:
        history = np.asarray(
            residual_history,
            dtype=np.float64,
        )

        if history.ndim != 1:
            raise ValueError(
                "residual_history must be one-dimensional"
            )

        if len(history) < self.minimum_history:
            raise ValueError(
                f"{self.name} requires at least "
                f"{self.minimum_history} observations"
            )

        latest_residual = float(history[-1])

        predicted_mean = (
            self.reversion_speed
            * (
                self.equilibrium
                - latest_residual
            )
        )

        return GaussianDistribution(
            location=predicted_mean,
            scale_squared=max(
                self.variance,
                self.variance_floor,
            ),
        )

    def update(
        self,
        observation: float,
    ) -> None:
        if not np.isfinite(observation):
            raise ValueError(
                "observation must be finite"
            )

        previous_equilibrium = self.equilibrium

        self.equilibrium = (
            (
                1.0
                - self.equilibrium_learning_rate
            )
            * self.equilibrium
            + self.equilibrium_learning_rate
            * observation
        )

        prediction_error = (
            observation - previous_equilibrium
        )

        self.variance = (
            (
                1.0
                - self.variance_learning_rate
            )
            * self.variance
            + self.variance_learning_rate
            * prediction_error**2
        )

        self.variance = max(
            self.variance,
            self.variance_floor,
        )
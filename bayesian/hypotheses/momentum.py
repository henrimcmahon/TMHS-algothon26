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
class MomentumHypothesis(PredictiveHypothesis):
    """
    Predicts continuation of the recent residual-return trend.

    Internal state is updated using exponentially weighted estimates of
    the trend and prediction-error variance.
    """

    trend: float = 0.0
    variance: float = 1e-4
    trend_learning_rate: float = 0.10
    variance_learning_rate: float = 0.05
    variance_floor: float = 1e-8
    label: str = "momentum"

    def __post_init__(self) -> None:
        if not np.isfinite(self.trend):
            raise ValueError("trend must be finite")

        if not np.isfinite(self.variance) or self.variance <= 0:
            raise ValueError(
                "variance must be finite and positive"
            )

        if not 0 < self.trend_learning_rate <= 1:
            raise ValueError(
                "trend_learning_rate must be in (0, 1]"
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

        return GaussianDistribution(
            location=self.trend,
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

        prediction_error = (
            observation - self.trend
        )

        self.variance = (
            (1.0 - self.variance_learning_rate)
            * self.variance
            + self.variance_learning_rate
            * prediction_error**2
        )

        self.trend = (
            (1.0 - self.trend_learning_rate)
            * self.trend
            + self.trend_learning_rate
            * observation
        )

        self.variance = max(
            self.variance,
            self.variance_floor,
        )
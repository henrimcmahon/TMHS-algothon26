from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesian.distributions.predictive_distribution import (
    PredictiveDistribution,
)


@dataclass(frozen=True, slots=True)
class GaussianDistribution(PredictiveDistribution):
    location: float
    scale_squared: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.location):
            raise ValueError("location must be finite")

        if (
            not np.isfinite(self.scale_squared)
            or self.scale_squared <= 0
        ):
            raise ValueError(
                "scale_squared must be finite and positive"
            )

    @property
    def mean(self) -> float:
        return self.location

    @property
    def variance(self) -> float:
        return self.scale_squared

    def log_probability(
        self,
        observation: float,
    ) -> float:
        error = observation - self.location

        return float(
            -0.5
            * (
                np.log(
                    2.0
                    * np.pi
                    * self.scale_squared
                )
                + error**2
                / self.scale_squared
            )
        )

    def sample(self) -> float:
        return float(
            np.random.normal(
                loc=self.location,
                scale=np.sqrt(
                    self.scale_squared
                ),
            )
        )
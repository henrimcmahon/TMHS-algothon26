from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from bayesian.distributions import PredictiveDistribution


class PredictiveHypothesis(ABC):
    """
    Base class for a stateful model that predicts the next residual return.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def minimum_history(self) -> int:
        return 2

    @abstractmethod
    def predict_distribution(
        self,
        residual_history: NDArray[np.float64],
    ) -> PredictiveDistribution:
        """
        Predict the next residual return before observing it.
        """
        ...

    @abstractmethod
    def update(
        self,
        observation: float,
    ) -> None:
        """
        Update the hypothesis's internal parameters after observing a return.
        """
        ...
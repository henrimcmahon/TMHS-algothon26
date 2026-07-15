from __future__ import annotations

from abc import ABC, abstractmethod


class PredictiveDistribution(ABC):
    @property
    @abstractmethod
    def mean(self) -> float:
        ...

    @property
    @abstractmethod
    def variance(self) -> float:
        ...

    @abstractmethod
    def log_probability(
        self,
        observation: float,
    ) -> float:
        ...

    @abstractmethod
    def sample(self) -> float:
        ...
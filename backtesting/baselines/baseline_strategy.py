from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class BaselineStrategy(ABC):
    """
    Common interface for simple benchmark strategies.

    A strategy receives all prices available through the current day and
    returns the desired end-of-day position vector.
    """

    number_of_instruments: int = 51

    def __init__(self) -> None:
        self.current_positions = np.zeros(
            self.number_of_instruments,
            dtype=np.int64,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        """
        Return the desired total share positions for the current day.
        """

    def reset(self) -> None:
        self.current_positions = np.zeros(
            self.number_of_instruments,
            dtype=np.int64,
        )

    @staticmethod
    def position_limits() -> FloatArray:
        limits = np.full(
            51,
            10_000.0,
            dtype=np.float64,
        )
        limits[0] = 100_000.0
        return limits

    @staticmethod
    def commission_rates() -> FloatArray:
        rates = np.full(
            51,
            0.0001,
            dtype=np.float64,
        )
        rates[0] = 0.00002
        return rates

    def maximum_shares(
        self,
        prices: FloatArray,
    ) -> IntArray:
        prices = np.asarray(
            prices,
            dtype=np.float64,
        )

        if prices.shape != (51,):
            raise ValueError(
                "prices must contain exactly 51 instruments"
            )

        if np.any(~np.isfinite(prices)):
            raise ValueError(
                "prices must be finite"
            )

        if np.any(prices <= 0):
            raise ValueError(
                "prices must be strictly positive"
            )

        return np.floor(
            self.position_limits() / prices
        ).astype(np.int64)

    def clip_positions(
        self,
        positions: IntArray,
        prices: FloatArray,
    ) -> IntArray:
        maximum = self.maximum_shares(
            prices
        )

        return np.clip(
            positions,
            -maximum,
            maximum,
        ).astype(np.int64)
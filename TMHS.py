from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Literal

# ============================================================================
# Inlined from backtesting.baselines.baseline_strategy
# ============================================================================
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
        self.current_positions = np.zeros(self.number_of_instruments, dtype=np.int64)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def get_positions(self, price_history: FloatArray) -> IntArray:
        """
        Return the desired total share positions for the current day.
        """

    def reset(self) -> None:
        self.current_positions = np.zeros(self.number_of_instruments, dtype=np.int64)

    @staticmethod
    def position_limits() -> FloatArray:
        limits = np.full(51, 10000.0, dtype=np.float64)
        limits[0] = 100000.0
        return limits

    @staticmethod
    def commission_rates() -> FloatArray:
        rates = np.full(51, 0.0001, dtype=np.float64)
        rates[0] = 2e-05
        return rates

    def maximum_shares(self, prices: FloatArray) -> IntArray:
        prices = np.asarray(prices, dtype=np.float64)
        if prices.shape != (51,):
            raise ValueError('prices must contain exactly 51 instruments')
        if np.any(~np.isfinite(prices)):
            raise ValueError('prices must be finite')
        if np.any(prices <= 0):
            raise ValueError('prices must be strictly positive')
        return np.floor(self.position_limits() / prices).astype(np.int64)

    def clip_positions(self, positions: IntArray, prices: FloatArray) -> IntArray:
        maximum = self.maximum_shares(prices)
        return np.clip(positions, -maximum, maximum).astype(np.int64)

# ============================================================================
# Inlined from backtesting.baselines.previous_return
# ============================================================================
class PreviousReturnStrategy(BaselineStrategy):

    def __init__(self, mode: Literal['momentum', 'reversal'], exposure_fraction: float=1.0) -> None:
        self.current_positions = np.zeros(self.number_of_instruments, dtype=np.int64)
        if mode not in {'momentum', 'reversal'}:
            raise ValueError("mode must be 'momentum' or 'reversal'")
        if not 0.0 <= exposure_fraction <= 1.0:
            raise ValueError('exposure_fraction must be between 0 and 1')
        self.mode = mode
        self.exposure_fraction = exposure_fraction

    @property
    def name(self) -> str:
        return 'Previous-day momentum' if self.mode == 'momentum' else 'Previous-day reversal'

    def get_positions(self, price_history: FloatArray) -> IntArray:
        prices = np.asarray(price_history, dtype=np.float64)
        if prices.shape[1] < 2:
            return np.zeros(51, dtype=np.int64)
        latest_prices = prices[:, -1]
        returns = np.log(prices[:, -1] / prices[:, -2])
        directions = np.sign(returns)
        if self.mode == 'reversal':
            directions = -directions
        maximum = self.maximum_shares(latest_prices)
        desired = np.trunc(directions * maximum * self.exposure_fraction).astype(np.int64)
        self.current_positions = self.clip_positions(desired, latest_prices)
        return self.current_positions.copy()

# =============================================================================
# Competition entry point
# =============================================================================

_STRATEGY = PreviousReturnStrategy(
    mode='momentum',
    exposure_fraction=1.0,
)


def getMyPosition(
    prices: np.ndarray,
) -> np.ndarray:
    """
    Return the desired integer share positions for all instruments.
    """
    price_history = np.asarray(
        prices,
        dtype=np.float64,
    )

    if price_history.ndim != 2:
        raise ValueError(
            "prices must have shape "
            "(number_of_instruments, number_of_days)"
        )

    positions = _STRATEGY.get_positions(
        price_history
    )

    positions = np.asarray(
        positions,
        dtype=np.int64,
    )

    expected_shape = (
        price_history.shape[0],
    )

    if positions.shape != expected_shape:
        raise ValueError(
            f"strategy returned shape {positions.shape}; "
            f"expected {expected_shape}"
        )

    return positions

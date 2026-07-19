from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class MarketConstraints:
    position_limits: FloatArray
    commission_rates: FloatArray

    def __post_init__(self) -> None:
        if self.position_limits.ndim != 1:
            raise ValueError("position_limits must be one-dimensional")

        if self.commission_rates.ndim != 1:
            raise ValueError("commission_rates must be one-dimensional")

        if self.position_limits.shape != self.commission_rates.shape:
            raise ValueError(
                "position_limits and commission_rates must have matching shapes"
            )

        if np.any(~np.isfinite(self.position_limits)):
            raise ValueError("position limits must be finite")

        if np.any(~np.isfinite(self.commission_rates)):
            raise ValueError("commission rates must be finite")

        if np.any(self.position_limits < 0):
            raise ValueError("position limits cannot be negative")

        if np.any(self.commission_rates < 0):
            raise ValueError("commission rates cannot be negative")

    def _validate_market_vector(
        self,
        values: NDArray[np.generic],
        *,
        name: str,
    ) -> None:
        if values.shape != self.position_limits.shape:
            raise ValueError(
                f"{name} must have shape {self.position_limits.shape}"
            )

    def position_limits_in_shares(
        self,
        prices: FloatArray,
    ) -> IntArray:
        """Convert fixed dollar limits into today's integer share limits."""
        prices = np.asarray(prices, dtype=np.float64)
        self._validate_market_vector(prices, name="prices")

        if np.any(~np.isfinite(prices)):
            raise ValueError("prices contain non-finite values")

        if np.any(prices <= 0):
            raise ValueError("prices must be strictly positive")

        return np.floor(self.position_limits / prices).astype(np.int64)

    def clip_positions(
        self,
        positions: IntArray,
        prices: FloatArray,
    ) -> IntArray:
        """Clip desired positions to today's long and short share limits."""
        positions = np.asarray(positions, dtype=np.int64)
        self._validate_market_vector(positions, name="positions")
        share_limits = self.position_limits_in_shares(prices)
        return np.clip(positions, -share_limits, share_limits).astype(np.int64)

    def trade_costs(
        self,
        trades: IntArray,
        prices: FloatArray,
    ) -> tuple[FloatArray, float, float]:
        """Return notional by ticker, total turnover, and commission."""
        trades = np.asarray(trades, dtype=np.int64)
        prices = np.asarray(prices, dtype=np.float64)
        self._validate_market_vector(trades, name="trades")
        self._validate_market_vector(prices, name="prices")

        if np.any(~np.isfinite(prices)):
            raise ValueError("prices contain non-finite values")

        if np.any(prices <= 0):
            raise ValueError("prices must be strictly positive")

        traded_notional = prices * np.abs(trades)
        turnover = float(np.sum(traded_notional))
        commission = float(np.dot(traded_notional, self.commission_rates))
        return traded_notional, turnover, commission

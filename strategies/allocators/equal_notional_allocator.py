from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.market_constraints import MarketConstraints

from strategies.allocators.position_allocator import (
    FloatArray,
    IntArray,
    PositionAllocator,
)


@dataclass
class EqualNotionalAllocator(PositionAllocator):
    """Allocate each signal as a fraction of its ticker's dollar limit."""

    def allocate(
        self,
        signal: FloatArray,
        latest_prices: FloatArray,
        current_positions: IntArray,
        constraints: MarketConstraints,
    ) -> IntArray:
        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional")

        if latest_prices.shape != signal.shape:
            raise ValueError(
                "latest_prices and signal must have matching shapes"
            )

        if current_positions.shape != signal.shape:
            raise ValueError(
                "current_positions and signal must have matching shapes"
            )

        if constraints.position_limits.shape != signal.shape:
            raise ValueError(
                "market constraints and signal must have matching shapes"
            )

        if np.any(~np.isfinite(signal)):
            raise ValueError("signal contains non-finite values")

        if np.any(latest_prices <= 0):
            raise ValueError("latest prices must be positive")

        share_limits = constraints.position_limits_in_shares(latest_prices)
        bounded_signal = np.clip(signal, -1.0, 1.0)
        return np.trunc(bounded_signal * share_limits).astype(np.int64)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategies.allocators.position_allocator import (
    FloatArray,
    IntArray,
    PositionAllocator,
)


@dataclass
class EqualNotionalAllocator(PositionAllocator):
    total_notional: float = 100_000.0

    def __post_init__(self) -> None:
        if self.total_notional <= 0:
            raise ValueError(
                "total_notional must be positive"
            )

    def allocate(
        self,
        signal: FloatArray,
        latest_prices: FloatArray,
        current_positions: IntArray,
    ) -> IntArray:
        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional")

        if latest_prices.shape != signal.shape:
            raise ValueError(
                "latest_prices and signal must have matching shapes"
            )

        if np.any(latest_prices <= 0):
            raise ValueError("latest prices must be positive")

        active = signal > 0

        if not np.any(active):
            return np.zeros(
                signal.shape[0],
                dtype=np.int64,
            )

        notional_per_ticker = (
            self.total_notional
            / np.count_nonzero(active)
        )

        target_positions = np.zeros(
            signal.shape[0],
            dtype=np.int64,
        )

        target_positions[active] = np.floor(
            notional_per_ticker
            / latest_prices[active]
        ).astype(np.int64)

        return target_positions

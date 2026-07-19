from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Ticker:
    index: int
    symbol: str
    position_limit: float
    commission_rate: float

    def get_prices(self, prices: FloatArray) -> FloatArray:
        """Return this ticker's price series from a 2D price matrix."""
        if prices.ndim != 2:
            raise ValueError("prices must have shape (n_tickers, n_days)")

        if not 0 <= self.index < prices.shape[0]:
            raise IndexError(
                f"Ticker index {self.index} is outside price matrix "
                f"with {prices.shape[0]} rows"
            )

        return prices[self.index]

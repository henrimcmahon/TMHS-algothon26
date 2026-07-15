from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


PriceArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Ticker:
    """
    Represents one instrument in the Algothon dataset.

    Attributes:
        symbol:
            Instrument symbol, such as "ALGO" or "AENO".

        index:
            Row index used by prcSoFar.

            The Algothon evaluator passes prices in the shape:

                (number_of_instruments, number_of_days)

            Therefore, this index identifies the instrument's row.

        prices:
            Full historical price series loaded from prices.txt.
    """

    symbol: str
    index: int
    prices: PriceArray

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Ticker symbol cannot be empty")

        if self.index < 0:
            raise ValueError("Ticker index cannot be negative")

        if self.prices.ndim != 1:
            raise ValueError("Ticker prices must be a one-dimensional array")

        if len(self.prices) == 0:
            raise ValueError("Ticker must contain at least one price")

        if not np.all(np.isfinite(self.prices)):
            raise ValueError(
                f"Ticker {self.symbol} contains non-finite prices"
            )

        if np.any(self.prices <= 0):
            raise ValueError(
                f"Ticker {self.symbol} contains non-positive prices"
            )

    @property
    def latest_price(self) -> float:
        """Return the most recent available price."""
        return float(self.prices[-1])

    @property
    def number_of_observations(self) -> int:
        """Return the number of recorded price observations."""
        return len(self.prices)

    def prices_so_far(self, day: int | None = None) -> PriceArray:
        """
        Return prices up to a specified day.

        Args:
            day:
                Number of observations to include. If omitted, the complete
                price history is returned.

        Returns:
            A copy of the requested price history.
        """
        if day is None:
            return self.prices.copy()

        if day < 0:
            raise ValueError("day cannot be negative")

        return self.prices[:day].copy()

    def simple_returns(self) -> PriceArray:
        """Calculate simple percentage returns."""
        if len(self.prices) < 2:
            return np.array([], dtype=np.float64)

        return np.diff(self.prices) / self.prices[:-1]

    def log_returns(self) -> PriceArray:
        """Calculate continuously compounded returns."""
        if len(self.prices) < 2:
            return np.array([], dtype=np.float64)

        return np.diff(np.log(self.prices))

    def __str__(self) -> str:
        return self.symbol
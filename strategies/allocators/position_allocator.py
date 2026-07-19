from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from models.market_constraints import MarketConstraints


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class PositionAllocator(ABC):
    @abstractmethod
    def allocate(
        self,
        signal: FloatArray,
        latest_prices: FloatArray,
        current_positions: IntArray,
        constraints: MarketConstraints,
    ) -> IntArray:
        """Convert a continuous signal into integer target positions."""
        raise NotImplementedError

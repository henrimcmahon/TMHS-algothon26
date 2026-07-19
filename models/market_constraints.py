from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


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

        if np.any(self.position_limits < 0):
            raise ValueError("position limits cannot be negative")

        if np.any(self.commission_rates < 0):
            raise ValueError("commission rates cannot be negative")

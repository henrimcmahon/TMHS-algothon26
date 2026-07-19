from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class SignalModel(ABC):
    @abstractmethod
    def compute(
        self,
        price_history: FloatArray,
    ) -> FloatArray:
        """Return one continuous signal value per ticker."""
        raise NotImplementedError

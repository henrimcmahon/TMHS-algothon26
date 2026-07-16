from __future__ import annotations

import numpy as np

from .baseline_strategy import (
    BaselineStrategy,
    FloatArray,
    IntArray,
)


class NoPositionStrategy(BaselineStrategy):
    @property
    def name(self) -> str:
        return "No position"

    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        self.current_positions = np.zeros(
            51,
            dtype=np.int64,
        )

        return self.current_positions.copy()
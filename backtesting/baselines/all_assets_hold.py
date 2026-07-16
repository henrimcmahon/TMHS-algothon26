from __future__ import annotations

from typing import Literal

import numpy as np

from .baseline_strategy import (
    BaselineStrategy,
    FloatArray,
    IntArray,
)


class AllAssetsHoldStrategy(BaselineStrategy):
    def __init__(
        self,
        direction: Literal["long", "short"],
    ) -> None:
        super().__init__()

        if direction not in {
            "long",
            "short",
        }:
            raise ValueError(
                "direction must be 'long' or 'short'"
            )

        self.direction = direction

    @property
    def name(self) -> str:
        return (
            "Max long all assets"
            if self.direction == "long"
            else "Max short all assets"
        )

    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        latest_prices = np.asarray(
            price_history[:, -1],
            dtype=np.float64,
        )

        maximum = self.maximum_shares(
            latest_prices
        )

        desired = (
            maximum
            if self.direction == "long"
            else -maximum
        )

        # Each day this naturally reapplies the moving dollar ceilings.
        self.current_positions = desired.astype(
            np.int64
        )

        return self.current_positions.copy()
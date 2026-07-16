from __future__ import annotations

from typing import Literal

import numpy as np

from .baseline_strategy import (
    BaselineStrategy,
    FloatArray,
    IntArray,
)


class AlgoHoldStrategy(BaselineStrategy):
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
            "ALGO long-and-hold"
            if self.direction == "long"
            else "ALGO short-and-hold"
        )

    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        prices = np.asarray(
            price_history,
            dtype=np.float64,
        )

        latest_prices = prices[:, -1]

        maximum = self.maximum_shares(
            latest_prices
        )

        if self.current_positions[0] == 0:
            desired_algo = (
                maximum[0]
                if self.direction == "long"
                else -maximum[0]
            )
        elif self.direction == "long":
            # Hold unless today's price forces a reduction.
            desired_algo = min(
                int(self.current_positions[0]),
                int(maximum[0]),
            )
        else:
            desired_algo = max(
                int(self.current_positions[0]),
                -int(maximum[0]),
            )

        positions = np.zeros(
            51,
            dtype=np.int64,
        )

        positions[0] = desired_algo

        self.current_positions = positions

        return positions.copy()
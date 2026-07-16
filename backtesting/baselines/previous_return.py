from __future__ import annotations

from typing import Literal

import numpy as np

from .baseline_strategy import (
    BaselineStrategy,
    FloatArray,
    IntArray,
)


class PreviousReturnStrategy(BaselineStrategy):
    def __init__(
        self,
        mode: Literal["momentum", "reversal"],
        exposure_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        if mode not in {
            "momentum",
            "reversal",
        }:
            raise ValueError(
                "mode must be 'momentum' or 'reversal'"
            )

        if not 0.0 <= exposure_fraction <= 1.0:
            raise ValueError(
                "exposure_fraction must be between 0 and 1"
            )

        self.mode = mode
        self.exposure_fraction = (
            exposure_fraction
        )

    @property
    def name(self) -> str:
        return (
            "Previous-day momentum"
            if self.mode == "momentum"
            else "Previous-day reversal"
        )

    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        prices = np.asarray(
            price_history,
            dtype=np.float64,
        )

        if prices.shape[1] < 2:
            return np.zeros(
                51,
                dtype=np.int64,
            )

        latest_prices = prices[:, -1]

        returns = np.log(
            prices[:, -1]
            / prices[:, -2]
        )

        directions = np.sign(
            returns
        )

        if self.mode == "reversal":
            directions = -directions

        maximum = self.maximum_shares(
            latest_prices
        )

        desired = np.trunc(
            directions
            * maximum
            * self.exposure_fraction
        ).astype(np.int64)

        self.current_positions = (
            self.clip_positions(
                desired,
                latest_prices,
            )
        )

        return self.current_positions.copy()
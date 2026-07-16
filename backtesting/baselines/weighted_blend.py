from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .baseline_strategy import (
    BaselineStrategy,
    FloatArray,
    IntArray,
)


class WeightedBlendStrategy(
    BaselineStrategy
):
    """
    Combine the desired dollar positions of several strategies.

    Weights control how much of each component's signal contributes
    before the final portfolio is clipped to competition limits.
    """

    def __init__(
        self,
        strategies: Sequence[
            BaselineStrategy
        ],
        weights: Sequence[float],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()

        if not strategies:
            raise ValueError(
                "at least one component strategy is required"
            )

        if len(strategies) != len(weights):
            raise ValueError(
                "strategies and weights must have equal length"
            )

        weight_array = np.asarray(
            weights,
            dtype=np.float64,
        )

        if np.any(
            ~np.isfinite(weight_array)
        ):
            raise ValueError(
                "weights must be finite"
            )

        if np.any(weight_array < 0.0):
            raise ValueError(
                "weights cannot be negative"
            )

        if float(
            np.sum(weight_array)
        ) <= 0.0:
            raise ValueError(
                "at least one weight must be positive"
            )

        self.strategies = list(
            strategies
        )

        self.weights = (
            weight_array
            / np.sum(weight_array)
        )

        self._name = (
            name
            if name is not None
            else "Weighted strategy blend"
        )

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        super().reset()

        for strategy in self.strategies:
            strategy.reset()

    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        prices = np.asarray(
            price_history,
            dtype=np.float64,
        )

        latest_prices = prices[
            :,
            -1,
        ]

        combined_dollars = np.zeros(
            self.number_of_instruments,
            dtype=np.float64,
        )

        for strategy, weight in zip(
            self.strategies,
            self.weights,
            strict=True,
        ):
            component_positions = (
                strategy.get_positions(
                    prices
                )
            )

            component_dollars = (
                component_positions
                * latest_prices
            )

            combined_dollars += (
                float(weight)
                * component_dollars
            )

        desired_positions = np.trunc(
            combined_dollars
            / latest_prices
        ).astype(
            np.int64
        )

        self.current_positions = (
            self.clip_positions(
                desired_positions,
                latest_prices,
            )
        )

        return (
            self.current_positions.copy()
        )
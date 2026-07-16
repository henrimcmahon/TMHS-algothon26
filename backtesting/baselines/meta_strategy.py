from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np

from .baseline_strategy import (
    BaselineStrategy,
    FloatArray,
    IntArray,
)


class MetaStrategy(BaselineStrategy):
    """
    Base class for strategies that allocate between other strategies.

    Each component strategy is run virtually. Its historical net PnL is
    calculated without look-ahead, using the position selected on day t
    against the price move from day t to day t + 1.

    The meta-strategy then combines today's component positions using
    weights calculated entirely from PnL observed before today's trade.
    """

    def __init__(
        self,
        strategies: Sequence[BaselineStrategy],
        *,
        minimum_observations: int = 20,
        annualisation_days: int = 250,
    ) -> None:
        super().__init__()

        if len(strategies) == 0:
            raise ValueError(
                "at least one component strategy is required"
            )

        if minimum_observations < 1:
            raise ValueError(
                "minimum_observations must be positive"
            )

        if annualisation_days <= 0:
            raise ValueError(
                "annualisation_days must be positive"
            )

        names = [
            strategy.name
            for strategy in strategies
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "component strategy names must be unique"
            )

        self.strategies = list(
            strategies
        )

        self.minimum_observations = (
            minimum_observations
        )

        self.annualisation_days = (
            annualisation_days
        )

        number_of_strategies = len(
            self.strategies
        )

        self._component_positions = np.zeros(
            (
                number_of_strategies,
                self.number_of_instruments,
            ),
            dtype=np.int64,
        )

        self._pending_commissions = np.zeros(
            number_of_strategies,
            dtype=np.float64,
        )

        self._pnl_history: list[
            list[float]
        ] = [
            []
            for _ in self.strategies
        ]

        self._last_number_of_days = 0

    def reset(self) -> None:
        super().reset()

        for strategy in self.strategies:
            strategy.reset()

        self._component_positions.fill(
            0
        )

        self._pending_commissions.fill(
            0.0
        )

        self._pnl_history = [
            []
            for _ in self.strategies
        ]

        self._last_number_of_days = 0

    @abstractmethod
    def _calculate_weights(
        self,
    ) -> FloatArray:
        """
        Return one portfolio weight per component strategy.

        This is called after all virtual PnL observations available before
        today's trade have been recorded.
        """

    def _validate_price_history(
        self,
        price_history: FloatArray,
    ) -> FloatArray:
        prices = np.asarray(
            price_history,
            dtype=np.float64,
        )

        if prices.ndim != 2:
            raise ValueError(
                "price_history must be two-dimensional"
            )

        if prices.shape[0] != (
            self.number_of_instruments
        ):
            raise ValueError(
                "price_history must contain exactly "
                "51 instruments"
            )

        if prices.shape[1] < 1:
            raise ValueError(
                "price_history must contain at least one day"
            )

        if np.any(~np.isfinite(prices)):
            raise ValueError(
                "price_history must be finite"
            )

        if np.any(prices <= 0):
            raise ValueError(
                "prices must be strictly positive"
            )

        return prices

    def _record_previous_day_pnl(
        self,
        prices: FloatArray,
    ) -> None:
        number_of_days = prices.shape[1]

        if self._last_number_of_days == 0:
            return

        if number_of_days == self._last_number_of_days:
            return

        if number_of_days != (
            self._last_number_of_days + 1
        ):
            raise ValueError(
                "price history must advance exactly "
                "one day at a time"
            )

        previous_prices = prices[
            :,
            -2,
        ]

        latest_prices = prices[
            :,
            -1,
        ]

        price_change = (
            latest_prices
            - previous_prices
        )

        for index in range(
            len(self.strategies)
        ):
            gross_pnl = float(
                np.dot(
                    self._component_positions[
                        index
                    ].astype(
                        np.float64
                    ),
                    price_change,
                )
            )

            net_pnl = (
                gross_pnl
                - self._pending_commissions[
                    index
                ]
            )

            self._pnl_history[
                index
            ].append(
                net_pnl
            )

    def _generate_component_positions(
        self,
        prices: FloatArray,
    ) -> IntArray:
        latest_prices = prices[
            :,
            -1,
        ]

        commission_rates = (
            self.commission_rates()
        )

        new_positions = np.zeros_like(
            self._component_positions
        )

        new_commissions = np.zeros(
            len(self.strategies),
            dtype=np.float64,
        )

        for index, strategy in enumerate(
            self.strategies
        ):
            desired = np.asarray(
                strategy.get_positions(
                    prices
                ),
                dtype=np.int64,
            )

            if desired.shape != (
                self.number_of_instruments,
            ):
                raise ValueError(
                    f"{strategy.name} must return "
                    "exactly 51 positions"
                )

            desired = self.clip_positions(
                desired,
                latest_prices,
            )

            shares_traded = (
                desired
                - self._component_positions[
                    index
                ]
            )

            dollar_volume = (
                np.abs(shares_traded)
                * latest_prices
            )

            new_commissions[index] = float(
                np.sum(
                    dollar_volume
                    * commission_rates
                )
            )

            new_positions[
                index
            ] = desired

        self._component_positions = (
            new_positions
        )

        self._pending_commissions = (
            new_commissions
        )

        return new_positions

    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        prices = self._validate_price_history(
            price_history
        )

        number_of_days = prices.shape[1]

        if (
            self._last_number_of_days > 0
            and number_of_days
            < self._last_number_of_days
        ):
            raise ValueError(
                "price history cannot move backwards"
            )

        if (
            number_of_days
            == self._last_number_of_days
        ):
            return self.current_positions.copy()

        self._record_previous_day_pnl(
            prices
        )

        component_positions = (
            self._generate_component_positions(
                prices
            )
        )

        weights = np.asarray(
            self._calculate_weights(),
            dtype=np.float64,
        )

        if weights.shape != (
            len(self.strategies),
        ):
            raise ValueError(
                "meta-strategy must return one weight "
                "per component strategy"
            )

        if np.any(~np.isfinite(weights)):
            raise ValueError(
                "strategy weights must be finite"
            )

        if np.any(weights < 0):
            raise ValueError(
                "strategy weights cannot be negative"
            )

        weight_sum = float(
            np.sum(weights)
        )

        if weight_sum > 1e-12:
            weights = (
                weights
                / weight_sum
            )
        else:
            weights = np.zeros_like(
                weights
            )

        combined_positions = np.rint(
            weights
            @ component_positions.astype(
                np.float64
            )
        ).astype(np.int64)

        self.current_positions = (
            self.clip_positions(
                combined_positions,
                prices[:, -1],
            )
        )

        self._last_number_of_days = (
            number_of_days
        )

        return self.current_positions.copy()

    def _metric_arrays(
        self,
        *,
        window: int | None = None,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """
        Return mean, volatility, Sharpe and score for every strategy.
        """
        means = np.full(
            len(self.strategies),
            np.nan,
            dtype=np.float64,
        )

        volatilities = np.full_like(
            means,
            np.nan,
        )

        sharpes = np.full_like(
            means,
            np.nan,
        )

        scores = np.full_like(
            means,
            np.nan,
        )

        for index, history in enumerate(
            self._pnl_history
        ):
            values = np.asarray(
                history,
                dtype=np.float64,
            )

            if window is not None:
                values = values[
                    -window:
                ]

            if len(values) < (
                self.minimum_observations
            ):
                continue

            mean = float(
                np.mean(values)
            )

            volatility = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            if volatility >= 1e-10:
                sharpe = float(
                    np.sqrt(
                        self.annualisation_days
                    )
                    * mean
                    / volatility
                )
            else:
                sharpe = np.nan

            if (
                mean >= 0
                and volatility >= 1e-10
            ):
                sharpe_squared = (
                    sharpe**2
                )

                score = float(
                    mean
                    * sharpe_squared
                    / (
                        sharpe_squared
                        + 1.0
                    )
                )
            else:
                score = mean

            means[index] = mean
            volatilities[index] = (
                volatility
            )
            sharpes[index] = sharpe
            scores[index] = score

        return (
            means,
            volatilities,
            sharpes,
            scores,
        )

    def _equal_weights(
        self,
    ) -> FloatArray:
        return np.full(
            len(self.strategies),
            1.0 / len(self.strategies),
            dtype=np.float64,
        )

    def _one_hot(
        self,
        index: int,
    ) -> FloatArray:
        weights = np.zeros(
            len(self.strategies),
            dtype=np.float64,
        )

        weights[index] = 1.0
        return weights
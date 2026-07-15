from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bayesian import BayesianSignalModel
from portfolio import AllocationResult, PortfolioAllocator
from statistics.market_statistics import MarketStatistics
from statistics.residual_statistics import ResidualStatistics


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TradingEngineResult:
    positions: IntArray
    signals: FloatArray
    allocation: AllocationResult
    end: int


class TradingEngine:
    """
    End-to-end trading pipeline.

    Pipeline:

        prices
            ↓
        MarketStatistics
            ↓
        ResidualStatistics
            ↓
        BayesianSignalModel
            ↓
        PortfolioAllocator
            ↓
        legal integer positions

    One TradingEngine instance should persist across calls so Bayesian
    beliefs and hypothesis parameters are not reset every day.
    """

    def __init__(
        self,
        *,
        residual_estimation_window: int = 60,
        forgetting_rate: float = 0.03,
        entropy_penalty: float = 0.5,
        signal_scale: float = 5.0,
        minimum_trade_dollars: float = 100.0,
        turnover_smoothing: float = 0.25,
        proxy_index: int = 0,
    ) -> None:
        if residual_estimation_window < 2:
            raise ValueError(
                "residual_estimation_window must be at least 2"
            )

        if not 0.0 <= forgetting_rate <= 1.0:
            raise ValueError(
                "forgetting_rate must be between 0 and 1"
            )

        if entropy_penalty < 0:
            raise ValueError(
                "entropy_penalty cannot be negative"
            )

        if proxy_index < 0:
            raise ValueError(
                "proxy_index cannot be negative"
            )

        self.residual_estimation_window = (
            residual_estimation_window
        )
        self.forgetting_rate = forgetting_rate
        self.entropy_penalty = entropy_penalty
        self.proxy_index = proxy_index

        self.allocator = (
            PortfolioAllocator.algothon_defaults(
                signal_scale=signal_scale,
                minimum_trade_dollars=(
                    minimum_trade_dollars
                ),
                turnover_smoothing=(
                    turnover_smoothing
                ),
            )
        )

        self.current_positions = np.zeros(
            51,
            dtype=np.int64,
        )

        self.market_statistics: (
            MarketStatistics | None
        ) = None

        self.residual_statistics: (
            ResidualStatistics | None
        ) = None

        self.signal_model: (
            BayesianSignalModel | None
        ) = None

        self._last_number_of_price_days = 0
        self._is_initialised = False

    @property
    def is_initialised(self) -> bool:
        return self._is_initialised

    @property
    def minimum_price_days(self) -> int:
        """
        Minimum number of price observations required before the
        Bayesian model can begin updating.

        Price days are one more than return days.
        """
        return (
            self.residual_estimation_window
            + 3
        )

    def _validate_prices(
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

        if prices.shape[0] != 51:
            raise ValueError(
                "price_history must contain exactly 51 instruments"
            )

        if np.any(~np.isfinite(prices)):
            raise ValueError(
                "price_history must be finite"
            )

        if np.any(prices <= 0):
            raise ValueError(
                "prices must be strictly positive"
            )

        if prices.shape[1] < self._last_number_of_price_days:
            raise ValueError(
                "price history cannot move backwards in time"
            )

        return prices

    def _build_models(
        self,
        prices: FloatArray,
    ) -> None:
        self.market_statistics = MarketStatistics(
            prices=prices
        )

        self.residual_statistics = ResidualStatistics(
            market_statistics=self.market_statistics,
            proxy=self.proxy_index,
        )

        self.signal_model = BayesianSignalModel(
            residual_statistics=self.residual_statistics,
            residual_estimation_window=(
                self.residual_estimation_window
            ),
            forgetting_rate=self.forgetting_rate,
        )

        self._is_initialised = True

    def _refresh_statistics(
        self,
        prices: FloatArray,
    ) -> None:
        """
        Rebuild the data/statistics wrappers around the latest matrix.

        The BayesianSignalModel itself must persist, so this method only
        replaces its underlying MarketStatistics and ResidualStatistics.
        """
        if self.signal_model is None:
            raise RuntimeError(
                "signal model has not been initialised"
            )

        market_statistics = MarketStatistics(
            prices=prices
        )

        residual_statistics = ResidualStatistics(
            market_statistics=market_statistics,
            proxy=self.proxy_index,
        )

        self.market_statistics = market_statistics
        self.residual_statistics = residual_statistics

        self.signal_model.market_statistics = (
            market_statistics
        )

        self.signal_model.residual_statistics = (
            residual_statistics
        )

        self.signal_model.symbols = (
            residual_statistics.symbols
        )

    def update(
        self,
        price_history: FloatArray,
    ) -> TradingEngineResult:
        prices = self._validate_prices(
            price_history
        )

        number_of_price_days = prices.shape[1]
        latest_prices = prices[:, -1]

        if number_of_price_days < self.minimum_price_days:
            allocation = self.allocator.allocate(
                signals=np.zeros(
                    51,
                    dtype=np.float64,
                ),
                prices=latest_prices,
                current_positions=self.current_positions,
            )

            self.current_positions = (
                allocation.positions
            )

            self._last_number_of_price_days = (
                number_of_price_days
            )

            return TradingEngineResult(
                positions=self.current_positions.copy(),
                signals=np.zeros(
                    51,
                    dtype=np.float64,
                ),
                allocation=allocation,
                end=number_of_price_days - 1,
            )

        if not self._is_initialised:
            self._build_models(prices)

            assert self.signal_model is not None

            number_of_return_days = (
                number_of_price_days - 1
            )

            self.signal_model.warm_start(
                end=number_of_return_days
            )

        else:
            self._refresh_statistics(prices)

            assert self.signal_model is not None

            previous_return_end = (
                self.signal_model.last_processed_end
            )

            current_return_end = (
                number_of_price_days - 1
            )

            if previous_return_end is None:
                self.signal_model.warm_start(
                    end=current_return_end
                )
            else:
                for end in range(
                    previous_return_end + 1,
                    current_return_end + 1,
                ):
                    self.signal_model.update(
                        end=end
                    )

        assert self.signal_model is not None

        current_return_end = (
            number_of_price_days - 1
        )

        signals = self.signal_model.signal_vector(
            end=current_return_end,
            entropy_penalty=self.entropy_penalty,
        )

        allocation = self.allocator.allocate(
            signals=signals,
            prices=latest_prices,
            current_positions=self.current_positions,
        )

        self.current_positions = (
            allocation.positions
        )

        self._last_number_of_price_days = (
            number_of_price_days
        )

        return TradingEngineResult(
            positions=self.current_positions.copy(),
            signals=signals.copy(),
            allocation=allocation,
            end=current_return_end,
        )
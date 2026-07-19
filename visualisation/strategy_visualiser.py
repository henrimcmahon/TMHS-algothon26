from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from models.ticker import Ticker
from strategies.strategy import Strategy


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class StrategyVisualizer:
    ticker: Ticker
    strategy: Strategy

    def plot_price(
        self,
        prices: FloatArray,
    ) -> Figure:
        ticker_prices = self.ticker.get_prices(prices)

        figure, axis = plt.subplots()
        axis.plot(ticker_prices)
        axis.set_title(f"{self.ticker.symbol} price")
        axis.set_xlabel("Day")
        axis.set_ylabel("Price")

        return figure

    def plot_positions(
        self,
        prices: FloatArray,
    ) -> Figure:
        positions = self._calculate_position_history(prices)

        figure, axis = plt.subplots()
        axis.step(
            np.arange(positions.shape[0]),
            positions[:, self.ticker.index],
            where="post",
        )
        axis.set_title(f"{self.ticker.symbol} positions")
        axis.set_xlabel("Day")
        axis.set_ylabel("Position")

        return figure

    def plot_price_and_positions(
        self,
        prices: FloatArray,
    ) -> Figure:
        ticker_prices = self.ticker.get_prices(prices)
        positions = self._calculate_position_history(prices)

        figure, price_axis = plt.subplots()
        position_axis = price_axis.twinx()

        price_axis.plot(ticker_prices)
        position_axis.step(
            np.arange(positions.shape[0]),
            positions[:, self.ticker.index],
            where="post",
        )

        price_axis.set_title(
            f"{self.ticker.symbol} price and positions"
        )
        price_axis.set_xlabel("Day")
        price_axis.set_ylabel("Price")
        position_axis.set_ylabel("Position")

        return figure

    def _calculate_position_history(
        self,
        prices: FloatArray,
    ) -> IntArray:
        self.strategy.reset()

        n_tickers, n_days = prices.shape
        history = np.zeros(
            (n_days, n_tickers),
            dtype=np.int64,
        )

        for day in range(n_days):
            history[day] = self.strategy.get_positions(
                prices[:, : day + 1]
            )

        return history

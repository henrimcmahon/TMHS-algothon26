from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from models.market_constraints import MarketConstraints
from models.ticker import Ticker


FloatArray = NDArray[np.float64]


@dataclass
class TickerUniverse:
    tickers: list[Ticker]
    prices: FloatArray
    constraints: MarketConstraints

    def __post_init__(self) -> None:
        if self.prices.ndim != 2:
            raise ValueError("prices must have shape (n_tickers, n_days)")

        if len(self.tickers) != self.prices.shape[0]:
            raise ValueError(
                "number of tickers must equal number of price-matrix rows"
            )

        if len(self.tickers) != len(self.constraints.position_limits):
            raise ValueError(
                "constraints must contain one entry per ticker"
            )

        indices = [ticker.index for ticker in self.tickers]

        if indices != list(range(len(self.tickers))):
            raise ValueError(
                "ticker indices must be consecutive and match matrix row order"
            )

        symbols = [ticker.symbol for ticker in self.tickers]

        if len(symbols) != len(set(symbols)):
            raise ValueError("ticker symbols must be unique")

    @classmethod
    def from_algothon(
        cls,
        prices: FloatArray,
        symbols: list[str],
        position_limits: FloatArray,
        commission_rates: FloatArray,
    ) -> TickerUniverse:
        constraints = MarketConstraints(
            position_limits=np.asarray(
                position_limits,
                dtype=np.float64,
            ),
            commission_rates=np.asarray(
                commission_rates,
                dtype=np.float64,
            ),
        )

        tickers = [
            Ticker(
                index=index,
                symbol=symbol,
                position_limit=float(
                    constraints.position_limits[index]
                ),
                commission_rate=float(
                    constraints.commission_rates[index]
                ),
            )
            for index, symbol in enumerate(symbols)
        ]

        return cls(
            tickers=tickers,
            prices=np.asarray(prices, dtype=np.float64),
            constraints=constraints,
        )

    def get(self, identifier: int | str) -> Ticker:
        if isinstance(identifier, int):
            return self.tickers[identifier]

        for ticker in self.tickers:
            if ticker.symbol == identifier:
                return ticker

        raise KeyError(f"Unknown ticker symbol: {identifier}")

    def as_price_matrix(self) -> FloatArray:
        return self.prices

    def __iter__(self) -> Iterator[Ticker]:
        return iter(self.tickers)

    def __len__(self) -> int:
        return len(self.tickers)

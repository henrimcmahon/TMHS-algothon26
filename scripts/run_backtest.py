from __future__ import annotations

import numpy as np

from backtesting import Backtester
from models.ticker_universe import TickerUniverse
from trading import TradingEngine


def main() -> None:
    universe = TickerUniverse("prices.txt")

    prices = np.asarray(
        universe.as_price_matrix(),
        dtype=np.float64,
    )

    engine = TradingEngine(
        residual_estimation_window=60,
        forgetting_rate=0.03,
        entropy_penalty=0.5,
        signal_scale=5.0,
        minimum_trade_dollars=100.0,
        turnover_smoothing=0.25,
    )

    backtester = Backtester(
        engine=engine,
        annualisation_days=250,
    )

    result = backtester.run(
        price_history=prices,
    )

    summary = result.summary()

    formatted = {
        key: f"{value:,.2f}"
        if isinstance(value, (float, np.floating))
        else f"{value:,d}"
        if isinstance(value, (int, np.integer))
        else value
        for key, value in summary.items()
    }

    print()
    for key, value in formatted.items():
        print(f"{key:<25} {value}")
    print()

    result.as_dataframe().to_csv(
        "data/backtest_results.csv"
    )


if __name__ == "__main__":
    main()
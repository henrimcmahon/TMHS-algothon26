from __future__ import annotations

from time import perf_counter

import numpy as np

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

    start = perf_counter()

    total_volume = 0.0
    total_commission = 0.0

    for day in range(1, prices.shape[1] + 1):
        result = engine.update(
            prices[:, :day]
        )

        total_volume += (
            result.allocation.total_dollar_volume
        )

        total_commission += (
            result.allocation.total_commission
        )

    elapsed = perf_counter() - start

    print(f"Days processed: {prices.shape[1]}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(
        "Milliseconds per day: "
        f"{1_000 * elapsed / prices.shape[1]:.3f}"
    )
    print(
        "Total dollar volume: "
        f"${total_volume:,.2f}"
    )
    print(
        "Total commissions: "
        f"${total_commission:,.2f}"
    )
    print(
        "Final gross exposure: "
        f"${np.sum(np.abs(result.positions * prices[:, -1])):,.2f}"
    )


if __name__ == "__main__":
    main()
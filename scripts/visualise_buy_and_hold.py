from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from backtesting.backtester import Backtester
from models.ticker_universe import TickerUniverse
from scripts.run_buy_and_hold import load_prices
from strategies.allocators.equal_notional_allocator import (
    EqualNotionalAllocator,
)
from strategies.signals.buy_and_hold_signal import BuyAndHoldSignal
from strategies.strategy import Strategy
from visualisation.strategy_visualiser import StrategyVisualiser


def main() -> None:
    prices, symbols = load_prices(Path("prices.txt"))
    n_tickers = prices.shape[0]

    position_limits = np.full(n_tickers, 10_000.0, dtype=np.float64)
    position_limits[0] = 100_000.0

    commission_rates = np.full(n_tickers, 0.0001, dtype=np.float64)
    commission_rates[0] = 0.00002

    universe = TickerUniverse.from_algothon(
        prices=prices,
        symbols=symbols,
        position_limits=position_limits,
        commission_rates=commission_rates,
    )
    strategy = Strategy(
        name="Buy and Hold",
        signal_model=BuyAndHoldSignal(),
        position_allocator=EqualNotionalAllocator(),
        rebalance_interval=1,
    )
    result = Backtester(num_test_days=749).run(strategy=strategy, universe=universe)
    StrategyVisualiser(universe=universe, result=result).show_dashboard()


if __name__ == "__main__":
    main()

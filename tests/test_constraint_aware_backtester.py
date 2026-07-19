from __future__ import annotations

import numpy as np

from backtesting.backtester import Backtester
from models.ticker_universe import TickerUniverse
from strategies.allocators.equal_notional_allocator import (
    EqualNotionalAllocator,
)
from strategies.signals.buy_and_hold_signal import BuyAndHoldSignal
from strategies.strategy import Strategy


def test_backtester_passes_daily_market_constraints_to_allocator() -> None:
    prices = np.asarray(
        [
            [100.0, 125.0, 80.0, 90.0],
            [20.0, 25.0, 10.0, 12.0],
        ]
    )
    universe = TickerUniverse.from_algothon(
        prices=prices,
        symbols=["ALGO", "OTHER"],
        position_limits=np.asarray([100_000.0, 10_000.0]),
        commission_rates=np.asarray([0.00002, 0.0001]),
    )
    strategy = Strategy(
        name="Full long",
        signal_model=BuyAndHoldSignal(),
        position_allocator=EqualNotionalAllocator(),
        rebalance_interval=1,
    )

    result = Backtester(num_test_days=3).run(strategy, universe)

    assert np.array_equal(result.positions[1], np.asarray([1_000, 500]))
    assert np.array_equal(result.positions[2], np.asarray([800, 400]))
    assert np.array_equal(result.positions[3], np.asarray([1_250, 1_000]))
    assert result.total_commissions > 0.0

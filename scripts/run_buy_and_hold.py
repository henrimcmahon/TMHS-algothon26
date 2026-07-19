from __future__ import annotations

from pathlib import Path

import numpy as np

from backtesting.backtester import Backtester
from models.ticker_universe import TickerUniverse
from strategies.allocators.equal_notional_allocator import (
    EqualNotionalAllocator,
)
from strategies.signals.buy_and_hold_signal import (
    BuyAndHoldSignal,
)
from strategies.strategy import Strategy


def load_prices(
    path: Path,
) -> tuple[np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8") as file:
        symbols = file.readline().strip().split()

    raw_prices = np.genfromtxt(
        path,
        dtype=np.float64,
        skip_header=1,
    )

    if raw_prices.ndim != 2:
        raise ValueError(
            "prices.txt must contain a two-dimensional price matrix"
        )

    if raw_prices.shape[1] != len(symbols):
        raise ValueError(
            "number of ticker symbols does not match price columns"
        )

    return raw_prices.T, symbols


def main() -> None:
    prices, symbols = load_prices(
        Path("prices.txt")
    )

    n_tickers = prices.shape[0]

    # Official Algothon dollar position limits.
    position_limits = np.full(
        n_tickers,
        10_000.0,
        dtype=np.float64,
    )
    position_limits[0] = 100_000.0

    # Official Algothon commission rates.
    commission_rates = np.full(
        n_tickers,
        0.0001,
        dtype=np.float64,
    )
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
        position_allocator=EqualNotionalAllocator(
            total_notional=100_000.0,
        ),
        rebalance_interval=prices.shape[1] + 1,
    )

    backtester = Backtester(
        num_test_days=250,
        annualisation_factor=250,
        score_parameter=1.0,
    )

    result = backtester.run(
        strategy=strategy,
        universe=universe,
    )

    print(f"{result.name} Backtest")
    print("-" * 50)
    print(f"Scoring start day:       {result.scoring_start_day}")
    print(f"Scoring days:            {result.scoring_days:,}")
    print()
    print(f"Total net P&L:           {result.total_pnl:,.2f}")
    print(f"Total gross P&L:         {result.total_gross_pnl:,.2f}")
    print(f"Total commissions:       {result.total_commissions:,.2f}")
    print(f"Mean daily P&L:          {result.mean_daily_pnl:,.4f}")
    print(f"P&L standard deviation: {result.pnl_std:,.4f}")
    print(f"Annualised Sharpe:       {result.annualised_sharpe:,.4f}")
    print(f"Sharpe multiplier:       {result.sharpe_multiplier:,.4f}")
    print(f"Score:                   {result.score:,.4f}")
    print(f"Maximum drawdown:        {result.maximum_drawdown:,.2f}")
    print()
    print(f"Total dollar volume:     {result.total_dollar_volume:,.2f}")
    print(f"Return on volume:        {result.return_on_volume:.6f}")
    print(f"Total shares traded:     {result.total_shares_traded:,.0f}")
    print(f"Average daily turnover:  {result.average_daily_turnover:,.2f}")
    print(f"Maximum daily turnover:  {result.maximum_daily_turnover:,.2f}")



if __name__ == "__main__":
    main()

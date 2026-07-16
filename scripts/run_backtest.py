from __future__ import annotations

import numpy as np

from backtesting import (

    Backtester,

    BaselineEvaluator,

)

from backtesting.baselines import (

    AlgoHoldStrategy,

    AllAssetsHoldStrategy,

    NoPositionStrategy,

    PreviousReturnStrategy,

)

from models.ticker_universe import TickerUniverse
from trading.trading_engine import TradingEngine
from visualisation import (

    BacktestVisualiser,

)

def backtest_algo_hold(
    prices: np.ndarray,
) -> dict[str, float]:
    algo_prices = prices[0]

    number_of_days = len(algo_prices)

    positions = np.zeros(
        number_of_days - 1,
        dtype=np.int64,
    )

    daily_pnl = np.zeros(
        number_of_days - 1,
        dtype=np.float64,
    )

    daily_commission = np.zeros(
        number_of_days - 1,
        dtype=np.float64,
    )

    previous_position = 0

    for day in range(number_of_days - 1):
        maximum_shares = int(
            np.floor(
                100_000.0 / algo_prices[day]
            )
        )

        if day == 0:
            position = maximum_shares
        else:
            # Hold the existing shares unless today's price makes
            # the old position exceed the $100,000 position limit.
            position = min(
                previous_position,
                maximum_shares,
            )

        shares_traded = (
            position - previous_position
        )

        commission = (
            abs(shares_traded)
            * algo_prices[day]
            * 0.00002
        )

        gross_pnl = (
            position
            * (
                algo_prices[day + 1]
                - algo_prices[day]
            )
        )

        daily_pnl[day] = (
            gross_pnl - commission
        )

        daily_commission[day] = commission
        positions[day] = position
        previous_position = position

    mean_pnl = float(
        np.mean(daily_pnl)
    )

    pnl_std = float(
        np.std(
            daily_pnl,
            ddof=1,
        )
    )

    if (
        mean_pnl >= 0
        and pnl_std >= 1e-10
    ):
        sharpe = float(
            np.sqrt(250)
            * mean_pnl
            / pnl_std
        )

        score = float(
            mean_pnl
            * sharpe**2
            / (
                sharpe**2 + 1
            )
        )
    else:
        sharpe = (
            float(
                np.sqrt(250)
                * mean_pnl
                / pnl_std
            )
            if pnl_std > 0
            else 0.0
        )

        score = mean_pnl

    return {
        "Total PnL": float(
            np.sum(daily_pnl)
        ),
        "Total Commission": float(
            np.sum(daily_commission)
        ),
        "Mean Daily PnL": mean_pnl,
        "Daily PnL Std": pnl_std,
        "Annualised Sharpe": sharpe,
        "Score": score,
        "Profitable Day Fraction": float(
            np.mean(daily_pnl > 0)
        ),
    }


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

    strategy_result = backtester.run(
        price_history=prices,
    )

    summary = strategy_result.summary()

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

    algo_baseline = backtest_algo_hold(
        prices
    )

    print("ALGO buy-and-hold baseline")
    print("-" * 32)

    for key, value in algo_baseline.items():
        print(
            f"{key:<25} {value:,.2f}"
        )

    print()

    baseline_strategies = [
        NoPositionStrategy(),
        AlgoHoldStrategy("long"),
        AlgoHoldStrategy("short"),
        AllAssetsHoldStrategy("long"),
        AllAssetsHoldStrategy("short"),
        PreviousReturnStrategy(
            mode="momentum",
            exposure_fraction=0.25,
        ),
        PreviousReturnStrategy(
            mode="reversal",
            exposure_fraction=0.25,
        ),
    ]

    baseline_evaluator = (
        BaselineEvaluator()
    )

    baseline_results = {
        baseline.name: (
            baseline_evaluator.evaluate(
                strategy=baseline,
                price_history=prices,
            )
        )
        for baseline in baseline_strategies
    }

    results = {
        "Bayesian strategy": strategy_result,
        **baseline_results,
    }

    visualiser = BacktestVisualiser(

        results=results,

    )

    visualiser.plot_dashboard(
        rolling_window=20,
        interval=80,
    )

    visualiser.plot_score_dashboard(
        annualisation_days=250,
        minimum_observations=20,
        rolling_score_window=50,
        interval=80,
    )

    visualiser.plot_score_changes(
        annualisation_days=250,
        minimum_observations=20,
    )

    visualiser.plot_pnl_distribution(
        bins=40
    )

    position_prices = prices[
        :,
        strategy_result.start_day:
        strategy_result.start_day + len(
            strategy_result.positions
        ),
    ].T

    visualiser.plot_exposure(
        position_prices
    )

    strategy_result.as_dataframe().to_csv(
        "data/backtest_results.csv"
    )

if __name__ == "__main__":
    main()
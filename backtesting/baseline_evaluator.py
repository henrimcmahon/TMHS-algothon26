from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from backtesting.baselines import (
    BaselineStrategy,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BaselineResult:
    strategy_name: str

    daily_pnl: FloatArray
    gross_daily_pnl: FloatArray
    daily_commissions: FloatArray
    daily_turnover: FloatArray
    cumulative_pnl: FloatArray
    positions: NDArray[np.int64]

    total_pnl: float
    gross_pnl: float
    total_commission: float
    total_turnover: float
    mean_daily_pnl: float
    daily_pnl_std: float
    annualised_sharpe: float
    score: float
    maximum_drawdown: float
    profitable_day_fraction: float

    start_day: int
    end_day: int


class BaselineEvaluator:
    def __init__(
        self,
        annualisation_days: int = 250,
    ) -> None:
        if annualisation_days <= 0:
            raise ValueError(
                "annualisation_days must be positive"
            )

        self.annualisation_days = (
            annualisation_days
        )

    def evaluate(
        self,
        strategy: BaselineStrategy,
        price_history: FloatArray,
    ) -> BaselineResult:
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
                "price_history must contain 51 instruments"
            )

        if prices.shape[1] < 2:
            raise ValueError(
                "at least two price days are required"
            )

        strategy.reset()

        number_of_intervals = (
            prices.shape[1] - 1
        )

        daily_gross_pnl = np.zeros(
            number_of_intervals,
            dtype=np.float64,
        )

        daily_commissions = np.zeros(
            number_of_intervals,
            dtype=np.float64,
        )

        daily_turnover = np.zeros(
            number_of_intervals,
            dtype=np.float64,
        )

        previous_positions = np.zeros(
            51,
            dtype=np.int64,
        )

        commission_rates = (
            strategy.commission_rates()
        )

        position_history = np.zeros(
            (
                number_of_intervals,
                51,
            ),
            dtype=np.int64,
        )

        for day in range(
            number_of_intervals
        ):
            positions = np.asarray(
                strategy.get_positions(
                    prices[:, :day + 1]
                ),
                dtype=np.int64,
            )

            positions = strategy.clip_positions(
                positions,
                prices[:, day],
            )

            position_history[day] = positions

            price_change = (
                prices[:, day + 1]
                - prices[:, day]
            )

            shares_traded = (
                positions
                - previous_positions
            )

            dollar_volume = (
                np.abs(shares_traded)
                * prices[:, day]
            )

            daily_gross_pnl[day] = float(
                np.dot(
                    positions.astype(np.float64),
                    price_change,
                )
            )

            daily_turnover[day] = float(
                np.sum(dollar_volume)
            )

            daily_commissions[day] = float(
                np.sum(
                    dollar_volume
                    * commission_rates
                )
            )

            previous_positions = (
                positions.copy()
            )

        daily_net_pnl = (
            daily_gross_pnl
            - daily_commissions
        )

        mean_daily_pnl = float(
            np.mean(daily_net_pnl)
        )

        daily_pnl_std = float(
            np.std(
                daily_net_pnl,
                ddof=1,
            )
        )

        if daily_pnl_std >= 1e-10:
            annualised_sharpe = float(
                np.sqrt(
                    self.annualisation_days
                )
                * mean_daily_pnl
                / daily_pnl_std
            )
        else:
            annualised_sharpe = 0.0

        if (
            mean_daily_pnl >= 0
            and daily_pnl_std >= 1e-10
        ):
            sharpe_squared = (
                annualised_sharpe**2
            )

            score = float(
                mean_daily_pnl
                * sharpe_squared
                / (
                    sharpe_squared + 1.0
                )
            )
        else:
            # Important: negative-mean strategies receive μ directly.
            score = mean_daily_pnl

        cumulative_pnl = np.cumsum(
            daily_net_pnl
        )

        equity = np.concatenate(
            (
                np.asarray([0.0]),
                cumulative_pnl,
            )
        )

        running_peak = np.maximum.accumulate(
            equity
        )

        maximum_drawdown = float(
            np.min(
                equity - running_peak
            )
        )

        return BaselineResult(
            strategy_name=strategy.name,

            daily_pnl=daily_net_pnl,
            gross_daily_pnl=daily_gross_pnl,
            daily_commissions=daily_commissions,
            daily_turnover=daily_turnover,
            cumulative_pnl=cumulative_pnl,
            positions=position_history,

            total_pnl=float(
                np.sum(daily_net_pnl)
            ),
            gross_pnl=float(
                np.sum(daily_gross_pnl)
            ),
            total_commission=float(
                np.sum(daily_commissions)
            ),
            total_turnover=float(
                np.sum(daily_turnover)
            ),
            mean_daily_pnl=mean_daily_pnl,
            daily_pnl_std=daily_pnl_std,
            annualised_sharpe=annualised_sharpe,
            score=score,
            maximum_drawdown=maximum_drawdown,
            profitable_day_fraction=float(
                np.mean(daily_net_pnl > 0)
            ),
            start_day=0,
            end_day=number_of_intervals - 1,
        )

    def compare(
        self,
        strategies: list[BaselineStrategy],
        price_history: FloatArray,
    ) -> pd.DataFrame:
        if not strategies:
            raise ValueError(
                "at least one strategy is required"
            )

        results = [
            self.evaluate(
                strategy=strategy,
                price_history=price_history,
            )
            for strategy in strategies
        ]

        rows = [
            {
                "Strategy": result.strategy_name,
                "Gross PnL": result.gross_pnl,
                "Total PnL": result.total_pnl,
                "Commission": (
                    result.total_commission
                ),
                "Turnover": (
                    result.total_turnover
                ),
                "Mean Daily PnL": (
                    result.mean_daily_pnl
                ),
                "Daily PnL Std": (
                    result.daily_pnl_std
                ),
                "Annualised Sharpe": (
                    result.annualised_sharpe
                ),
                "Score": result.score,
                "Maximum Drawdown": (
                    result.maximum_drawdown
                ),
                "Profitable Day Fraction": (
                    result
                    .profitable_day_fraction
                ),
            }
            for result in results
        ]

        return (
            pd.DataFrame(rows)
            .set_index("Strategy")
            .sort_values(
                "Score",
                ascending=False,
            )
        )
    
    def results_dataframe(
        self,
        results: dict[
            str,
            BaselineResult,
        ],
    ) -> pd.DataFrame:
        if not results:
            raise ValueError(
                "at least one result is required"
            )

        rows = [
            {
                "Strategy": result.strategy_name,
                "Gross PnL": result.gross_pnl,
                "Total PnL": result.total_pnl,
                "Commission": (
                    result.total_commission
                ),
                "Turnover": result.total_turnover,
                "Mean Daily PnL": (
                    result.mean_daily_pnl
                ),
                "Daily PnL Std": (
                    result.daily_pnl_std
                ),
                "Annualised Sharpe": (
                    result.annualised_sharpe
                ),
                "Score": result.score,
                "Maximum Drawdown": (
                    result.maximum_drawdown
                ),
                "Profitable Day Fraction": (
                    result.profitable_day_fraction
                ),
            }
            for result in results.values()
        ]

        return (
            pd.DataFrame(rows)
            .set_index("Strategy")
            .sort_values(
                "Score",
                ascending=False,
            )
        )
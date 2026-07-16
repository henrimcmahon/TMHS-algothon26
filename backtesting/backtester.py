from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from trading import TradingEngine


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    daily_pnl: FloatArray
    gross_daily_pnl: FloatArray
    daily_commissions: FloatArray
    daily_turnover: FloatArray
    positions: IntArray

    cumulative_pnl: FloatArray
    total_pnl: float
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

    def as_dataframe(self) -> pd.DataFrame:
        days = np.arange(
            self.start_day,
            self.end_day + 1,
            dtype=np.int64,
        )

        return pd.DataFrame(
            {
                "Day": days,
                "Gross PnL": self.gross_daily_pnl,
                "Commission": self.daily_commissions,
                "Net PnL": self.daily_pnl,
                "Turnover": self.daily_turnover,
                "Cumulative PnL": self.cumulative_pnl,
            }
        ).set_index("Day")

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "Start Day": self.start_day,
                "End Day": self.end_day,
                "Total PnL": self.total_pnl,
                "Total Commission": self.total_commission,
                "Total Turnover": self.total_turnover,
                "Mean Daily PnL": self.mean_daily_pnl,
                "Daily PnL Std": self.daily_pnl_std,
                "Annualised Sharpe": self.annualised_sharpe,
                "Score": self.score,
                "Maximum Drawdown": self.maximum_drawdown,
                "Profitable Day Fraction": (
                    self.profitable_day_fraction
                ),
            },
            name="Backtest Summary",
        )


class Backtester:
    """
    Replay the trading engine through historical prices and calculate
    daily PnL after commissions.

    Timing convention:

        At the end of day t:
            the strategy observes prices through t and chooses position q_t.

        From day t to t + 1:
            q_t earns:

                q_t * (P_{t+1} - P_t)

        Commission for changing from q_{t-1} to q_t is charged at day-t
        prices:

                commission_t
                = rate * P_t * abs(q_t - q_{t-1})

    This avoids look-ahead bias because positions selected using day-t
    information only earn returns beginning after day t.
    """

    def __init__(
        self,
        engine: TradingEngine,
        annualisation_days: int = 250,
    ) -> None:
        if annualisation_days <= 0:
            raise ValueError(
                "annualisation_days must be positive"
            )

        self.engine = engine
        self.annualisation_days = annualisation_days

    def run(
        self,
        price_history: FloatArray,
        start_day: int | None = None,
        end_day: int | None = None,
    ) -> BacktestResult:
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

        number_of_days = prices.shape[1]

        if number_of_days < 2:
            raise ValueError(
                "at least two price days are required"
            )

        if start_day is None:
            start_day = 0

        if end_day is None:
            end_day = number_of_days - 1

        if not 0 <= start_day < end_day < number_of_days:
            raise ValueError(
                "require 0 <= start_day < end_day < number_of_days"
            )

        number_of_pnl_days = end_day - start_day

        gross_daily_pnl = np.zeros(
            number_of_pnl_days,
            dtype=np.float64,
        )

        daily_commissions = np.zeros(
            number_of_pnl_days,
            dtype=np.float64,
        )

        daily_turnover = np.zeros(
            number_of_pnl_days,
            dtype=np.float64,
        )

        position_history = np.zeros(
            (
                number_of_pnl_days,
                prices.shape[0],
            ),
            dtype=np.int64,
        )

        previous_positions = np.zeros(
            prices.shape[0],
            dtype=np.int64,
        )

        for output_index, day in enumerate(
            range(start_day, end_day)
        ):
            engine_result = self.engine.update(
                prices[:, :day + 1]
            )

            positions = np.asarray(
                engine_result.positions,
                dtype=np.int64,
            )

            price_change = (
                prices[:, day + 1]
                - prices[:, day]
            )

            gross_pnl = float(
                np.dot(
                    positions.astype(np.float64),
                    price_change,
                )
            )

            shares_traded = (
                positions - previous_positions
            )

            dollar_turnover = float(
                np.sum(
                    np.abs(shares_traded)
                    * prices[:, day]
                )
            )

            commission = float(
                np.sum(
                    np.abs(shares_traded)
                    * prices[:, day]
                    * self.engine
                    .allocator
                    .commission_rates
                )
            )

            gross_daily_pnl[output_index] = gross_pnl
            daily_commissions[output_index] = commission
            daily_turnover[output_index] = dollar_turnover
            position_history[output_index] = positions

            previous_positions = positions.copy()

        daily_pnl = (
            gross_daily_pnl
            - daily_commissions
        )

        cumulative_pnl = np.cumsum(
            daily_pnl
        )

        mean_daily_pnl = float(
            np.mean(daily_pnl)
        )

        daily_pnl_std = float(
            np.std(
                daily_pnl,
                ddof=1,
            )
        )

        if daily_pnl_std > 0:
            annualised_sharpe = float(
                np.sqrt(self.annualisation_days)
                * mean_daily_pnl
                / daily_pnl_std
            )
        else:
            annualised_sharpe = 0.0

        sharpe_squared = annualised_sharpe**2

        score = float(
            mean_daily_pnl
            * (
                sharpe_squared
                / (sharpe_squared + 1.0)
            )
        )

        running_peak = np.maximum.accumulate(
            np.concatenate(
                (
                    np.asarray([0.0]),
                    cumulative_pnl,
                )
            )
        )

        equity_path = np.concatenate(
            (
                np.asarray([0.0]),
                cumulative_pnl,
            )
        )

        drawdowns = (
            equity_path - running_peak
        )

        maximum_drawdown = float(
            np.min(drawdowns)
        )

        profitable_day_fraction = float(
            np.mean(daily_pnl > 0)
        )

        return BacktestResult(
            daily_pnl=daily_pnl,
            gross_daily_pnl=gross_daily_pnl,
            daily_commissions=daily_commissions,
            daily_turnover=daily_turnover,
            positions=position_history,
            cumulative_pnl=cumulative_pnl,
            total_pnl=float(
                np.sum(daily_pnl)
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
            profitable_day_fraction=(
                profitable_day_fraction
            ),
            start_day=start_day,
            end_day=end_day - 1,
        )
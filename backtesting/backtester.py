from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from backtesting.backtest_result import BacktestResult
from models.ticker_universe import TickerUniverse
from strategies.strategy import Strategy


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class Backtester:
    """
    Backtester matching the Algothon 2026 evaluator.

    The final num_test_days price changes are scored. Strategies receive
    all prices available up to the current day and return integer target
    positions.
    """

    num_test_days: int = 250
    annualisation_factor: int = 250
    score_parameter: float = 1.0

    def __post_init__(self) -> None:
        if self.num_test_days < 1:
            raise ValueError("num_test_days must be at least 1")

        if self.annualisation_factor < 1:
            raise ValueError(
                "annualisation_factor must be at least 1"
            )

        if self.score_parameter <= 0:
            raise ValueError(
                "score_parameter must be positive"
            )

    def run(
        self,
        strategy: Strategy,
        universe: TickerUniverse,
    ) -> BacktestResult:
        prices = np.asarray(
            universe.as_price_matrix(),
            dtype=np.float64,
        )

        if prices.ndim != 2:
            raise ValueError(
                "prices must have shape (n_tickers, n_days)"
            )

        n_tickers, n_days = prices.shape

        if n_days <= self.num_test_days:
            raise ValueError(
                f"Need more than {self.num_test_days} price days; "
                f"received {n_days}"
            )

        if np.any(~np.isfinite(prices)):
            raise ValueError("prices contain non-finite values")

        if np.any(prices <= 0):
            raise ValueError("prices must be strictly positive")

        commission_rates = np.asarray(
            universe.constraints.commission_rates,
            dtype=np.float64,
        )

        dollar_position_limits = np.asarray(
            universe.constraints.position_limits,
            dtype=np.float64,
        )

        expected_shape = (n_tickers,)

        if commission_rates.shape != expected_shape:
            raise ValueError(
                "commission_rates must have shape (n_tickers,)"
            )

        if dollar_position_limits.shape != expected_shape:
            raise ValueError(
                "position_limits must have shape (n_tickers,)"
            )

        strategy.reset()

        # The official evaluator starts calling the strategy here.
        scoring_start_day = n_days - self.num_test_days

        positions = np.zeros(
            (n_days, n_tickers),
            dtype=np.int64,
        )

        trades = np.zeros(
            (n_days, n_tickers),
            dtype=np.int64,
        )

        gross_daily_pnl = np.zeros(
            n_days,
            dtype=np.float64,
        )

        daily_commissions = np.zeros(
            n_days,
            dtype=np.float64,
        )

        daily_pnl = np.zeros(
            n_days,
            dtype=np.float64,
        )

        daily_turnover = np.zeros(
            n_days,
            dtype=np.float64,
        )

        portfolio_values = np.zeros(
            n_days,
            dtype=np.float64,
        )

        cash_history = np.zeros(
            n_days,
            dtype=np.float64,
        )

        current_positions = np.zeros(
            n_tickers,
            dtype=np.int64,
        )

        cash = 0.0
        previous_value = 0.0
        total_dollar_volume = 0.0

        # This reproduces:
        #
        # for t in range(startDay, nt + 1)
        #
        # The extra iteration at n_days marks the final held position
        # at the last available prices without requesting another trade.
        for day in range(scoring_start_day, n_days + 1):
            price_history = prices[:, :day]
            current_prices = price_history[:, -1]

            if day < n_days:
                proposed_positions = np.asarray(
                    strategy.get_positions(price_history),
                    dtype=np.int64,
                )

                if proposed_positions.shape != expected_shape:
                    raise ValueError(
                        "strategy must return shape (n_tickers,)"
                    )

                position_limits_in_shares = np.floor(
                    dollar_position_limits
                    / current_prices
                ).astype(np.int64)

                new_positions = np.clip(
                    proposed_positions,
                    -position_limits_in_shares,
                    position_limits_in_shares,
                ).astype(np.int64)
            else:
                new_positions = current_positions.copy()

            position_change = (
                new_positions - current_positions
            )

            traded_notional_by_ticker = (
                current_prices
                * np.abs(position_change)
            )

            turnover = float(
                np.sum(traded_notional_by_ticker)
            )

            commission = float(
                np.sum(
                    traded_notional_by_ticker
                    * commission_rates
                )
            )

            # Purchasing or selling shares changes cash by trade value,
            # and commissions are charged immediately.
            cash -= float(
                np.dot(
                    current_prices,
                    position_change,
                )
            )
            cash -= commission

            total_dollar_volume += turnover

            current_positions = new_positions.copy()

            position_value = float(
                np.dot(
                    current_positions,
                    current_prices,
                )
            )

            portfolio_value = cash + position_value
            net_pnl = portfolio_value - previous_value

            # Gross P&L adds the current day's commission back.
            gross_pnl = net_pnl + commission

            if day < n_days:
                positions[day] = current_positions
                trades[day] = position_change
                daily_commissions[day] = commission
                daily_turnover[day] = turnover
                portfolio_values[day] = portfolio_value
                cash_history[day] = cash

            # The first iteration establishes the starting portfolio.
            # The official evaluator only scores when day > startDay.
            if day > scoring_start_day:
                pnl_index = day - 1

                daily_pnl[pnl_index] = net_pnl
                gross_daily_pnl[pnl_index] = gross_pnl
                portfolio_values[pnl_index] = portfolio_value
                cash_history[pnl_index] = cash

            previous_value = portfolio_value

        scoring_slice = slice(
            scoring_start_day,
            n_days,
        )

        scored_daily_pnl = daily_pnl[scoring_slice]
        scored_gross_pnl = gross_daily_pnl[scoring_slice]
        scored_commissions = daily_commissions[scoring_slice]
        scored_turnover = daily_turnover[scoring_slice]
        scored_trades = trades[scoring_slice]

        cumulative_pnl = np.cumsum(daily_pnl)

        mean_daily_pnl = self._mean(
            scored_daily_pnl
        )

        pnl_std = self._population_std(
            scored_daily_pnl
        )

        annualised_sharpe = self._annualised_sharpe(
            mean_daily_pnl,
            pnl_std,
        )

        sharpe_multiplier = self._sharpe_multiplier(
            annualised_sharpe
        )

        score = self._score(
            mean_daily_pnl,
            pnl_std,
        )

        total_pnl = float(
            np.sum(scored_daily_pnl)
        )

        total_gross_pnl = float(
            np.sum(scored_gross_pnl)
        )

        total_commissions = float(
            np.sum(scored_commissions)
        )

        maximum_drawdown = self._maximum_drawdown(
            scored_daily_pnl
        )

        total_shares_traded = float(
            np.sum(
                np.abs(scored_trades)
            )
        )

        average_daily_turnover = self._mean(
            scored_turnover
        )

        maximum_daily_turnover = (
            float(np.max(scored_turnover))
            if scored_turnover.size > 0
            else 0.0
        )

        final_value = (
            float(portfolio_values[n_days - 1])
            if n_days > 0
            else 0.0
        )

        return_on_volume = (
            final_value / total_dollar_volume
            if total_dollar_volume > 0
            else 0.0
        )

        return BacktestResult(
            name=strategy.name,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            gross_daily_pnl=gross_daily_pnl,
            daily_commissions=daily_commissions,
            daily_turnover=daily_turnover,
            positions=positions,
            trades=trades,
            portfolio_values=portfolio_values,
            cash_history=cash_history,
            total_pnl=total_pnl,
            total_gross_pnl=total_gross_pnl,
            total_commissions=total_commissions,
            mean_daily_pnl=mean_daily_pnl,
            pnl_std=pnl_std,
            annualised_sharpe=annualised_sharpe,
            sharpe_multiplier=sharpe_multiplier,
            score=score,
            maximum_drawdown=maximum_drawdown,
            total_dollar_volume=total_dollar_volume,
            total_shares_traded=total_shares_traded,
            average_daily_turnover=average_daily_turnover,
            maximum_daily_turnover=maximum_daily_turnover,
            return_on_volume=return_on_volume,
            scoring_start_day=scoring_start_day,
            scoring_days=scored_daily_pnl.size,
        )

    @staticmethod
    def _mean(
        values: FloatArray,
    ) -> float:
        if values.size == 0:
            return 0.0

        return float(np.mean(values))

    @staticmethod
    def _population_std(
        values: FloatArray,
    ) -> float:
        """
        Official evaluator uses np.std(values), meaning ddof=0.
        """
        if values.size == 0:
            return 0.0

        return float(
            np.std(values, ddof=0)
        )

    def _annualised_sharpe(
        self,
        mean_daily_pnl: float,
        pnl_std: float,
    ) -> float:
        if pnl_std <= 0:
            return 0.0

        return float(
            np.sqrt(self.annualisation_factor)
            * mean_daily_pnl
            / pnl_std
        )

    def _sharpe_multiplier(
        self,
        annualised_sharpe: float,
    ) -> float:
        sharpe_squared = annualised_sharpe**2
        parameter_squared = self.score_parameter**2

        denominator = (
            sharpe_squared
            + parameter_squared
        )

        if denominator == 0:
            return 0.0

        return float(
            sharpe_squared / denominator
        )

    def _score(
        self,
        mean_daily_pnl: float,
        pnl_std: float,
    ) -> float:
        """
        Match the official Algothon score function.

        score = mean P&L * Sharpe multiplier

        For non-positive mean P&L or approximately zero volatility,
        the score is simply the mean P&L.
        """
        if (
            mean_daily_pnl <= 0
            or pnl_std < 1e-10
        ):
            return mean_daily_pnl

        annualised_sharpe = self._annualised_sharpe(
            mean_daily_pnl,
            pnl_std,
        )

        multiplier = self._sharpe_multiplier(
            annualised_sharpe
        )

        return float(
            mean_daily_pnl * multiplier
        )

    @staticmethod
    def _maximum_drawdown(
        daily_pnl: FloatArray,
    ) -> float:
        if daily_pnl.size == 0:
            return 0.0

        cumulative_pnl = np.concatenate(
            (
                np.array([0.0]),
                np.cumsum(daily_pnl),
            )
        )

        running_peak = np.maximum.accumulate(
            cumulative_pnl
        )

        drawdown = (
            running_peak - cumulative_pnl
        )

        return float(
            np.max(drawdown)
        )

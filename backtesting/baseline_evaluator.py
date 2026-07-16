from __future__ import annotations

from collections.abc import Sequence
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
    name: str

    daily_pnl: np.ndarray
    cumulative_pnl: np.ndarray
    gross_daily_pnl: np.ndarray
    daily_commissions: np.ndarray
    positions: np.ndarray

    total_pnl: float
    total_gross_pnl: float
    total_commissions: float
    mean_daily_pnl: float
    pnl_std: float
    annualised_sharpe: float
    score: float
    maximum_drawdown: float

    scoring_start_day: int
    scoring_days: int


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

    @staticmethod
    def _competition_score(
        mean_daily_pnl: float,
        pnl_std: float,
        parameter: float = 1.0,
    ) -> float:
        if (
            mean_daily_pnl <= 0.0
            or pnl_std < 1e-10
        ):
            return mean_daily_pnl

        annualised_sharpe = (
            np.sqrt(250.0)
            * mean_daily_pnl
            / pnl_std
        )

        score_fraction = (
            annualised_sharpe**2
            / (
                annualised_sharpe**2
                + parameter**2
            )
        )

        return float(
            mean_daily_pnl
            * score_fraction
        )

    def evaluate(
        self,
        strategy: BaselineStrategy,
        price_history: np.ndarray,
        *,
        num_test_days: int | None = None,
    ) -> BaselineResult:
        """
        Backtest one strategy.

        The strategy is run using the full price history. When
        num_test_days is supplied, summary metrics are calculated only
        from the final num_test_days observations.
        """

        strategy.reset()

        number_of_days = price_history.shape[1]

        if num_test_days is None:
            num_test_days = number_of_days

        if num_test_days <= 0:
            raise ValueError(
                "num_test_days must be positive"
            )

        if num_test_days > number_of_days:
            raise ValueError(
                "num_test_days cannot exceed the number of observations"
            )

        scoring_start_index = (
            number_of_days - num_test_days
        )

        prices = np.asarray(
            price_history,
            dtype=np.float64,
        )

        if prices.ndim != 2:
            raise ValueError(
                "price_history must have shape "
                "(number_of_instruments, number_of_days)"
            )

        number_of_instruments, number_of_days = (
            prices.shape
        )

        if number_of_days < 2:
            raise ValueError(
                "price_history must contain at least two days"
            )

        if num_test_days is None:
            num_test_days = number_of_days

        if num_test_days <= 0:
            raise ValueError(
                "num_test_days must be positive"
            )

        if num_test_days > number_of_days:
            raise ValueError(
                "num_test_days cannot exceed the "
                "number of price observations"
            )

        scoring_start_index = (
            number_of_days - num_test_days
        )

        dollar_position_limits = np.asarray(
            strategy.position_limits(),
            dtype=np.float64,
        )

        commission_rates = np.asarray(
            strategy.commission_rates(),
            dtype=np.float64,
        )

        if dollar_position_limits.shape != (
            number_of_instruments,
        ):
            raise ValueError(
                f"{strategy.name!r} has dollar position limits "
                f"with shape {dollar_position_limits.shape}; "
                f"expected {(number_of_instruments,)}"
            )

        if commission_rates.shape != (
            number_of_instruments,
        ):
            raise ValueError(
                f"{strategy.name!r} has commission rates "
                f"with shape {commission_rates.shape}; "
                f"expected {(number_of_instruments,)}"
            )

        # ----------------------------------------------------------
        # Run the existing full-history backtest here.
        # ----------------------------------------------------------

        positions = np.zeros(
            (
                number_of_days - 1,
                number_of_instruments,
            ),
            dtype=np.int64,
        )

        gross_daily_pnl = np.zeros(
            number_of_days - 1,
            dtype=np.float64,
        )

        daily_commissions = np.zeros(
            number_of_days - 1,
            dtype=np.float64,
        )

        current_positions = np.zeros(
            number_of_instruments,
            dtype=np.int64,
        )

        for day_index in range(
            number_of_days - 1
        ):
            prices_so_far = prices[
                :,
                : day_index + 1,
            ]

            current_prices = prices[
                :,
                day_index,
            ]

            next_prices = prices[
                :,
                day_index + 1,
            ]

            proposed_positions = np.asarray(
                strategy.get_positions(
                    prices_so_far
                ),
                dtype=np.int64,
            )

            if proposed_positions.shape != (
                number_of_instruments,
            ):
                raise ValueError(
                    f"{strategy.name!r} returned positions "
                    f"with shape {proposed_positions.shape}; "
                    f"expected {(number_of_instruments,)}"
                )

            new_positions = strategy.clip_positions(
                proposed_positions,
                current_prices,
            )

            traded_positions = (
                new_positions
                - current_positions
            )

            traded_dollars = (
                current_prices
                * np.abs(traded_positions)
            )

            commission = float(
                np.sum(
                    traded_dollars
                    * commission_rates
                )
            )

            gross_pnl = float(
                np.dot(
                    new_positions,
                    next_prices - current_prices,
                )
            )

            positions[
                day_index
            ] = new_positions

            gross_daily_pnl[
                day_index
            ] = gross_pnl

            daily_commissions[
                day_index
            ] = commission

            current_positions = new_positions

        daily_pnl = (
            gross_daily_pnl
            - daily_commissions
        )

        cumulative_pnl = np.cumsum(
            daily_pnl
        )

        # ----------------------------------------------------------
        # Select only the official scoring window.
        # ----------------------------------------------------------

        scoring_daily_pnl = daily_pnl[
            scoring_start_index:
        ]

        scoring_gross_pnl = gross_daily_pnl[
            scoring_start_index:
        ]

        scoring_commissions = daily_commissions[
            scoring_start_index:
        ]

        scoring_positions = positions[
            scoring_start_index:
        ]

        scoring_cumulative_pnl = np.cumsum(
            scoring_daily_pnl
        )

        mean_daily_pnl = float(
            np.mean(scoring_daily_pnl)
        )

        pnl_std = float(
            np.std(
                scoring_daily_pnl,
                ddof=0,
            )
        )

        annualised_sharpe = (
            0.0
            if pnl_std < 1e-10
            else float(
                np.sqrt(250.0)
                * mean_daily_pnl
                / pnl_std
            )
        )

        score = self._competition_score(
            mean_daily_pnl,
            pnl_std,
        )

        scoring_cumulative_pnl = np.concatenate(
            (
                np.array(
                    [0.0],
                    dtype=np.float64,
                ),
                np.cumsum(
                    scoring_daily_pnl
                ),
            )
        )

        running_peak = np.maximum.accumulate(
            scoring_cumulative_pnl
        )

        maximum_drawdown = float(
            np.max(
                running_peak
                - scoring_cumulative_pnl
            )
        )

        return BaselineResult(
            name=strategy.name,
            daily_pnl=scoring_daily_pnl,
            cumulative_pnl=scoring_cumulative_pnl,
            gross_daily_pnl=scoring_gross_pnl,
            daily_commissions=scoring_commissions,
            positions=scoring_positions,
            total_pnl=float(
                np.sum(scoring_daily_pnl)
            ),
            total_gross_pnl=float(
                np.sum(scoring_gross_pnl)
            ),
            total_commissions=float(
                np.sum(scoring_commissions)
            ),
            mean_daily_pnl=mean_daily_pnl,
            pnl_std=pnl_std,
            annualised_sharpe=annualised_sharpe,
            score=score,
            maximum_drawdown=maximum_drawdown,
            scoring_start_day = (
                0
                if num_test_days is None
                else number_of_days - num_test_days
            ),
            scoring_days=len(
                scoring_daily_pnl
            ),
        )

    def compare(
        self,
        strategies: Sequence[BaselineStrategy],
        price_history: np.ndarray,
        *,
        num_test_days: int | None = None,
    ) -> dict[str, BaselineResult]:
        return {
            strategy.name: self.evaluate(
                strategy,
                price_history,
                num_test_days=num_test_days,
            )
            for strategy in strategies
        }
    
    def results_dataframe(
        self,
        results: Sequence[BaselineResult]
        | dict[str, BaselineResult],
    ) -> pd.DataFrame:
        if isinstance(results, dict):
            result_values = results.values()
        else:
            result_values = results

        rows = [
            {
                "Strategy": result.name,
                "Total PnL": result.total_pnl,
                "Gross PnL": result.total_gross_pnl,
                "Commissions": result.total_commissions,
                "Mean Daily PnL": result.mean_daily_pnl,
                "PnL Std": result.pnl_std,
                "Annualised Sharpe": result.annualised_sharpe,
                "Score": result.score,
                "Maximum Drawdown": result.maximum_drawdown,
                "Scoring Start Day": result.scoring_start_day,
                "Scoring Days": result.scoring_days,
            }
            for result in result_values
        ]

        return (
            pd.DataFrame(rows)
            .sort_values(
                by="Score",
                ascending=False,
            )
            .reset_index(drop=True)
        )
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from backtesting.baseline_evaluator import (
    BaselineResult,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OracleResult:
    daily_pnl: FloatArray
    cumulative_pnl: FloatArray
    selected_strategies: tuple[str, ...]

    total_pnl: float
    mean_daily_pnl: float
    daily_pnl_std: float
    annualised_sharpe: float
    score: float
    maximum_drawdown: float
    profitable_day_fraction: float

    scoring_start_day: int
    scoring_days: int

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "Total PnL": self.total_pnl,
                "Mean Daily PnL": self.mean_daily_pnl,
                "Daily PnL Std": self.daily_pnl_std,
                "Annualised Sharpe": (
                    self.annualised_sharpe
                ),
                "Score": self.score,
                "Maximum Drawdown": (
                    self.maximum_drawdown
                ),
                "Profitable Day Fraction": (
                    self.profitable_day_fraction
                ),
            },
            name="Perfect-foresight oracle",
            dtype=np.float64,
        )


class OracleEvaluator:
    """
    Perfect-foresight upper bound over existing strategy results.

    This is not tradeable. On each day it selects whichever component
    strategy produced the highest realised net PnL that same day.
    """

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
        results: Mapping[
            str,
            BaselineResult,
        ],
    ) -> OracleResult:
        if not results:
            raise ValueError(
                "at least one baseline result is required"
            )

        strategy_names = list(results)

        first_result = next(iter(results.values()))

        scoring_start_day = (
            first_result.scoring_start_day
        )

        scoring_days = (
            first_result.scoring_days
        )

        lengths = {
            len(result.daily_pnl)
            for result in results.values()
        }

        if len(lengths) != 1:
            raise ValueError(
                "all results must have equal length"
            )

        pnl_matrix = np.vstack(
            [
                np.asarray(
                    results[name].daily_pnl,
                    dtype=np.float64,
                )
                for name in strategy_names
            ]
        )

        winning_indices = np.argmax(
            pnl_matrix,
            axis=0,
        )

        day_indices = np.arange(
            pnl_matrix.shape[1]
        )

        daily_pnl = pnl_matrix[
            winning_indices,
            day_indices,
        ]

        selected_strategies = tuple(
            strategy_names[index]
            for index in winning_indices
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
            score = mean_daily_pnl

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

        return OracleResult(
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            selected_strategies=selected_strategies,

            total_pnl=float(np.sum(daily_pnl)),
            mean_daily_pnl=mean_daily_pnl,
            daily_pnl_std=daily_pnl_std,
            annualised_sharpe=annualised_sharpe,
            score=score,
            maximum_drawdown=maximum_drawdown,
            profitable_day_fraction=float(
                np.mean(daily_pnl > 0)
            ),

            scoring_start_day=scoring_start_day,
            scoring_days=scoring_days,
        )
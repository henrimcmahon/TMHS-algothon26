from typing import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray as FloatArray

from .baseline_strategy import BaselineStrategy
from .meta_strategy import MetaStrategy


class LeaderStrategy(MetaStrategy):
    """
    Follow the strategy with the highest historical metric.
    """

    def __init__(
        self,
        strategies: Sequence[BaselineStrategy],
        *,
        metric: Literal[
            "score",
            "sharpe",
            "mean",
            "lowest_volatility",
        ] = "score",
        window: int | None = None,
        minimum_metric: float | None = None,
        minimum_observations: int = 20,
        annualisation_days: int = 250,
    ) -> None:
        super().__init__(
            strategies,
            minimum_observations=(
                minimum_observations
            ),
            annualisation_days=(
                annualisation_days
            ),
        )

        valid_metrics = {
            "score",
            "sharpe",
            "mean",
            "lowest_volatility",
        }

        if metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of "
                f"{sorted(valid_metrics)}"
            )

        if window is not None and window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        self.minimum_metric = minimum_metric
        self.metric = metric
        self.window = window

    @property
    def name(self) -> str:
        prefix = (
            f"{self.window}-day rolling"
            if self.window is not None
            else "Expanding"
        )

        descriptions = {
            "score": "score leader",
            "sharpe": "Sharpe leader",
            "mean": "mean-PnL leader",
            "lowest_volatility": (
                "lowest-volatility leader"
            ),
        }

        suffix = (
            f" above {self.minimum_metric:g}"
            if self.minimum_metric is not None
            else ""
        )

        return (
            f"{prefix} "
            f"{descriptions[self.metric]}"
            f"{suffix}"
        )

    def _calculate_weights(
        self,
    ) -> FloatArray:
        (
            means,
            volatilities,
            sharpes,
            scores,
        ) = self._metric_arrays(
            window=self.window
        )

        metric_values = {
            "score": scores,
            "sharpe": sharpes,
            "mean": means,
            "lowest_volatility": (
                -volatilities
            ),
        }[self.metric]

        valid = np.isfinite(
            metric_values
        )

        if not np.any(valid):
            return self._equal_weights()

        valid_indices = np.flatnonzero(
            valid
        )

        winner = valid_indices[
            np.argmax(
                metric_values[valid]
            )
        ]

        winner_value = float(
            metric_values[winner]
        )

        if (
            self.minimum_metric is not None
            and winner_value < self.minimum_metric
        ):
            return np.zeros(
                len(self.strategies),
                dtype=np.float64,
            )

        return self._one_hot(
            int(winner)
        )
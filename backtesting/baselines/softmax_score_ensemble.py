from typing import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray as FloatArray

from .meta_strategy import MetaStrategy, BaselineStrategy


class SoftmaxScoreEnsemble(MetaStrategy):
    """
    Smoothly allocate toward higher-scoring component strategies.
    """

    def __init__(
        self,
        strategies: Sequence[BaselineStrategy],
        *,
        temperature: float = 10.0,
        window: int | None = None,
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

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive"
            )

        if window is not None and window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        self.temperature = temperature
        self.window = window

    @property
    def name(self) -> str:
        if self.window is None:
            return "Softmax expanding-score ensemble"

        return (
            f"Softmax {self.window}-day "
            "score ensemble"
        )

    def _calculate_weights(
        self,
    ) -> FloatArray:
        _, _, _, scores = (
            self._metric_arrays(
                window=self.window
            )
        )

        valid = np.isfinite(
            scores
        )

        if not np.any(valid):
            return self._equal_weights()

        weights = np.zeros(
            len(self.strategies),
            dtype=np.float64,
        )

        valid_scores = scores[
            valid
        ]

        scaled = (
            valid_scores
            / self.temperature
        )

        # Numerical stability.
        scaled -= np.max(
            scaled
        )

        exponentials = np.exp(
            scaled
        )

        weights[valid] = (
            exponentials
            / np.sum(exponentials)
        )

        return weights
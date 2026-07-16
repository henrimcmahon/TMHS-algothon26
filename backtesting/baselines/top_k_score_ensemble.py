from typing import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray as FloatArray

from .meta_strategy import MetaStrategy, BaselineStrategy


class TopKScoreEnsemble(MetaStrategy):
    """
    Equally weight the top-k strategies by historical score.
    """

    def __init__(
        self,
        strategies: Sequence[BaselineStrategy],
        *,
        top_k: int = 2,
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

        if not (
            1
            <= top_k
            <= len(strategies)
        ):
            raise ValueError(
                "top_k must be between 1 and the "
                "number of component strategies"
            )

        if window is not None and window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        self.top_k = top_k
        self.window = window

    @property
    def name(self) -> str:
        prefix = (
            f"{self.window}-day"
            if self.window is not None
            else "Expanding"
        )

        return (
            f"{prefix} top-{self.top_k} "
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

        valid_indices = np.flatnonzero(
            np.isfinite(scores)
        )

        if len(valid_indices) == 0:
            return self._equal_weights()

        count = min(
            self.top_k,
            len(valid_indices),
        )

        ordered = valid_indices[
            np.argsort(
                scores[valid_indices]
            )
        ]

        selected = ordered[
            -count:
        ]

        weights = np.zeros(
            len(self.strategies),
            dtype=np.float64,
        )

        weights[selected] = (
            1.0 / count
        )

        return weights
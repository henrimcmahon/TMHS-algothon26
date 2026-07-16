from typing import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray as FloatArray

from .meta_strategy import MetaStrategy


class PositiveScoreEnsemble(MetaStrategy):
    """
    Equally weight every strategy whose historical score is positive.
    """

    @property
    def name(self) -> str:
        return "Positive-score ensemble"

    def _calculate_weights(
        self,
    ) -> FloatArray:
        _, _, _, scores = (
            self._metric_arrays()
        )

        selected = (
            np.isfinite(scores)
            & (scores > 0)
        )

        if not np.any(selected):
            return np.zeros(
                len(self.strategies),
                dtype=np.float64,
            )

        weights = selected.astype(
            np.float64
        )

        return weights / np.sum(
            weights
        )
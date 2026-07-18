from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


DiscretisationMethod = Literal["quantile", "volatility"]


@dataclass(frozen=True)
class ReturnDiscretiser:
    """
    Convert continuous returns into three discrete states:

        -1: down
         0: neutral
         1: up

    Quantile mode gives approximately balanced state frequencies.
    Volatility mode defines neutral moves relative to each ticker's volatility.
    """

    method: DiscretisationMethod = "quantile"
    lower_quantile: float = 1.0 / 3.0
    upper_quantile: float = 2.0 / 3.0
    volatility_threshold: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.lower_quantile < self.upper_quantile < 1.0:
            raise ValueError(
                "Expected 0 < lower_quantile < upper_quantile < 1."
            )

        if self.volatility_threshold < 0.0:
            raise ValueError("volatility_threshold must be non-negative.")

    def transform(
        self,
        returns: NDArray[np.floating],
    ) -> NDArray[np.int8]:
        """
        Parameters
        ----------
        returns:
            Matrix shaped (n_tickers, n_observations).

        Returns
        -------
        NDArray[np.int8]
            State matrix with the same shape as returns.
        """
        values = np.asarray(returns, dtype=float)

        if values.ndim != 2:
            raise ValueError(
                "returns must have shape (n_tickers, n_observations)."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("returns contains non-finite values.")

        if self.method == "quantile":
            return self._quantile_transform(values)

        if self.method == "volatility":
            return self._volatility_transform(values)

        raise ValueError(f"Unknown discretisation method: {self.method}")

    def _quantile_transform(
        self,
        returns: NDArray[np.float64],
    ) -> NDArray[np.int8]:
        lower = np.quantile(
            returns,
            self.lower_quantile,
            axis=1,
            keepdims=True,
        )
        upper = np.quantile(
            returns,
            self.upper_quantile,
            axis=1,
            keepdims=True,
        )

        states = np.zeros(returns.shape, dtype=np.int8)
        states[returns < lower] = -1
        states[returns > upper] = 1

        return states

    def _volatility_transform(
        self,
        returns: NDArray[np.float64],
    ) -> NDArray[np.int8]:
        volatility = np.std(returns, axis=1, ddof=1, keepdims=True)
        volatility = np.where(volatility > 0.0, volatility, 1.0)

        boundary = self.volatility_threshold * volatility

        states = np.zeros(returns.shape, dtype=np.int8)
        states[returns < -boundary] = -1
        states[returns > boundary] = 1

        return states
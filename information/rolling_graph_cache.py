from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from information import ReturnDiscretiser, mutual_information_matrix
from information.discretiser import DiscretisationMethod

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class InformationMatrixKey:
    """
    Identifies one rolling information matrix.

    day_count is the number of price observations currently available.
    """

    day_count: int
    window: int
    lag: int
    method: str
    bias_correct: bool


@dataclass(frozen=True, slots=True)
class FilteredGraphKey:
    """
    Identifies one filtered graph derived from an information matrix.
    """

    matrix_key: InformationMatrixKey
    percentile: float
    top_k: int


class RollingInformationGraphCache:
    """
    Cache rolling mutual-information matrices and filtered graphs.

    The expensive MI matrix depends on:

        current day
        rolling window
        lag
        discretisation method
        bias correction

    It does not depend on the strategy's trading interpretation, so the same
    matrix can be reused by LeaderFollowerStrategy, centrality strategies,
    diffusion strategies, and parameter variants.
    """

    def __init__(self) -> None:
        self._matrix_cache: dict[
            InformationMatrixKey,
            FloatArray,
        ] = {}

        self._graph_cache: dict[
            FilteredGraphKey,
            FloatArray,
        ] = {}

        self.matrix_hits = 0
        self.matrix_misses = 0

        self.graph_hits = 0
        self.graph_misses = 0

    def get_information_matrix(
        self,
        price_history: FloatArray,
        *,
        window: int,
        lag: int,
        method: DiscretisationMethod,
        bias_correct: bool,
    ) -> FloatArray:
        """
        Return the rolling lagged-MI matrix for the current day.
        """
        prices = np.asarray(
            price_history,
            dtype=np.float64,
        )

        if prices.ndim != 2:
            raise ValueError(
                "price_history must have shape "
                "(n_tickers, n_days)."
            )

        if window < 2:
            raise ValueError("window must be at least 2.")

        if lag < 0:
            raise ValueError("lag cannot be negative.")

        required_price_days = window + 1

        if prices.shape[1] < required_price_days:
            raise ValueError(
                f"At least {required_price_days} price days are "
                f"required for window={window}."
            )

        key = InformationMatrixKey(
            day_count=prices.shape[1],
            window=window,
            lag=lag,
            method=str(method),
            bias_correct=bias_correct,
        )

        cached = self._matrix_cache.get(key)

        if cached is not None:
            self.matrix_hits += 1
            return cached

        self.matrix_misses += 1

        # Only calculate returns for the required rolling window rather than
        # recomputing returns over the complete price history.
        window_prices = prices[
            :,
            -required_price_days:,
        ]

        if (
            np.any(window_prices <= 0.0)
            or not np.all(np.isfinite(window_prices))
        ):
            raise ValueError(
                "Prices must be finite and strictly positive."
            )

        window_returns = np.diff(
            np.log(window_prices),
            axis=1,
        )

        discretiser = ReturnDiscretiser(
            method=method,
        )

        states = discretiser.transform(
            window_returns
        )

        matrix = np.asarray(
            mutual_information_matrix(
                states,
                lag=lag,
                normalised=False,
            ),
            dtype=np.float64,
        )

        np.fill_diagonal(matrix, 0.0)

        matrix = np.where(
            np.isfinite(matrix),
            matrix,
            0.0,
        )

        matrix = np.maximum(
            matrix,
            0.0,
        )

        if bias_correct:
            effective_n = max(
                states.shape[1] - lag,
                1,
            )

            # Approximate finite-sample bias for two three-state variables.
            bias = 4.0 / (
                2.0
                * effective_n
                * np.log(2.0)
            )

            matrix = np.maximum(
                matrix - bias,
                0.0,
            )

        # Prevent accidental mutation of cached values.
        matrix.setflags(write=False)

        self._matrix_cache[key] = matrix

        return matrix

    def get_filtered_graph(
        self,
        price_history: FloatArray,
        *,
        window: int,
        lag: int,
        method: DiscretisationMethod,
        percentile: float,
        top_k: int,
        bias_correct: bool,
    ) -> FloatArray:
        """
        Return a thresholded, top-k incoming information graph.
        """
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(
                "percentile must be between 0 and 100."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        matrix_key = InformationMatrixKey(
            day_count=np.asarray(
                price_history
            ).shape[1],
            window=window,
            lag=lag,
            method=str(method),
            bias_correct=bias_correct,
        )

        graph_key = FilteredGraphKey(
            matrix_key=matrix_key,
            percentile=round(
                float(percentile),
                8,
            ),
            top_k=top_k,
        )

        cached = self._graph_cache.get(
            graph_key
        )

        if cached is not None:
            self.graph_hits += 1
            return cached

        self.graph_misses += 1

        matrix = self.get_information_matrix(
            price_history,
            window=window,
            lag=lag,
            method=method,
            bias_correct=bias_correct,
        )

        positive = matrix[
            matrix > 0.0
        ]

        if positive.size == 0:
            graph = np.zeros_like(matrix)
            graph.setflags(write=False)

            self._graph_cache[
                graph_key
            ] = graph

            return graph

        threshold = float(
            np.percentile(
                positive,
                percentile,
            )
        )

        graph = np.zeros_like(matrix)

        # Keep only the strongest incoming leaders for each target.
        for target in range(
            matrix.shape[1]
        ):
            column = matrix[
                :,
                target,
            ].copy()

            column[target] = 0.0

            candidates = np.flatnonzero(
                column >= threshold
            )

            if candidates.size > top_k:
                ordering = np.argsort(
                    column[candidates]
                )[::-1]

                candidates = candidates[
                    ordering[:top_k]
                ]

            graph[
                candidates,
                target,
            ] = column[candidates]

        graph.setflags(write=False)

        self._graph_cache[
            graph_key
        ] = graph

        return graph

    def clear(self) -> None:
        self._matrix_cache.clear()
        self._graph_cache.clear()

        self.matrix_hits = 0
        self.matrix_misses = 0
        self.graph_hits = 0
        self.graph_misses = 0

    @property
    def matrix_count(self) -> int:
        return len(
            self._matrix_cache
        )

    @property
    def graph_count(self) -> int:
        return len(
            self._graph_cache
        )

    def statistics(self) -> dict[str, int | float]:
        matrix_requests = (
            self.matrix_hits
            + self.matrix_misses
        )

        graph_requests = (
            self.graph_hits
            + self.graph_misses
        )

        matrix_hit_rate = (
            self.matrix_hits
            / matrix_requests
            if matrix_requests > 0
            else 0.0
        )

        graph_hit_rate = (
            self.graph_hits
            / graph_requests
            if graph_requests > 0
            else 0.0
        )

        return {
            "Matrices Cached": self.matrix_count,
            "Filtered Graphs Cached": self.graph_count,
            "Matrix Hits": self.matrix_hits,
            "Matrix Misses": self.matrix_misses,
            "Matrix Hit Rate": matrix_hit_rate,
            "Graph Hits": self.graph_hits,
            "Graph Misses": self.graph_misses,
            "Graph Hit Rate": graph_hit_rate,
        }
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from backtesting.baselines import BaselineStrategy

from information.rolling_graph_cache import (
    RollingInformationGraphCache,
)
from information.discretiser import DiscretisationMethod

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

from abc import ABC, abstractmethod

SignalVersion: TypeAlias = Literal[
    "raw",
    "tanh",
    "weighted",
    "latest_raw",
]

def _safe_log_returns(prices: FloatArray) -> FloatArray:
    values = np.asarray(prices, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError("prices must have shape (n_tickers, n_days).")

    if values.shape[1] < 2:
        return np.empty((values.shape[0], 0), dtype=np.float64)

    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("prices must be finite and strictly positive.")

    return np.diff(np.log(values), axis=1)


def _cross_sectional_zscore(values: FloatArray) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(vector))
    std = float(np.std(vector))

    if std < 1e-12:
        return np.zeros_like(vector)

    return (vector - mean) / std


def _normalise_signal(signal: FloatArray) -> FloatArray:
    vector = np.asarray(signal, dtype=np.float64)
    vector = np.where(np.isfinite(vector), vector, 0.0)
    vector -= float(np.mean(vector))

    gross = float(np.sum(np.abs(vector)))

    if gross < 1e-12:
        return np.zeros_like(vector)

    return vector / gross


def _positions_from_signal(
    signal: FloatArray,
    latest_prices: FloatArray,
    *,
    gross_notional: float,
    max_notional_per_ticker: float,
) -> IntArray:
    weights = _normalise_signal(signal)

    target_notional = weights * gross_notional
    target_notional = np.clip(
        target_notional,
        -max_notional_per_ticker,
        max_notional_per_ticker,
    )

    positions = np.divide(
        target_notional,
        latest_prices,
        out=np.zeros_like(target_notional),
        where=latest_prices > 0.0,
    )

    return np.rint(positions).astype(np.int64)

@dataclass
@dataclass
class InformationStrategy(BaselineStrategy, ABC):
    window: int = 150
    lag: int = 1
    method: DiscretisationMethod = "volatility"
    percentile: float = 95.0
    top_k: int = 3
    bias_correct: bool = True

    gross_notional: float = 100_000.0
    max_notional_per_ticker: float = 10_000.0
    minimum_days: int = 170
    rebalance_interval: int = 1

    forecast_threshold: float = 0.5
    time_decay: float = 0.6

    graph_cache: RollingInformationGraphCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    _current_positions: IntArray | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _last_rebalance_day: int = field(
        default=-1,
        init=False,
        repr=False,
    )

    def _get_graph_cache(
        self,
    ) -> RollingInformationGraphCache:
        if self.graph_cache is None:
            self.graph_cache = RollingInformationGraphCache()

        return self.graph_cache

    @abstractmethod
    def compute_signal(
        self,
        graph: FloatArray,
        returns: FloatArray,
    ) -> FloatArray:
        raise NotImplementedError

    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        values = np.asarray(
            price_history,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "price_history must have shape "
                "(n_tickers, n_days)."
            )

        n_tickers, n_days = values.shape

        if self.rebalance_interval < 1:
            raise ValueError(
                "rebalance_interval must be at least 1."
            )

        if (
            self._current_positions is not None
            and self._last_rebalance_day >= 0
            and (
                n_days
                - self._last_rebalance_day
            ) < self.rebalance_interval
        ):
            return self._current_positions.copy()

        required_days = max(
            self.minimum_days,
            self.window + self.lag + 1,
            self.required_signal_days,
        )

        if n_days < required_days:
            positions = np.zeros(
                n_tickers,
                dtype=np.int64,
            )

            self._current_positions = positions
            return positions

        returns = _safe_log_returns(values)

        cache = self._get_graph_cache()

        graph = cache.get_filtered_graph(
            values,
            window=self.window,
            lag=self.lag,
            method=self.method,
            percentile=self.percentile,
            top_k=self.top_k,
            bias_correct=self.bias_correct,
        )

        signal = np.asarray(
            self.compute_signal(
                graph,
                returns,
            ),
            dtype=np.float64,
        )

        if signal.shape != (n_tickers,):
            raise ValueError(
                "compute_signal must return shape "
                f"({n_tickers},), got {signal.shape}."
            )

        positions = _positions_from_signal(
            signal,
            values[:, -1],
            gross_notional=self.gross_notional,
            max_notional_per_ticker=(
                self.max_notional_per_ticker
            ),
        )

        self._current_positions = positions
        self._last_rebalance_day = n_days

        return positions.copy()

    @property
    def required_signal_days(self) -> int:
        return 2

    def getMyPosition(
        self,
        prcSoFar: FloatArray,
    ) -> IntArray:
        return self.get_positions(prcSoFar)
    
@dataclass
class LeaderFollowerStrategy(
    InformationStrategy
):
    signal_lookback: int = 2
    self_move_penalty: float = 0.15
    normalise_incoming_weights: bool = True

    signal_version: SignalVersion = "raw"

    @property
    def required_signal_days(self) -> int:
        return self.signal_lookback + 1

    @property
    def name(self) -> str:
        return (
            "LeaderFollower"
            f"_V{self.signal_version}"
            f"_W{self.window}"
            f"_L{self.lag}"
            f"_K{self.top_k}"
            f"_P{self.percentile:g}"
            f"_S{self.signal_lookback}"
            f"_R{self.rebalance_interval}"
            f"_M{self.self_move_penalty:.3f}"
            f"_D{self.time_decay:.2f}"
            f"_T{self.forecast_threshold:.2f}"
        )

    def compute_signal(
        self,
        graph: FloatArray,
        returns: FloatArray,
    ) -> FloatArray:
        working_graph = np.asarray(
            graph,
            dtype=np.float64,
        )

        if working_graph.ndim != 2:
            raise ValueError(
                "graph must have shape "
                "(n_tickers, n_tickers)."
            )

        if returns.ndim != 2:
            raise ValueError(
                "returns must have shape "
                "(n_tickers, n_days)."
            )

        if returns.shape[1] < self.signal_lookback:
            return np.zeros(
                returns.shape[0],
                dtype=np.float64,
            )

        if self.normalise_incoming_weights:
            incoming_weight = np.sum(
                working_graph,
                axis=0,
            )

            working_graph = np.divide(
                working_graph,
                incoming_weight[np.newaxis, :],
                out=np.zeros_like(working_graph),
                where=(
                    incoming_weight[np.newaxis, :]
                    > 1e-12
                ),
            )

        recent_returns = returns[
            :,
            -self.signal_lookback:,
        ]

        #
        # ===== EXACT RECOVERED LOGIC =====
        #
        if self.signal_version == "raw":

            # These two lines intentionally remain,
            # even though the weights are unused.
            time_weights = (
                self.time_decay
                ** np.arange(
                    self.signal_lookback - 1,
                    -1,
                    -1,
                    dtype=np.float64,
                )
            )

            time_weights /= np.sum(time_weights)

            recent_move = np.sum(
                recent_returns,
                axis=1,
            )

        #
        # ===== EXPERIMENTS =====
        #
        elif self.signal_version == "latest_raw":

            recent_move = returns[:, -1]

        elif self.signal_version == "weighted":

            time_weights = (
                self.time_decay
                ** np.arange(
                    self.signal_lookback - 1,
                    -1,
                    -1,
                    dtype=np.float64,
                )
            )

            time_weights /= np.sum(time_weights)

            recent_move = np.sum(
                recent_returns
                * time_weights[np.newaxis, :],
                axis=1,
            )

        elif self.signal_version == "tanh":

            recent_move = np.sum(
                recent_returns,
                axis=1,
            )

        else:
            raise ValueError(
                f"Unknown signal version: "
                f"{self.signal_version}"
            )

        source_move = _cross_sectional_zscore(
            recent_move
        )

        forecast = (
            working_graph.T
            @ source_move
        )

        forecast -= (
            self.self_move_penalty
            * source_move
        )

        forecast = np.where(
            np.isfinite(forecast),
            forecast,
            0.0,
        )

        if self.signal_version == "tanh":
            forecast = np.tanh(forecast)

        return forecast
    
@dataclass
class InformationCentralityMomentumStrategy(
    InformationStrategy
):
    momentum_lookback: int = 3
    centrality_power: float = 1.5

    @property
    def required_signal_days(self) -> int:
        return self.momentum_lookback + 1

    @property
    def name(self) -> str:
        return (
            "Centrality"
            f"_W{self.window}"
            f"_M{self.momentum_lookback}"
            f"_K{self.top_k}"
            f"_L{self.lag}"
            f"_R{self.rebalance_interval}"
        )
    def compute_signal(
        self,
        graph: FloatArray,
        returns: FloatArray,
    ) -> FloatArray:
        outgoing_strength = np.sum(
            graph,
            axis=1,
        )

        maximum = float(
            np.max(outgoing_strength)
        )

        if maximum < 1e-12:
            return np.zeros_like(
                outgoing_strength
            )

        centrality = (
            outgoing_strength
            / maximum
        ) ** self.centrality_power

        recent_move = np.sum(
            returns[
                :,
                -self.momentum_lookback:
            ],
            axis=1,
        )

        momentum = _cross_sectional_zscore(
            recent_move
        )

        return centrality * momentum
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from models.market_constraints import MarketConstraints
from strategies.allocators.position_allocator import PositionAllocator
from strategies.signals.signal_model import SignalModel


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class Strategy:
    name: str
    signal_model: SignalModel
    position_allocator: PositionAllocator
    rebalance_interval: int = 1

    current_positions: IntArray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        init=False,
        repr=False,
    )
    last_rebalance_day: int = field(
        default=-1,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.rebalance_interval < 1:
            raise ValueError("rebalance_interval must be at least 1")

    def get_positions(
        self,
        price_history: FloatArray,
        constraints: MarketConstraints,
    ) -> IntArray:
        if price_history.ndim != 2:
            raise ValueError(
                "price_history must have shape (n_tickers, n_days)"
            )

        n_tickers, n_days = price_history.shape

        if n_days == 0:
            raise ValueError("price_history must contain at least one day")

        if self.current_positions.shape != (n_tickers,):
            self.current_positions = np.zeros(
                n_tickers,
                dtype=np.int64,
            )

        should_rebalance = (
            self.last_rebalance_day < 0
            or n_days - self.last_rebalance_day
            >= self.rebalance_interval
        )

        if not should_rebalance:
            return self.current_positions.copy()

        signal = self.signal_model.compute(price_history)

        if signal.shape != (n_tickers,):
            raise ValueError(
                "signal model must return shape (n_tickers,)"
            )

        target_positions = self.position_allocator.allocate(
            signal=signal,
            latest_prices=price_history[:, -1],
            current_positions=self.current_positions,
            constraints=constraints,
        )

        if target_positions.shape != (n_tickers,):
            raise ValueError(
                "position allocator must return shape (n_tickers,)"
            )

        self.current_positions = np.asarray(
            target_positions,
            dtype=np.int64,
        )
        self.last_rebalance_day = n_days

        return self.current_positions.copy()

    def reset(self) -> None:
        self.current_positions = np.empty(
            0,
            dtype=np.int64,
        )
        self.last_rebalance_day = -1

from __future__ import annotations

import numpy as np
import pytest

from models.market_constraints import MarketConstraints
from strategies.allocators.equal_notional_allocator import (
    EqualNotionalAllocator,
)


def test_signal_allocates_fraction_of_each_tickers_position_limit() -> None:
    constraints = MarketConstraints(
        position_limits=np.asarray(
            [100_000.0, 10_000.0, 10_000.0, 10_000.0]
        ),
        commission_rates=np.asarray([0.00002, 0.0001, 0.0001, 0.0001]),
    )
    allocator = EqualNotionalAllocator()

    positions = allocator.allocate(
        signal=np.asarray([1.0, -1.0, 0.5, 0.0]),
        latest_prices=np.asarray([100.0, 20.0, 25.0, 50.0]),
        current_positions=np.zeros(4, dtype=np.int64),
        constraints=constraints,
    )

    assert np.array_equal(positions, np.asarray([1_000, -500, 200, 0]))


def test_signal_is_clipped_to_full_long_or_short() -> None:
    constraints = MarketConstraints(
        position_limits=np.asarray([10_000.0, 10_000.0]),
        commission_rates=np.zeros(2),
    )

    positions = EqualNotionalAllocator().allocate(
        signal=np.asarray([2.0, -3.0]),
        latest_prices=np.asarray([100.0, 50.0]),
        current_positions=np.zeros(2, dtype=np.int64),
        constraints=constraints,
    )

    assert np.array_equal(positions, np.asarray([100, -200]))


def test_allocator_rejects_non_finite_signal() -> None:
    constraints = MarketConstraints(
        position_limits=np.asarray([10_000.0]),
        commission_rates=np.asarray([0.0001]),
    )

    with pytest.raises(ValueError, match="non-finite"):
        EqualNotionalAllocator().allocate(
            signal=np.asarray([np.nan]),
            latest_prices=np.asarray([100.0]),
            current_positions=np.zeros(1, dtype=np.int64),
            constraints=constraints,
        )

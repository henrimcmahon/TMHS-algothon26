from __future__ import annotations

import numpy as np
import pytest

from models.market_constraints import MarketConstraints


def make_constraints() -> MarketConstraints:
    return MarketConstraints(
        position_limits=np.asarray([100_000.0, 10_000.0]),
        commission_rates=np.asarray([0.00002, 0.0001]),
    )


def test_position_limits_are_converted_to_daily_share_limits() -> None:
    constraints = make_constraints()

    limits = constraints.position_limits_in_shares(
        np.asarray([125.0, 25.0])
    )

    assert np.array_equal(limits, np.asarray([800, 400]))


def test_positions_are_clipped_long_and_short() -> None:
    constraints = make_constraints()

    positions = constraints.clip_positions(
        np.asarray([2_000, -1_000]),
        np.asarray([100.0, 20.0]),
    )

    assert np.array_equal(positions, np.asarray([1_000, -500]))


def test_trade_costs_use_per_ticker_commission_rates() -> None:
    constraints = make_constraints()

    notionals, turnover, commission = constraints.trade_costs(
        trades=np.asarray([100, -50]),
        prices=np.asarray([100.0, 20.0]),
    )

    assert np.array_equal(notionals, np.asarray([10_000.0, 1_000.0]))
    assert turnover == 11_000.0
    assert commission == pytest.approx(0.3)

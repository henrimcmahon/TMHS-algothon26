import numpy as np

from portfolio import PortfolioAllocator


def test_algothon_defaults_have_correct_limits() -> None:
    allocator = (
        PortfolioAllocator.algothon_defaults()
    )

    assert allocator.position_limits.shape == (
        51,
    )

    assert allocator.position_limits[0] == (
        100_000.0
    )

    assert np.all(
        allocator.position_limits[1:]
        == 10_000.0
    )

    assert allocator.commission_rates[0] == (
        0.00002
    )

    assert np.all(
        allocator.commission_rates[1:]
        == 0.0001
    )


def test_allocator_returns_exactly_51_integers() -> None:
    allocator = (
        PortfolioAllocator.algothon_defaults()
    )

    signals = np.linspace(
        -2.0,
        2.0,
        51,
        dtype=np.float64,
    )

    prices = np.full(
        51,
        100.0,
        dtype=np.float64,
    )

    result = allocator.allocate(
        signals=signals,
        prices=prices,
    )

    assert result.positions.shape == (51,)

    assert np.issubdtype(
        result.positions.dtype,
        np.integer,
    )


def test_allocator_respects_all_position_limits() -> None:
    allocator = (
        PortfolioAllocator.algothon_defaults(
            signal_scale=1.0,
        )
    )

    signals = np.full(
        51,
        1_000.0,
        dtype=np.float64,
    )

    prices = np.linspace(
        10.0,
        200.0,
        51,
        dtype=np.float64,
    )

    result = allocator.allocate(
        signals=signals,
        prices=prices,
    )

    dollar_positions = (
        np.abs(result.positions)
        * prices
    )

    assert np.all(
        dollar_positions
        <= allocator.position_limits
    )


def test_allocator_respects_short_limits() -> None:
    allocator = (
        PortfolioAllocator.algothon_defaults(
            signal_scale=1.0,
        )
    )

    signals = np.full(
        51,
        -1_000.0,
        dtype=np.float64,
    )

    prices = np.full(
        51,
        75.0,
        dtype=np.float64,
    )

    result = allocator.allocate(
        signals=signals,
        prices=prices,
    )

    assert np.all(
        result.positions <= 0
    )

    assert np.all(
        np.abs(result.positions) * prices
        <= allocator.position_limits
    )


def test_price_rise_forces_position_reduction() -> None:
    allocator = (
        PortfolioAllocator.algothon_defaults(
            minimum_trade_dollars=10_000.0,
        )
    )

    current_positions = np.zeros(
        51,
        dtype=np.int64,
    )

    # Previously legal: 1,000 shares × $100 = $100,000.
    current_positions[0] = 1_000

    prices = np.full(
        51,
        100.0,
        dtype=np.float64,
    )

    # ALGO rises to $125. Its new ceiling is:
    # floor(100,000 / 125) = 800 shares.
    prices[0] = 125.0

    signals = np.zeros(
        51,
        dtype=np.float64,
    )

    result = allocator.allocate(
        signals=signals,
        prices=prices,
        current_positions=current_positions,
    )

    assert result.positions[0] <= 800

    assert (
        abs(result.positions[0])
        * prices[0]
        <= 100_000.0
    )


def test_commission_is_calculated_on_dollars_traded() -> None:
    allocator = (
        PortfolioAllocator.algothon_defaults(
            minimum_trade_dollars=0.0,
        )
    )

    current_positions = np.zeros(
        51,
        dtype=np.int64,
    )

    prices = np.full(
        51,
        100.0,
        dtype=np.float64,
    )

    signals = np.zeros(
        51,
        dtype=np.float64,
    )

    signals[0] = 1_000.0
    signals[1] = 1_000.0

    result = allocator.allocate(
        signals=signals,
        prices=prices,
        current_positions=current_positions,
    )

    expected_algo_commission = (
        result.dollar_volume_traded[0]
        * 0.00002
    )

    expected_asset_commission = (
        result.dollar_volume_traded[1]
        * 0.0001
    )

    assert np.isclose(
        result.commissions[0],
        expected_algo_commission,
    )

    assert np.isclose(
        result.commissions[1],
        expected_asset_commission,
    )


def test_minimum_trade_filter_reduces_small_turnover() -> None:
    allocator = (
        PortfolioAllocator.algothon_defaults(
            signal_scale=10.0,
            minimum_trade_dollars=500.0,
        )
    )

    current_positions = np.zeros(
        51,
        dtype=np.int64,
    )

    prices = np.full(
        51,
        100.0,
        dtype=np.float64,
    )

    tiny_signals = np.full(
        51,
        0.001,
        dtype=np.float64,
    )

    result = allocator.allocate(
        signals=tiny_signals,
        prices=prices,
        current_positions=current_positions,
    )

    assert np.array_equal(
        result.positions,
        current_positions,
    )
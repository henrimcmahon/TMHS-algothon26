import numpy as np

from models.ticker_universe import TickerUniverse
from trading import TradingEngine


def load_prices() -> np.ndarray:
    universe = TickerUniverse(
        "prices.txt"
    )

    return np.asarray(
        universe.as_price_matrix(),
        dtype=np.float64,
    )


def test_engine_returns_51_integer_positions() -> None:
    prices = load_prices()

    engine = TradingEngine(
        residual_estimation_window=60,
    )

    result = engine.update(
        prices[:, :100]
    )

    assert result.positions.shape == (51,)

    assert np.issubdtype(
        result.positions.dtype,
        np.integer,
    )


def test_engine_respects_daily_position_limits() -> None:
    prices = load_prices()

    engine = TradingEngine(
        residual_estimation_window=60,
    )

    result = engine.update(
        prices[:, :100]
    )

    latest_prices = prices[:, 99]

    dollar_positions = (
        np.abs(result.positions)
        * latest_prices
    )

    assert dollar_positions[0] <= 100_000.0

    assert np.all(
        dollar_positions[1:]
        <= 10_000.0
    )


def test_engine_preserves_state_across_days() -> None:
    prices = load_prices()

    engine = TradingEngine(
        residual_estimation_window=60,
    )

    first = engine.update(
        prices[:, :100]
    )

    second = engine.update(
        prices[:, :101]
    )

    assert first.end == 99
    assert second.end == 100

    assert engine.is_initialised


def test_engine_returns_zero_before_enough_history() -> None:
    prices = load_prices()

    engine = TradingEngine(
        residual_estimation_window=60,
    )

    result = engine.update(
        prices[:, :20]
    )

    assert np.array_equal(
        result.positions,
        np.zeros(
            51,
            dtype=np.int64,
        ),
    )


def test_engine_tracks_commissions_and_volume() -> None:
    prices = load_prices()

    engine = TradingEngine(
        residual_estimation_window=60,
    )

    result = engine.update(
        prices[:, :100]
    )

    assert (
        result.allocation.total_dollar_volume
        >= 0
    )

    assert (
        result.allocation.total_commission
        >= 0
    )


def test_engine_rejects_backward_history() -> None:
    prices = load_prices()

    engine = TradingEngine(
        residual_estimation_window=60,
    )

    engine.update(
        prices[:, :100]
    )

    try:
        engine.update(
            prices[:, :99]
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected backward history to fail"
        )
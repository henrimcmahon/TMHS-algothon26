import numpy as np

from backtesting import Backtester
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


def create_backtester() -> Backtester:
    engine = TradingEngine(
        residual_estimation_window=60,
        forgetting_rate=0.03,
        entropy_penalty=0.5,
        signal_scale=5.0,
        minimum_trade_dollars=100.0,
        turnover_smoothing=0.25,
    )

    return Backtester(engine)


def test_backtester_produces_one_pnl_per_interval() -> None:
    prices = load_prices()

    result = create_backtester().run(
        prices[:, :100]
    )

    assert result.daily_pnl.shape == (99,)
    assert result.positions.shape == (99, 51)


def test_backtester_net_pnl_equals_gross_minus_commission() -> None:
    prices = load_prices()

    result = create_backtester().run(
        prices[:, :100]
    )

    assert np.allclose(
        result.daily_pnl,
        (
            result.gross_daily_pnl
            - result.daily_commissions
        ),
    )


def test_backtester_summary_metrics_are_finite() -> None:
    prices = load_prices()

    result = create_backtester().run(
        prices[:, :100]
    )

    values = np.asarray(
        [
            result.total_pnl,
            result.total_commission,
            result.total_turnover,
            result.mean_daily_pnl,
            result.daily_pnl_std,
            result.annualised_sharpe,
            result.score,
            result.maximum_drawdown,
            result.profitable_day_fraction,
        ],
        dtype=np.float64,
    )

    assert np.all(np.isfinite(values))


def test_backtester_meets_position_limits() -> None:
    prices = load_prices()

    result = create_backtester().run(
        prices[:, :100]
    )

    for index, day in enumerate(
        range(0, 99)
    ):
        dollar_positions = (
            np.abs(result.positions[index])
            * prices[:, day]
        )

        assert dollar_positions[0] <= 100_000.0

        assert np.all(
            dollar_positions[1:]
            <= 10_000.0
        )


def test_backtester_activity_matches_turnover() -> None:
    prices = load_prices()

    result = create_backtester().run(
        prices[:, :100]
    )

    assert result.total_turnover == (
        np.sum(result.daily_turnover)
    )

    assert result.total_turnover >= 0
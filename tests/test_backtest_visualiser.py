import matplotlib.pyplot as plt
import numpy as np

from backtesting import BacktestResult
from visualisation import BacktestVisualiser


def create_result() -> BacktestResult:
    daily_pnl = np.asarray(
        [10.0, -5.0, 20.0, -2.0],
        dtype=np.float64,
    )

    commissions = np.asarray(
        [1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    gross = daily_pnl + commissions

    return BacktestResult(
        daily_pnl=daily_pnl,
        gross_daily_pnl=gross,
        daily_commissions=commissions,
        daily_turnover=np.full(
            4,
            10_000.0,
            dtype=np.float64,
        ),
        positions=np.zeros(
            (4, 51),
            dtype=np.int64,
        ),
        cumulative_pnl=np.cumsum(
            daily_pnl
        ),
        total_pnl=float(
            np.sum(daily_pnl)
        ),
        total_commission=4.0,
        total_turnover=40_000.0,
        mean_daily_pnl=float(
            np.mean(daily_pnl)
        ),
        daily_pnl_std=float(
            np.std(daily_pnl, ddof=1)
        ),
        annualised_sharpe=0.0,
        score=0.0,
        maximum_drawdown=-5.0,
        profitable_day_fraction=0.5,
        start_day=0,
        end_day=3,
    )


def test_visualiser_constructs() -> None:
    visualiser = BacktestVisualiser(
        {
            "Test strategy": create_result(),
        }
    )

    assert len(visualiser.days) == 4


def test_rolling_mean_shape() -> None:
    values = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    rolling = (
        BacktestVisualiser
        ._rolling_mean(
            values,
            window=2,
        )
    )

    assert rolling.shape == values.shape

    assert np.allclose(
        rolling[1:],
        [1.5, 2.5, 3.5],
    )

    plt.close("all")
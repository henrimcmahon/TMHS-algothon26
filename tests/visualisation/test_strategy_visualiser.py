from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from backtesting.backtest_result import BacktestResult
from models.ticker_universe import TickerUniverse
from visualisation.strategy_visualiser import StrategyVisualiser


def make_visualiser(
    *,
    positions: np.ndarray | None = None,
    daily_pnl: np.ndarray | None = None,
    commissions: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
) -> StrategyVisualiser:
    prices = np.asarray(
        [
            [10.0, 11.0, 10.5, 12.0, 12.5],
            [20.0, 19.0, 21.0, 20.5, 22.0],
        ],
        dtype=np.float64,
    )
    n_tickers, n_days = prices.shape
    positions = (
        np.asarray(positions, dtype=np.int64)
        if positions is not None
        else np.asarray(
            [[0, 0], [2, -1], [2, -1], [3, 0], [3, 1]],
            dtype=np.int64,
        )
    )
    daily_pnl = (
        np.asarray(daily_pnl, dtype=np.float64)
        if daily_pnl is not None
        else np.asarray([0.0, 1.0, -0.5, 3.0, 1.5], dtype=np.float64)
    )
    commissions = (
        np.asarray(commissions, dtype=np.float64)
        if commissions is not None
        else np.asarray([0.0, 0.2, 0.0, 0.1, 0.1], dtype=np.float64)
    )
    turnover = (
        np.asarray(turnover, dtype=np.float64)
        if turnover is not None
        else np.asarray([0.0, 41.0, 0.0, 12.0, 22.0], dtype=np.float64)
    )
    cumulative_pnl = np.cumsum(daily_pnl)
    scoring_start_day = 1
    scored = daily_pnl[scoring_start_day:]

    universe = TickerUniverse.from_algothon(
        prices=prices,
        symbols=["AAA", "BBB"],
        position_limits=np.full(n_tickers, 10_000.0),
        commission_rates=np.full(n_tickers, 0.001),
    )
    result = BacktestResult(
        name="Test Strategy",
        daily_pnl=daily_pnl,
        cumulative_pnl=cumulative_pnl,
        gross_daily_pnl=daily_pnl + commissions,
        daily_commissions=commissions,
        daily_turnover=turnover,
        positions=positions,
        trades=np.zeros((n_days, n_tickers), dtype=np.int64),
        portfolio_values=cumulative_pnl.copy(),
        cash_history=np.zeros(n_days),
        total_pnl=float(np.sum(scored)),
        total_gross_pnl=float(np.sum(scored + commissions[scoring_start_day:])),
        total_commissions=float(np.sum(commissions[scoring_start_day:])),
        mean_daily_pnl=float(np.mean(scored)),
        pnl_std=float(np.std(scored)),
        annualised_sharpe=1.2,
        sharpe_multiplier=0.8,
        score=0.7,
        maximum_drawdown=-0.5,
        total_dollar_volume=float(np.sum(turnover)),
        total_shares_traded=8.0,
        average_daily_turnover=float(np.mean(turnover[scoring_start_day:])),
        maximum_daily_turnover=float(np.max(turnover)),
        return_on_volume=0.1,
        scoring_start_day=scoring_start_day,
        scoring_days=n_days - scoring_start_day,
    )
    return StrategyVisualiser(universe=universe, result=result)


def test_constructor_rejects_invalid_position_shape() -> None:
    visualiser = make_visualiser()
    invalid_result = replace(
        visualiser.result,
        positions=np.zeros((2, 5), dtype=np.int64),
    )

    with pytest.raises(ValueError, match="result.positions"):
        StrategyVisualiser(visualiser.universe, invalid_result)


def test_constructor_rejects_invalid_price_shape() -> None:
    visualiser = make_visualiser()
    visualiser.universe.prices = np.ones((3, 5), dtype=np.float64)

    with pytest.raises(ValueError, match="universe.prices"):
        StrategyVisualiser(visualiser.universe, visualiser.result)


@pytest.mark.parametrize("identifier", [0, "AAA"])
def test_ticker_lookup_by_index_and_symbol(identifier: int | str) -> None:
    figure = make_visualiser().plot_ticker(identifier)
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_ticker_lookup_by_ticker() -> None:
    visualiser = make_visualiser()
    figure = visualiser.plot_ticker(visualiser.universe.tickers[1])
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_unknown_ticker_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown ticker"):
        make_visualiser().plot_ticker("MISSING")


def test_every_plotting_method_returns_figure() -> None:
    visualiser = make_visualiser()
    figures = [
        visualiser.plot_summary(),
        visualiser.plot_cumulative_pnl(),
        visualiser.plot_cumulative_pnl(show_gross=False),
        visualiser.plot_daily_pnl(),
        visualiser.plot_commissions(),
        visualiser.plot_turnover(),
        visualiser.plot_drawdown(),
        visualiser.plot_position_heatmap(),
        visualiser.plot_position_heatmap(normalise=True),
        visualiser.plot_ticker(0),
        visualiser.plot_exposure(),
        visualiser.plot_pnl_distribution(),
        visualiser.plot_pnl_distribution(scored_only=False),
    ]

    assert all(isinstance(figure, Figure) for figure in figures)
    dashboard = visualiser.create_dashboard()
    assert len(dashboard) == 1
    assert isinstance(dashboard[0], Figure)
    plt.close("all")


def test_dashboard_uses_subplots_and_interactive_window_slider() -> None:
    figure = make_visualiser().create_dashboard()[0]
    slider = figure._strategy_window_slider

    assert len(figure.axes) >= 7
    slider.set_val((2, 4))

    time_series_axes = [
        axis
        for axis in figure.axes
        if axis.get_title()
        in {
            "Cumulative PnL",
            "Daily Net PnL",
            "Drawdown",
            "Daily Turnover",
            "Portfolio Exposure Heatmap",
        }
    ]
    assert len(time_series_axes) == 5
    assert all(
        axis.get_xlim() == pytest.approx((1.5, 4.5))
        for axis in time_series_axes
    )
    plt.close(figure)


def test_position_heatmap_defaults_to_normalised_notional_exposure() -> None:
    visualiser = make_visualiser()
    figure = visualiser.plot_position_heatmap()
    image = figure.axes[0].images[0]
    expected = (
        visualiser.result.positions
        * visualiser.universe.prices.T
        / visualiser.universe.constraints.position_limits[np.newaxis, :]
    ).T

    assert np.asarray(image.get_array()) == pytest.approx(expected)
    assert image.get_clim() == pytest.approx((-1.0, 1.0))
    assert figure.axes[1].get_ylabel() == "Normalised Exposure"
    assert figure.axes[0].get_xlabel() == "Trading day"
    plt.close(figure)


def test_non_normalised_heatmap_displays_notional_exposure() -> None:
    visualiser = make_visualiser()
    figure = visualiser.plot_position_heatmap(normalise=False)
    image = figure.axes[0].images[0]
    expected = (
        visualiser.result.positions * visualiser.universe.prices.T
    ).T
    expected_limit = float(np.max(np.abs(expected)))

    assert np.asarray(image.get_array()) == pytest.approx(expected)
    assert image.get_clim() == pytest.approx((-expected_limit, expected_limit))
    assert figure.axes[1].get_ylabel() == "Notional Exposure ($)"
    plt.close(figure)


def test_position_heatmap_thins_large_ticker_universe_labels() -> None:
    visualiser = make_visualiser()
    number_of_tickers = 50
    prices = np.tile(visualiser.universe.prices[0], (number_of_tickers, 1))
    universe = TickerUniverse.from_algothon(
        prices=prices,
        symbols=[f"TICKER_{index}" for index in range(number_of_tickers)],
        position_limits=np.full(number_of_tickers, 10_000.0),
        commission_rates=np.zeros(number_of_tickers),
    )
    result = replace(
        visualiser.result,
        positions=np.zeros((prices.shape[1], number_of_tickers), dtype=np.int64),
    )

    figure = StrategyVisualiser(universe, result).plot_position_heatmap()

    assert len(figure.axes[0].get_yticklabels()) <= 20
    plt.close(figure)


def test_save_all_saves_every_dashboard_figure(tmp_path) -> None:
    paths = make_visualiser().save_all(tmp_path)

    assert len(paths) == 1
    assert paths[0].name == "dashboard.png"
    assert all(path.suffix == ".png" for path in paths)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_flat_pnl() -> None:
    visualiser = make_visualiser(daily_pnl=np.zeros(5))
    figures = [
        visualiser.plot_cumulative_pnl(),
        visualiser.plot_drawdown(),
        visualiser.plot_pnl_distribution(),
    ]
    assert all(isinstance(figure, Figure) for figure in figures)
    plt.close("all")


def test_zero_trades_and_zero_commissions() -> None:
    visualiser = make_visualiser(
        turnover=np.zeros(5),
        commissions=np.zeros(5),
    )
    assert isinstance(visualiser.plot_turnover(), Figure)
    assert isinstance(visualiser.plot_commissions(), Figure)
    plt.close("all")


@pytest.mark.parametrize(
    "positions",
    [
        np.asarray([[1, 2], [1, 2], [2, 3], [2, 3], [3, 4]]),
        np.asarray([[1, -2], [1, -2], [-2, 3], [2, -3], [3, -4]]),
    ],
    ids=["all-long", "long-short"],
)
def test_position_plots_support_all_long_and_long_short(
    positions: np.ndarray,
) -> None:
    visualiser = make_visualiser(positions=positions)
    assert isinstance(visualiser.plot_position_heatmap(), Figure)
    assert isinstance(visualiser.plot_exposure(), Figure)
    plt.close("all")

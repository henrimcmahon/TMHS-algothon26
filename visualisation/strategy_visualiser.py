from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import RangeSlider

from backtesting.backtest_result import BacktestResult
from models.ticker import Ticker
from models.ticker_universe import TickerUniverse


TickerIdentifier = int | str | Ticker


@dataclass(slots=True, frozen=True)
class StrategyVisualiser:
    """Visualise one completed backtest without running strategy logic."""

    universe: TickerUniverse
    result: BacktestResult

    def __post_init__(self) -> None:
        prices = np.asarray(self.universe.prices)
        n_tickers = len(self.universe.tickers)

        if prices.ndim != 2:
            raise ValueError(
                "universe.prices must have shape (n_tickers, n_days)"
            )

        n_days = prices.shape[1]
        expected_price_shape = (n_tickers, n_days)

        if prices.shape != expected_price_shape:
            raise ValueError(
                "universe.prices must have shape (n_tickers, n_days); "
                f"expected {expected_price_shape}, received {prices.shape}"
            )

        expected_position_shape = (n_days, n_tickers)

        if self.result.positions.shape != expected_position_shape:
            raise ValueError(
                "result.positions must have shape (n_days, n_tickers); "
                f"expected {expected_position_shape}, "
                f"received {self.result.positions.shape}"
            )

        series = {
            "daily_pnl": self.result.daily_pnl,
            "cumulative_pnl": self.result.cumulative_pnl,
            "gross_daily_pnl": self.result.gross_daily_pnl,
            "daily_commissions": self.result.daily_commissions,
            "daily_turnover": self.result.daily_turnover,
        }
        for name, values in series.items():
            if np.asarray(values).shape != (n_days,):
                raise ValueError(
                    f"result.{name} must have shape (n_days,); "
                    f"expected {(n_days,)}, received {np.asarray(values).shape}"
                )

        if not 0 <= self.result.scoring_start_day <= n_days:
            raise ValueError(
                "result.scoring_start_day must be between 0 and n_days"
            )

    @property
    def _days(self) -> np.ndarray:
        return np.arange(self.universe.prices.shape[1], dtype=np.int64)

    def _resolve_ticker(self, identifier: TickerIdentifier) -> Ticker:
        if isinstance(identifier, Ticker):
            index = identifier.index
            if not 0 <= index < len(self.universe.tickers):
                raise KeyError(f"Unknown ticker: {identifier.symbol}")

            universe_ticker = self.universe.tickers[index]
            if universe_ticker != identifier:
                raise KeyError(f"Unknown ticker: {identifier.symbol}")
            return universe_ticker

        if isinstance(identifier, (int, str)):
            if isinstance(identifier, int) and not 0 <= identifier < len(
                self.universe.tickers
            ):
                raise KeyError(f"Unknown ticker: {identifier}")
            try:
                return self.universe.get(identifier)
            except (IndexError, KeyError) as error:
                raise KeyError(f"Unknown ticker: {identifier}") from error

        raise TypeError("ticker must be an int, str, or Ticker")

    @staticmethod
    def _new_figure(
        *,
        title: str,
        ylabel: str,
        figsize: tuple[float, float] = (11.0, 5.5),
    ) -> tuple[Figure, Axes]:
        figure, axis = plt.subplots(figsize=figsize)
        axis.set_title(title)
        axis.set_xlabel("Day")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        return figure, axis

    def _mark_scoring_start(self, axis: Axes) -> None:
        axis.axvline(
            self.result.scoring_start_day,
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            label="Scoring start",
        )

    @staticmethod
    def _format_currency_axis(axis: Axes) -> None:
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"${value:,.0f}")
        )

    @staticmethod
    def _drawdown(cumulative_pnl: np.ndarray) -> np.ndarray:
        equity = np.concatenate(
            (np.asarray([0.0]), np.asarray(cumulative_pnl, dtype=np.float64))
        )
        running_peak = np.maximum.accumulate(equity)
        return (equity - running_peak)[1:]

    def _exposure_heatmap_data(
        self,
        *,
        normalise: bool,
    ) -> tuple[np.ndarray, float, str]:
        notional_positions = (
            np.asarray(self.result.positions, dtype=np.float64)
            * np.asarray(self.universe.prices, dtype=np.float64).T
        )

        if normalise:
            position_limits = np.asarray(
                self.universe.constraints.position_limits,
                dtype=np.float64,
            )
            exposures = np.divide(
                notional_positions,
                position_limits[np.newaxis, :],
                out=np.zeros_like(notional_positions),
                where=position_limits[np.newaxis, :] != 0.0,
            )
            return exposures.T, 1.0, "Normalised Exposure"

        limit = max(float(np.max(np.abs(notional_positions))), 1.0)
        return notional_positions.T, limit, "Notional Exposure ($)"

    def _set_thinned_ticker_labels(self, axis: Axes) -> None:
        number_of_tickers = len(self.universe.tickers)
        label_step = max(1, int(np.ceil(number_of_tickers / 20)))
        tick_indices = np.arange(0, number_of_tickers, label_step)
        axis.set_yticks(tick_indices)
        axis.set_yticklabels(
            [self.universe.tickers[index].symbol for index in tick_indices],
            fontsize=7,
        )

    def plot_summary(self) -> Figure:
        figure, axis = plt.subplots(figsize=(10.0, 6.0))
        axis.axis("off")
        axis.set_title(f"{self.result.name} — Backtest Summary", pad=18)

        rows = (
            ("Scoring start day", f"{self.result.scoring_start_day:,}"),
            ("Scoring days", f"{self.result.scoring_days:,}"),
            ("Total net PnL", f"${self.result.total_pnl:,.2f}"),
            ("Total gross PnL", f"${self.result.total_gross_pnl:,.2f}"),
            ("Total commissions", f"${self.result.total_commissions:,.2f}"),
            ("Mean daily PnL", f"${self.result.mean_daily_pnl:,.2f}"),
            ("PnL standard deviation", f"${self.result.pnl_std:,.2f}"),
            ("Annualised Sharpe", f"{self.result.annualised_sharpe:,.3f}"),
            ("Score", f"{self.result.score:,.3f}"),
            ("Maximum drawdown", f"${self.result.maximum_drawdown:,.2f}"),
            ("Total dollar volume", f"${self.result.total_dollar_volume:,.2f}"),
            ("Return on volume", f"{self.result.return_on_volume:.6f}"),
        )

        table = axis.table(
            cellText=rows,
            colLabels=("Metric", "Value"),
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.45)
        figure.tight_layout()
        return figure

    def plot_cumulative_pnl(self, show_gross: bool = True) -> Figure:
        figure, axis = self._new_figure(
            title="Cumulative PnL",
            ylabel="Cumulative PnL",
        )
        axis.plot(self._days, self.result.cumulative_pnl, label="Net PnL")

        if show_gross:
            gross_cumulative_pnl = np.cumsum(
                np.asarray(self.result.gross_daily_pnl, dtype=np.float64)
            )
            axis.plot(
                self._days,
                gross_cumulative_pnl,
                label="Gross PnL",
                alpha=0.8,
            )

        axis.axhline(0.0, color="0.25", linewidth=0.8)
        self._mark_scoring_start(axis)
        self._format_currency_axis(axis)
        axis.legend()
        figure.tight_layout()
        return figure

    def plot_daily_pnl(self) -> Figure:
        figure, axis = self._new_figure(
            title="Daily Net PnL",
            ylabel="Daily PnL",
        )
        values = np.asarray(self.result.daily_pnl)
        colours = np.where(values >= 0.0, "tab:green", "tab:red")
        axis.bar(self._days, values, color=colours, width=0.9, alpha=0.75)
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        self._mark_scoring_start(axis)
        self._format_currency_axis(axis)
        figure.tight_layout()
        return figure

    def plot_commissions(self) -> Figure:
        figure, axis = self._new_figure(
            title="Daily Commissions",
            ylabel="Commission",
        )
        axis.bar(
            self._days,
            self.result.daily_commissions,
            width=0.9,
            color="tab:orange",
            alpha=0.8,
        )
        self._mark_scoring_start(axis)
        self._format_currency_axis(axis)
        figure.tight_layout()
        return figure

    def plot_turnover(self) -> Figure:
        figure, axis = self._new_figure(
            title="Daily Turnover",
            ylabel="Dollar turnover",
        )
        axis.plot(self._days, self.result.daily_turnover, color="tab:purple")
        axis.fill_between(
            self._days,
            self.result.daily_turnover,
            0.0,
            color="tab:purple",
            alpha=0.18,
        )
        self._mark_scoring_start(axis)
        self._format_currency_axis(axis)
        figure.tight_layout()
        return figure

    def plot_drawdown(self) -> Figure:
        figure, axis = self._new_figure(
            title="Drawdown",
            ylabel="Drawdown",
        )
        drawdown = self._drawdown(self.result.cumulative_pnl)
        axis.plot(self._days, drawdown, color="tab:red")
        axis.fill_between(
            self._days,
            drawdown,
            0.0,
            color="tab:red",
            alpha=0.22,
        )
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        self._mark_scoring_start(axis)
        self._format_currency_axis(axis)
        figure.tight_layout()
        return figure

    def plot_position_heatmap(self, normalise: bool = True) -> Figure:
        exposures, limit, colour_label = self._exposure_heatmap_data(
            normalise=normalise
        )

        figure, axis = plt.subplots(figsize=(12.0, 6.0))
        image = axis.imshow(
            exposures,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            origin="upper",
        )
        axis.set_title("Portfolio Exposure Heatmap")
        axis.set_xlabel("Trading day")
        axis.set_ylabel("Ticker")
        self._set_thinned_ticker_labels(axis)
        self._mark_scoring_start(axis)
        figure.colorbar(image, ax=axis, label=colour_label)
        figure.tight_layout()
        return figure

    def plot_ticker(self, ticker: TickerIdentifier) -> Figure:
        resolved = self._resolve_ticker(ticker)
        figure, price_axis = self._new_figure(
            title=f"{resolved.symbol} Price and Position",
            ylabel="Price",
        )
        position_axis = price_axis.twinx()
        price_axis.plot(
            self._days,
            self.universe.prices[resolved.index],
            color="tab:blue",
            label="Price",
        )
        position_axis.step(
            self._days,
            self.result.positions[:, resolved.index],
            where="post",
            color="tab:orange",
            label="Position",
        )
        position_axis.set_ylabel("Position (shares)")
        self._mark_scoring_start(price_axis)
        price_axis.legend(loc="upper left")
        position_axis.legend(loc="upper right")
        figure.tight_layout()
        return figure

    def plot_exposure(self) -> Figure:
        dollar_positions = (
            np.asarray(self.result.positions, dtype=np.float64)
            * np.asarray(self.universe.prices, dtype=np.float64).T
        )
        long_exposure = np.sum(np.maximum(dollar_positions, 0.0), axis=1)
        short_exposure = np.sum(np.minimum(dollar_positions, 0.0), axis=1)
        gross_exposure = np.sum(np.abs(dollar_positions), axis=1)
        net_exposure = np.sum(dollar_positions, axis=1)

        figure, axis = self._new_figure(
            title="Portfolio Exposure",
            ylabel="Dollar exposure",
        )
        axis.plot(self._days, gross_exposure, label="Gross", linewidth=2.0)
        axis.plot(self._days, net_exposure, label="Net", linewidth=1.5)
        axis.plot(self._days, long_exposure, label="Long", alpha=0.7)
        axis.plot(self._days, short_exposure, label="Short", alpha=0.7)
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        self._mark_scoring_start(axis)
        self._format_currency_axis(axis)
        axis.legend()
        figure.tight_layout()
        return figure

    def plot_pnl_distribution(self, scored_only: bool = True) -> Figure:
        pnl = np.asarray(self.result.daily_pnl, dtype=np.float64)
        if scored_only:
            pnl = pnl[self.result.scoring_start_day :]

        figure, axis = plt.subplots(figsize=(10.0, 5.5))
        axis.hist(pnl, bins=30, color="tab:blue", alpha=0.75)
        if pnl.size:
            axis.axvline(
                float(np.mean(pnl)),
                color="tab:orange",
                linestyle="--",
                linewidth=1.5,
                label="Mean",
            )
            axis.legend()
        axis.set_title("Distribution of Daily Net PnL")
        axis.set_xlabel("Daily net PnL")
        axis.set_ylabel("Frequency")
        axis.grid(alpha=0.2)
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"${value:,.0f}")
        )
        figure.tight_layout()
        return figure

    def create_dashboard(self) -> list[Figure]:
        figure = plt.figure(figsize=(16.0, 7.0))
        grid = figure.add_gridspec(
            3,
            2,
            left=0.055,
            right=0.965,
            top=0.91,
            bottom=0.14,
            hspace=0.42,
            wspace=0.22,
        )
        summary_axis = figure.add_subplot(grid[0, 0])
        cumulative_axis = figure.add_subplot(grid[0, 1])
        daily_axis = figure.add_subplot(grid[1, 0])
        drawdown_axis = figure.add_subplot(grid[1, 1])
        turnover_axis = figure.add_subplot(grid[2, 0])
        heatmap_axis = figure.add_subplot(grid[2, 1])
        figure.suptitle(f"{self.result.name} — Backtest Dashboard", fontsize=16)

        summary_axis.axis("off")
        summary_axis.set_title("Full-result Summary")
        summary_rows = (
            ("Net PnL", f"${self.result.total_pnl:,.2f}"),
            ("Gross PnL", f"${self.result.total_gross_pnl:,.2f}"),
            ("Commissions", f"${self.result.total_commissions:,.2f}"),
            ("Mean daily PnL", f"${self.result.mean_daily_pnl:,.2f}"),
            ("PnL standard deviation", f"${self.result.pnl_std:,.2f}"),
            ("Annualised Sharpe", f"{self.result.annualised_sharpe:,.3f}"),
            ("Score", f"{self.result.score:,.3f}"),
            ("Maximum drawdown", f"${self.result.maximum_drawdown:,.2f}"),
            ("Dollar volume", f"${self.result.total_dollar_volume:,.2f}"),
            ("Return on volume", f"{self.result.return_on_volume:.6f}"),
        )
        summary_table = summary_axis.table(
            cellText=summary_rows,
            colLabels=("Metric", "Value"),
            cellLoc="left",
            colLoc="left",
            bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 0.88),
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(8)
        summary_table.scale(1.0, 1.05)

        gross_cumulative_pnl = np.cumsum(
            np.asarray(self.result.gross_daily_pnl, dtype=np.float64)
        )
        cumulative_axis.plot(
            self._days,
            self.result.cumulative_pnl,
            label="Net PnL",
        )
        cumulative_axis.plot(
            self._days,
            gross_cumulative_pnl,
            label="Gross PnL",
            alpha=0.8,
        )
        cumulative_axis.axhline(0.0, color="0.25", linewidth=0.8)
        cumulative_axis.set_title("Cumulative PnL")
        cumulative_axis.set_ylabel("PnL")
        cumulative_axis.legend(fontsize=8)
        self._format_currency_axis(cumulative_axis)

        daily_pnl = np.asarray(self.result.daily_pnl, dtype=np.float64)
        daily_colours = np.where(daily_pnl >= 0.0, "tab:green", "tab:red")
        daily_axis.bar(
            self._days,
            daily_pnl,
            color=daily_colours,
            width=0.9,
            alpha=0.75,
        )
        daily_axis.axhline(0.0, color="0.25", linewidth=0.8)
        daily_axis.set_title("Daily Net PnL")
        daily_axis.set_ylabel("PnL")
        self._format_currency_axis(daily_axis)

        drawdown = self._drawdown(self.result.cumulative_pnl)
        drawdown_axis.plot(self._days, drawdown, color="tab:red")
        drawdown_axis.fill_between(
            self._days,
            drawdown,
            0.0,
            color="tab:red",
            alpha=0.22,
        )
        drawdown_axis.axhline(0.0, color="0.25", linewidth=0.8)
        drawdown_axis.set_title("Drawdown")
        drawdown_axis.set_ylabel("Drawdown")
        self._format_currency_axis(drawdown_axis)

        daily_turnover = np.asarray(
            self.result.daily_turnover,
            dtype=np.float64,
        )
        turnover_axis.plot(
            self._days,
            daily_turnover,
            color="tab:purple",
        )
        turnover_axis.fill_between(
            self._days,
            daily_turnover,
            0.0,
            color="tab:purple",
            alpha=0.18,
        )
        turnover_axis.set_title("Daily Turnover")
        turnover_axis.set_xlabel("Day")
        turnover_axis.set_ylabel("Dollar turnover")
        self._format_currency_axis(turnover_axis)

        exposures, exposure_limit, exposure_label = (
            self._exposure_heatmap_data(normalise=True)
        )
        position_image = heatmap_axis.imshow(
            exposures,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-exposure_limit,
            vmax=exposure_limit,
            origin="upper",
        )
        heatmap_axis.set_title("Portfolio Exposure Heatmap")
        heatmap_axis.set_xlabel("Trading day")
        heatmap_axis.set_ylabel("Ticker")
        self._set_thinned_ticker_labels(heatmap_axis)
        figure.colorbar(
            position_image,
            ax=heatmap_axis,
            label=exposure_label,
            fraction=0.046,
            pad=0.04,
        )

        time_axes = (
            cumulative_axis,
            daily_axis,
            drawdown_axis,
            turnover_axis,
            heatmap_axis,
        )
        for axis in time_axes:
            self._mark_scoring_start(axis)

        for axis in (
            cumulative_axis,
            daily_axis,
            drawdown_axis,
            turnover_axis,
        ):
            axis.grid(alpha=0.25)

        number_of_days = len(self._days)
        slider_axis = figure.add_axes((0.15, 0.045, 0.70, 0.025))
        initial_start = min(
            self.result.scoring_start_day,
            number_of_days - 1,
        )
        window_slider = RangeSlider(
            ax=slider_axis,
            label="Rolling window",
            valmin=0,
            valmax=number_of_days - 1,
            valinit=(initial_start, number_of_days - 1),
            valstep=1,
        )

        series_by_axis = {
            cumulative_axis: np.vstack(
                (self.result.cumulative_pnl, gross_cumulative_pnl)
            ),
            daily_axis: daily_pnl[np.newaxis, :],
            drawdown_axis: drawdown[np.newaxis, :],
            turnover_axis: daily_turnover[np.newaxis, :],
        }

        def set_window(values: tuple[float, float]) -> None:
            start_day = int(round(values[0]))
            end_day = int(round(values[1]))
            if end_day <= start_day:
                return

            for axis in time_axes:
                axis.set_xlim(start_day - 0.5, end_day + 0.5)

            for axis, series in series_by_axis.items():
                visible = series[:, start_day : end_day + 1]
                lower = min(float(np.min(visible)), 0.0)
                upper = max(float(np.max(visible)), 0.0)
                padding = max((upper - lower) * 0.08, 1.0)
                axis.set_ylim(lower - padding, upper + padding)

            figure.canvas.draw_idle()

        window_slider.on_changed(set_window)
        set_window(window_slider.val)

        # Matplotlib widgets must remain referenced for interaction to work.
        figure._strategy_window_slider = window_slider  # type: ignore[attr-defined]

        return [figure]

    def show_dashboard(self) -> list[Figure]:
        figures = self.create_dashboard()
        plt.show()
        return figures

    def save_all(self, directory: str | Path = "visualisations") -> list[Path]:
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        names = ("dashboard",)
        figures = self.create_dashboard()
        paths: list[Path] = []

        for name, figure in zip(names, figures, strict=True):
            path = output_directory / f"{name}.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            paths.append(path)
            plt.close(figure)

        return paths

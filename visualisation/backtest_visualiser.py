from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from collections.abc import Mapping

from dataclasses import dataclass
from typing import Callable

from matplotlib.figure import Figure
from matplotlib.text import Text

from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import (
    Button,
    CheckButtons,
    Slider,
)
from matplotlib.lines import Line2D

from numpy.typing import NDArray

from backtesting import BacktestResult
from backtesting.baseline_evaluator import BaselineResult


FloatArray = NDArray[np.float64]

BacktestPlotResult = (

    BacktestResult

    | BaselineResult

)

@dataclass(slots=True)
class _DashboardShell:
    figure: Figure
    axes: tuple[
        Axes,
        Axes,
        Axes,
        Axes,
    ]

    check_buttons: CheckButtons
    day_slider: Slider
    play_button: Button

    information_day_text: Text
    information_left_text: Text
    information_right_text: Text

LineDictionary = dict[str, Line2D]
LineGroups = tuple[
    LineDictionary,
    ...,
]

class BacktestVisualiser:
    """
    Visualise the outputs of a completed BacktestResult.

    This class performs presentation calculations only. The backtester
    remains responsible for PnL, commissions, turnover and scoring.
    """

    def __init__(
        self,
        results: Mapping[
            str,
            BacktestPlotResult,
        ],
    ) -> None:
        if not results:
            raise ValueError(
                "at least one backtest result is required"
            )

        self.results = dict(results)

        self.strategy_colours = {
            name: colour
            for name, colour in zip(
                self.results,
                plt.rcParams[
                    "axes.prop_cycle"
                ].by_key()["color"],
                strict=False,
            )
        }

        lengths = {
            len(result.daily_pnl)
            for result in self.results.values()
        }

        if len(lengths) != 1:
            raise ValueError(
                "all results must contain the same "
                "number of observations"
            )

        start_days = {
            result.start_day
            for result in self.results.values()
        }

        if len(start_days) != 1:
            raise ValueError(
                "all results must use the same start day"
            )

        self.number_of_days = next(
            iter(lengths)
        )

        self.start_day = next(
            iter(start_days)
        )

        self.days = np.arange(
            self.start_day,
            self.start_day
            + self.number_of_days,
            dtype=np.int64,
        )

    def _create_dashboard_shell(
        self,
        *,
        title: str,
        interval: int,
        right_boundary: float = 0.715,
    ) -> _DashboardShell:
        if interval < 1:
            raise ValueError(
                "interval must be positive"
            )

        strategy_names = list(
            self.results
        )

        initial_index = (
            self.number_of_days - 1
        )

        figure, axes_array = plt.subplots(
            2,
            2,
            figsize=(16, 8),
            sharex=True,
        )

        figure.subplots_adjust(
            left=0.055,
            right=right_boundary,
            top=0.91,
            bottom=0.14,
            wspace=0.18,
            hspace=0.24,
        )

        axes = (
            axes_array[0, 0],
            axes_array[0, 1],
            axes_array[1, 0],
            axes_array[1, 1],
        )

        for axis in axes:
            axis.axhline(
                0,
                linewidth=1,
            )

            axis.grid(
                alpha=0.25
            )

        # ==========================================================
        # Strategy selector
        # ==========================================================

        checkbox_axis = figure.add_axes(
            (
                0.74,
                0.56,
                0.245,
                0.33,
            )
        )

        check_buttons = CheckButtons(
            checkbox_axis,
            strategy_names,
            actives=[
                True
                for _ in strategy_names
            ],
        )

        checkbox_axis.set_title(
            "Strategies",
            fontsize=11,
            pad=10,
        )

        for label, strategy_name in zip(
            check_buttons.labels,
            strategy_names,
            strict=True,
        ):
            label.set_color(
                self.strategy_colours[
                    strategy_name
                ]
            )

            label.set_fontweight(
                "medium"
            )

        strategy_count = len(
            strategy_names
        )

        for index, strategy_name in enumerate(
            strategy_names
        ):
            row_y = (
                1.0
                - (
                    index + 0.5
                )
                / strategy_count
            )

            checkbox_axis.plot(
                [0.11, 0.18],
                [row_y, row_y],
                transform=(
                    checkbox_axis.transAxes
                ),
                color=self.strategy_colours[
                    strategy_name
                ],
                linewidth=3,
                solid_capstyle="round",
                clip_on=False,
            )

        # ==========================================================
        # Selected-day information panel
        # ==========================================================

        information_axis = figure.add_axes(
            (
                0.74,
                0.15,
                0.245,
                0.35,
            )
        )

        information_axis.set_title(
            "Selected Day",
            fontsize=11,
            pad=10,
        )

        information_axis.set_xticks([])
        information_axis.set_yticks([])

        for spine in (
            information_axis
            .spines
            .values()
        ):
            spine.set_alpha(
                0.35
            )

        information_day_text = (
            information_axis.text(
                0.50,
                0.96,
                "",
                fontsize=9,
                fontweight="medium",
                va="top",
                ha="center",
                transform=(
                    information_axis
                    .transAxes
                ),
            )
        )

        information_left_text = (
            information_axis.text(
                0.025,
                0.86,
                "",
                fontsize=7.2,
                family="monospace",
                va="top",
                ha="left",
                transform=(
                    information_axis
                    .transAxes
                ),
            )
        )

        information_right_text = (
            information_axis.text(
                0.515,
                0.86,
                "",
                fontsize=7.2,
                family="monospace",
                va="top",
                ha="left",
                transform=(
                    information_axis
                    .transAxes
                ),
            )
        )

        information_axis.plot(
            [0.49, 0.49],
            [0.03, 0.88],
            transform=(
                information_axis.transAxes
            ),
            linewidth=0.8,
            alpha=0.30,
        )

        # ==========================================================
        # Shared controls
        # ==========================================================

        day_slider_axis = figure.add_axes(
            (
                0.23,
                0.055,
                0.38,
                0.025,
            )
        )

        day_slider = Slider(
            ax=day_slider_axis,
            label="Selected day",
            valmin=0,
            valmax=(
                self.number_of_days - 1
            ),
            valinit=initial_index,
            valstep=1,
        )

        play_axis = figure.add_axes(
            (
                0.66,
                0.038,
                0.08,
                0.055,
            )
        )

        play_button = Button(
            play_axis,
            "Play",
        )

        figure.suptitle(
            title,
            fontsize=16,
        )

        return _DashboardShell(
            figure=figure,
            axes=axes,
            check_buttons=check_buttons,
            day_slider=day_slider,
            play_button=play_button,
            information_day_text=(
                information_day_text
            ),
            information_left_text=(
                information_left_text
            ),
            information_right_text=(
                information_right_text
            ),
        )
    
    @staticmethod
    def _split_active_strategies(
        strategy_names: list[str],
        active_names: set[str],
    ) -> tuple[
        list[str],
        list[str],
    ]:
        selected = [
            name
            for name in strategy_names
            if name in active_names
        ]

        midpoint = (
            len(selected) + 1
        ) // 2

        return (
            selected[:midpoint],
            selected[midpoint:],
        )
    
    @staticmethod
    def _set_strategy_visible(
        strategy_name: str,
        visible: bool,
        line_groups: LineGroups,
    ) -> None:
        for line_group in line_groups:
            line_group[
                strategy_name
            ].set_visible(
                visible
            )


    @staticmethod
    def _autoscale_axes(
        axes: tuple[
            Axes,
            ...,
        ],
    ) -> None:
        for axis in axes:
            axis.relim(
                visible_only=True
            )

            axis.autoscale_view()

            axis.margins(
                x=0.02,
                y=0.05,
            )

    def _connect_dashboard_interactions(
        self,
        *,
        shell: _DashboardShell,
        line_groups: LineGroups,
        vertical_markers: list[Line2D],
        update_points: Callable[
            [int, int],
            None,
        ],
        update_information: Callable[
            [int, set[str]],
            None,
        ],
        interval: int,
    ) -> FuncAnimation:
        strategy_names = list(
            self.results
        )

        state: dict[
            str,
            bool | set[str],
        ] = {
            "playing": False,
            "active": set(
                strategy_names
            ),
        }

        def active_names() -> set[str]:
            value = state["active"]

            if not isinstance(
                value,
                set,
            ):
                raise RuntimeError(
                    "active strategy state is invalid"
                )

            return value

        def update_day(
            slider_value: float,
        ) -> None:
            index = int(
                slider_value
            )

            day = int(
                self.days[index]
            )

            for marker in vertical_markers:
                marker.set_xdata(
                    [day, day]
                )

            update_points(
                index,
                day,
            )

            update_information(
                index,
                active_names(),
            )

            shell.figure.canvas.draw_idle()

        def update_strategy_visibility(
            _: str | None = None,
        ) -> None:
            statuses = (
                shell
                .check_buttons
                .get_status()
            )

            selected = {
                name
                for name, enabled in zip(
                    strategy_names,
                    statuses,
                    strict=True,
                )
                if enabled
            }

            state["active"] = selected

            for (
                label,
                strategy_name,
                enabled,
            ) in zip(
                shell.check_buttons.labels,
                strategy_names,
                statuses,
                strict=True,
            ):
                self._set_strategy_visible(
                    strategy_name=(
                        strategy_name
                    ),
                    visible=enabled,
                    line_groups=line_groups,
                )

                label.set_alpha(
                    1.0
                    if enabled
                    else 0.30
                )

            self._autoscale_axes(
                shell.axes
            )

            update_information(
                int(shell.day_slider.val),
                selected,
            )

            shell.figure.canvas.draw_idle()

        def toggle_play(
            _: object,
        ) -> None:
            playing = bool(
                state["playing"]
            )

            state["playing"] = (
                not playing
            )

            shell.play_button.label.set_text(
                "Pause"
                if not playing
                else "Play"
            )

        def animate(
            _: int,
        ) -> None:
            if not bool(
                state["playing"]
            ):
                return

            next_index = (
                int(shell.day_slider.val)
                + 1
            )

            if (
                next_index
                >= self.number_of_days
            ):
                next_index = 0

            shell.day_slider.set_val(
                next_index
            )

        shell.day_slider.on_changed(
            update_day
        )

        shell.check_buttons.on_clicked(
            update_strategy_visibility
        )

        shell.play_button.on_clicked(
            toggle_play
        )

        update_strategy_visibility()

        update_day(
            float(
                self.number_of_days - 1
            )
        )

        return FuncAnimation(
            shell.figure,
            animate,
            interval=interval,
            cache_frame_data=False,
        )

    @staticmethod
    def _running_mean(
        values: FloatArray,
    ) -> FloatArray:
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        counts = np.arange(
            1,
            len(values) + 1,
            dtype=np.float64,
        )

        return np.cumsum(values) / counts


    @staticmethod
    def _running_std(
        values: FloatArray,
    ) -> FloatArray:
        """
        Expanding sample standard deviation.

        The first value is zero because only one observation exists.
        """
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        output = np.zeros(
            len(values),
            dtype=np.float64,
        )

        if len(values) < 2:
            return output

        cumulative_sum = np.cumsum(
            values
        )

        cumulative_squared_sum = np.cumsum(
            values**2
        )

        counts = np.arange(
            1,
            len(values) + 1,
            dtype=np.float64,
        )

        variance_numerator = (
            cumulative_squared_sum
            - cumulative_sum**2 / counts
        )

        sample_variance = np.zeros(
            len(values),
            dtype=np.float64,
        )

        sample_variance[1:] = (
            variance_numerator[1:]
            / (counts[1:] - 1.0)
        )

        sample_variance = np.maximum(
            sample_variance,
            0.0,
        )

        output[1:] = np.sqrt(
            sample_variance[1:]
        )

        return output


    @classmethod
    def _running_sharpe(
        cls,
        values: FloatArray,
        annualisation_days: int = 250,
        minimum_observations: int = 20,
    ) -> FloatArray:
        if minimum_observations < 2:
            raise ValueError(
                "minimum_observations must be at least 2"
            )

        running_mean = cls._running_mean(
            values
        )

        running_std = cls._running_std(
            values
        )

        sharpe = np.full(
            len(values),
            np.nan,
            dtype=np.float64,
        )

        counts = np.arange(
            1,
            len(values) + 1,
        )

        valid = (
            (counts >= minimum_observations)
            & (running_std >= 1e-10)
        )

        sharpe[valid] = (
            np.sqrt(annualisation_days)
            * running_mean[valid]
            / running_std[valid]
        )

        return sharpe


    @classmethod
    def _running_score(
        cls,
        values: FloatArray,
        annualisation_days: int = 250,
        minimum_observations: int = 20,
    ) -> FloatArray:
        running_mean = cls._running_mean(
            values
        )

        running_std = cls._running_std(
            values
        )

        running_sharpe = cls._running_sharpe(
            values,
            annualisation_days=annualisation_days,
            minimum_observations=minimum_observations,
        )

        score = np.full(
            len(values),
            np.nan,
            dtype=np.float64,
        )

        counts = np.arange(
            1,
            len(values) + 1,
        )

        enough_history = (
            counts >= minimum_observations
        )

        fallback = (
            enough_history
            & (
                (running_mean < 0)
                | (running_std < 1e-10)
            )
        )

        score[fallback] = (
            running_mean[fallback]
        )

        positive_valid = (
            enough_history
            & (running_mean >= 0)
            & (running_std >= 1e-10)
            & np.isfinite(running_sharpe)
        )

        sharpe_squared = (
            running_sharpe[
                positive_valid
            ] ** 2
        )

        score[positive_valid] = (
            running_mean[
                positive_valid
            ]
            * sharpe_squared
            / (
                sharpe_squared + 1.0
            )
        )

        return score

    @staticmethod
    def _rolling_mean(
        values: FloatArray,
        window: int,
    ) -> FloatArray:
        if window < 1:
            raise ValueError(
                "window must be positive"
            )

        output = np.full(
            len(values),
            np.nan,
            dtype=np.float64,
        )

        if len(values) < window:
            return output

        cumulative = np.concatenate(
            (
                np.asarray([0.0]),
                np.cumsum(values),
            )
        )

        output[window - 1:] = (
            cumulative[window:]
            - cumulative[:-window]
        ) / window

        return output

    @classmethod
    def _running_mean_confidence_interval(
        cls,
        values: FloatArray,
        confidence_multiplier: float = 1.96,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Approximate confidence interval around the expanding mean.

        Uses:

            running_mean ± z * running_std / sqrt(n)
        """
        if confidence_multiplier <= 0:
            raise ValueError(
                "confidence_multiplier must be positive"
            )

        running_mean = cls._running_mean(
            values
        )

        running_std = cls._running_std(
            values
        )

        counts = np.arange(
            1,
            len(values) + 1,
            dtype=np.float64,
        )

        standard_error = (
            running_std
            / np.sqrt(counts)
        )

        margin = (
            confidence_multiplier
            * standard_error
        )

        return (
            running_mean - margin,
            running_mean + margin,
        )


    @classmethod
    def _rolling_score(
        cls,
        values: FloatArray,
        window: int,
        annualisation_days: int = 250,
    ) -> FloatArray:
        """
        Calculate the competition score over a moving PnL window.
        """
        if window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        output = np.full(
            len(values),
            np.nan,
            dtype=np.float64,
        )

        if len(values) < window:
            return output

        for end_index in range(
            window - 1,
            len(values),
        ):
            window_values = values[
                end_index - window + 1:
                end_index + 1
            ]

            mean_pnl = float(
                np.mean(window_values)
            )

            pnl_std = float(
                np.std(
                    window_values,
                    ddof=1,
                )
            )

            if (
                mean_pnl >= 0
                and pnl_std >= 1e-10
            ):
                sharpe = float(
                    np.sqrt(
                        annualisation_days
                    )
                    * mean_pnl
                    / pnl_std
                )

                sharpe_squared = (
                    sharpe**2
                )

                output[end_index] = (
                    mean_pnl
                    * sharpe_squared
                    / (
                        sharpe_squared
                        + 1.0
                    )
                )
            else:
                output[end_index] = mean_pnl

        return output


    @staticmethod
    def _score_change(
        running_score: FloatArray,
    ) -> FloatArray:
        score = np.asarray(
            running_score,
            dtype=np.float64,
        )

        output = np.full(
            len(score),
            np.nan,
            dtype=np.float64,
        )

        if len(score) < 2:
            return output

        valid = (
            np.isfinite(score[1:])
            & np.isfinite(score[:-1])
        )

        differences = (
            score[1:]
            - score[:-1]
        )

        output[1:][valid] = (
            differences[valid]
        )

        return output

    @staticmethod
    def _drawdown(
        cumulative_pnl: FloatArray,
    ) -> FloatArray:
        equity = np.concatenate(
            (
                np.asarray([0.0]),
                cumulative_pnl,
            )
        )

        running_peak = np.maximum.accumulate(
            equity
        )

        return (
            equity - running_peak
        )[1:]

    @staticmethod
    def _format_currency_axis(
        axis: Axes,
    ) -> None:
        axis.ticklabel_format(
            axis="y",
            style="plain",
            useOffset=False,
        )

        axis.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: f"${value:,.0f}"
            )
        )

    def plot_dashboard(
        self,
        rolling_window: int = 20,
        interval: int = 80,
    ) -> None:
        """
        Compare strategy PnL, drawdown, turnover and commissions.

        Includes:
            - strategy multiselect;
            - selected-day slider;
            - rolling-window slider;
            - play/pause animation.
        """
        if rolling_window < 2:
            raise ValueError(
                "rolling_window must be at least 2"
            )

        shell = self._create_dashboard_shell(
            title="Interactive Strategy Comparison",
            interval=interval,
        )

        (
            equity_axis,
            daily_axis,
            drawdown_axis,
            cost_axis,
        ) = shell.axes

        commission_axis = cost_axis.twinx()

        strategy_names = list(
            self.results
        )

        initial_index = (
            self.number_of_days - 1
        )

        # ==========================================================
        # Stored data
        # ==========================================================

        drawdown_series: dict[
            str,
            FloatArray,
        ] = {}

        # ==========================================================
        # Lines and selected-day points
        # ==========================================================

        equity_lines: LineDictionary = {}
        equity_points: LineDictionary = {}

        rolling_lines: LineDictionary = {}
        daily_points: LineDictionary = {}

        drawdown_lines: LineDictionary = {}
        drawdown_points: LineDictionary = {}

        turnover_lines: LineDictionary = {}
        turnover_points: LineDictionary = {}

        commission_lines: LineDictionary = {}
        commission_points: LineDictionary = {}

        for strategy_name, result in (
            self.results.items()
        ):
            colour = self.strategy_colours[
                strategy_name
            ]

            cumulative_pnl = np.asarray(
                result.cumulative_pnl,
                dtype=np.float64,
            )

            daily_pnl = np.asarray(
                result.daily_pnl,
                dtype=np.float64,
            )

            rolling_pnl = self._rolling_mean(
                daily_pnl,
                rolling_window,
            )

            drawdown = self._drawdown(
                cumulative_pnl
            )

            drawdown_series[
                strategy_name
            ] = drawdown

            equity_line, = equity_axis.plot(
                self.days,
                cumulative_pnl,
                linewidth=2.0,
                label=strategy_name,
                color=colour,
            )

            equity_point, = equity_axis.plot(
                [self.days[initial_index]],
                [cumulative_pnl[initial_index]],
                marker="o",
                linestyle="None",
                color=colour,
            )

            rolling_line, = daily_axis.plot(
                self.days,
                rolling_pnl,
                linewidth=1.8,
                label=strategy_name,
                color=colour,
            )

            daily_point, = daily_axis.plot(
                [self.days[initial_index]],
                [daily_pnl[initial_index]],
                marker="o",
                linestyle="None",
                color=colour,
            )

            drawdown_line, = drawdown_axis.plot(
                self.days,
                drawdown,
                linewidth=1.5,
                label=strategy_name,
                color=colour,
            )

            drawdown_point, = drawdown_axis.plot(
                [self.days[initial_index]],
                [drawdown[initial_index]],
                marker="o",
                linestyle="None",
                color=colour,
            )

            turnover_line, = cost_axis.plot(
                self.days,
                result.daily_turnover,
                linewidth=1.2,
                label=strategy_name,
                color=colour,
            )

            turnover_point, = cost_axis.plot(
                [self.days[initial_index]],
                [
                    result.daily_turnover[
                        initial_index
                    ]
                ],
                marker="o",
                linestyle="None",
                color=colour,
            )

            commission_line, = (
                commission_axis.plot(
                    self.days,
                    result.daily_commissions,
                    linewidth=1.0,
                    linestyle="--",
                    label=strategy_name,
                    color=colour,
                )
            )

            commission_point, = (
                commission_axis.plot(
                    [self.days[initial_index]],
                    [
                        result.daily_commissions[
                            initial_index
                        ]
                    ],
                    marker="o",
                    linestyle="None",
                    color=colour,
                )
            )

            equity_lines[
                strategy_name
            ] = equity_line

            equity_points[
                strategy_name
            ] = equity_point

            rolling_lines[
                strategy_name
            ] = rolling_line

            daily_points[
                strategy_name
            ] = daily_point

            drawdown_lines[
                strategy_name
            ] = drawdown_line

            drawdown_points[
                strategy_name
            ] = drawdown_point

            turnover_lines[
                strategy_name
            ] = turnover_line

            turnover_points[
                strategy_name
            ] = turnover_point

            commission_lines[
                strategy_name
            ] = commission_line

            commission_points[
                strategy_name
            ] = commission_point

        # ==========================================================
        # Axis configuration
        # ==========================================================

        equity_axis.set_title(
            "Cumulative Net PnL"
        )
        equity_axis.set_ylabel(
            "Cumulative PnL"
        )

        daily_axis.set_title(
            "Rolling Mean Daily PnL"
        )
        daily_axis.set_ylabel(
            "Daily PnL"
        )

        drawdown_axis.set_title(
            "Drawdown"
        )
        drawdown_axis.set_ylabel(
            "Drawdown"
        )

        cost_axis.set_title(
            "Turnover and Commission"
        )
        cost_axis.set_ylabel(
            "Daily turnover"
        )

        commission_axis.set_ylabel(
            "Daily commission"
        )

        for axis in (
            equity_axis,
            daily_axis,
            drawdown_axis,
            cost_axis,
            commission_axis,
        ):
            self._format_currency_axis(
                axis
            )

        # The strategy selector acts as the shared legend.
        for axis in shell.axes:
            legend = axis.get_legend()

            if legend is not None:
                legend.remove()

        # ==========================================================
        # Vertical selected-day markers
        # ==========================================================

        selected_day = int(
            self.days[initial_index]
        )

        vertical_markers = [
            axis.axvline(
                selected_day,
                linestyle="--",
                linewidth=1.2,
            )
            for axis in shell.axes
        ]

        # ==========================================================
        # Dashboard-specific rolling-window control
        # ==========================================================

        window_slider_axis = (
            shell.figure.add_axes(
                (
                    0.055,
                    0.055,
                    0.13,
                    0.025,
                )
            )
        )

        maximum_window = max(
            2,
            min(
                100,
                self.number_of_days,
            ),
        )

        window_slider = Slider(
            ax=window_slider_axis,
            label="Rolling window",
            valmin=2,
            valmax=maximum_window,
            valinit=min(
                rolling_window,
                maximum_window,
            ),
            valstep=1,
        )

        # ==========================================================
        # Shared interaction data
        # ==========================================================

        line_groups: LineGroups = (
            equity_lines,
            equity_points,
            rolling_lines,
            daily_points,
            drawdown_lines,
            drawdown_points,
            turnover_lines,
            turnover_points,
            commission_lines,
            commission_points,
        )

        # ==========================================================
        # Dashboard-specific callbacks
        # ==========================================================

        def update_dashboard_points(
            index: int,
            day: int,
        ) -> None:
            for strategy_name, result in (
                self.results.items()
            ):
                equity_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        result.cumulative_pnl[
                            index
                        ]
                    ],
                )

                daily_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        result.daily_pnl[
                            index
                        ]
                    ],
                )

                drawdown_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        drawdown_series[
                            strategy_name
                        ][index]
                    ],
                )

                turnover_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        result.daily_turnover[
                            index
                        ]
                    ],
                )

                commission_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        result.daily_commissions[
                            index
                        ]
                    ],
                )

        def update_dashboard_information(
            index: int,
            active_names: set[str],
        ) -> None:
            left_names, right_names = (
                self._split_active_strategies(
                    strategy_names,
                    active_names,
                )
            )

            def format_column(
                names: list[str],
            ) -> str:
                lines: list[str] = []

                for strategy_name in names:
                    result = self.results[
                        strategy_name
                    ]

                    daily_pnl = float(
                        result.daily_pnl[index]
                    )

                    cumulative_pnl = float(
                        result.cumulative_pnl[
                            index
                        ]
                    )

                    drawdown = float(
                        drawdown_series[
                            strategy_name
                        ][index]
                    )

                    lines.extend(
                        [
                            strategy_name,
                            (
                                " Daily  "
                                f"${daily_pnl:>10,.2f}"
                            ),
                            (
                                " Total  "
                                f"${cumulative_pnl:>10,.2f}"
                            ),
                            (
                                " DD     "
                                f"${drawdown:>10,.2f}"
                            ),
                            "",
                        ]
                    )

                return "\n".join(
                    lines
                )

            shell.information_day_text.set_text(
                f"Day {int(self.days[index])}"
            )

            shell.information_left_text.set_text(
                format_column(
                    left_names
                )
            )

            shell.information_right_text.set_text(
                format_column(
                    right_names
                )
            )

        def update_window(
            slider_value: float,
        ) -> None:
            window = int(
                slider_value
            )

            for strategy_name, result in (
                self.results.items()
            ):
                updated_rolling_pnl = (
                    self._rolling_mean(
                        np.asarray(
                            result.daily_pnl,
                            dtype=np.float64,
                        ),
                        window,
                    )
                )

                rolling_lines[
                    strategy_name
                ].set_ydata(
                    updated_rolling_pnl
                )

            daily_axis.relim(
                visible_only=True
            )
            daily_axis.autoscale_view()
            daily_axis.margins(
                x=0.02,
                y=0.05,
            )

            shell.figure.canvas.draw_idle()

        def rescale_commission_axis(
            _: str | None = None,
        ) -> None:
            commission_axis.relim(
                visible_only=True
            )
            commission_axis.autoscale_view()
            commission_axis.margins(
                x=0.02,
                y=0.05,
            )

            shell.figure.canvas.draw_idle()

        # ==========================================================
        # Connect interactions
        # ==========================================================

        self._comparison_animation = (
            self._connect_dashboard_interactions(
                shell=shell,
                line_groups=line_groups,
                vertical_markers=(
                    vertical_markers
                ),
                update_points=(
                    update_dashboard_points
                ),
                update_information=(
                    update_dashboard_information
                ),
                interval=interval,
            )
        )

        window_slider.on_changed(
            update_window
        )

        # Commission uses a secondary axis, so it needs separate
        # rescaling whenever strategy visibility changes.
        shell.check_buttons.on_clicked(
            rescale_commission_axis
        )

        rescale_commission_axis()

        self._comparison_day_slider = (
            shell.day_slider
        )

        self._comparison_window_slider = (
            window_slider
        )

        self._comparison_check_buttons = (
            shell.check_buttons
        )

        self._comparison_play_button = (
            shell.play_button
        )

        plt.show()

    def plot_score_dashboard(
        self,
        annualisation_days: int = 250,
        minimum_observations: int = 20,
        rolling_score_window: int = 50,
        interval: int = 80,
    ) -> None:
        """
        Compare the evolution of the competition objective.

        Panels:
            - expanding mean daily PnL with confidence intervals;
            - expanding PnL standard deviation;
            - expanding annualised Sharpe ratio;
            - expanding and rolling competition score.

        The currently highest-scoring strategy is highlighted at the
        selected day.
        """
        if annualisation_days <= 0:
            raise ValueError(
                "annualisation_days must be positive"
            )

        if minimum_observations < 2:
            raise ValueError(
                "minimum_observations must be at least 2"
            )

        if rolling_score_window < 2:
            raise ValueError(
                "rolling_score_window must be at least 2"
            )

        shell = self._create_dashboard_shell(
            title="Competition Objective Evolution",
            interval=interval,
        )

        (
            mean_axis,
            volatility_axis,
            sharpe_axis,
            score_axis,
        ) = shell.axes

        strategy_names = list(
            self.results
        )

        initial_index = (
            self.number_of_days - 1
        )

        # ==========================================================
        # Calculated series
        # ==========================================================

        running_means: dict[
            str,
            FloatArray,
        ] = {}

        mean_lower_bounds: dict[
            str,
            FloatArray,
        ] = {}

        mean_upper_bounds: dict[
            str,
            FloatArray,
        ] = {}

        running_volatilities: dict[
            str,
            FloatArray,
        ] = {}

        running_sharpes: dict[
            str,
            FloatArray,
        ] = {}

        running_scores: dict[
            str,
            FloatArray,
        ] = {}

        rolling_scores: dict[
            str,
            FloatArray,
        ] = {}

        # ==========================================================
        # Main artists
        # ==========================================================

        mean_lines: LineDictionary = {}
        mean_points: LineDictionary = {}

        volatility_lines: LineDictionary = {}
        volatility_points: LineDictionary = {}

        sharpe_lines: LineDictionary = {}
        sharpe_points: LineDictionary = {}

        score_lines: LineDictionary = {}
        score_points: LineDictionary = {}

        rolling_score_lines: LineDictionary = {}

        confidence_bands: dict[
            str,
            object,
        ] = {}

        for strategy_name, result in (
            self.results.items()
        ):
            colour = self.strategy_colours[
                strategy_name
            ]

            daily_pnl = np.asarray(
                result.daily_pnl,
                dtype=np.float64,
            )

            running_mean = self._running_mean(
                daily_pnl
            )

            (
                mean_lower,
                mean_upper,
            ) = (
                self
                ._running_mean_confidence_interval(
                    daily_pnl
                )
            )

            running_volatility = (
                self._running_std(
                    daily_pnl
                )
            )

            running_sharpe = (
                self._running_sharpe(
                    daily_pnl,
                    annualisation_days=(
                        annualisation_days
                    ),
                    minimum_observations=(
                        minimum_observations
                    ),
                )
            )

            running_score = (
                self._running_score(
                    daily_pnl,
                    annualisation_days=(
                        annualisation_days
                    ),
                    minimum_observations=(
                        minimum_observations
                    ),
                )
            )

            rolling_score = (
                self._rolling_score(
                    daily_pnl,
                    window=rolling_score_window,
                    annualisation_days=(
                        annualisation_days
                    ),
                )
            )

            running_means[
                strategy_name
            ] = running_mean

            mean_lower_bounds[
                strategy_name
            ] = mean_lower

            mean_upper_bounds[
                strategy_name
            ] = mean_upper

            running_volatilities[
                strategy_name
            ] = running_volatility

            running_sharpes[
                strategy_name
            ] = running_sharpe

            running_scores[
                strategy_name
            ] = running_score

            rolling_scores[
                strategy_name
            ] = rolling_score

            confidence_band = (
                mean_axis.fill_between(
                    self.days,
                    mean_lower,
                    mean_upper,
                    color=colour,
                    alpha=0.08,
                    linewidth=0,
                )
            )

            mean_line, = mean_axis.plot(
                self.days,
                running_mean,
                linewidth=1.6,
                label=strategy_name,
                color=colour,
            )

            mean_point, = mean_axis.plot(
                [self.days[initial_index]],
                [running_mean[initial_index]],
                marker="o",
                linestyle="None",
                color=colour,
            )

            volatility_line, = (
                volatility_axis.plot(
                    self.days,
                    running_volatility,
                    linewidth=1.6,
                    label=strategy_name,
                    color=colour,
                )
            )

            volatility_point, = (
                volatility_axis.plot(
                    [self.days[initial_index]],
                    [
                        running_volatility[
                            initial_index
                        ]
                    ],
                    marker="o",
                    linestyle="None",
                    color=colour,
                )
            )

            sharpe_line, = sharpe_axis.plot(
                self.days,
                running_sharpe,
                linewidth=1.6,
                label=strategy_name,
                color=colour,
            )

            sharpe_point, = sharpe_axis.plot(
                [self.days[initial_index]],
                [
                    running_sharpe[
                        initial_index
                    ]
                ],
                marker="o",
                linestyle="None",
                color=colour,
            )

            score_line, = score_axis.plot(
                self.days,
                running_score,
                linewidth=1.8,
                label=strategy_name,
                color=colour,
            )

            rolling_score_line, = (
                score_axis.plot(
                    self.days,
                    rolling_score,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.65,
                    color=colour,
                )
            )

            score_point, = score_axis.plot(
                [self.days[initial_index]],
                [
                    running_score[
                        initial_index
                    ]
                ],
                marker="o",
                linestyle="None",
                color=colour,
            )

            confidence_bands[
                strategy_name
            ] = confidence_band

            mean_lines[
                strategy_name
            ] = mean_line

            mean_points[
                strategy_name
            ] = mean_point

            volatility_lines[
                strategy_name
            ] = volatility_line

            volatility_points[
                strategy_name
            ] = volatility_point

            sharpe_lines[
                strategy_name
            ] = sharpe_line

            sharpe_points[
                strategy_name
            ] = sharpe_point

            score_lines[
                strategy_name
            ] = score_line

            rolling_score_lines[
                strategy_name
            ] = rolling_score_line

            score_points[
                strategy_name
            ] = score_point

        # ==========================================================
        # Axis configuration
        # ==========================================================

        mean_axis.set_title(
            "Running Mean Daily PnL"
        )
        mean_axis.set_ylabel(
            "Mean daily PnL"
        )

        volatility_axis.set_title(
            "Running PnL Volatility"
        )
        volatility_axis.set_ylabel(
            "Daily PnL standard deviation"
        )

        sharpe_axis.set_title(
            "Running Annualised Sharpe"
        )
        sharpe_axis.set_ylabel(
            "Annualised Sharpe"
        )

        score_axis.set_title(
            (
                "Competition Score "
                f"(dashed = {rolling_score_window}-day)"
            )
        )
        score_axis.set_ylabel(
            "Competition score"
        )

        self._format_currency_axis(
            mean_axis
        )

        self._format_currency_axis(
            volatility_axis
        )

        selected_day = int(
            self.days[initial_index]
        )

        vertical_markers = [
            axis.axvline(
                selected_day,
                linestyle="--",
                linewidth=1.2,
            )
            for axis in shell.axes
        ]

        line_groups: LineGroups = (
            mean_lines,
            mean_points,
            volatility_lines,
            volatility_points,
            sharpe_lines,
            sharpe_points,
            score_lines,
            rolling_score_lines,
            score_points,
        )

        # ==========================================================
        # Winner highlighting
        # ==========================================================

        def current_winner(
            index: int,
            active_names: set[str],
        ) -> str | None:
            candidates = {
                strategy_name: (
                    running_scores[
                        strategy_name
                    ][index]
                )
                for strategy_name in active_names
                if np.isfinite(
                    running_scores[
                        strategy_name
                    ][index]
                )
            }

            if not candidates:
                return None

            return max(
                candidates,
                key=candidates.__getitem__,
            )

        def highlight_winner(
            winner: str | None,
            active_names: set[str],
        ) -> None:
            main_groups = (
                mean_lines,
                volatility_lines,
                sharpe_lines,
                score_lines,
            )

            for strategy_name in (
                strategy_names
            ):
                enabled = (
                    strategy_name
                    in active_names
                )

                is_winner = (
                    enabled
                    and strategy_name == winner
                )

                for line_group in main_groups:
                    line_group[
                        strategy_name
                    ].set_linewidth(
                        2.8
                        if is_winner
                        else 1.4
                    )

                    line_group[
                        strategy_name
                    ].set_alpha(
                        1.0
                        if is_winner
                        else 0.65
                        if enabled
                        else 0.0
                    )

                rolling_score_lines[
                    strategy_name
                ].set_linewidth(
                    2.0
                    if is_winner
                    else 1.0
                )

                rolling_score_lines[
                    strategy_name
                ].set_alpha(
                    0.85
                    if is_winner
                    else 0.40
                    if enabled
                    else 0.0
                )

        # ==========================================================
        # Updating selected points
        # ==========================================================

        def update_score_points(
            index: int,
            day: int,
        ) -> None:
            for strategy_name in (
                strategy_names
            ):
                mean_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        running_means[
                            strategy_name
                        ][index]
                    ],
                )

                volatility_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        running_volatilities[
                            strategy_name
                        ][index]
                    ],
                )

                sharpe_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        running_sharpes[
                            strategy_name
                        ][index]
                    ],
                )

                score_points[
                    strategy_name
                ].set_data(
                    [day],
                    [
                        running_scores[
                            strategy_name
                        ][index]
                    ],
                )

        # ==========================================================
        # Sidebar information and ranking
        # ==========================================================

        def update_score_information(
            index: int,
            active_names: set[str],
        ) -> None:
            winner = current_winner(
                index,
                active_names,
            )

            highlight_winner(
                winner,
                active_names,
            )

            ranked_names = sorted(
                active_names,
                key=lambda strategy_name: (
                    running_scores[
                        strategy_name
                    ][index]
                    if np.isfinite(
                        running_scores[
                            strategy_name
                        ][index]
                    )
                    else -np.inf
                ),
                reverse=True,
            )

            left_names, right_names = (
                self._split_active_strategies(
                    ranked_names,
                    set(ranked_names),
                )
            )

            rank_lookup = {
                strategy_name: rank
                for rank, strategy_name in enumerate(
                    ranked_names,
                    start=1,
                )
            }

            def format_column(
                names: list[str],
            ) -> str:
                lines: list[str] = []

                for strategy_name in names:
                    mean_value = (
                        running_means[
                            strategy_name
                        ][index]
                    )

                    volatility_value = (
                        running_volatilities[
                            strategy_name
                        ][index]
                    )

                    sharpe_value = (
                        running_sharpes[
                            strategy_name
                        ][index]
                    )

                    score_value = (
                        running_scores[
                            strategy_name
                        ][index]
                    )

                    winner_marker = (
                        " ★"
                        if strategy_name == winner
                        else ""
                    )

                    lines.extend(
                        [
                            (
                                f"#{rank_lookup[strategy_name]} "
                                f"{strategy_name}"
                                f"{winner_marker}"
                            ),
                            (
                                " Mean   "
                                f"${mean_value:>9,.2f}"
                            ),
                            (
                                " Vol    "
                                f"${volatility_value:>9,.2f}"
                            ),
                            (
                                " Sharpe  "
                                f"{sharpe_value:>9.3f}"
                            ),
                            (
                                " Score   "
                                f"{score_value:>9.3f}"
                            ),
                            "",
                        ]
                    )

                return "\n".join(
                    lines
                )

            heading = (
                f"Day {int(self.days[index])}"
            )

            if winner is not None:
                heading += (
                    f" — Leader: {winner}"
                )

            shell.information_day_text.set_text(
                heading
            )

            shell.information_left_text.set_text(
                format_column(
                    left_names
                )
            )

            shell.information_right_text.set_text(
                format_column(
                    right_names
                )
            )

        self._score_animation = (
            self._connect_dashboard_interactions(
                shell=shell,
                line_groups=line_groups,
                vertical_markers=(
                    vertical_markers
                ),
                update_points=(
                    update_score_points
                ),
                update_information=(
                    update_score_information
                ),
                interval=interval,
            )
        )

        self._score_day_slider = (
            shell.day_slider
        )

        self._score_check_buttons = (
            shell.check_buttons
        )

        self._score_play_button = (
            shell.play_button
        )

        plt.show()

    def plot_score_changes(
        self,
        annualisation_days: int = 250,
        minimum_observations: int = 20,
    ) -> None:
        """
        Plot the one-day change in each strategy's expanding score.
        """
        figure, axis = plt.subplots(
            figsize=(14, 7)
        )

        for strategy_name, result in (
            self.results.items()
        ):
            daily_pnl = np.asarray(
                result.daily_pnl,
                dtype=np.float64,
            )

            running_score = self._running_score(
                daily_pnl,
                annualisation_days=annualisation_days,
                minimum_observations=minimum_observations,
            )

            score_change = self._score_change(
                running_score
            )

            axis.plot(
                self.days,
                score_change,
                linewidth=1.0,
                alpha=0.75,
                label=strategy_name,
                color=self.strategy_colours[
                    strategy_name
                ],
            )

        axis.axhline(
            0,
            linewidth=1,
        )

        axis.set_title(
            "Daily Change in Running Competition Score"
        )

        axis.set_xlabel(
            "Day"
        )

        axis.set_ylabel(
            "Change in score"
        )

        axis.grid(
            alpha=0.25
        )

        axis.legend(
            fontsize=8
        )

        figure.tight_layout()

        plt.show()

    def plot_pnl_distribution(
        self,
        bins: int = 40,
    ) -> None:
        if bins < 2:
            raise ValueError(
                "bins must be at least 2"
            )

        figure, axis = plt.subplots(
            figsize=(12, 6)
        )

        for strategy_name, result in self.results.items():
            axis.hist(
                result.daily_pnl,
                bins=bins,
                alpha=0.35,
                density=False,
                label=strategy_name,
            )

            mean_pnl = float(
                np.mean(result.daily_pnl)
            )

            axis.axvline(
                mean_pnl,
                linestyle="--",
                linewidth=2,
                color=self.strategy_colours[strategy_name],
            )

        axis.set_title(
            "Distribution of Daily Net PnL"
        )

        axis.set_xlabel(
            "Daily Net PnL"
        )

        axis.set_ylabel(
            "Frequency"
        )

        axis.grid(alpha=0.20)

        axis.legend()

        figure.tight_layout()

        plt.show()

    def plot_exposure(
        self,
        prices: FloatArray,
    ) -> None:

        figure, axis = plt.subplots(
            figsize=(14, 7)
        )

        for strategy_name, result in self.results.items():

            price_values = np.asarray(
                prices,
                dtype=np.float64,
            )

            if price_values.shape != result.positions.shape:
                raise ValueError(
                    "prices must match "
                    f"{strategy_name}"
                )

            dollar_positions = (
                result.positions
                * price_values
            )

            gross = np.sum(
                np.abs(dollar_positions),
                axis=1,
            )

            axis.plot(
                self.days,
                gross,
                linewidth=2,
                label=strategy_name,
            )

        axis.axhline(
            0,
            linewidth=1,
        )

        axis.grid(alpha=0.25)

        axis.legend()

        axis.set_title(
            "Gross Dollar Exposure"
        )

        axis.set_xlabel(
            "Day"
        )

        axis.set_ylabel(
            "Gross Exposure"
        )

        self._format_currency_axis(
            axis
        )

        figure.tight_layout()

        plt.show()
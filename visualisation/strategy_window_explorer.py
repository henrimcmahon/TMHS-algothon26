from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import (
    CheckButtons,
    RadioButtons,
    RangeSlider,
)
from sklearn import metrics

from backtesting.baseline_evaluator import (
    BaselineResult,
)


class StrategyWindowExplorer:
    """
    Interactively compare strategies over arbitrary historical windows.

    Slider values represent price-history boundaries. For example,
    selecting 500–750 evaluates the 250 PnLs labelled 501–750.
    """

    def __init__(
        self,
        results: Mapping[
            str,
            BaselineResult,
        ],
        *,
        total_price_days: int,
        initial_window_days: int = 250,
        top_n: int = 7,
        annualisation_days: int = 250,
        rolling_window_days: int = 50,
        rolling_step_days: int = 5,
    ) -> None:
        if not results:
            raise ValueError(
                "at least one strategy result is required"
            )

        if total_price_days < 2:
            raise ValueError(
                "total_price_days must be at least 2"
            )

        if not (
            1
            <= initial_window_days
            <= total_price_days
        ):
            raise ValueError(
                "initial_window_days must be between "
                "1 and total_price_days"
            )

        lengths = {
            len(result.daily_pnl)
            for result in results.values()
        }

        if len(lengths) != 1:
            raise ValueError(
                "all strategy results must have equal length"
            )

        expected_pnl_days = (
            total_price_days - 1
        )

        result_length = next(
            iter(lengths)
        )

        if result_length != expected_pnl_days:
            raise ValueError(
                "StrategyWindowExplorer requires full-history "
                "results. Expected "
                f"{expected_pnl_days} PnLs but received "
                f"{result_length}. Evaluate with "
                "num_test_days=None first."
            )

        self.results = dict(results)
        self.strategy_names = tuple(
            self.results
        )

        self.sort_metric = "Score"
        self._last_base_metric = "Score"

        colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        self.strategy_colours = {
            name: colour_cycle[i % len(colour_cycle)]
            for i, name in enumerate(self.strategy_names)
        }

        self.total_price_days = (
            total_price_days
        )

        self.initial_window_days = (
            initial_window_days
        )

        self.top_n = max(
            1,
            min(
                top_n,
                len(self.strategy_names),
            ),
        )

        self.annualisation_days = (
            annualisation_days
        )

        if rolling_window_days <= 0:
            raise ValueError(
                "rolling_window_days must be positive"
            )

        if rolling_window_days > total_price_days:
            raise ValueError(
                "rolling_window_days cannot exceed total_price_days"
            )

        if rolling_step_days <= 0:
            raise ValueError(
                "rolling_step_days must be positive"
            )

        self.rolling_window_days = (
            rolling_window_days
        )

        self.rolling_step_days = (
            rolling_step_days
        )

        self.heatmap_axis = None

        self.pnl_matrix = np.vstack(
            [
                np.asarray(
                    self.results[name].daily_pnl,
                    dtype=np.float64,
                )
                for name in self.strategy_names
            ]
        )

        colour_cycle = (
            plt.rcParams[
                "axes.prop_cycle"
            ]
            .by_key()[
                "color"
            ]
        )

        self.strategy_colours = {
            name: colour_cycle[
                index % len(colour_cycle)
            ]
            for index, name in enumerate(
                self.strategy_names
            )
        }

        self.normalise_by_drawdown = False
        self.sort_metric = "Score"

        self.heatmap_axis = None
        self.regime_axis = None
        self.sort_axis = None
        self.toggle_axis = None
        self.sort_buttons = None
        self.normalise_toggle = None

        (
            self.rolling_window_ends,
            self.rolling_score_matrix,
            self.rolling_winner_indices,
            self.rolling_win_counts,
        ) = self._calculate_rolling_scores()

        self.figure = None
        self.pnl_axis = None
        self.score_axis = None
        self.slider = None

    def _score(
        self,
        daily_pnl: np.ndarray,
    ) -> float:
        mean_pnl = float(
            np.mean(daily_pnl)
        )

        pnl_std = float(
            np.std(
                daily_pnl,
                ddof=0,
            )
        )

        if (
            mean_pnl <= 0.0
            or pnl_std < 1e-10
        ):
            return mean_pnl

        sharpe = float(
            np.sqrt(
                self.annualisation_days
            )
            * mean_pnl
            / pnl_std
        )

        return float(
            mean_pnl
            * sharpe**2
            / (
                sharpe**2 + 1.0
            )
        )
    
    def _metric_value(
        self,
        metrics: dict[str, float],
        metric_name: str | None = None,
    ) -> float:
        metric = (
            self.sort_metric
            if metric_name is None
            else metric_name
        )

        if metric == "Score":
            return metrics["score"]

        if metric == "Sharpe":
            return metrics["sharpe"]

        if metric == "Total PnL":
            return metrics["total_pnl"]

        if metric == "Max DD":
            # Smaller drawdown is better, so negate it.
            return -metrics["maximum_drawdown"]

        raise ValueError(
            f"Unsupported rolling metric: {metric}"
        )

    def _metrics(
        self,
        daily_pnl: np.ndarray,
    ) -> dict[str, float]:
        mean_pnl = float(
            np.mean(daily_pnl)
        )

        pnl_std = float(
            np.std(
                daily_pnl,
                ddof=0,
            )
        )

        sharpe = (
            0.0
            if pnl_std < 1e-10
            else float(
                np.sqrt(
                    self.annualisation_days
                )
                * mean_pnl
                / pnl_std
            )
        )

        cumulative = np.concatenate(
            (
                np.asarray(
                    [0.0],
                    dtype=np.float64,
                ),
                np.cumsum(
                    daily_pnl
                ),
            )
        )

        running_peak = (
            np.maximum.accumulate(
                cumulative
            )
        )

        maximum_drawdown = float(
            np.max(
                running_peak
                - cumulative
            )
        )

        return {
            "total_pnl": float(
                np.sum(daily_pnl)
            ),
            "mean_pnl": mean_pnl,
            "pnl_std": pnl_std,
            "sharpe": sharpe,
            "score": self._score(
                daily_pnl
            ),
            "maximum_drawdown": (
                maximum_drawdown
            ),
        }
    
    def _calculate_rolling_scores(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        end_days = np.arange(
            self.rolling_window_days,
            self.total_price_days + 1,
            self.rolling_step_days,
            dtype=np.int64,
        )

        number_of_strategies = len(
            self.strategy_names
        )

        score_matrix = np.full(
            (
                number_of_strategies,
                len(end_days),
            ),
            np.nan,
            dtype=np.float64,
        )

        winner_indices = np.full(
            len(end_days),
            -1,
            dtype=np.int64,
        )

        for window_index, end_day in enumerate(
            end_days
        ):
            start_day = (
                int(end_day)
                - self.rolling_window_days
            )

            start_index, end_index = (
                self._window_indices(
                    start_day,
                    int(end_day),
                )
            )

            for strategy_index in range(
                number_of_strategies
            ):
                daily_pnl = self.pnl_matrix[
                    strategy_index,
                    start_index:end_index,
                ]

                if daily_pnl.size == 0:
                    continue

                score_matrix[
                    strategy_index,
                    window_index,
                ] = self._score(
                    daily_pnl
                )

            scores = score_matrix[
                :,
                window_index,
            ]

            valid = np.isfinite(
                scores
            )

            if np.any(valid):
                valid_indices = np.flatnonzero(
                    valid
                )

                winner_indices[
                    window_index
                ] = int(
                    valid_indices[
                        np.argmax(
                            scores[
                                valid_indices
                            ]
                        )
                    ]
                )

        win_counts = np.zeros(
            number_of_strategies,
            dtype=np.int64,
        )

        valid_winners = winner_indices[
            winner_indices >= 0
        ]

        for strategy_index in valid_winners:
            win_counts[
                strategy_index
            ] += 1

        return (
            end_days,
            score_matrix,
            winner_indices,
            win_counts,
        )
    
    def _rolling_winners_for_metric(
        self,
    ) -> np.ndarray:
        """
        Return the winning strategy index for each rolling window,
        using the currently selected table metric.
        """
        # Window wins cannot define its own rolling winners because that
        # would be circular. Keep the most recently selected base metric,
        # defaulting to Score.
        metric_name = self.sort_metric

        if metric_name == "Window wins":
            metric_name = getattr(
                self,
                "_last_base_metric",
                "Score",
            )

        winner_indices = np.full(
            len(self.rolling_window_ends),
            -1,
            dtype=np.int64,
        )

        for window_number, end_day in enumerate(
            self.rolling_window_ends
        ):
            start_day = (
                int(end_day)
                - self.rolling_window_days
            )

            start_index, end_index = (
                self._window_indices(
                    start_day,
                    int(end_day),
                )
            )

            metric_values = np.full(
                len(self.strategy_names),
                np.nan,
                dtype=np.float64,
            )

            for strategy_index in range(
                len(self.strategy_names)
            ):
                daily_pnl = self.pnl_matrix[
                    strategy_index,
                    start_index:end_index,
                ]

                if daily_pnl.size == 0:
                    continue

                strategy_metrics = self._metrics(
                    daily_pnl
                )

                metric_values[
                    strategy_index
                ] = self._metric_value(
                    strategy_metrics,
                    metric_name,
                )

            valid = np.isfinite(
                metric_values
            )

            if not np.any(valid):
                continue

            valid_indices = np.flatnonzero(
                valid
            )

            winner_indices[
                window_number
            ] = int(
                valid_indices[
                    np.argmax(
                        metric_values[
                            valid_indices
                        ]
                    )
                ]
            )

        return winner_indices
    
    def _ranking_order(
        self,
        metrics: list[
            dict[str, float]
        ],
    ) -> np.ndarray:
        if self.sort_metric == "Window wins":
            visible_counts = getattr(
                self,
                "_current_visible_win_counts",
                self.rolling_win_counts,
            )

            return np.argsort(
                visible_counts
            )[::-1]

        values = np.asarray(
            [
                self._metric_value(
                    strategy_metrics
                )
                for strategy_metrics in metrics
            ],
            dtype=np.float64,
        )

        return np.argsort(
            values
        )[::-1]

    def _window_indices(
        self,
        start_day: int,
        end_day: int,
    ) -> tuple[int, int]:
        """
        Convert boundary days to PnL-array indices.

        Example:
            boundaries 500–750
            -> daily_pnl[499:749]
            -> PnLs labelled 501–750
        """
        start_index = max(
            0,
            start_day - 1,
        )

        end_index = min(
            self.pnl_matrix.shape[1],
            end_day - 1,
        )

        if end_index <= start_index:
            end_index = min(
                start_index + 1,
                self.pnl_matrix.shape[1],
            )

        return (
            start_index,
            end_index,
        )
    
    def _window_win_counts(
        self,
        start_day: int,
        end_day: int,
        winner_indices: np.ndarray,
    ) -> np.ndarray:
        visible_mask = (
            (
                self.rolling_window_ends
                >= start_day
            )
            & (
                self.rolling_window_ends
                <= end_day
            )
        )

        counts = np.zeros(
            len(self.strategy_names),
            dtype=np.int64,
        )

        for winner_index in winner_indices[
            visible_mask
        ]:
            if winner_index >= 0:
                counts[
                    int(winner_index)
                ] += 1

        return counts
    
    def _render_score_heatmap(
        self,
    ) -> None:
        if self.heatmap_axis is None:
            return

        self.heatmap_axis.clear()

        # Strategies with the most rolling-window wins are shown first.
        strategy_order = np.argsort(
            self.rolling_win_counts
        )[::-1]

        ordered_scores = (
            self.rolling_score_matrix[
                strategy_order
            ]
        )

        ordered_names = [
            self.strategy_names[index]
            for index in strategy_order
        ]

        self.heatmap_axis.imshow(
            ordered_scores,
            aspect="auto",
            interpolation="nearest",
            extent=(
                self.rolling_window_ends[0],
                self.rolling_window_ends[-1],
                len(ordered_names) - 0.5,
                -0.5,
            ),
        )

        self.heatmap_axis.set_yticks(
            np.arange(
                len(ordered_names)
            )
        )

        self.heatmap_axis.set_yticklabels(
            [
                (
                    f"{name} "
                    f"({self.rolling_win_counts[index]} wins)"
                )
                for name, index in zip(
                    ordered_names,
                    strategy_order,
                    strict=True,
                )
            ],
            fontsize=7,
        )

        self.heatmap_axis.set_title(
            (
                "Rolling Strategy Scores "
                f"({self.rolling_window_days}-day windows, "
                f"{self.rolling_step_days}-day step)"
            ),
            fontsize=11,
        )

        self.heatmap_axis.set_xlabel(
            "Window end day"
        )

        self.heatmap_axis.set_ylabel(
            "Strategy"
        )

        # Mark the currently selected slider window.
        if self.slider is not None:
            start_day, end_day = (
                self.slider.val
            )

            self.heatmap_axis.axvspan(
                start_day,
                end_day,
                alpha=0.12,
            )

    def _render_regime_views(
        self,
        selected_start_day: int,
        selected_end_day: int,
        rolling_winner_indices: np.ndarray,
        visible_win_counts: np.ndarray,
    ) -> None:
        if (
            self.heatmap_axis is None
            or self.regime_axis is None
        ):
            return

        self.heatmap_axis.clear()
        self.regime_axis.clear()

        # Only keep rolling windows whose end day lies inside
        # the currently selected evaluation window.
        visible_mask = (
            (self.rolling_window_ends >= selected_start_day)
            & (self.rolling_window_ends <= selected_end_day)
        )

        visible_end_days = self.rolling_window_ends[
            visible_mask
        ]

        visible_winner_indices = (
            rolling_winner_indices[
                visible_mask
            ]
        )

        if visible_end_days.size == 0:
            self.heatmap_axis.text(
                0.5,
                0.5,
                "No rolling windows in selected range",
                transform=self.heatmap_axis.transAxes,
                ha="center",
                va="center",
            )

            self.regime_axis.text(
                0.5,
                0.5,
                "No regime data",
                transform=self.regime_axis.transAxes,
                ha="center",
                va="center",
            )

            self.heatmap_axis.set_xlim(
                selected_start_day,
                selected_end_day,
            )

            self.regime_axis.set_xlim(
                selected_start_day,
                selected_end_day,
            )

            return

        # Recalculate win counts using only the visible window.
        visible_win_counts = np.zeros(
            len(self.strategy_names),
            dtype=np.int64,
        )

        for winner_index in visible_winner_indices:
            if winner_index >= 0:
                visible_win_counts[
                    int(winner_index)
                ] += 1

        strategy_order = np.argsort(
            visible_win_counts
        )[::-1]

        ordered_names = [
            self.strategy_names[index]
            for index in strategy_order
        ]

        ordered_lookup = {
            original_index: ordered_index
            for ordered_index, original_index
            in enumerate(strategy_order)
        }

        number_of_strategies = len(
            strategy_order
        )

        number_of_windows = len(
            visible_end_days
        )

        white = np.asarray(
            to_rgba("white")
        )

        winner_image = np.broadcast_to(
            white,
            (
                number_of_strategies,
                number_of_windows,
                4,
            ),
        ).copy()

        regime_image = np.broadcast_to(
            white,
            (
                1,
                number_of_windows,
                4,
            ),
        ).copy()

        for window_index, winner_index in enumerate(
            visible_winner_indices
        ):
            if winner_index < 0:
                continue

            winner_index = int(
                winner_index
            )

            ordered_row = ordered_lookup[
                winner_index
            ]

            colour = np.asarray(
                to_rgba(
                    self.strategy_colours[
                        self.strategy_names[
                            winner_index
                        ]
                    ]
                )
            )

            winner_image[
                ordered_row,
                window_index,
            ] = colour

            regime_image[
                0,
                window_index,
            ] = colour

        first_end = int(
            visible_end_days[0]
        )

        last_end = int(
            visible_end_days[-1]
        )

        # Add half a step so each heatmap block is centred
        # on its rolling-window end day.
        half_step = (
            self.rolling_step_days
            / 2.0
        )

        left_edge = max(
            selected_start_day,
            first_end - half_step,
        )

        right_edge = min(
            selected_end_day,
            last_end + half_step,
        )

        self.heatmap_axis.imshow(
            winner_image,
            aspect="auto",
            interpolation="nearest",
            extent=(
                left_edge,
                right_edge,
                number_of_strategies - 0.5,
                -0.5,
            ),
        )

        self.heatmap_axis.set_yticks(
            np.arange(
                number_of_strategies
            )
        )

        self.heatmap_axis.set_yticklabels(
            [
                (
                    f"{name} "
                    f"({visible_win_counts[index]} wins)"
                )
                for name, index in zip(
                    ordered_names,
                    strategy_order,
                    strict=True,
                )
            ],
            fontsize=7,
        )

        self.heatmap_axis.set_xlim(
            selected_start_day,
            selected_end_day,
        )

        metric_label = self.sort_metric

        if metric_label == "Window wins":
            metric_label = (
                "Window wins based on "
                + getattr(
                    self,
                    "_last_base_metric",
                    "Score",
                )
            )

        self.heatmap_axis.set_title(
            (
                f"Rolling-window Winners by {metric_label} "
                f"({self.rolling_window_days}-day windows, "
                f"{self.rolling_step_days}-day step)"
            )
        )

        self.heatmap_axis.set_ylabel(
            "Winning strategy"
        )

        self.heatmap_axis.set_xlabel(
            ""
        )

        self.regime_axis.imshow(
            regime_image,
            aspect="auto",
            interpolation="nearest",
            extent=(
                left_edge,
                right_edge,
                0,
                1,
            ),
        )

        self.regime_axis.set_xlim(
            selected_start_day,
            selected_end_day,
        )

        self.regime_axis.set_yticks(
            []
        )

        self.regime_axis.set_xlabel(
            "Window end day"
        )

        self.regime_axis.set_ylabel(
            "Regime",
            rotation=0,
            labelpad=25,
            va="center",
        )

    def _render_window(
        self,
        start_day: int,
        end_day: int,
    ) -> None:
        if (
            self.pnl_axis is None
            or self.score_axis is None
            or self.figure is None
        ):
            return

        start_index, end_index = (
            self._window_indices(
                start_day,
                end_day,
            )
        )

        rolling_winner_indices = (
            self._rolling_winners_for_metric()
        )

        visible_win_counts = (
            self._window_win_counts(
                start_day,
                end_day,
                rolling_winner_indices,
            )
        )

        self._current_visible_win_counts = (
            visible_win_counts
        )

        window_pnl = self.pnl_matrix[
            :,
            start_index:end_index,
        ]

        if window_pnl.shape[1] == 0:
            self.pnl_axis.clear()
            self.score_axis.clear()

            self.score_axis.axis("off")

            self.score_axis.text(
                0.5,
                0.5,
                "No observations in selected window",
                ha="center",
                va="center",
                fontsize=14,
            )

            self.figure.canvas.draw_idle()
            return

        displayed_days = np.arange(
            start_day + 1,
            start_day + window_pnl.shape[1] + 1,
            dtype=np.int64,
        )

        metrics = [
            self._metrics(
                window_pnl[
                    strategy_index
                ]
            )
            for strategy_index in range(
                len(self.strategy_names)
            )
        ]

        rolling_winner_indices = (
            self._rolling_winners_for_metric()
        )

        visible_win_counts = (
            self._window_win_counts(
                start_day,
                end_day,
                rolling_winner_indices,
            )
        )

        self._current_visible_win_counts = (
            visible_win_counts
        )

        # The displayed table can be sorted by any chosen metric.
        ordering = self._ranking_order(
            metrics
        )

        top_indices = ordering[
            :self.top_n
        ]

        best_index = int(
            ordering[0]
        )

        best_name = self.strategy_names[
            best_index
        ]

        best_metrics = metrics[
            best_index
        ]

        self.pnl_axis.clear()
        self.score_axis.clear()

        # ==========================================================
        # Cumulative PnL plot
        # ==========================================================

        for displayed_rank, strategy_index in enumerate(
            top_indices,
            start=1,
        ):
            name = self.strategy_names[
                strategy_index
            ]

            cumulative_pnl = np.cumsum(
                window_pnl[
                    strategy_index
                ]
            )

            if self.normalise_by_drawdown:
                drawdown = metrics[
                    strategy_index
                ]["maximum_drawdown"]

                plotted_values = (
                    cumulative_pnl
                    / max(
                        drawdown,
                        1e-10,
                    )
                )
            else:
                plotted_values = cumulative_pnl

            is_winner = (
                strategy_index == best_index
            )

            self.pnl_axis.plot(
                displayed_days,
                plotted_values,
                color=self.strategy_colours[
                    name
                ],
                linewidth=(
                    2.8
                    if is_winner
                    else 1.5
                ),
                alpha=(
                    1.0
                    if is_winner
                    else 0.85
                ),
                label=(
                    f"#{displayed_rank} {name}"
                ),
            )

        self.pnl_axis.axhline(
            0.0,
            linewidth=1.0,
        )

        self.pnl_axis.set_xlim(
            start_day,
            end_day,
        )

        self.pnl_axis.set_title(
            (
                f"Window {start_day}–{end_day}\n"
                f"Winner by {self.sort_metric}: "
                f"{best_name}"
            )
        )

        self.pnl_axis.set_xlabel(
            "Day"
        )

        if self.normalise_by_drawdown:
            self.pnl_axis.set_ylabel(
                "Cumulative PnL / Max Drawdown"
            )

            self.pnl_axis.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _:
                    f"{value:.2f}×"
                )
            )
        else:
            self.pnl_axis.set_ylabel(
                "Cumulative net PnL"
            )

            self.pnl_axis.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _:
                    f"${value:,.0f}"
                )
            )

        self.pnl_axis.grid(
            alpha=0.22,
        )

        self.pnl_axis.legend(
            fontsize=8,
            loc="best",
        )

        # ==========================================================
        # Ranked table
        # ==========================================================

        self.score_axis.axis(
            "off"
        )

        table_rows: list[list[object]] = []

        for displayed_rank, strategy_index in enumerate(
            top_indices,
            start=1,
        ):
            name = self.strategy_names[
                strategy_index
            ]

            strategy_metrics = metrics[
                strategy_index
            ]

            table_rows.append(
                [
                    displayed_rank,
                    name,
                    (
                        f"{strategy_metrics['score']:,.2f}"
                    ),
                    (
                        f"${strategy_metrics['total_pnl']:,.0f}"
                    ),
                    (
                        f"{strategy_metrics['sharpe']:.2f}"
                    ),
                    (
                        f"${strategy_metrics['maximum_drawdown']:,.0f}"
                    ),
                    int(
                        visible_win_counts[
                            strategy_index
                        ]
                    ),
                ]
            )

        if table_rows:
            table = self.score_axis.table(
                cellText=table_rows,
                colLabels=[
                    "Rank",
                    "Strategy",
                    "Score",
                    "Total PnL",
                    "Sharpe",
                    "Max DD",
                    (
                        f"{self.rolling_window_days}d "
                        "window wins"
                    ),
                ],
                loc="center",
                cellLoc="left",
                colLoc="left",
                colWidths=[
                    0.06,
                    0.38,
                    0.10,
                    0.13,
                    0.09,
                    0.12,
                    0.12,
                ],
            )

            table.auto_set_font_size(
                False
            )

            table.set_fontsize(
                8
            )

            table.scale(
                1.0,
                1.55,
            )
        else:
            self.score_axis.text(
                0.5,
                0.5,
                "No strategies in selected window",
                ha="center",
                va="center",
                fontsize=14,
            )

        self.score_axis.set_title(
            (
                "Best strategy in selected window\n"
                f"Score {best_metrics['score']:,.2f} • "
                f"PnL ${best_metrics['total_pnl']:,.0f} • "
                f"Sharpe {best_metrics['sharpe']:.2f}"
            ),
            fontsize=11,
            pad=12,
        )

        self._render_regime_views(
            start_day,
            end_day,
            rolling_winner_indices,
            visible_win_counts,
        )

        self.figure.canvas.draw_idle()

    def _sort_changed(
        self,
        label: str,
    ) -> None:
        self.sort_metric = label

        if label != "Window wins":
            self._last_base_metric = label

        if self.slider is None:
            return

        start_day, end_day = (
            self.slider.val
        )

        self._render_window(
            int(round(start_day)),
            int(round(end_day)),
        )


    def _normalise_changed(
        self,
        _: str,
    ) -> None:
        self.normalise_by_drawdown = (
            not self.normalise_by_drawdown
        )

        if self.slider is None:
            return

        start_day, end_day = (
            self.slider.val
        )

        self._render_window(
            int(round(start_day)),
            int(round(end_day)),
        )

    def _slider_changed(
        self,
        values: tuple[float, float],
    ) -> None:
        start_day = int(
            round(
                values[0]
            )
        )

        end_day = int(
            round(
                values[1]
            )
        )

        if end_day <= start_day:
            return

        self._render_window(
            start_day,
            end_day,
        )

    def show(self) -> None:
        self.figure = plt.figure(
            figsize=(18, 12)
        )

        grid = self.figure.add_gridspec(
            4,
            1,
            height_ratios=[
                3.0,
                1.75,
                2.2,
                0.35,
            ],
            left=0.17,
            right=0.96,
            top=0.92,
            bottom=0.13,
            hspace=0.47,
        )

        self.pnl_axis = (
            self.figure.add_subplot(
                grid[0]
            )
        )

        self.score_axis = (
            self.figure.add_subplot(
                grid[1]
            )
        )

        self.heatmap_axis = (
            self.figure.add_subplot(
                grid[2]
            )
        )

        self.regime_axis = (
            self.figure.add_subplot(
                grid[3]
            )
        )

        slider_axis = (
            self.figure.add_axes(
                [
                    0.20,
                    0.045,
                    0.62,
                    0.024,
                ]
            )
        )

        self.sort_axis = (
            self.figure.add_axes(
                [
                    0.015,
                    0.56,
                    0.13,
                    0.19,
                ]
            )
        )

        self.toggle_axis = (
            self.figure.add_axes(
                [
                    0.015,
                    0.47,
                    0.14,
                    0.055,
                ]
            )
        )

        initial_start = max(
            0,
            self.total_price_days
            - self.initial_window_days,
        )

        self.slider = RangeSlider(
            ax=slider_axis,
            label="Evaluation window",
            valmin=0,
            valmax=self.total_price_days,
            valinit=(
                initial_start,
                self.total_price_days,
            ),
            valstep=1,
        )

        self.sort_buttons = RadioButtons(
            self.sort_axis,
            (
                "Score",
                "Sharpe",
                "Total PnL",
                "Max DD",
                "Window wins",
            ),
            active=0,
        )

        self.sort_axis.set_title(
            "Sort table by",
            fontsize=9,
        )

        self.normalise_toggle = CheckButtons(
            self.toggle_axis,
            [
                "Normalise by Max DD",
            ],
            [
                False,
            ],
        )

        self.slider.on_changed(
            self._slider_changed
        )

        self.sort_buttons.on_clicked(
            self._sort_changed
        )

        self.normalise_toggle.on_clicked(
            self._normalise_changed
        )

        self.figure.suptitle(
            "Interactive Strategy Window Explorer",
            fontsize=16,
        )

        self._render_window(
            initial_start,
            self.total_price_days,
        )

        plt.show()
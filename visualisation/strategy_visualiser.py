from __future__ import annotations

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseEvent
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram

from statistics.market_statistics import MarketStatistics


class ExplorerState(TypedDict):
    selected_ticker: int
    playing: bool
    cluster_order: NDArray[np.intp]


class Strategyvisualiser:
    """
    Displays analytics calculated by MarketStatistics.

    This class contains presentation logic only. It does not calculate
    correlations, PCA, clustering, beta, returns or trading features.
    """

    def __init__(self, statistics: MarketStatistics) -> None:
        self.statistics = statistics

        # Keep widget and animation references alive.
        self._animation: FuncAnimation | None = None
        self._slider: Slider | None = None
        self._play_button: Button | None = None

    def plot_interactive_dashboard(
        self,
        window: int = 60,
        beta_window: int = 30,
        interval: int = 120,
        neighbour_count: int = 3,
    ) -> None:
        stats = self.statistics

        n_tickers = stats.number_of_tickers
        n_days = stats.number_of_return_days
        symbols = np.asarray(stats.symbols, dtype=str)

        if not 2 <= window <= n_days:
            raise ValueError(
                f"window must be between 2 and {n_days}"
            )

        if not 2 <= beta_window <= n_days:
            raise ValueError(
                f"beta_window must be between 2 and {n_days}"
            )

        if neighbour_count < 1:
            raise ValueError(
                "neighbour_count must be positive"
            )

        if neighbour_count >= n_tickers:
            raise ValueError(
                "neighbour_count must be smaller than "
                "the number of tickers"
            )

        rolling = stats.rolling_market_structure(
            window=window
        )

        end_days = np.asarray(
            rolling["end_days"],
            dtype=np.float64,
        )

        average_correlations = np.asarray(
            rolling["average_correlation"],
            dtype=np.float64,
        )

        pc1_ratios = np.asarray(
            rolling["pc1_ratio"],
            dtype=np.float64,
        )

        effective_dimensions = np.asarray(
            rolling["effective_dimension"],
            dtype=np.float64,
        )

        low_dimension_threshold = float(
            np.quantile(effective_dimensions, 0.33)
        )

        high_dimension_threshold = float(
            np.quantile(effective_dimensions, 0.67)
        )

        # ==========================================================
        # Layout
        # ==========================================================

        figure = plt.figure(figsize=(18, 9))

        grid = figure.add_gridspec(
            4,
            2,
            width_ratios=(1.05, 2.2),
            height_ratios=(0.75, 1.0, 1.0, 0.12),
            left=0.025,
            right=0.97,
            top=0.94,
            bottom=0.105,
            wspace=0.18,
            hspace=0.42,
        )

        dendrogram_axis = figure.add_subplot(
            grid[0, 0]
        )

        heatmap_axis = figure.add_subplot(
            grid[1:3, 0]
        )

        eigen_axis = figure.add_subplot(
            grid[0, 1]
        )

        structure_axis = figure.add_subplot(
            grid[1, 1]
        )

        lower_grid = grid[2, 1].subgridspec(
            1,
            2,
            width_ratios=(2.2, 1.0),
            wspace=0.20,
        )

        comparison_axis = figure.add_subplot(
            lower_grid[0, 0]
        )

        beta_axis = figure.add_subplot(
            lower_grid[0, 1]
        )

        # ==========================================================
        # Initial market state
        # ==========================================================

        initial_end = window

        initial_snapshot = (
            stats.market_structure_snapshot(
                end=initial_end,
                window=window,
            )
        )

        initial_clustered = (
            initial_snapshot.correlation_matrix[
                np.ix_(
                    initial_snapshot.cluster_order,
                    initial_snapshot.cluster_order,
                )
            ]
        )

        state: ExplorerState = {
            "selected_ticker": 0,
            "playing": False,
            "cluster_order": (
                initial_snapshot.cluster_order
            ),
        }

        # ==========================================================
        # Dendrogram
        # ==========================================================

        dendrogram(
            initial_snapshot.linkage_matrix,
            ax=dendrogram_axis,
            orientation="top",
            no_labels=True,
        )

        dendrogram_axis.set_title(
            "Correlation Clusters"
        )
        dendrogram_axis.set_xticks([])
        dendrogram_axis.set_ylabel("Distance")

        # ==========================================================
        # Correlation heatmap
        # ==========================================================

        heatmap = heatmap_axis.imshow(
            initial_clustered,
            vmin=-1,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )

        heatmap_axis.set_xticks([])
        heatmap_axis.set_yticks(
            np.arange(n_tickers)
        )
        heatmap_axis.set_yticklabels(
            symbols[
                initial_snapshot.cluster_order
            ],
            fontsize=5,
        )

        heatmap_axis.set_title(
            f"{window}-day clustered correlation: "
            f"days 1–{initial_end}"
        )
        heatmap_axis.set_ylabel("Ticker")

        figure.colorbar(
            heatmap,
            ax=heatmap_axis,
            label="Correlation",
            fraction=0.045,
            pad=0.015,
        )

        selected_rectangle = Rectangle(
            (-0.5, -0.5),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5,
            zorder=20,
        )

        heatmap_axis.add_patch(
            selected_rectangle
        )

        # ==========================================================
        # Eigenvalue spectrum
        # ==========================================================

        displayed_components = min(
            15,
            n_tickers,
        )

        eigen_x = np.arange(
            1,
            displayed_components + 1,
        )

        eigen_bars = eigen_axis.bar(
            eigen_x,
            initial_snapshot.eigenvalue_ratios[
                :displayed_components
            ],
        )

        eigen_axis.set_title(
            "Eigenvalue Spectrum"
        )
        eigen_axis.set_xlabel(
            "Principal Component"
        )
        eigen_axis.set_ylabel(
            "Explained Variance Ratio"
        )
        eigen_axis.grid(
            axis="y",
            alpha=0.25,
        )

        # ==========================================================
        # Rolling market structure
        # ==========================================================

        concentrated_mask = (
            effective_dimensions
            <= low_dimension_threshold
        )

        intermediate_mask = (
            (
                effective_dimensions
                > low_dimension_threshold
            )
            & (
                effective_dimensions
                < high_dimension_threshold
            )
        )

        dispersed_mask = (
            effective_dimensions
            >= high_dimension_threshold
        )

        structure_axis.fill_between(
            end_days,
            0,
            1,
            where=concentrated_mask.tolist(),
            transform=(
                structure_axis
                .get_xaxis_transform()
            ),
            alpha=0.08,
            label="Concentrated regime",
        )

        structure_axis.fill_between(
            end_days,
            0,
            1,
            where=intermediate_mask.tolist(),
            transform=(
                structure_axis
                .get_xaxis_transform()
            ),
            alpha=0.05,
            label="Intermediate regime",
        )

        structure_axis.fill_between(
            end_days,
            0,
            1,
            where=dispersed_mask.tolist(),
            transform=(
                structure_axis
                .get_xaxis_transform()
            ),
            alpha=0.05,
            label="Dispersed regime",
        )

        structure_axis.plot(
            end_days,
            average_correlations,
            label="Average correlation",
        )

        structure_axis.plot(
            end_days,
            pc1_ratios,
            label="PC1 explained variance",
        )

        structure_marker = (
            structure_axis.axvline(
                initial_end,
                linestyle="--",
                linewidth=1.5,
            )
        )

        structure_axis.set_title(
            "Rolling Dependence and Factor Strength"
        )
        structure_axis.set_xlabel("Day")
        structure_axis.set_ylabel("Ratio")
        structure_axis.grid(alpha=0.25)

        dimension_axis = (
            structure_axis.twinx()
        )

        dimension_axis.plot(
            end_days,
            effective_dimensions,
            linestyle=":",
            linewidth=2,
            label="Effective dimension",
        )

        dimension_axis.set_ylabel(
            r"Effective Dimension $N_{\mathrm{eff}}$"
        )

        left_handles, left_labels = (
            structure_axis
            .get_legend_handles_labels()
        )

        right_handles, right_labels = (
            dimension_axis
            .get_legend_handles_labels()
        )

        structure_axis.legend(
            left_handles + right_handles,
            left_labels + right_labels,
            loc="upper left",
            ncols=2,
            fontsize=8,
        )

        # ==========================================================
        # Ticker comparison
        # ==========================================================

        comparison_lines = []

        maximum_lines = neighbour_count + 1

        for _ in range(maximum_lines):
            line, = comparison_axis.plot(
                [],
                [],
            )

            comparison_lines.append(line)

        comparison_axis.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        comparison_axis.set_xlabel("Day")
        comparison_axis.set_ylabel(
            "Cumulative Log Return"
        )
        comparison_axis.grid(alpha=0.25)

        # ==========================================================
        # Rolling beta
        # ==========================================================

        beta_line, = beta_axis.plot(
            [],
            [],
        )

        beta_axis.axhline(
            1,
            linestyle="--",
            linewidth=1,
        )

        beta_axis.axhline(
            0,
            linewidth=1,
        )

        beta_axis.set_xlabel("Day")
        beta_axis.set_ylabel("Beta")
        beta_axis.grid(alpha=0.25)

        # ==========================================================
        # Text and controls
        # ==========================================================

        information_text = figure.text(
            0.025,
            0.075,
            "",
            fontsize=8,
            family="monospace",
            va="bottom",
        )

        slider_axis = figure.add_axes(
            (0.26, 0.027, 0.55, 0.025)
        )

        day_slider = Slider(
            ax=slider_axis,
            label="Window end day",
            valmin=window,
            valmax=n_days,
            valinit=initial_end,
            valstep=1,
        )

        button_axis = figure.add_axes(
            (0.84, 0.014, 0.10, 0.05)
        )

        play_button = Button(
            button_axis,
            "Play",
        )

        # ==========================================================
        # Presentation updates
        # ==========================================================

        def update_selected_rectangle() -> None:
            order = state["cluster_order"]
            selected = state["selected_ticker"]

            locations = np.flatnonzero(
                order == selected
            )

            if len(locations) == 0:
                selected_rectangle.set_visible(
                    False
                )
                return

            cluster_index = int(
                locations[0]
            )

            selected_rectangle.set_visible(
                True
            )

            selected_rectangle.set_xy(
                (
                    cluster_index - 0.5,
                    cluster_index - 0.5,
                )
            )

        def update_heatmap(
            snapshot,
        ) -> None:
            order = snapshot.cluster_order

            clustered = (
                snapshot.correlation_matrix[
                    np.ix_(order, order)
                ]
            )

            state["cluster_order"] = order

            heatmap.set_data(clustered)

            heatmap_axis.set_xlim(
                -0.5,
                n_tickers - 0.5,
            )

            heatmap_axis.set_ylim(
                n_tickers - 0.5,
                -0.5,
            )

            heatmap_axis.set_yticks(
                np.arange(n_tickers)
            )

            heatmap_axis.set_yticklabels(
                symbols[order],
                fontsize=5,
            )

            heatmap_axis.set_title(
                f"{window}-day clustered correlation: "
                f"days "
                f"{snapshot.end - window + 1}"
                f"–{snapshot.end}"
            )

            update_selected_rectangle()

        def update_dendrogram(
            snapshot,
        ) -> None:
            dendrogram_axis.clear()

            dendrogram(
                snapshot.linkage_matrix,
                ax=dendrogram_axis,
                orientation="top",
                no_labels=True,
            )

            dendrogram_axis.set_title(
                "Correlation Clusters"
            )
            dendrogram_axis.set_xticks([])
            dendrogram_axis.set_ylabel(
                "Distance"
            )
            dendrogram_axis.set_xlim(
                0,
                10 * n_tickers,
            )

        def update_eigenvalues(
            snapshot,
        ) -> None:
            ratios = (
                snapshot.eigenvalue_ratios[
                    :displayed_components
                ]
            )

            for bar, height in zip(
                eigen_bars,
                ratios,
            ):
                bar.set_height(float(height))

            maximum = float(
                np.max(ratios)
            )

            eigen_axis.set_ylim(
                0,
                max(0.05, maximum * 1.15),
            )

        def update_comparison(
            end: int,
            selected: int,
        ) -> None:
            strongest, _ = (
                stats.related_tickers(
                    ticker_index=selected,
                    end=end,
                    window=window,
                    count=neighbour_count,
                )
            )

            displayed = np.concatenate(
                (
                    np.asarray(
                        [selected],
                        dtype=np.intp,
                    ),
                    strongest,
                )
            )

            for line, ticker_index in zip(
                comparison_lines,
                displayed,
            ):
                index = int(ticker_index)

                values = (
                    stats.cumulative_log_returns(
                        ticker_index=index,
                        end=end,
                    )
                )

                line.set_xdata(
                    np.arange(len(values))
                )
                line.set_ydata(values)
                line.set_label(
                    symbols[index]
                )
                line.set_visible(True)

            # Hide unused lines if neighbour_count changes.
            for line in comparison_lines[
                len(displayed):
            ]:
                line.set_visible(False)

            comparison_axis.set_xlim(
                0,
                max(1, end),
            )

            comparison_axis.relim()
            comparison_axis.autoscale_view(
                scalex=False,
                scaley=True,
            )

            comparison_axis.set_title(
                f"{symbols[selected]} "
                "and Closest Peers"
            )

            comparison_axis.legend(
                loc="best",
                fontsize=8,
            )

        def update_beta(
            end: int,
            selected: int,
        ) -> None:
            beta_days, beta_values = (
                stats.rolling_beta(
                    ticker_index=selected,
                    end=end,
                    window=beta_window,
                )
            )

            beta_line.set_xdata(beta_days)
            beta_line.set_ydata(beta_values)

            beta_axis.set_xlim(
                beta_window,
                max(beta_window + 1, end),
            )

            beta_axis.relim()
            beta_axis.autoscale_view(
                scalex=False,
                scaley=True,
            )

            beta_axis.set_title(
                f"{symbols[selected]} "
                "Rolling Beta"
            )

        def update_information(
            snapshot,
            selected: int,
        ) -> None:
            strongest, weakest = (
                stats.related_tickers(
                    ticker_index=selected,
                    end=snapshot.end,
                    window=window,
                    count=neighbour_count,
                )
            )

            strongest_pair, weakest_pair = (
                stats.strongest_pairs(
                    end=snapshot.end,
                    window=window,
                )
            )

            ticker_features = (
                stats.ticker_snapshot(
                    ticker_index=selected,
                    end=snapshot.end,
                    window=window,
                    neighbour_count=(
                        neighbour_count
                    ),
                )
            )

            top_text = ", ".join(
                (
                    f"{symbols[int(index)]} "
                    f"("
                    f"{snapshot.correlation_matrix[
                        selected,
                        int(index),
                    ]:+.2f}"
                    f")"
                )
                for index in strongest
            )

            lowest_text = ", ".join(
                (
                    f"{symbols[int(index)]} "
                    f"("
                    f"{snapshot.correlation_matrix[
                        selected,
                        int(index),
                    ]:+.2f}"
                    f")"
                )
                for index in weakest
            )

            information_text.set_text(
                f"Selected: "
                f"{ticker_features.symbol}    "
                f"Mean: "
                f"{ticker_features.mean_return:+.4f}    "
                f"Vol: "
                f"{ticker_features.volatility:.4f}    "
                f"Sharpe: "
                f"{ticker_features.sharpe:+.3f}    "
                f"Beta: "
                f"{ticker_features.beta:+.3f}    "
                f"Drawdown: "
                f"{ticker_features.drawdown:+.3f}\n"
                f"Top: {top_text}    "
                f"Lowest: {lowest_text}\n"
                f"Market μ: "
                f"{snapshot.mean_return:+.4f}    "
                f"σ: "
                f"{snapshot.volatility:.4f}    "
                f"Breadth: "
                f"{snapshot.breadth:.2%}    "
                f"PC1: "
                f"{snapshot.pc1_ratio:.3f}    "
                f"Neff: "
                f"{snapshot.effective_dimension:.2f}    "
                f"Strongest pair: "
                f"{symbols[strongest_pair[0]]}"
                f" ↔ "
                f"{symbols[strongest_pair[1]]}"
                f" ({strongest_pair[2]:+.2f})    "
                f"Weakest pair: "
                f"{symbols[weakest_pair[0]]}"
                f" ↔ "
                f"{symbols[weakest_pair[1]]}"
                f" ({weakest_pair[2]:+.2f})"
            )

        def update_dashboard(
            end_value: float,
        ) -> None:
            end = int(end_value)
            selected = state["selected_ticker"]

            snapshot = (
                stats.market_structure_snapshot(
                    end=end,
                    window=window,
                )
            )

            update_heatmap(snapshot)
            update_dendrogram(snapshot)
            update_eigenvalues(snapshot)
            update_comparison(
                end=end,
                selected=selected,
            )
            update_beta(
                end=end,
                selected=selected,
            )
            update_information(
                snapshot=snapshot,
                selected=selected,
            )

            structure_marker.set_xdata(
                [end, end]
            )

            figure.canvas.draw_idle()

        def on_heatmap_click(
            event: MouseEvent,
        ) -> None:
            if event.inaxes is not heatmap_axis:
                return

            if event.ydata is None:
                return

            clustered_index = int(
                np.floor(event.ydata + 0.5)
            )

            if not (
                0
                <= clustered_index
                < n_tickers
            ):
                return

            order = state["cluster_order"]

            state["selected_ticker"] = int(
                order[clustered_index]
            )

            update_dashboard(
                float(day_slider.val)
            )

        def toggle_play(_: object) -> None:
            state["playing"] = not state[
                "playing"
            ]

            play_button.label.set_text(
                "Pause"
                if state["playing"]
                else "Play"
            )

            figure.canvas.draw_idle()

        def animate(_: int) -> None:
            if not state["playing"]:
                return

            next_day = (
                int(day_slider.val) + 1
            )

            if next_day > n_days:
                next_day = window

            day_slider.set_val(next_day)

        day_slider.on_changed(
            update_dashboard
        )

        play_button.on_clicked(
            toggle_play
        )

        figure.canvas.mpl_connect(
            "button_press_event",
            on_heatmap_click,
        )

        # Initial render of all dynamic panels.
        update_dashboard(
            float(initial_end)
        )

        self._slider = day_slider
        self._play_button = play_button
        self._animation = FuncAnimation(
            figure,
            animate,
            interval=interval,
            cache_frame_data=False,
        )

        figure.suptitle(
            "Interactive Market Statistics Dashboard",
            fontsize=16,
            y=0.985,
        )

        plt.show()
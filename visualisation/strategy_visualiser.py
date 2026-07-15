from __future__ import annotations

from typing import Literal,TypedDict

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.animation import FuncAnimation

from matplotlib.backend_bases import MouseEvent

from matplotlib.patches import Rectangle

from matplotlib.widgets import Button, Slider

from numpy.typing import NDArray

from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

from scipy.spatial.distance import squareform

from models.ticker_universe import TickerUniverse

from models.ticker import Ticker

class ExplorerState(TypedDict):

    selected_ticker: int

    playing: bool

    cluster_order: NDArray[np.intp]

class StrategyVisualizer:
    def plot_ticker(self, ticker: Ticker) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(ticker.prices)

        plt.title(f"{ticker.symbol} Price History")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.show()

    def plot_normalised_tickers(
        self,
        universe,
        start: int = 0,
        end: int | None = None,
        scale: Literal["indexed", "log_return"] = "indexed",
        show_mean: bool = True,
        show_sigma_bands: bool = True,
        show_best_fit: bool = False,
    ) -> None:
        if scale not in {"indexed", "log_return"}:
            raise ValueError(
                "scale must be either 'indexed' or 'log_return'"
            )

        series = []

        for ticker in universe:
            prices = np.asarray(ticker.prices[start:end], dtype=float)

            if len(prices) == 0:
                continue

            if np.any(prices <= 0):
                raise ValueError(
                    f"{ticker.symbol} contains non-positive prices"
                )

            if scale == "indexed":
                values = 100 * prices / prices[0]
            else:
                values = np.log(prices / prices[0])

            series.append(values)

        if not series:
            raise ValueError("No ticker data available")

        values_matrix = np.vstack(series)
        days = np.arange(start, start + values_matrix.shape[1])

        mean_values = np.mean(values_matrix, axis=0)
        std_values = np.std(values_matrix, axis=0)

        figure, axis = plt.subplots(figsize=(14, 8))

        for values in values_matrix:
            axis.plot(
                days,
                values,
                alpha=0.35,
                linewidth=0.9,
            )

        median_values = np.median(values_matrix, axis=0)

        axis.plot(

            days,

            median_values,

            linewidth=2,

            linestyle=":",

            label="Cross-sectional median",

        )

        if show_sigma_bands:
            axis.fill_between(
                days,
                mean_values - 3 * std_values,
                mean_values + 3 * std_values,
                alpha=0.08,
                label="Mean ± 3σ",
            )
            axis.fill_between(
                days,
                mean_values - 2 * std_values,
                mean_values + 2 * std_values,
                alpha=0.12,
                label="Mean ± 2σ",
            )
            axis.fill_between(
                days,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.18,
                label="Mean ± 1σ",
            )

        if show_mean:
            axis.plot(
                days,
                mean_values,
                linewidth=2.5,
                label="Cross-sectional mean",
            )

        if show_best_fit:
            slope, intercept = np.polyfit(days, mean_values, deg=1)
            fitted_mean = slope * days + intercept

            axis.plot(
                days,
                fitted_mean,
                linestyle="--",
                linewidth=2,
                label="Linear fit of mean",
            )

        axis.set_title(
            "Normalised Ticker Performance"
            if scale == "indexed"
            else "Cumulative Log Returns"
        )
        axis.set_xlabel("Day")
        axis.set_ylabel(
            "Indexed price (start = 100)"
            if scale == "indexed"
            else "Cumulative log return"
        )

        if scale == "log_return":
            axis.axhline(0, linewidth=1)

        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        plt.show()

    def plot_market_structure_dashboard(

        self,

        universe: TickerUniverse,

        window: int = 60,

        interval: int = 100,

    ) -> None:

        """

        Display an interactive market-structure dashboard.

        The slider selects the final return observation used in the rolling

        correlation window. The play button animates the window through time.

        Args:

            universe:

                Collection of all Algothon tickers.

            window:

                Number of daily returns used for each rolling calculation.

            interval:

                Animation delay in milliseconds.

        """

        prices = universe.as_price_matrix()

        if prices.ndim != 2:

            raise ValueError("Price matrix must be two-dimensional")

        if np.any(prices <= 0):

            raise ValueError("All prices must be strictly positive")

        symbols = list(universe.symbols)

        returns = np.diff(np.log(prices), axis=1)

        n_tickers, n_returns = returns.shape

        if window < 2:

            raise ValueError("window must be at least 2")

        if window > n_returns:

            raise ValueError(

                f"window={window} exceeds the {n_returns} available returns"

            )

        end_days = np.arange(window, n_returns + 1)

        average_correlations = np.empty(len(end_days))

        pc1_ratios = np.empty(len(end_days))

        effective_dimensions = np.empty(len(end_days))

        def correlation_at(end: int) -> np.ndarray:

            window_returns = returns[:, end - window:end]

            correlation = np.corrcoef(window_returns)

            # Handle a constant ticker if one ever appears.

            correlation = np.nan_to_num(

                correlation,

                nan=0.0,

                posinf=0.0,

                neginf=0.0,

            )

            np.fill_diagonal(correlation, 1.0)

            return correlation

        def calculate_structure_metrics(

            correlation: np.ndarray,

        ) -> tuple[float, float, float]:

            upper_triangle = correlation[

                np.triu_indices_from(correlation, k=1)

            ]

            average_correlation = float(np.mean(upper_triangle))

            # Correlation matrices are symmetric, so eigvalsh is appropriate.

            eigenvalues = np.linalg.eigvalsh(correlation)

            eigenvalues = np.clip(eigenvalues, 0.0, None)

            total_variance = float(np.sum(eigenvalues))

            if total_variance == 0:

                return average_correlation, 0.0, 0.0

            variance_ratios = eigenvalues / total_variance

            pc1_ratio = float(np.max(variance_ratios))

            effective_dimension = float(

                1.0 / np.sum(variance_ratios**2)

            )

            return (

                average_correlation,

                pc1_ratio,

                effective_dimension,

            )

        # Precompute the rolling scalar metrics.

        for index, end in enumerate(end_days):

            correlation = correlation_at(end)

            (

                average_correlations[index],

                pc1_ratios[index],

                effective_dimensions[index],

            ) = calculate_structure_metrics(correlation)

        # --------------------------------------------------------------

        # Create dashboard

        # --------------------------------------------------------------

        figure, axes = plt.subplots(

            2,

            2,

            figsize=(14, 8),

        )

        figure.subplots_adjust(

            bottom=0.16,

            hspace=0.34,

            wspace=0.25,

        )

        heatmap_axis = axes[0, 0]

        correlation_axis = axes[0, 1]

        pc1_axis = axes[1, 0]

        dimension_axis = axes[1, 1]

        initial_end = window

        initial_correlation = correlation_at(initial_end)

        heatmap = heatmap_axis.imshow(

            initial_correlation,

            vmin=-1,

            vmax=1,

            aspect="auto",

            interpolation="nearest",

        )

        heatmap_axis.set_title(

            f"{window}-day correlation matrix: day {initial_end}"

        )

        figure.colorbar(

            heatmap,

            ax=heatmap_axis,

            label="Correlation",

            fraction=0.046,

            pad=0.04,

        )

        # --------------------------------------------------------------

        # Average correlation

        # --------------------------------------------------------------

        correlation_axis.plot(

            end_days,

            average_correlations,

        )

        correlation_marker = correlation_axis.axvline(

            initial_end,

            linestyle="--",

        )

        correlation_axis.axhline(

            0,

            linewidth=1,

        )

        correlation_axis.set_title(

            "Rolling Average Pairwise Correlation"

        )

        correlation_axis.set_xlabel("Day")

        correlation_axis.set_ylabel("Average correlation")

        # --------------------------------------------------------------

        # PC1 ratio

        # --------------------------------------------------------------

        pc1_axis.plot(

            end_days,

            pc1_ratios,

        )

        pc1_marker = pc1_axis.axvline(

            initial_end,

            linestyle="--",

        )

        pc1_axis.set_title(

            "Rolling PC1 Explained Variance"

        )

        pc1_axis.set_xlabel("Day")

        pc1_axis.set_ylabel("Variance ratio")

        # --------------------------------------------------------------

        # Effective dimension

        # --------------------------------------------------------------

        dimension_axis.plot(

            end_days,

            effective_dimensions,

        )

        dimension_marker = dimension_axis.axvline(

            initial_end,

            linestyle="--",

        )

        dimension_axis.set_title("Rolling Effective Dimension")

        dimension_axis.set_xlabel("Day")

        dimension_axis.set_ylabel(r"$N_{\mathrm{eff}}$")

        for axis in (

            correlation_axis,

            pc1_axis,

            dimension_axis,

        ):

            axis.grid(alpha=0.25)

        # --------------------------------------------------------------

        # Slider and playback controls

        # --------------------------------------------------------------

        slider_axis = figure.add_axes(
            (0.18, 0.065, 0.62, 0.025)
        )

        day_slider = Slider(

            ax=slider_axis,

            label="Window end day",

            valmin=window,

            valmax=n_returns,

            valinit=initial_end,

            valstep=1,

        )

        button_axis = figure.add_axes(
            (0.83, 0.05, 0.10, 0.055)
        )

        play_button = Button(

            button_axis,

            "Play",

        )

        animation_state = {

            "playing": False,

        }

        def update_dashboard(end_value: float) -> None:

            end = int(end_value)

            correlation = correlation_at(end)

            heatmap.set_data(correlation)

            heatmap_axis.set_title(

                f"{window}-day correlation matrix: "

                f"days {end - window + 1}–{end}"

            )

            correlation_marker.set_xdata([end, end])

            pc1_marker.set_xdata([end, end])

            dimension_marker.set_xdata([end, end])

            figure.canvas.draw_idle()

        def animation_frame(_: int) -> None:

            if not animation_state["playing"]:

                return

            next_day = int(day_slider.val) + 1

            if next_day > n_returns:

                next_day = window

            day_slider.set_val(next_day)

        def toggle_animation(_: object) -> None:

            animation_state["playing"] = not animation_state["playing"]

            play_button.label.set_text(

                "Pause"

                if animation_state["playing"]

                else "Play"

            )

            figure.canvas.draw_idle()

        day_slider.on_changed(update_dashboard)

        play_button.on_clicked(toggle_animation)

        # Store references on self so Matplotlib does not garbage-collect them.

        self._market_structure_animation = FuncAnimation(

            figure,

            animation_frame,

            interval=interval,

            cache_frame_data=False,

        )

        self._market_structure_slider = day_slider

        self._market_structure_button = play_button

        figure.suptitle(

            "Rolling Market Structure Dashboard",

            fontsize=16,

        )

        plt.show()

    def plot_interactive_market_explorer(
        self,
        universe: TickerUniverse,
        window: int = 60,
        interval: int = 120,
        beta_window: int = 30,
    ) -> None:
        prices = np.asarray(
            universe.as_price_matrix(),
            dtype=np.float64,
        )

        symbols = np.asarray(
            universe.symbols,
            dtype=str,
        )

        if prices.ndim != 2:
            raise ValueError("Price matrix must be two-dimensional")

        if np.any(prices <= 0):
            raise ValueError("All prices must be strictly positive")

        returns = np.diff(np.log(prices), axis=1)

        n_tickers, n_days = returns.shape

        if not 2 <= window <= n_days:
            raise ValueError(
                f"window must be between 2 and {n_days}"
            )

        if not 2 <= beta_window <= n_days:
            raise ValueError(
                f"beta_window must be between 2 and {n_days}"
            )

        end_days = np.arange(window, n_days + 1)

        # ==============================================================
        # Calculations
        # ==============================================================

        def correlation_at(end: int) -> NDArray[np.float64]:
            sample = returns[:, end - window:end]

            correlation = np.corrcoef(sample)

            correlation = np.nan_to_num(
                correlation,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            correlation = np.clip(correlation, -1.0, 1.0)
            np.fill_diagonal(correlation, 1.0)

            return correlation

        def linkage_from_correlation(
            correlation: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            distance = np.clip(
                1.0 - correlation,
                0.0,
                2.0,
            )

            np.fill_diagonal(distance, 0.0)

            condensed = squareform(
                distance,
                checks=False,
            )

            return linkage(
                condensed,
                method="average",
            )

        def cluster_correlation(
            correlation: NDArray[np.float64],
        ) -> tuple[
            NDArray[np.float64],
            NDArray[np.intp],
            NDArray[np.float64],
        ]:
            linkage_matrix = linkage_from_correlation(correlation)

            dendrogram_result = dendrogram(
                linkage_matrix,
                no_plot=True,
            )

            order = np.asarray(
                dendrogram_result["leaves"],
                dtype=np.intp,
            )

            clustered = correlation[np.ix_(order, order)]

            return clustered, order, linkage_matrix

        def pairwise_values(
            correlation: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            indices = np.triu_indices_from(
                correlation,
                k=1,
            )

            return correlation[indices]

        def eigen_statistics(
            correlation: NDArray[np.float64],
        ) -> tuple[
            NDArray[np.float64],
            float,
            float,
        ]:
            eigenvalues = np.linalg.eigvalsh(correlation)

            eigenvalues = np.sort(
                np.clip(eigenvalues, 0.0, None)
            )[::-1]

            total = float(np.sum(eigenvalues))

            if total <= 0:
                ratios = np.zeros_like(eigenvalues)
                return ratios, 0.0, 0.0

            ratios = eigenvalues / total

            pc1_ratio = float(ratios[0])

            concentration = float(
                np.sum(ratios**2)
            )

            effective_dimension = (
                1.0 / concentration
                if concentration > 0
                else 0.0
            )

            return (
                ratios,
                pc1_ratio,
                effective_dimension,
            )

        def strongest_pairs(
            correlation: NDArray[np.float64],
        ) -> tuple[
            tuple[int, int, float],
            tuple[int, int, float],
        ]:
            values = correlation.copy()
            np.fill_diagonal(values, np.nan)

            strongest_flat = int(
                np.nanargmax(values)
            )

            weakest_flat = int(
                np.nanargmin(values)
            )

            strongest_row, strongest_col = np.unravel_index(
                strongest_flat,
                values.shape,
            )

            weakest_row, weakest_col = np.unravel_index(
                weakest_flat,
                values.shape,
            )

            strongest = (
                int(strongest_row),
                int(strongest_col),
                float(values[strongest_row, strongest_col]),
            )

            weakest = (
                int(weakest_row),
                int(weakest_col),
                float(values[weakest_row, weakest_col]),
            )

            return strongest, weakest

        def related_tickers(
            correlation: NDArray[np.float64],
            ticker_index: int,
            count: int = 3,
        ) -> tuple[
            NDArray[np.intp],
            NDArray[np.intp],
        ]:
            row = correlation[ticker_index].copy()
            row[ticker_index] = np.nan

            valid = np.flatnonzero(~np.isnan(row))

            sorted_indices = valid[
                np.argsort(row[valid])
            ]

            weakest = sorted_indices[:count]
            strongest = sorted_indices[-count:][::-1]

            return (
                strongest.astype(np.intp),
                weakest.astype(np.intp),
            )

        def cumulative_log_returns(
            ticker_index: int,
            end: int,
        ) -> NDArray[np.float64]:
            ticker_prices = prices[ticker_index, :end + 1]

            return np.log(
                ticker_prices / ticker_prices[0]
            )

        def rolling_beta(
            ticker_index: int,
            end_day: int,
        ) -> tuple[
            NDArray[np.int64],
            NDArray[np.float64],
        ]:
            market_return = np.mean(
                returns,
                axis=0,
            )

            ticker_return = returns[ticker_index]

            final_end = min(
                end_day,
                n_days,
            )

            beta_days: list[int] = []
            beta_values: list[float] = []

            for end in range(
                beta_window,
                final_end + 1,
            ):
                market_window = market_return[
                    end - beta_window:end
                ]

                ticker_window = ticker_return[
                    end - beta_window:end
                ]

                market_variance = float(
                    np.var(
                        market_window,
                        ddof=1,
                    )
                )

                if market_variance <= 0:
                    beta = np.nan
                else:
                    covariance = float(
                        np.cov(
                            ticker_window,
                            market_window,
                            ddof=1,
                        )[0, 1]
                    )

                    beta = covariance / market_variance

                beta_days.append(end)
                beta_values.append(beta)

            return (
                np.asarray(beta_days, dtype=np.int64),
                np.asarray(beta_values, dtype=np.float64),
            )

        # ==============================================================
        # Precompute rolling structure
        # ==============================================================

        average_correlations = np.empty(
            len(end_days),
            dtype=np.float64,
        )

        pc1_ratios = np.empty(
            len(end_days),
            dtype=np.float64,
        )

        effective_dimensions = np.empty(
            len(end_days),
            dtype=np.float64,
        )

        for index, end in enumerate(end_days):
            correlation = correlation_at(int(end))

            average_correlations[index] = float(
                np.mean(pairwise_values(correlation))
            )

            (
                _,
                pc1_ratios[index],
                effective_dimensions[index],
            ) = eigen_statistics(correlation)

        # Dynamic regime boundaries based on the observed distribution.
        low_dimension_threshold = float(
            np.quantile(effective_dimensions, 0.33)
        )

        high_dimension_threshold = float(
            np.quantile(effective_dimensions, 0.67)
        )

        # ==============================================================
        # Figure layout
        # ==============================================================

        figure = plt.figure(figsize=(18, 8))

        grid = figure.add_gridspec(
            4,
            2,
            width_ratios=(1.05, 2.2),
            height_ratios=(0.75, 1.0, 1.0, 0.12),
            left=0.035,
            right=0.985,
            top=0.91,
            bottom=0.055,
            wspace=0.16,
            hspace=0.42,
        )

        dendrogram_axis = figure.add_subplot(grid[0, 0])
        heatmap_axis = figure.add_subplot(grid[1:3, 0])

        eigen_axis = figure.add_subplot(grid[0, 1])
        structure_axis = figure.add_subplot(grid[1, 1])

        lower_grid = grid[2, 1].subgridspec(
            1,
            2,
            width_ratios=(2.2, 1.0),
            wspace=0.20,
        )

        comparison_axis = figure.add_subplot(lower_grid[0, 0])
        beta_axis = figure.add_subplot(lower_grid[0, 1])

        # Almost edge-to-edge layout.
        figure.subplots_adjust(
            left=0.025,
            right=0.995,
            top=0.94,
            bottom=0.105,
            wspace=0.24,
            hspace=0.34,
        )

        # ==============================================================
        # Initial state
        # ==============================================================

        initial_end = window
        initial_correlation = correlation_at(initial_end)

        (
            clustered_correlation,
            initial_order,
            initial_linkage,
        ) = cluster_correlation(initial_correlation)


        state: ExplorerState = {
            "selected_ticker": 0,
            "playing": False,
            "cluster_order": initial_order,
        }

        # ==============================================================
        # Dendrogram
        # ==============================================================

        dendrogram(
            initial_linkage,
            ax=dendrogram_axis,
            orientation="top",
            no_labels=True,
        )

        dendrogram_axis.set_title("Correlation Clusters")
        dendrogram_axis.set_xticks([])
        dendrogram_axis.set_ylabel("Distance")

        # ==============================================================
        # Heatmap
        # ==============================================================

        heatmap = heatmap_axis.imshow(
            clustered_correlation,
            vmin=-1,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )

        ordered_symbols = symbols[initial_order]

        heatmap_axis.set_xticks([])
        heatmap_axis.set_xlabel("")

        heatmap_axis.set_yticks(np.arange(n_tickers))
        heatmap_axis.set_yticklabels(
            ordered_symbols,
            fontsize=5,
        )

        heatmap_axis.set_title(
            f"{window}-day clustered correlation: "
            f"days 1–{window}"
        )

        heatmap_axis.set_xlabel("Ticker")
        heatmap_axis.set_ylabel("Ticker")

        colorbar = figure.colorbar(
            heatmap,
            ax=heatmap_axis,
            label="Correlation",
            fraction=0.045,
            pad=0.015,
        )

        # Selected ticker markers.

        selected_rectangle = Rectangle(
            (-0.5, -0.5),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5,
            zorder=20,
        )

        heatmap_axis.add_patch(selected_rectangle)

        # ==============================================================
        # Eigenvalue spectrum
        # ==============================================================

        initial_ratios, _, _ = eigen_statistics(
            initial_correlation
        )

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
            initial_ratios[:displayed_components],
        )

        eigen_axis.set_title("Eigenvalue Spectrum")
        eigen_axis.set_xlabel("Principal Component")
        eigen_axis.set_ylabel("Explained Variance Ratio")
        eigen_axis.grid(axis="y", alpha=0.25)

        # ==============================================================
        # Rolling structure
        # ==============================================================

        # Regime shading.
        structure_axis.fill_between(
            end_days,
            0,
            1,
            where=(effective_dimensions
            <= low_dimension_threshold).tolist(),
            transform=structure_axis.get_xaxis_transform(),
            alpha=0.08,
            label="Concentrated regime",
        )

        structure_axis.fill_between(
            end_days,
            0,
            1,
            where=(
                (effective_dimensions > low_dimension_threshold)
                & (
                    effective_dimensions
                    < high_dimension_threshold
                ).tolist()
            ),
            transform=structure_axis.get_xaxis_transform(),
            alpha=0.05,
            label="Intermediate regime",
        )

        structure_axis.fill_between(
            end_days,
            0,
            1,
            where=(effective_dimensions
            >= high_dimension_threshold).tolist(),
            transform=structure_axis.get_xaxis_transform(),
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

        structure_marker = structure_axis.axvline(
            initial_end,
            linestyle="--",
            linewidth=1.5,
        )

        structure_axis.set_title(
            "Rolling Dependence and Factor Strength"
        )

        structure_axis.set_xlabel("Day")
        structure_axis.set_ylabel("Ratio")
        structure_axis.grid(alpha=0.25)
        structure_axis.legend(
            loc="upper left",
            ncols=2,
            fontsize=8,
        )

        dimension_axis = structure_axis.twinx()

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

        dimension_axis.legend(
            loc="upper right",
            fontsize=8,
        )

        # ==============================================================
        # Ticker comparison plot
        # ==============================================================

        comparison_lines = []

        for _ in range(4):
            line, = comparison_axis.plot(
                np.arange(prices.shape[1]),
                np.zeros(prices.shape[1]),
            )

            comparison_lines.append(line)

        comparison_axis.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        comparison_axis.set_title(
            "Selected Ticker and Closest Peers"
        )

        comparison_axis.set_xlabel("Day")
        comparison_axis.set_ylabel("Cumulative Log Return")
        comparison_axis.grid(alpha=0.25)

        # ==============================================================
        # Rolling beta
        # ==============================================================

        initial_beta_days, initial_beta = rolling_beta(0, initial_end)

        beta_line, = beta_axis.plot(
            initial_beta_days,
            initial_beta,
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

        beta_axis.set_title(
            f"{symbols[0]} Rolling Beta"
        )

        beta_axis.set_xlabel("Day")
        beta_axis.set_ylabel("Beta")
        beta_axis.grid(alpha=0.25)

        information_text = figure.text(
            0.04,
            0.085,
            "",
            fontsize=8,
            family="monospace",
            va="bottom",
        )

        # ==============================================================
        # Controls
        # ==============================================================

        slider_axis = figure.add_axes(
            (0.26, 0.035, 0.55, 0.025)
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
            (0.84, 0.022, 0.10, 0.05)
        )

        play_button = Button(
            button_axis,
            "Play",
        )

        # ==============================================================
        # Updates
        # ==============================================================

        def update_heatmap_markers() -> None:
            order = state["cluster_order"]
            selected = state["selected_ticker"]

            locations = np.flatnonzero(order == selected)

            if len(locations) == 0:
                selected_rectangle.set_visible(False)
                return

            selected_cluster_index = int(locations[0])

            selected_rectangle.set_visible(True)
            selected_rectangle.set_xy(
                (
                    selected_cluster_index - 0.5,
                    selected_cluster_index - 0.5,
                )
            )

            selected_rectangle.set_zorder(20)

        def update_comparison_plot(
            correlation: NDArray[np.float64],
            end: int,
        ) -> None:
            selected = state["selected_ticker"]

            strongest_indices, _ = related_tickers(
                correlation,
                selected,
                count=3,
            )

            displayed_indices = np.concatenate(
                (
                    np.asarray([selected], dtype=np.intp),
                    strongest_indices,
                )
            )

            for line, ticker_index in zip(
                comparison_lines,
                displayed_indices,
            ):
                index = int(ticker_index)

                values = cumulative_log_returns(
                    index,
                    end,
                )

                line.set_xdata(
                    np.arange(len(values))
                )

                line.set_ydata(values)
                line.set_label(symbols[index])

            comparison_axis.set_xlim(
                0,
                max(1, end),
            )

            comparison_axis.relim()
            comparison_axis.autoscale_view(
                scalex=False,
                scaley=True,
            )

            comparison_axis.legend(
                loc="best",
                fontsize=8,
            )

            comparison_axis.set_title(
                f"{symbols[selected]} and Closest Peers"
            )

        def update_beta_plot(end: int) -> None:
            selected = state["selected_ticker"]

            beta_days, beta_values = rolling_beta(
                selected,
                end,
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
                f"{symbols[selected]} Rolling Beta"
            )

        def update_information(
            correlation: NDArray[np.float64],
        ) -> None:
            selected = state["selected_ticker"]

            strongest_indices, weakest_indices = related_tickers(
                correlation,
                selected,
                count=3,
            )

            strongest_pair, weakest_pair = strongest_pairs(
                correlation
            )

            strongest_text = ", ".join(
                f"{symbols[int(index)]} "
                f"({correlation[selected, int(index)]:+.2f})"
                for index in strongest_indices
            )

            weakest_text = ", ".join(
                f"{symbols[int(index)]} "
                f"({correlation[selected, int(index)]:+.2f})"
                for index in weakest_indices
            )

            information_text.set_text(
                f"Selected: {symbols[selected]}    "
                f"Top: {strongest_text}    "
                f"Lowest: {weakest_text}\n"
                f"Strongest pair: "
                f"{symbols[strongest_pair[0]]} ↔ "
                f"{symbols[strongest_pair[1]]} "
                f"({strongest_pair[2]:+.2f})    "
                f"Weakest pair: "
                f"{symbols[weakest_pair[0]]} ↔ "
                f"{symbols[weakest_pair[1]]} "
                f"({weakest_pair[2]:+.2f})"
            )

        def update_dashboard(end_value: float) -> None:
            end = int(end_value)

            correlation = correlation_at(end)

            (
                clustered,
                order,
                linkage_matrix,
            ) = cluster_correlation(correlation)

            state["cluster_order"] = order

            heatmap.set_data(clustered)

            ordered_symbols = symbols[order]

            heatmap_axis.set_xlim(-0.5, n_tickers - 0.5)

            heatmap_axis.set_ylim(n_tickers - 0.5, -0.5)

            heatmap_axis.set_xticks([])

            heatmap_axis.set_yticks(

                np.arange(n_tickers)

            )

            heatmap_axis.set_yticklabels(

                ordered_symbols,

                fontsize=5,

            )

            heatmap_axis.set_title(

                f"{window}-day clustered correlation: "

                f"days {end - window + 1}–{end}"

            )

            update_heatmap_markers()

            # Redraw dendrogram.
            dendrogram_axis.clear()

            dendrogram(
                linkage_matrix,
                ax=dendrogram_axis,
                orientation="top",
                no_labels=True,
            )

            dendrogram_axis.set_title(
                "Correlation Clusters"
            )

            dendrogram_axis.set_xticks([])
            dendrogram_axis.set_ylabel("Distance")

            dendrogram_axis.set_xlim(0, 10 * n_tickers)

            # Update eigenvalues.
            ratios, _, _ = eigen_statistics(
                correlation
            )

            for bar, height in zip(
                eigen_bars,
                ratios[:displayed_components],
            ):
                bar.set_height(float(height))

            maximum_ratio = float(
                np.max(ratios[:displayed_components])
            )

            eigen_axis.set_ylim(
                0,
                max(0.05, maximum_ratio * 1.15),
            )

            eigen_axis.autoscale_view()

            structure_marker.set_xdata(
                [end, end]
            )

            update_heatmap_markers()
            update_comparison_plot(correlation, end)
            update_beta_plot(end)
            update_information(correlation)

            figure.canvas.draw_idle()

        def on_heatmap_click(
            event: MouseEvent,
        ) -> None:
            if event.inaxes is not heatmap_axis:
                return

            if event.xdata is None or event.ydata is None:
                return

            clustered_index = int(
                round(event.ydata)
            )

            if not 0 <= clustered_index < n_tickers:
                return

            order = state["cluster_order"]

            state["selected_ticker"] = int(
                order[clustered_index]
            )

            correlation = correlation_at(
                int(day_slider.val)
            )

            update_heatmap_markers()
            update_comparison_plot(
                initial_correlation,
                initial_end,
            )
            update_beta_plot(initial_end)
            update_information(correlation)

            figure.canvas.draw_idle()

        def toggle_play(_: object) -> None:
            state["playing"] = not state["playing"]

            play_button.label.set_text(
                "Pause"
                if state["playing"]
                else "Play"
            )

            figure.canvas.draw_idle()

        def animate(_: int) -> None:
            if not state["playing"]:
                return

            next_day = int(day_slider.val) + 1

            if next_day > n_days:
                next_day = window

            day_slider.set_val(next_day)

        day_slider.on_changed(update_dashboard)
        play_button.on_clicked(toggle_play)

        figure.canvas.mpl_connect(
            "button_press_event",
            on_heatmap_click,
        )

        update_heatmap_markers()
        update_comparison_plot(
            initial_correlation,
            initial_end,
        )
        update_beta_plot(initial_end)
        update_information(initial_correlation)

        self._market_explorer_slider = day_slider
        self._market_explorer_button = play_button

        self._market_explorer_animation = FuncAnimation(
            figure,
            animate,
            interval=interval,
            cache_frame_data=False,
        )

        figure.suptitle(
            "Interactive Market Structure Explorer",
            fontsize=16,
            y=0.985,
        )

        plt.show()
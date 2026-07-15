from __future__ import annotations

from collections.abc import Iterable

from sklearn.decomposition import PCA

import pandas as pd

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import entropy, kurtosis, skew

from models.ticker_universe import TickerUniverse

from models.ticker import Ticker

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class MarketStructureSnapshot:
    """
    Market-level features calculated at one point in time.
    """

    end: int
    window: int

    mean_return: float
    median_return: float
    volatility: float
    skewness: float
    excess_kurtosis: float
    breadth: float

    average_correlation: float
    pc1_ratio: float
    effective_dimension: float

    correlation_matrix: FloatArray
    eigenvalue_ratios: FloatArray
    cluster_order: IntArray
    linkage_matrix: FloatArray


@dataclass(frozen=True, slots=True)
class TickerFeatureSnapshot:
    """
    Features for one ticker at one point in time.
    """

    ticker_index: int
    symbol: str
    end: int
    window: int

    mean_return: float
    volatility: float
    sharpe: float
    skewness: float
    excess_kurtosis: float

    beta: float
    market_correlation: float

    momentum: float
    relative_strength: float
    cumulative_log_return: float
    drawdown: float

    strongest_correlations: tuple[int, ...]
    weakest_correlations: tuple[int, ...]


class MarketStatistics:
    """
    Central calculation engine for market analytics and trading features.

    Data orientation:

        rows    = tickers
        columns = time
    """

    def __init__(
        self,
        universe: TickerUniverse | None = None,
        prices: FloatArray | None = None,
    ) -> None:
        if universe is None and prices is None:
            raise ValueError(
                "Provide either a TickerUniverse or a price matrix"
            )

        if universe is not None and prices is not None:
            raise ValueError(
                "Provide universe or prices, not both"
            )

        self.universe = universe

        if universe is not None:
            loaded_prices = universe.as_price_matrix()
            self.symbols = tuple(universe.symbols)
        else:
            assert prices is not None

            loaded_prices = prices
            self.symbols = tuple(
                str(index)
                for index in range(prices.shape[0])
            )

        self.prices = np.asarray(
            loaded_prices,
            dtype=np.float64,
        )

        if self.prices.ndim != 2:
            raise ValueError(
                "prices must have shape "
                "(number_of_tickers, number_of_days)"
            )

        if np.any(~np.isfinite(self.prices)):
            raise ValueError("Prices must be finite")

        if np.any(self.prices <= 0):
            raise ValueError("Prices must be positive")

        self.log_returns = np.diff(
            np.log(self.prices),
            axis=1,
        )

    # ==============================================================
    # Basic properties
    # ==============================================================

    @property
    def number_of_tickers(self) -> int:
        return self.prices.shape[0]

    @property
    def number_of_price_days(self) -> int:
        return self.prices.shape[1]

    @property
    def number_of_return_days(self) -> int:
        return self.log_returns.shape[1]

    @property
    def data(self) -> FloatArray:
        """
        Alias retained for compatibility with your existing code.
        """
        return self.log_returns

    # ==============================================================
    # Cross-sectional statistics through time
    # ==============================================================

    @property
    def mu(self) -> FloatArray:
        return np.mean(self.log_returns, axis=0)

    @property
    def median(self) -> FloatArray:
        return np.median(self.log_returns, axis=0)

    @property
    def sigma(self) -> FloatArray:
        return np.std(
            self.log_returns,
            axis=0,
            ddof=1,
        )

    @property
    def skewness(self) -> FloatArray:
        return np.asarray(
            skew(
                self.log_returns,
                axis=0,
                bias=False,
            ),
            dtype=np.float64,
        )

    @property
    def kurtosis(self) -> FloatArray:
        return np.asarray(
            kurtosis(
                self.log_returns,
                axis=0,
                fisher=True,
                bias=False,
            ),
            dtype=np.float64,
        )

    @property
    def positive_fraction(self) -> FloatArray:
        return np.mean(
            self.log_returns > 0,
            axis=0,
        )

    def percentile(self, q: float) -> FloatArray:
        if not 0 <= q <= 100:
            raise ValueError(
                "q must be between 0 and 100"
            )

        return np.percentile(
            self.log_returns,
            q=q,
            axis=0,
        )

    @property
    def percentile_5(self) -> FloatArray:
        return self.percentile(5)

    @property
    def percentile_25(self) -> FloatArray:
        return self.percentile(25)

    @property
    def percentile_75(self) -> FloatArray:
        return self.percentile(75)

    @property
    def percentile_95(self) -> FloatArray:
        return self.percentile(95)

    @property
    def iqr(self) -> FloatArray:
        return (
            self.percentile_75
            - self.percentile_25
        )

    @property
    def tail_spread(self) -> FloatArray:
        return (
            self.percentile_95
            - self.percentile_5
        )

    @property
    def mean_absolute_return(self) -> FloatArray:
        return np.mean(
            np.abs(self.log_returns),
            axis=0,
        )

    def cross_sectional_entropy(
        self,
        bins: int = 10,
    ) -> FloatArray:
        if bins < 2:
            raise ValueError(
                "bins must be at least 2"
            )

        results = np.empty(
            self.number_of_return_days,
            dtype=np.float64,
        )

        for day in range(
            self.number_of_return_days
        ):
            counts, _ = np.histogram(
                self.log_returns[:, day],
                bins=bins,
            )

            probabilities = (
                counts[counts > 0]
                / counts.sum()
            )

            results[day] = entropy(
                probabilities,
                base=2,
            )

        return results

    # ==============================================================
    # Validation and rolling windows
    # ==============================================================

    def _validate_end(
        self,
        end: int | None,
    ) -> int:
        if end is None:
            return self.number_of_return_days

        if not 1 <= end <= self.number_of_return_days:
            raise ValueError(
                f"end must be between 1 and "
                f"{self.number_of_return_days}"
            )

        return end

    def _window_returns(
        self,
        end: int | None,
        window: int,
    ) -> FloatArray:
        validated_end = self._validate_end(end)

        if window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        if window > validated_end:
            raise ValueError(
                "window cannot exceed available history"
            )

        return self.log_returns[
            :,
            validated_end - window:
            validated_end,
        ]

    # ==============================================================
    # Correlation and clustering
    # ==============================================================

    def correlation_matrix(
        self,
        end: int | None = None,
        window: int = 60,
    ) -> FloatArray:
        sample = self._window_returns(
            end=end,
            window=window,
        )

        matrix = np.corrcoef(sample)

        matrix = np.nan_to_num(
            matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        matrix = np.clip(
            matrix,
            -1.0,
            1.0,
        )

        np.fill_diagonal(matrix, 1.0)

        return np.asarray(
            matrix,
            dtype=np.float64,
        )

    @staticmethod
    def pairwise_correlations(
        correlation: FloatArray,
    ) -> FloatArray:
        indices = np.triu_indices_from(
            correlation,
            k=1,
        )

        return correlation[indices]

    def average_pairwise_correlation(
        self,
        end: int | None = None,
        window: int = 60,
    ) -> float:
        correlation = self.correlation_matrix(
            end=end,
            window=window,
        )

        return float(
            np.mean(
                self.pairwise_correlations(
                    correlation
                )
            )
        )

    @staticmethod
    def linkage_from_correlation(
        correlation: FloatArray,
    ) -> FloatArray:
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

        return np.asarray(
            linkage(
                condensed,
                method="average",
            ),
            dtype=np.float64,
        )

    def clustered_correlation(
        self,
        end: int | None = None,
        window: int = 60,
    ) -> tuple[
        FloatArray,
        IntArray,
        FloatArray,
    ]:
        correlation = self.correlation_matrix(
            end=end,
            window=window,
        )

        linkage_matrix = (
            self.linkage_from_correlation(
                correlation
            )
        )

        result = dendrogram(
            linkage_matrix,
            no_plot=True,
        )

        order = np.asarray(
            result["leaves"],
            dtype=np.intp,
        )

        clustered = correlation[
            np.ix_(order, order)
        ]

        return (
            clustered,
            order,
            linkage_matrix,
        )

    def strongest_pairs(
        self,
        end: int | None = None,
        window: int = 60,
    ) -> tuple[
        tuple[int, int, float],
        tuple[int, int, float],
    ]:
        correlation = self.correlation_matrix(
            end=end,
            window=window,
        )

        values = correlation.copy()
        np.fill_diagonal(values, np.nan)

        strongest_flat = int(
            np.nanargmax(values)
        )
        weakest_flat = int(
            np.nanargmin(values)
        )

        strong_row, strong_column = (
            np.unravel_index(
                strongest_flat,
                values.shape,
            )
        )

        weak_row, weak_column = (
            np.unravel_index(
                weakest_flat,
                values.shape,
            )
        )

        strongest = (
            int(strong_row),
            int(strong_column),
            float(
                values[
                    strong_row,
                    strong_column,
                ]
            ),
        )

        weakest = (
            int(weak_row),
            int(weak_column),
            float(
                values[
                    weak_row,
                    weak_column,
                ]
            ),
        )

        return strongest, weakest

    def related_tickers(
        self,
        ticker_index: int,
        end: int | None = None,
        window: int = 60,
        count: int = 3,
    ) -> tuple[IntArray, IntArray]:
        if not 0 <= ticker_index < self.number_of_tickers:
            raise IndexError(
                "ticker_index is out of range"
            )

        correlation = self.correlation_matrix(
            end=end,
            window=window,
        )

        row = correlation[ticker_index].copy()
        row[ticker_index] = np.nan

        valid = np.flatnonzero(
            ~np.isnan(row)
        )

        ordered = valid[
            np.argsort(row[valid])
        ]

        weakest = ordered[:count]
        strongest = ordered[-count:][::-1]

        return (
            strongest.astype(np.intp),
            weakest.astype(np.intp),
        )

    # ==============================================================
    # PCA and market structure
    # ==============================================================

    @staticmethod
    def eigen_statistics(
        correlation: FloatArray,
    ) -> tuple[
        FloatArray,
        float,
        float,
    ]:
        eigenvalues = np.linalg.eigvalsh(
            correlation
        )

        eigenvalues = np.sort(
            np.clip(
                eigenvalues,
                0.0,
                None,
            )
        )[::-1]

        total = float(
            np.sum(eigenvalues)
        )

        if total <= 0:
            ratios = np.zeros_like(
                eigenvalues
            )

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

    def market_structure_snapshot(
        self,
        end: int | None = None,
        window: int = 60,
    ) -> MarketStructureSnapshot:
        validated_end = self._validate_end(
            end
        )

        sample = self._window_returns(
            end=validated_end,
            window=window,
        )

        latest_cross_section = sample[:, -1]

        correlation = self.correlation_matrix(
            end=validated_end,
            window=window,
        )

        (
            eigenvalue_ratios,
            pc1_ratio,
            effective_dimension,
        ) = self.eigen_statistics(
            correlation
        )

        (
            _,
            order,
            linkage_matrix,
        ) = self.clustered_correlation(
            end=validated_end,
            window=window,
        )

        return MarketStructureSnapshot(
            end=validated_end,
            window=window,
            mean_return=float(
                np.mean(latest_cross_section)
            ),
            median_return=float(
                np.median(latest_cross_section)
            ),
            volatility=float(
                np.std(
                    latest_cross_section,
                    ddof=1,
                )
            ),
            skewness=float(
                skew(
                    latest_cross_section,
                    bias=False,
                )
            ),
            excess_kurtosis=float(
                kurtosis(
                    latest_cross_section,
                    fisher=True,
                    bias=False,
                )
            ),
            breadth=float(
                np.mean(
                    latest_cross_section > 0
                )
            ),
            average_correlation=float(
                np.mean(
                    self.pairwise_correlations(
                        correlation
                    )
                )
            ),
            pc1_ratio=pc1_ratio,
            effective_dimension=(
                effective_dimension
            ),
            correlation_matrix=correlation,
            eigenvalue_ratios=(
                eigenvalue_ratios
            ),
            cluster_order=order,
            linkage_matrix=linkage_matrix,
        )

    def rolling_market_structure(
        self,
        window: int = 60,
    ) -> dict[str, FloatArray]:
        end_days = np.arange(
            window,
            self.number_of_return_days + 1,
        )

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

        for position, end in enumerate(
            end_days
        ):
            correlation = (
                self.correlation_matrix(
                    end=int(end),
                    window=window,
                )
            )

            average_correlations[position] = (
                np.mean(
                    self.pairwise_correlations(
                        correlation
                    )
                )
            )

            (
                _,
                pc1_ratios[position],
                effective_dimensions[position],
            ) = self.eigen_statistics(
                correlation
            )

        return {
            "end_days": end_days,
            "average_correlation": (
                average_correlations
            ),
            "pc1_ratio": pc1_ratios,
            "effective_dimension": (
                effective_dimensions
            ),
        }

    # ==============================================================
    # Ticker-level calculations
    # ==============================================================

    def cumulative_log_returns(
        self,
        ticker_index: int,
        end: int | None = None,
    ) -> FloatArray:
        validated_end = self._validate_end(
            end
        )

        # Return day t corresponds to price columns 0 ... t.
        ticker_prices = self.prices[
            ticker_index,
            :validated_end + 1,
        ]

        return np.log(
            ticker_prices
            / ticker_prices[0]
        )

    def rolling_beta(
        self,
        ticker_index: int,
        end: int | None = None,
        window: int = 30,
    ) -> tuple[
        NDArray[np.int64],
        FloatArray,
    ]:
        validated_end = self._validate_end(
            end
        )

        if window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        market_returns = np.mean(
            self.log_returns,
            axis=0,
        )

        ticker_returns = self.log_returns[
            ticker_index
        ]

        days: list[int] = []
        values: list[float] = []

        for current_end in range(
            window,
            validated_end + 1,
        ):
            ticker_sample = ticker_returns[
                current_end - window:
                current_end
            ]

            market_sample = market_returns[
                current_end - window:
                current_end
            ]

            market_variance = float(
                np.var(
                    market_sample,
                    ddof=1,
                )
            )

            if market_variance <= 0:
                beta = np.nan
            else:
                covariance = float(
                    np.cov(
                        ticker_sample,
                        market_sample,
                        ddof=1,
                    )[0, 1]
                )

                beta = (
                    covariance
                    / market_variance
                )

            days.append(current_end)
            values.append(beta)

        return (
            np.asarray(
                days,
                dtype=np.int64,
            ),
            np.asarray(
                values,
                dtype=np.float64,
            ),
        )

    def ticker_snapshot(
        self,
        ticker_index: int,
        end: int | None = None,
        window: int = 60,
        neighbour_count: int = 3,
    ) -> TickerFeatureSnapshot:
        validated_end = self._validate_end(
            end
        )

        sample = self._window_returns(
            end=validated_end,
            window=window,
        )

        ticker_sample = sample[ticker_index]
        market_sample = np.mean(
            sample,
            axis=0,
        )

        mean_return = float(
            np.mean(ticker_sample)
        )

        volatility = float(
            np.std(
                ticker_sample,
                ddof=1,
            )
        )

        sharpe = (
            mean_return / volatility
            if volatility > 0
            else 0.0
        )

        market_variance = float(
            np.var(
                market_sample,
                ddof=1,
            )
        )

        if market_variance > 0:
            beta = float(
                np.cov(
                    ticker_sample,
                    market_sample,
                    ddof=1,
                )[0, 1]
                / market_variance
            )
        else:
            beta = 0.0

        if (
            np.std(ticker_sample) > 0
            and np.std(market_sample) > 0
        ):
            market_correlation = float(
                np.corrcoef(
                    ticker_sample,
                    market_sample,
                )[0, 1]
            )
        else:
            market_correlation = 0.0

        cumulative = self.cumulative_log_returns(
            ticker_index,
            end=validated_end,
        )

        current_cumulative = float(
            cumulative[-1]
        )

        peak_cumulative = float(
            np.max(cumulative)
        )

        drawdown = (
            current_cumulative
            - peak_cumulative
        )

        market_momentum = float(
            np.sum(market_sample)
        )

        ticker_momentum = float(
            np.sum(ticker_sample)
        )

        strongest, weakest = (
            self.related_tickers(
                ticker_index=ticker_index,
                end=validated_end,
                window=window,
                count=neighbour_count,
            )
        )

        return TickerFeatureSnapshot(
            ticker_index=ticker_index,
            symbol=self.symbols[ticker_index],
            end=validated_end,
            window=window,
            mean_return=mean_return,
            volatility=volatility,
            sharpe=sharpe,
            skewness=float(
                skew(
                    ticker_sample,
                    bias=False,
                )
            ),
            excess_kurtosis=float(
                kurtosis(
                    ticker_sample,
                    fisher=True,
                    bias=False,
                )
            ),
            beta=beta,
            market_correlation=(
                market_correlation
            ),
            momentum=ticker_momentum,
            relative_strength=(
                ticker_momentum
                - market_momentum
            ),
            cumulative_log_return=(
                current_cumulative
            ),
            drawdown=drawdown,
            strongest_correlations=tuple(
                int(index)
                for index in strongest
            ),
            weakest_correlations=tuple(
                int(index)
                for index in weakest
            ),
        )

    # ==============================================================
    # Direct trading-algorithm interface
    # ==============================================================

    def feature_matrix(
        self,
        end: int | None = None,
        window: int = 60,
    ) -> FloatArray:
        """
        Return one feature vector per ticker.

        Shape:
            (number_of_tickers, number_of_features)

        Columns:
            0 mean return
            1 volatility
            2 Sharpe
            3 skewness
            4 excess kurtosis
            5 beta
            6 market correlation
            7 momentum
            8 relative strength
            9 cumulative log return
            10 drawdown
        """
        rows = []

        for ticker_index in range(
            self.number_of_tickers
        ):
            snapshot = self.ticker_snapshot(
                ticker_index=ticker_index,
                end=end,
                window=window,
            )

            rows.append(
                [
                    snapshot.mean_return,
                    snapshot.volatility,
                    snapshot.sharpe,
                    snapshot.skewness,
                    snapshot.excess_kurtosis,
                    snapshot.beta,
                    snapshot.market_correlation,
                    snapshot.momentum,
                    snapshot.relative_strength,
                    snapshot.cumulative_log_return,
                    snapshot.drawdown,
                ]
            )

        return np.asarray(
            rows,
            dtype=np.float64,
        )

    def trading_state(
        self,
        end: int | None = None,
        window: int = 60,
    ) -> tuple[
        MarketStructureSnapshot,
        FloatArray,
    ]:
        """
        Main plug-in point for a trading strategy.

        Returns:
            market_snapshot:
                Global regime information.

            ticker_features:
                Feature matrix with one row per instrument.
        """
        return (
            self.market_structure_snapshot(
                end=end,
                window=window,
            ),
            self.feature_matrix(
                end=end,
                window=window,
            ),
        )
    
    def _resolve_ticker_indices(
        self,
        tickers: (
            str
            | int
            | Ticker
            | Iterable[str | int | Ticker]
        ),
    ) -> list[int]:
        """
        Convert ticker symbols, indices or Ticker objects into ticker indices.
        """

        if isinstance(tickers, (str, int, Ticker)):
            ticker_items = [tickers]
        else:
            ticker_items = list(tickers)

        if not ticker_items:
            raise ValueError("At least one ticker must be provided")

        symbol_to_index = {
            symbol: index
            for index, symbol in enumerate(self.symbols)
        }

        resolved_indices: list[int] = []

        for ticker in ticker_items:
            if isinstance(ticker, int):
                ticker_index = ticker

            elif isinstance(ticker, str):
                try:
                    ticker_index = symbol_to_index[ticker]
                except KeyError as error:
                    raise KeyError(
                        f"Unknown ticker symbol: {ticker}"
                    ) from error

            elif isinstance(ticker, Ticker):
                try:
                    ticker_index = symbol_to_index[ticker.symbol]
                except KeyError as error:
                    raise KeyError(
                        f"Unknown ticker symbol: {ticker.symbol}"
                    ) from error

            else:
                raise TypeError(
                    "Each ticker must be a symbol, integer index, "
                    f"or Ticker object—not {type(ticker).__name__}"
                )

            if not 0 <= ticker_index < self.number_of_tickers:
                raise IndexError(
                    f"Ticker index {ticker_index} is outside the valid "
                    f"range 0–{self.number_of_tickers - 1}"
                )

            if ticker_index not in resolved_indices:
                resolved_indices.append(ticker_index)

        return resolved_indices
    
    def ticker_statistics(
        self,
        tickers: (
            str
            | int
            | Ticker
            | Iterable[str | int | Ticker]
        ),
        window: int = 60,
        beta_window: int = 30,
        start: int | None = None,
        end: int | None = None,
    ) -> pd.DataFrame:
        """
        Return ticker-level and market-level statistics through time.

        Args:
            tickers:
                One ticker or an iterable of tickers. Each ticker may be:

                - a symbol, such as "ALGO"
                - an integer index, such as 0
                - a Ticker object

            window:
                Rolling window for ticker statistics, correlations and PCA.

            beta_window:
                Rolling window used to estimate market beta.

            start:
                First return-day index to include. Defaults to the earliest day
                at which both rolling windows are available.

            end:
                Final return-day index to include. Defaults to the final
                available return day.

        Returns:
            A DataFrame indexed by ticker and day.

        Example:
            stats.ticker_statistics("ALGO")

            stats.ticker_statistics(
                ["ALGO", "LSST", "SRNA"]
            )

            stats.ticker_statistics(universe)
        """

        ticker_indices = self._resolve_ticker_indices(tickers)

        if window < 2:
            raise ValueError("window must be at least 2")

        if beta_window < 2:
            raise ValueError("beta_window must be at least 2")

        minimum_end = max(window, beta_window)

        if start is None:
            start = minimum_end

        if end is None:
            end = self.number_of_return_days

        if start < minimum_end:
            raise ValueError(
                f"start must be at least {minimum_end} so both rolling "
                "windows contain enough observations"
            )

        if end > self.number_of_return_days:
            raise ValueError(
                f"end cannot exceed {self.number_of_return_days}"
            )

        if start > end:
            raise ValueError("start cannot be greater than end")

        # --------------------------------------------------------------
        # Precompute market-level rolling statistics once
        # --------------------------------------------------------------

        market_structure = self.rolling_market_structure(
            window=window
        )

        structure_days = np.asarray(
            market_structure["end_days"],
            dtype=int,
        )

        structure_lookup = {
            int(day): position
            for position, day in enumerate(structure_days)
        }

        market_mean = self.mu
        market_median = self.median
        market_sigma = self.sigma
        market_skewness = self.skewness
        market_kurtosis = self.kurtosis
        market_breadth = self.positive_fraction
        market_iqr = self.iqr
        market_tail_spread = self.tail_spread
        market_absolute_return = self.mean_absolute_return

        rows: list[dict[str, object]] = []

        # --------------------------------------------------------------
        # Build one observation per ticker per day
        # --------------------------------------------------------------

        for ticker_index in ticker_indices:
            symbol = self.symbols[ticker_index]

            ticker_returns = self.log_returns[ticker_index]
            ticker_prices = self.prices[ticker_index]

            beta_days, beta_values = self.rolling_beta(
                ticker_index=ticker_index,
                end=end,
                window=beta_window,
            )

            beta_lookup = {
                int(day): float(beta)
                for day, beta in zip(beta_days, beta_values)
            }

            for current_end in range(start, end + 1):
                return_position = current_end - 1

                ticker_window = ticker_returns[
                    current_end - window:
                    current_end
                ]

                market_window = np.mean(
                    self.log_returns[
                        :,
                        current_end - window:
                        current_end,
                    ],
                    axis=0,
                )

                mean_return = float(
                    np.mean(ticker_window)
                )

                volatility = float(
                    np.std(ticker_window, ddof=1)
                )

                sharpe = (
                    mean_return / volatility
                    if volatility > 0
                    else 0.0
                )

                ticker_skewness = float(
                    skew(
                        ticker_window,
                        bias=False,
                    )
                )

                ticker_kurtosis = float(
                    kurtosis(
                        ticker_window,
                        fisher=True,
                        bias=False,
                    )
                )

                if (
                    np.std(ticker_window, ddof=1) > 0
                    and np.std(market_window, ddof=1) > 0
                ):
                    market_correlation = float(
                        np.corrcoef(
                            ticker_window,
                            market_window,
                        )[0, 1]
                    )
                else:
                    market_correlation = 0.0

                momentum = float(
                    np.sum(ticker_window)
                )

                market_momentum = float(
                    np.sum(market_window)
                )

                relative_strength = (
                    momentum - market_momentum
                )

                current_price = float(
                    ticker_prices[current_end]
                )

                cumulative_log_return = float(
                    np.log(
                        current_price
                        / ticker_prices[0]
                    )
                )

                historical_prices = ticker_prices[
                    :current_end + 1
                ]

                running_peak = float(
                    np.max(historical_prices)
                )

                drawdown = (
                    current_price / running_peak
                ) - 1.0

                correlation_matrix = self.correlation_matrix(
                    end=current_end,
                    window=window,
                )

                related_row = correlation_matrix[
                    ticker_index
                ].copy()

                related_row[ticker_index] = np.nan

                valid_indices = np.flatnonzero(
                    ~np.isnan(related_row)
                )

                ordered_indices = valid_indices[
                    np.argsort(
                        related_row[valid_indices]
                    )
                ]

                strongest_index = int(
                    ordered_indices[-1]
                )

                weakest_index = int(
                    ordered_indices[0]
                )

                structure_position = structure_lookup[
                    current_end
                ]

                rows.append(
                    {
                        "Ticker": symbol,
                        "Ticker Index": ticker_index,
                        "Day": current_end,
                        "Price": current_price,
                        "Log Return": float(
                            ticker_returns[return_position]
                        ),
                        "Rolling Mean Return": mean_return,
                        "Rolling Volatility": volatility,
                        "Rolling Sharpe": sharpe,
                        "Rolling Skewness": ticker_skewness,
                        "Rolling Excess Kurtosis": ticker_kurtosis,
                        "Rolling Beta": beta_lookup.get(
                            current_end,
                            np.nan,
                        ),
                        "Market Correlation": market_correlation,
                        "Momentum": momentum,
                        "Relative Strength": relative_strength,
                        "Cumulative Log Return": (
                            cumulative_log_return
                        ),
                        "Drawdown": drawdown,
                        "Strongest Correlated Ticker": (
                            self.symbols[strongest_index]
                        ),
                        "Strongest Correlation": float(
                            related_row[strongest_index]
                        ),
                        "Weakest Correlated Ticker": (
                            self.symbols[weakest_index]
                        ),
                        "Weakest Correlation": float(
                            related_row[weakest_index]
                        ),
                        "Market Mean Return": float(
                            market_mean[return_position]
                        ),
                        "Market Median Return": float(
                            market_median[return_position]
                        ),
                        "Market Volatility": float(
                            market_sigma[return_position]
                        ),
                        "Market Skewness": float(
                            market_skewness[return_position]
                        ),
                        "Market Excess Kurtosis": float(
                            market_kurtosis[return_position]
                        ),
                        "Market Breadth": float(
                            market_breadth[return_position]
                        ),
                        "Market IQR": float(
                            market_iqr[return_position]
                        ),
                        "Market Tail Spread": float(
                            market_tail_spread[return_position]
                        ),
                        "Market Mean Absolute Return": float(
                            market_absolute_return[
                                return_position
                            ]
                        ),
                        "Average Pairwise Correlation": float(
                            market_structure[
                                "average_correlation"
                            ][structure_position]
                        ),
                        "PC1 Explained Variance": float(
                            market_structure[
                                "pc1_ratio"
                            ][structure_position]
                        ),
                        "Effective Dimension": float(
                            market_structure[
                                "effective_dimension"
                            ][structure_position]
                        ),
                    }
                )

        dataframe = pd.DataFrame(rows)

        return (
            dataframe
            .set_index(["Ticker", "Day"])
            .sort_index()
        )
    

    def analyse_market_proxy(
        self,
        ticker: str | int | Ticker,
        max_lag: int = 5,
        periods_per_year: int = 252,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyse whether a ticker behaves like a broad market proxy.

        The proxy ticker is excluded when constructing:
            - the equal-weight market return,
            - market breadth,
            - the first principal component.

        Lead-lag convention:
            lag > 0:
                corr(proxy_t, market_{t + lag})
                The proxy leads the market.

            lag < 0:
                corr(proxy_t, market_{t + lag})
                The market leads the proxy.

            lag = 0:
                contemporaneous correlation.

        Returns:
            summary:
                One-row DataFrame containing market-proxy diagnostics.

            lead_lag:
                DataFrame containing proxy/market and proxy/breadth
                correlations at each lag.
        """

        if max_lag < 0:
            raise ValueError("max_lag must be non-negative")

        if periods_per_year < 1:
            raise ValueError(
                "periods_per_year must be positive"
            )

        proxy_index = self._resolve_ticker_indices(
            ticker
        )[0]

        proxy_symbol = self.symbols[proxy_index]

        proxy_returns = np.asarray(
            self.log_returns[proxy_index],
            dtype=np.float64,
        )

        other_indices = np.asarray(
            [
                index
                for index in range(self.number_of_tickers)
                if index != proxy_index
            ],
            dtype=np.intp,
        )

        if len(other_indices) < 2:
            raise ValueError(
                "At least two non-proxy tickers are required"
            )

        other_returns = np.asarray(
            self.log_returns[other_indices],
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Equal-weight market excluding the proxy
        # --------------------------------------------------------------

        market_returns = np.mean(
            other_returns,
            axis=0,
        )

        market_breadth = np.mean(
            other_returns > 0,
            axis=0,
        )

        # --------------------------------------------------------------
        # Correlation helper
        # --------------------------------------------------------------

        def safe_correlation(
            x: NDArray[np.float64],
            y: NDArray[np.float64],
        ) -> float:
            valid = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            x_valid = x[valid]
            y_valid = y[valid]

            if len(x_valid) < 2:
                return np.nan

            if (
                np.std(x_valid, ddof=1) == 0
                or np.std(y_valid, ddof=1) == 0
            ):
                return np.nan

            return float(
                np.corrcoef(
                    x_valid,
                    y_valid,
                )[0, 1]
            )

        # --------------------------------------------------------------
        # Regression:
        #
        # proxy_return = alpha + beta * market_return + residual
        # --------------------------------------------------------------

        design_matrix = np.column_stack(
            (
                np.ones(len(market_returns)),
                market_returns,
            )
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix,
            proxy_returns,
            rcond=None,
        )

        daily_alpha = float(coefficients[0])
        beta = float(coefficients[1])

        fitted_returns = (
            daily_alpha
            + beta * market_returns
        )

        residuals = (
            proxy_returns
            - fitted_returns
        )

        residual_sum_squares = float(
            np.sum(residuals**2)
        )

        total_sum_squares = float(
            np.sum(
                (
                    proxy_returns
                    - np.mean(proxy_returns)
                ) ** 2
            )
        )

        r_squared = (
            1.0
            - residual_sum_squares
            / total_sum_squares
            if total_sum_squares > 0
            else np.nan
        )

        market_correlation = safe_correlation(
            proxy_returns,
            market_returns,
        )

        breadth_correlation = safe_correlation(
            proxy_returns,
            market_breadth,
        )

        # --------------------------------------------------------------
        # PCA of all other tickers
        # --------------------------------------------------------------

        pca_input = other_returns.T

        means = np.mean(
            pca_input,
            axis=0,
        )

        standard_deviations = np.std(
            pca_input,
            axis=0,
            ddof=1,
        )

        standard_deviations = np.where(
            standard_deviations == 0,
            1.0,
            standard_deviations,
        )

        standardised_returns = (
            pca_input - means
        ) / standard_deviations

        pca = PCA(n_components=1)

        pc1_scores = pca.fit_transform(
            standardised_returns
        )[:, 0]

        # PCA sign is arbitrary. Orient PC1 so that it has positive
        # correlation with the equal-weight market.
        pc1_market_correlation = safe_correlation(
            pc1_scores,
            market_returns,
        )

        if (
            np.isfinite(pc1_market_correlation)
            and pc1_market_correlation < 0
        ):
            pc1_scores = -pc1_scores
            pc1_market_correlation = (
                -pc1_market_correlation
            )

        proxy_pc1_correlation = safe_correlation(
            proxy_returns,
            pc1_scores,
        )

        # --------------------------------------------------------------
        # Risk and tracking statistics
        # --------------------------------------------------------------

        proxy_volatility = float(
            np.std(
                proxy_returns,
                ddof=1,
            )
        )

        market_volatility = float(
            np.std(
                market_returns,
                ddof=1,
            )
        )

        residual_volatility = float(
            np.std(
                residuals,
                ddof=1,
            )
        )

        tracking_difference = (
            proxy_returns
            - market_returns
        )

        tracking_error = float(
            np.std(
                tracking_difference,
                ddof=1,
            )
        )

        annualisation_factor = np.sqrt(
            periods_per_year
        )

        annualised_proxy_volatility = (
            proxy_volatility
            * annualisation_factor
        )

        annualised_market_volatility = (
            market_volatility
            * annualisation_factor
        )

        annualised_residual_volatility = (
            residual_volatility
            * annualisation_factor
        )

        annualised_tracking_error = (
            tracking_error
            * annualisation_factor
        )

        annualised_alpha = (
            daily_alpha
            * periods_per_year
        )

        mean_proxy_return = float(
            np.mean(proxy_returns)
        )

        mean_market_return = float(
            np.mean(market_returns)
        )

        # --------------------------------------------------------------
        # Lead-lag correlations
        # --------------------------------------------------------------

        lead_lag_rows: list[
            dict[str, float | int]
        ] = []

        for lag in range(
            -max_lag,
            max_lag + 1,
        ):
            if lag > 0:
                # Proxy today versus market/breadth in the future.
                proxy_sample = proxy_returns[:-lag]
                market_sample = market_returns[lag:]
                breadth_sample = market_breadth[lag:]

            elif lag < 0:
                # Market/breadth today versus proxy in the future.
                shift = -lag

                proxy_sample = proxy_returns[shift:]
                market_sample = market_returns[:-shift]
                breadth_sample = market_breadth[:-shift]

            else:
                proxy_sample = proxy_returns
                market_sample = market_returns
                breadth_sample = market_breadth

            lead_lag_rows.append(
                {
                    "Lag": lag,
                    "Proxy-Market Correlation": (
                        safe_correlation(
                            proxy_sample,
                            market_sample,
                        )
                    ),
                    "Proxy-Breadth Correlation": (
                        safe_correlation(
                            proxy_sample,
                            breadth_sample,
                        )
                    ),
                }
            )

        lead_lag = pd.DataFrame(
            lead_lag_rows
        ).set_index("Lag")

        nonzero_lags = lead_lag.loc[
            lead_lag.index != 0
        ]

        market_predictive_strength = (
            nonzero_lags[
                "Proxy-Market Correlation"
            ].abs()
        )

        breadth_predictive_strength = (
            nonzero_lags[
                "Proxy-Breadth Correlation"
            ].abs()
        )

        strongest_market_lag = int(
            market_predictive_strength.idxmax()
        )

        strongest_breadth_lag = int(
            breadth_predictive_strength.idxmax()
        )

        strongest_market_lag_correlation = float(
            lead_lag.loc[
                strongest_market_lag,
                "Proxy-Market Correlation",
            ]
        )

        strongest_breadth_lag_correlation = float(
            lead_lag.loc[
                strongest_breadth_lag,
                "Proxy-Breadth Correlation",
            ]
        )

        # --------------------------------------------------------------
        # Summary table
        # --------------------------------------------------------------

        summary = pd.DataFrame(
            [
                {
                    "Ticker": proxy_symbol,
                    "Ticker Index": proxy_index,
                    "Observations": len(proxy_returns),

                    "Mean Daily Return": (
                        mean_proxy_return
                    ),
                    "Market Mean Daily Return": (
                        mean_market_return
                    ),
                    "Daily Alpha": daily_alpha,
                    "Annualised Alpha": (
                        annualised_alpha
                    ),

                    "Market Beta": beta,
                    "Market Correlation": (
                        market_correlation
                    ),
                    "Regression R Squared": (
                        r_squared
                    ),

                    "PC1 Correlation": (
                        proxy_pc1_correlation
                    ),
                    "PC1-Market Correlation": (
                        pc1_market_correlation
                    ),
                    "PC1 Explained Variance": float(
                        pca.explained_variance_ratio_[0]
                    ),

                    "Breadth Correlation": (
                        breadth_correlation
                    ),

                    "Daily Volatility": (
                        proxy_volatility
                    ),
                    "Market Daily Volatility": (
                        market_volatility
                    ),
                    "Residual Volatility": (
                        residual_volatility
                    ),
                    "Tracking Error": (
                        tracking_error
                    ),

                    "Annualised Volatility": (
                        annualised_proxy_volatility
                    ),
                    "Market Annualised Volatility": (
                        annualised_market_volatility
                    ),
                    "Annualised Residual Volatility": (
                        annualised_residual_volatility
                    ),
                    "Annualised Tracking Error": (
                        annualised_tracking_error
                    ),

                    "Strongest Nonzero Market Lag": (
                        strongest_market_lag
                    ),
                    "Strongest Nonzero Market Lag Correlation": (
                        strongest_market_lag_correlation
                    ),

                    "Strongest Nonzero Breadth Lag": (
                        strongest_breadth_lag
                    ),
                    "Strongest Nonzero Breadth Lag Correlation": (
                        strongest_breadth_lag_correlation
                    ),
                }
            ]
        ).set_index("Ticker")

        return summary, lead_lag
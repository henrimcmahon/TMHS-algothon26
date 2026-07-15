from __future__ import annotations

import numpy as np
from scipy.stats import entropy, kurtosis, skew
from sklearn.decomposition import PCA

from models.ticker_universe import TickerUniverse


class MarketStatistics:
    def __init__(
        self,
        universe: TickerUniverse,
        log_returns: bool = True,
    ) -> None:
        self.universe = universe
        self.log_returns = log_returns

        prices = universe.as_price_matrix()

        if np.any(prices <= 0):
            raise ValueError("Prices must be positive")

        if log_returns:
            self.data = np.diff(np.log(prices), axis=1)
        else:
            self.data = prices

    @property
    def mu(self) -> np.ndarray:
        return np.mean(self.data, axis=0)

    @property
    def median(self) -> np.ndarray:
        return np.median(self.data, axis=0)

    @property
    def sigma(self) -> np.ndarray:
        return np.std(self.data, axis=0, ddof=1)

    @property
    def skewness(self) -> np.ndarray:
        return skew(
            self.data,
            axis=0,
            bias=False,
        )

    @property
    def kurtosis(self) -> np.ndarray:
        return kurtosis(
            self.data,
            axis=0,
            fisher=True,
            bias=False,
        )

    @property
    def positive_fraction(self) -> np.ndarray:
        """
        Fraction of tickers with a positive return on each day.

        0.50 means half of all tickers rose.
        """
        return np.mean(self.data > 0, axis=0)

    def percentile(self, q: float) -> np.ndarray:
        """
        Cross-sectional percentile as a function of time.

        Example:
            stats.percentile(5)
            stats.percentile(95)
        """
        if not 0 <= q <= 100:
            raise ValueError("q must be between 0 and 100")

        return np.percentile(
            self.data,
            q=q,
            axis=0,
        )

    @property
    def percentile_5(self) -> np.ndarray:
        return self.percentile(5)

    @property
    def percentile_25(self) -> np.ndarray:
        return self.percentile(25)

    @property
    def percentile_75(self) -> np.ndarray:
        return self.percentile(75)

    @property
    def percentile_95(self) -> np.ndarray:
        return self.percentile(95)

    @property
    def iqr(self) -> np.ndarray:
        return self.percentile_75 - self.percentile_25

    def cross_sectional_entropy(
        self,
        bins: int = 10,
    ) -> np.ndarray:
        """
        Histogram entropy of cross-sectional returns on each day.

        Higher entropy:
            returns are more broadly dispersed across bins.

        Lower entropy:
            returns are concentrated in fewer bins.
        """
        if bins < 2:
            raise ValueError("bins must be at least 2")

        entropy_values = np.zeros(self.data.shape[1])

        for day in range(self.data.shape[1]):
            daily_returns = self.data[:, day]

            counts, _ = np.histogram(
                daily_returns,
                bins=bins,
            )

            probabilities = counts / counts.sum()
            probabilities = probabilities[probabilities > 0]

            entropy_values[day] = entropy(
                probabilities,
                base=2,
            )

        return entropy_values

    def rolling_pca_explained_variance(
        self,
        window: int = 60,
        n_components: int = 3,
    ) -> np.ndarray:
        """
        Rolling PCA explained-variance ratios.

        Returns an array with shape:

            (number_of_valid_windows, n_components)

        Column 0 is the fraction of variance explained by PC1.
        """
        if window < 2:
            raise ValueError("window must be at least 2")

        n_instruments, n_days = self.data.shape

        if window > n_days:
            raise ValueError(
                "window cannot exceed the number of observations"
            )

        max_components = min(
            n_components,
            n_instruments,
            window,
        )

        results = []

        for end in range(window, n_days + 1):
            window_data = self.data[:, end - window:end].T

            # Standardise each ticker inside the rolling window.
            means = np.mean(window_data, axis=0)
            standard_deviations = np.std(
                window_data,
                axis=0,
                ddof=1,
            )

            # Avoid division by zero for constant instruments.
            standard_deviations[standard_deviations == 0] = 1.0

            standardised = (
                window_data - means
            ) / standard_deviations

            pca = PCA(n_components=max_components)
            pca.fit(standardised)

            results.append(
                pca.explained_variance_ratio_
            )

        return np.asarray(results)

    def rolling_pc1_ratio(
        self,
        window: int = 60,
    ) -> np.ndarray:
        """
        Rolling share of total return variance explained by PC1.
        """
        return self.rolling_pca_explained_variance(
            window=window,
            n_components=1,
        )[:, 0]
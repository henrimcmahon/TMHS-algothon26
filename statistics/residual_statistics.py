from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

from models.ticker import Ticker
from statistics.market_statistics import MarketStatistics


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ResidualSnapshot:
    ticker_index: int
    symbol: str
    end: int
    window: int

    alpha: float
    beta: float
    residual_return: float
    residual_mean: float
    residual_volatility: float
    residual_sharpe: float
    residual_z_score: float
    residual_momentum: float
    residual_skewness: float
    residual_excess_kurtosis: float
    residual_autocorrelation: float
    residual_cumulative_return: float


class ResidualStatistics:
    """
    Calculates ticker returns after removing exposure to a market proxy.

    By default, ALGO is used as the market proxy:

        residual_i(t)
            = return_i(t)
            - alpha_i
            - beta_i * return_ALGO(t)

    This class uses MarketStatistics for all price, return and ticker
    resolution logic.
    """

    def __init__(
        self,
        market_statistics: MarketStatistics,
        proxy: str | int | Ticker = "ALGO",
    ) -> None:
        self.market_statistics = market_statistics

        self.proxy_index = (
            market_statistics
            ._resolve_ticker_indices(proxy)[0]
        )

        self.proxy_symbol = (
            market_statistics.symbols[
                self.proxy_index
            ]
        )

    @property
    def number_of_tickers(self) -> int:
        return (
            self.market_statistics
            .number_of_tickers
        )

    @property
    def number_of_return_days(self) -> int:
        return (
            self.market_statistics
            .number_of_return_days
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.market_statistics.symbols

    @property
    def returns(self) -> FloatArray:
        return np.asarray(
            self.market_statistics.data,
            dtype=np.float64,
        )

    @property
    def proxy_returns(self) -> FloatArray:
        return self.returns[self.proxy_index]

    def _validate_ticker(
        self,
        ticker: str | int | Ticker,
    ) -> int:
        ticker_index = (
            self.market_statistics
            ._resolve_ticker_indices(ticker)[0]
        )

        if ticker_index == self.proxy_index:
            raise ValueError(
                f"{self.proxy_symbol} is the proxy itself, "
                "so its residual against itself is always zero"
            )

        return ticker_index

    def _validate_window(
        self,
        end: int | None,
        window: int,
    ) -> int:
        if end is None:
            end = self.number_of_return_days

        if not 1 <= end <= self.number_of_return_days:
            raise ValueError(
                f"end must be between 1 and "
                f"{self.number_of_return_days}"
            )

        if window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        if window > end:
            raise ValueError(
                "window cannot exceed available history"
            )

        return end

    # ==============================================================
    # Regression and residual construction
    # ==============================================================

    def regression_coefficients(
        self,
        ticker: str | int | Ticker,
        end: int | None = None,
        window: int = 60,
    ) -> tuple[float, float]:
        """
        Estimate rolling alpha and beta:

            r_i = alpha + beta * r_proxy + epsilon
        """
        ticker_index = self._validate_ticker(
            ticker
        )

        validated_end = self._validate_window(
            end=end,
            window=window,
        )

        ticker_sample = self.returns[
            ticker_index,
            validated_end - window:
            validated_end,
        ]

        proxy_sample = self.proxy_returns[
            validated_end - window:
            validated_end
        ]

        design_matrix = np.column_stack(
            (
                np.ones(window),
                proxy_sample,
            )
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix,
            ticker_sample,
            rcond=None,
        )

        alpha = float(coefficients[0])
        beta = float(coefficients[1])

        return alpha, beta

    def residual_returns(
        self,
        ticker: str | int | Ticker,
        end: int | None = None,
        estimation_window: int = 60,
        expanding: bool = False,
    ) -> FloatArray:
        """
        Generate a historical residual-return series.

        For each day t, alpha and beta are estimated using only data
        available before or through t.

        Args:
            ticker:
                Ticker whose market exposure should be removed.

            end:
                Final return observation to include.

            estimation_window:
                Window used to estimate alpha and beta.

            expanding:
                If True, use all available prior observations after the
                minimum estimation window. Otherwise use a rolling window.

        Returns:
            Residual returns beginning at estimation_window.
        """
        ticker_index = self._validate_ticker(
            ticker
        )

        final_end = self._validate_window(
            end=end,
            window=estimation_window,
        )

        ticker_returns = self.returns[
            ticker_index
        ]

        proxy_returns = self.proxy_returns

        residuals: list[float] = []

        for current_end in range(
            estimation_window,
            final_end + 1,
        ):
            start = (
                0
                if expanding
                else current_end
                - estimation_window
            )

            ticker_sample = ticker_returns[
                start:current_end
            ]

            proxy_sample = proxy_returns[
                start:current_end
            ]

            design_matrix = np.column_stack(
                (
                    np.ones(len(proxy_sample)),
                    proxy_sample,
                )
            )

            coefficients, _, _, _ = (
                np.linalg.lstsq(
                    design_matrix,
                    ticker_sample,
                    rcond=None,
                )
            )

            alpha = float(coefficients[0])
            beta = float(coefficients[1])

            current_ticker_return = float(
                ticker_returns[current_end - 1]
            )

            current_proxy_return = float(
                proxy_returns[current_end - 1]
            )

            residual = (
                current_ticker_return
                - alpha
                - beta * current_proxy_return
            )

            residuals.append(residual)

        return np.asarray(
            residuals,
            dtype=np.float64,
        )

    def residual_matrix(
        self,
        end: int | None = None,
        estimation_window: int = 60,
    ) -> FloatArray:
        """
        Return residual histories for every ticker.

        Shape:
            (number_of_tickers, number_of_valid_days)

        The proxy row is filled with zeros.
        """
        if end is None:
            end = self.number_of_return_days

        number_of_valid_days = (
            end - estimation_window + 1
        )

        if number_of_valid_days < 1:
            raise ValueError(
                "Not enough observations for the "
                "requested estimation window"
            )

        matrix = np.zeros(
            (
                self.number_of_tickers,
                number_of_valid_days,
            ),
            dtype=np.float64,
        )

        for ticker_index in range(
            self.number_of_tickers
        ):
            if ticker_index == self.proxy_index:
                continue

            matrix[ticker_index] = (
                self.residual_returns(
                    ticker=ticker_index,
                    end=end,
                    estimation_window=(
                        estimation_window
                    ),
                )
            )

        return matrix

    def residual_matrix_fast(
        self,
        estimation_window: int = 60,
    ) -> FloatArray:
        """
        Calculate rolling one-step residuals for every ticker simultaneously.

        Returns shape:

            (number_of_tickers, number_of_valid_residuals)

        Column j is the residual for the observation at:

            estimation_window - 1 + j

        Alpha and beta are estimated from the rolling window ending at that
        same observation, matching the current residual_returns behaviour.
        """
        returns = np.asarray(
            self.returns,
            dtype=np.float64,
        )

        number_of_tickers, number_of_days = returns.shape

        if estimation_window < 2:
            raise ValueError(
                "estimation_window must be at least 2"
            )

        if estimation_window > number_of_days:
            raise ValueError(
                "estimation_window exceeds available returns"
            )

        proxy = returns[self.proxy_index]

        number_of_windows = (
            number_of_days - estimation_window + 1
        )

        # Prefix sums allow every rolling sum to be calculated in one pass.
        ticker_prefix = np.concatenate(
            (
                np.zeros(
                    (number_of_tickers, 1),
                    dtype=np.float64,
                ),
                np.cumsum(
                    returns,
                    axis=1,
                ),
            ),
            axis=1,
        )

        proxy_prefix = np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(proxy),
            )
        )

        proxy_squared_prefix = np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(proxy**2),
            )
        )

        cross_prefix = np.concatenate(
            (
                np.zeros(
                    (number_of_tickers, 1),
                    dtype=np.float64,
                ),
                np.cumsum(
                    returns * proxy[None, :],
                    axis=1,
                ),
            ),
            axis=1,
        )

        starts = np.arange(number_of_windows)
        ends = starts + estimation_window

        ticker_sums = (
            ticker_prefix[:, ends]
            - ticker_prefix[:, starts]
        )

        proxy_sums = (
            proxy_prefix[ends]
            - proxy_prefix[starts]
        )

        proxy_squared_sums = (
            proxy_squared_prefix[ends]
            - proxy_squared_prefix[starts]
        )

        cross_sums = (
            cross_prefix[:, ends]
            - cross_prefix[:, starts]
        )

        ticker_means = (
            ticker_sums / estimation_window
        )

        proxy_means = (
            proxy_sums / estimation_window
        )

        covariance_numerators = (
            cross_sums
            - estimation_window
            * ticker_means
            * proxy_means[None, :]
        )

        variance_numerators = (
            proxy_squared_sums
            - estimation_window
            * proxy_means**2
        )

        safe_variance = np.where(
            np.abs(variance_numerators)
            > np.finfo(np.float64).eps,
            variance_numerators,
            np.inf,
        )

        betas = (
            covariance_numerators
            / safe_variance[None, :]
        )

        alphas = (
            ticker_means
            - betas * proxy_means[None, :]
        )

        observed_returns = returns[
            :,
            estimation_window - 1:
        ]

        observed_proxy_returns = proxy[
            estimation_window - 1:
        ]

        residuals = (
            observed_returns
            - alphas
            - betas
            * observed_proxy_returns[None, :]
        )

        residuals[self.proxy_index] = 0.0

        return np.asarray(
            residuals,
            dtype=np.float64,
        )
    # ==============================================================
    # Residual statistics
    # ==============================================================

    def residual_snapshot(
        self,
        ticker: str | int | Ticker,
        end: int | None = None,
        estimation_window: int = 60,
        signal_window: int = 20,
        autocorrelation_lag: int = 1,
    ) -> ResidualSnapshot:
        ticker_index = self._validate_ticker(
            ticker
        )

        if signal_window < 2:
            raise ValueError(
                "signal_window must be at least 2"
            )

        if autocorrelation_lag < 1:
            raise ValueError(
                "autocorrelation_lag must be positive"
            )

        residuals = self.residual_returns(
            ticker=ticker_index,
            end=end,
            estimation_window=estimation_window,
        )

        if len(residuals) < signal_window:
            raise ValueError(
                "Not enough residual observations for "
                "the requested signal window"
            )

        residual_window = residuals[
            -signal_window:
        ]

        residual_mean = float(
            np.mean(residual_window)
        )

        residual_volatility = float(
            np.std(
                residual_window,
                ddof=1,
            )
        )

        residual_sharpe = (
            residual_mean
            / residual_volatility
            if residual_volatility > 0
            else 0.0
        )

        residual_return = float(
            residuals[-1]
        )

        residual_z_score = (
            (
                residual_return
                - residual_mean
            )
            / residual_volatility
            if residual_volatility > 0
            else 0.0
        )

        residual_momentum = float(
            np.sum(residual_window)
        )

        if len(residual_window) > autocorrelation_lag:
            lagged = residual_window[
                :-autocorrelation_lag
            ]

            current = residual_window[
                autocorrelation_lag:
            ]

            if (
                np.std(lagged, ddof=1) > 0
                and np.std(current, ddof=1) > 0
            ):
                residual_autocorrelation = float(
                    np.corrcoef(
                        lagged,
                        current,
                    )[0, 1]
                )
            else:
                residual_autocorrelation = 0.0
        else:
            residual_autocorrelation = np.nan

        alpha, beta = (
            self.regression_coefficients(
                ticker=ticker_index,
                end=end,
                window=estimation_window,
            )
        )

        validated_end = (
            self.number_of_return_days
            if end is None
            else end
        )

        return ResidualSnapshot(
            ticker_index=ticker_index,
            symbol=self.symbols[ticker_index],
            end=validated_end,
            window=signal_window,
            alpha=alpha,
            beta=beta,
            residual_return=residual_return,
            residual_mean=residual_mean,
            residual_volatility=(
                residual_volatility
            ),
            residual_sharpe=residual_sharpe,
            residual_z_score=residual_z_score,
            residual_momentum=residual_momentum,
            residual_skewness=float(
                skew(
                    residual_window,
                    bias=False,
                )
            ),
            residual_excess_kurtosis=float(
                kurtosis(
                    residual_window,
                    fisher=True,
                    bias=False,
                )
            ),
            residual_autocorrelation=(
                residual_autocorrelation
            ),
            residual_cumulative_return=float(
                np.sum(residuals)
            ),
        )

    # ==============================================================
    # Strategy-ready features
    # ==============================================================

    def feature_matrix(
        self,
        end: int | None = None,
        estimation_window: int = 60,
        signal_window: int = 20,
    ) -> FloatArray:
        """
        Return one residual feature vector per ticker.

        Columns:
            0 alpha
            1 beta
            2 current residual return
            3 residual mean
            4 residual volatility
            5 residual Sharpe
            6 residual z-score
            7 residual momentum
            8 residual skewness
            9 residual excess kurtosis
            10 residual autocorrelation
            11 cumulative residual return

        The proxy row contains zeros.
        """
        matrix = np.zeros(
            (
                self.number_of_tickers,
                12,
            ),
            dtype=np.float64,
        )

        for ticker_index in range(
            self.number_of_tickers
        ):
            if ticker_index == self.proxy_index:
                continue

            snapshot = self.residual_snapshot(
                ticker=ticker_index,
                end=end,
                estimation_window=(
                    estimation_window
                ),
                signal_window=signal_window,
            )

            matrix[ticker_index] = [
                snapshot.alpha,
                snapshot.beta,
                snapshot.residual_return,
                snapshot.residual_mean,
                snapshot.residual_volatility,
                snapshot.residual_sharpe,
                snapshot.residual_z_score,
                snapshot.residual_momentum,
                snapshot.residual_skewness,
                snapshot.residual_excess_kurtosis,
                snapshot.residual_autocorrelation,
                snapshot.residual_cumulative_return,
            ]

        return matrix

    def residual_statistics(
        self,
        tickers: (
            str
            | int
            | Ticker
            | list[str | int | Ticker]
            | tuple[str | int | Ticker, ...]
        ),
        end: int | None = None,
        estimation_window: int = 60,
        signal_window: int = 20,
    ) -> pd.DataFrame:
        """
        Return the latest residual statistics as a DataFrame.
        """
        indices = (
            self.market_statistics
            ._resolve_ticker_indices(tickers)
        )

        rows: list[dict[str, object]] = []

        for ticker_index in indices:
            if ticker_index == self.proxy_index:
                continue

            snapshot = self.residual_snapshot(
                ticker=ticker_index,
                end=end,
                estimation_window=(
                    estimation_window
                ),
                signal_window=signal_window,
            )

            rows.append(
                {
                    "Ticker": snapshot.symbol,
                    "Ticker Index": (
                        snapshot.ticker_index
                    ),
                    "End": snapshot.end,
                    "Estimation Window": (
                        estimation_window
                    ),
                    "Signal Window": (
                        signal_window
                    ),
                    "Alpha": snapshot.alpha,
                    "Beta": snapshot.beta,
                    "Residual Return": (
                        snapshot.residual_return
                    ),
                    "Residual Mean": (
                        snapshot.residual_mean
                    ),
                    "Residual Volatility": (
                        snapshot.residual_volatility
                    ),
                    "Residual Sharpe": (
                        snapshot.residual_sharpe
                    ),
                    "Residual Z-Score": (
                        snapshot.residual_z_score
                    ),
                    "Residual Momentum": (
                        snapshot.residual_momentum
                    ),
                    "Residual Skewness": (
                        snapshot.residual_skewness
                    ),
                    "Residual Excess Kurtosis": (
                        snapshot
                        .residual_excess_kurtosis
                    ),
                    "Residual Autocorrelation": (
                        snapshot
                        .residual_autocorrelation
                    ),
                    "Cumulative Residual Return": (
                        snapshot
                        .residual_cumulative_return
                    ),
                }
            )

        return (
            pd.DataFrame(rows)
            .set_index("Ticker")
            .sort_index()
        )
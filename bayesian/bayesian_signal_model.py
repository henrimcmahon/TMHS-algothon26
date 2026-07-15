from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

import pandas as pd

from .bayesian_updater import (
    BayesianPrediction,
    BayesianUpdateDiagnostics,
    BayesianUpdater,
)
from .belief_state import BeliefState

from bayesian.hypotheses import (
    MeanReversionHypothesis,
    MomentumHypothesis,
    NoiseHypothesis,
)
from statistics.residual_statistics import ResidualStatistics


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TickerBayesianPrediction:
    ticker_index: int
    symbol: str
    predictive_mean: float
    predictive_variance: float
    posterior_probabilities: FloatArray
    posterior_entropy: float | None
    negative_log_likelihood: float | None
    information_gain: float | None


class BayesianSignalModel:
    """
    Applies one independent BayesianUpdater to each non-proxy ticker.

    ResidualStatistics supplies market-neutral residual returns.
    Each updater maintains persistent beliefs over:

        - momentum
        - mean reversion
        - noise
    """

    def __init__(
        self,
        residual_statistics: ResidualStatistics,
        forgetting_rate: float = 0.03,
        residual_estimation_window: int = 60,
        updater_factory: Callable[[], BayesianUpdater] | None = None,
    ) -> None:
        if not 0.0 <= forgetting_rate <= 1.0:
            raise ValueError(
                "forgetting_rate must be between 0 and 1"
            )

        if residual_estimation_window < 2:
            raise ValueError(
                "residual_estimation_window must be at least 2"
            )

        self.residual_statistics = residual_statistics
        self.market_statistics = (
            residual_statistics.market_statistics
        )

        self.forgetting_rate = forgetting_rate
        self.residual_estimation_window = residual_estimation_window

        self.proxy_index = residual_statistics.proxy_index
        self.symbols = residual_statistics.symbols
        self.number_of_tickers = (
            residual_statistics.number_of_tickers
        )

        self._updater_factory = (
            updater_factory
            if updater_factory is not None
            else self._create_default_updater
        )

        self._updaters: dict[int, BayesianUpdater] = {
            ticker_index: self._updater_factory()
            for ticker_index in range(
                self.number_of_tickers
            )
            if ticker_index != self.proxy_index
        }

        self._latest_predictions: dict[
            int,
            BayesianPrediction,
        ] = {}

        self._latest_diagnostics: dict[
            int,
            BayesianUpdateDiagnostics,
        ] = {}

        self._last_processed_end: int | None = None

    def _create_default_updater(
        self,
    ) -> BayesianUpdater:
        return BayesianUpdater(
            hypotheses=[
                MomentumHypothesis(
                    trend_learning_rate=0.15,
                    variance_learning_rate=0.10,
                    variance=1e-4,
                ),
                MeanReversionHypothesis(
                    reversion_speed=0.70,
                    equilibrium_learning_rate=0.02,
                    variance_learning_rate=0.10,
                    variance=1e-4,
                ),
                NoiseHypothesis(
                    variance=1e-4,
                    learning_rate=0.10,
                ),
            ],
            belief_state=BeliefState.uniform(
                number_of_hypotheses=3,
                forgetting_rate=self.forgetting_rate,
            ),
        )

    def _validate_end(
        self,
        end: int | None,
    ) -> int:
        if end is None:
            return (
                self.market_statistics
                .number_of_return_days
            )

        if not 1 <= end <= (
            self.market_statistics
            .number_of_return_days
        ):
            raise ValueError(
                "end is outside the available return history"
            )

        return end

    def residual_history(
        self,
        ticker_index: int,
        end: int,
    ) -> FloatArray:
        residuals = (
            self.residual_statistics
            .residual_returns(
                ticker=ticker_index,
                end=end,
                estimation_window=(
                    self.residual_estimation_window
                ),
            )
        )

        return np.asarray(
            residuals,
            dtype=np.float64,
        )

    @property
    def last_processed_end(self) -> int | None:
        return self._last_processed_end
    
    def warm_start(
        self,
        end: int | None = None,
    ) -> None:
        """
        Replay historical residuals so every updater reaches a state
        consistent with the available history.

        Call once before requesting live predictions.
        """
        final_end = self._validate_end(end)

        required_history = max(
            hypothesis.minimum_history
            for updater in self._updaters.values()
            for hypothesis in updater.hypotheses
        )

        first_update_end = (
            self.residual_estimation_window
            + required_history
        )

        if final_end < first_update_end:
            raise ValueError(
                "Not enough observations to warm-start the model"
            )

        for current_end in range(
            first_update_end,
            final_end + 1,
        ):
            self.update(end=current_end)

    def predict(
        self,
        end: int | None = None,
    ) -> list[TickerBayesianPrediction]:
        """
        Predict the next residual return for each non-proxy ticker.

        This method does not update beliefs.
        """
        validated_end = self._validate_end(end)

        output: list[TickerBayesianPrediction] = []

        self._latest_predictions.clear()

        for ticker_index, updater in self._updaters.items():
            history = self.residual_history(
                ticker_index=ticker_index,
                end=validated_end,
            )

            prediction = updater.predict(history)

            self._latest_predictions[
                ticker_index
            ] = prediction

            diagnostics = self._latest_diagnostics.get(
                ticker_index
            )

            output.append(
                TickerBayesianPrediction(
                    ticker_index=ticker_index,
                    symbol=self.symbols[ticker_index],
                    predictive_mean=float(
                        prediction
                        .combined_distribution
                        .mean
                    ),
                    predictive_variance=float(
                        prediction
                        .combined_distribution
                        .variance
                    ),
                    posterior_probabilities=(
                        updater.probabilities
                    ),
                    posterior_entropy=(
                        diagnostics.posterior_entropy
                        if diagnostics is not None
                        else None
                    ),
                    negative_log_likelihood=(
                        diagnostics.negative_log_likelihood
                        if diagnostics is not None
                        else None
                    ),
                    information_gain=(
                        diagnostics.information_gain
                        if diagnostics is not None
                        else None
                    ),
                )
            )

        return output

    def update(
        self,
        end: int,
    ) -> None:
        """
        Incorporate the residual return observed at `end`.

        For each ticker:

        1. predict using residual history through end - 1;
        2. observe the residual at end;
        3. update posterior beliefs and hypothesis state.
        """
        validated_end = self._validate_end(end)

        if validated_end <= self.residual_estimation_window:
            raise ValueError(
                "end must exceed residual_estimation_window"
            )

        if (
            self._last_processed_end is not None
            and validated_end
            <= self._last_processed_end
        ):
            raise ValueError(
                "updates must be processed in strictly increasing order"
            )

        for ticker_index, updater in self._updaters.items():
            full_residual_history = (
                self.residual_history(
                    ticker_index=ticker_index,
                    end=validated_end,
                )
            )

            required_history = max(
                hypothesis.minimum_history
                for hypothesis in updater.hypotheses
            )

            if len(full_residual_history) < required_history + 1:
                continue

            history = full_residual_history[:-1]
            observation = float(
                full_residual_history[-1]
            )

            prediction = updater.predict(history)

            diagnostics = (
                updater.observe_with_diagnostics(
                    observation=observation,
                    prediction=prediction,
                )
            )

            self._latest_predictions[
                ticker_index
            ] = prediction

            self._latest_diagnostics[
                ticker_index
            ] = diagnostics

        self._last_processed_end = validated_end

    def signal_vector(
        self,
        end: int | None = None,
        entropy_penalty: float = 0.0,
        predictions: (
            list[TickerBayesianPrediction]
            | None
        ) = None,
    ) -> FloatArray:
        """
        Return one risk-adjusted Bayesian signal per ticker.
        """
        if entropy_penalty < 0:
            raise ValueError(
                "entropy_penalty cannot be negative"
            )

        active_predictions = (
            predictions
            if predictions is not None
            else self.predict(end=end)
        )

        signals = np.zeros(
            self.number_of_tickers,
            dtype=np.float64,
        )

        for result in active_predictions:
            denominator = max(
                result.predictive_variance,
                np.finfo(np.float64).eps,
            )

            signal = (
                result.predictive_mean
                / denominator
            )

            if (
                entropy_penalty > 0
                and result.posterior_entropy
                is not None
            ):
                signal *= np.exp(
                    -entropy_penalty
                    * result.posterior_entropy
                )

            signals[result.ticker_index] = signal

        signals[self.proxy_index] = 0.0

        return signals

    def belief_matrix(
        self,
    ) -> FloatArray:
        """
        Shape:

            (number_of_tickers, number_of_hypotheses)

        The proxy row is zero.
        """
        hypothesis_count = len(
            next(iter(self._updaters.values()))
            .hypotheses
        )

        matrix = np.zeros(
            (
                self.number_of_tickers,
                hypothesis_count,
            ),
            dtype=np.float64,
        )

        for ticker_index, updater in self._updaters.items():
            matrix[ticker_index] = (
                updater.probabilities
            )

        return matrix

    def diagnostic_matrix(
        self,
    ) -> FloatArray:
        """
        Columns:

            0 posterior entropy
            1 negative log likelihood
            2 information gain
            3 predictive mean
            4 predictive variance

        Missing and proxy values are NaN.
        """
        matrix = np.full(
            (
                self.number_of_tickers,
                5,
            ),
            np.nan,
            dtype=np.float64,
        )

        for ticker_index, diagnostics in (
            self._latest_diagnostics.items()
        ):
            matrix[ticker_index] = [
                diagnostics.posterior_entropy,
                diagnostics.negative_log_likelihood,
                diagnostics.information_gain,
                diagnostics.predictive_mean,
                diagnostics.predictive_variance,
            ]

        return matrix
    
    def as_dataframe(
        self,
        end: int | None = None,
        entropy_penalty: float = 0.0,
    ) -> pd.DataFrame:
        """
        Return the latest Bayesian predictions, beliefs, diagnostics,
        and signals as a labelled DataFrame.

        The proxy ticker is included with a zero signal and NaN diagnostics.
        """
        predictions = self.predict(end=end)

        signals = self.signal_vector(
            entropy_penalty=entropy_penalty,
            predictions=predictions,
        )

        rows: list[dict[str, object]] = []

        prediction_lookup = {
            prediction.ticker_index: prediction
            for prediction in predictions
        }

        for ticker_index, symbol in enumerate(
            self.symbols
        ):
            if ticker_index == self.proxy_index:
                rows.append(
                    {
                        "Ticker": symbol,
                        "Ticker Index": ticker_index,
                        "Predictive Mean": 0.0,
                        "Predictive Variance": np.nan,
                        "Momentum Probability": np.nan,
                        "Mean-Reversion Probability": np.nan,
                        "Noise Probability": np.nan,
                        "Winning Hypothesis": "proxy",
                        "Winning Probability": np.nan,
                        "Posterior Entropy": np.nan,
                        "Negative Log Likelihood": np.nan,
                        "Information Gain": np.nan,
                        "Signal": 0.0,
                    }
                )
                continue

            prediction = prediction_lookup[
                ticker_index
            ]

            probabilities = (
                prediction.posterior_probabilities
            )

            hypothesis_names = tuple(
                self._updaters[ticker_index].hypothesis_names
            )

            winning_index = int(
                np.argmax(probabilities)
            )

            winning_hypothesis = hypothesis_names[
                winning_index
            ]

            winning_probability = float(
                probabilities[winning_index]
            )

            if len(probabilities) != 3:
                raise ValueError(
                    "as_dataframe currently expects exactly "
                    "three hypotheses"
                )

            rows.append(
                {
                    "Ticker": symbol,
                    "Ticker Index": ticker_index,
                    "Predictive Mean": prediction.predictive_mean,
                    "Predictive Variance": prediction.predictive_variance,
                    "Momentum Probability": float(probabilities[0]),
                    "Mean-Reversion Probability": float(probabilities[1]),
                    "Noise Probability": float(probabilities[2]),
                    "Winning Hypothesis": hypothesis_names[
                        winning_index
                    ],
                    "Winning Probability": float(
                        probabilities[winning_index]
                    ),
                    "Posterior Entropy": prediction.posterior_entropy,
                    "Negative Log Likelihood": (
                        prediction.negative_log_likelihood
                    ),
                    "Information Gain": prediction.information_gain,
                    "Signal": float(signals[ticker_index]),
                }
            )

        return (
            pd.DataFrame(rows)
            .set_index("Ticker")
            .sort_values(
                "Signal",
                ascending=False,
            )
        )
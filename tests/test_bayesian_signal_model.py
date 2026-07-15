import numpy as np

from bayesian import BayesianSignalModel
from models.ticker_universe import TickerUniverse
from statistics.market_statistics import MarketStatistics
from statistics.residual_statistics import ResidualStatistics


def test_signal_model_outputs_one_signal_per_ticker() -> None:
    universe = TickerUniverse("prices.txt")

    market_statistics = MarketStatistics(
        universe=universe
    )

    residual_statistics = ResidualStatistics(
        market_statistics=market_statistics,
        proxy="ALGO",
    )

    model = BayesianSignalModel(
        residual_statistics=residual_statistics,
        residual_estimation_window=60,
        forgetting_rate=0.03,
    )

    model.warm_start(end=100)

    signals = model.signal_vector(end=100)

    assert signals.shape == (
        market_statistics.number_of_tickers,
    )

    assert np.all(np.isfinite(signals))

    assert signals[
        residual_statistics.proxy_index
    ] == 0.0


def test_belief_matrix_is_normalised() -> None:
    universe = TickerUniverse("prices.txt")

    market_statistics = MarketStatistics(
        universe=universe
    )

    residual_statistics = ResidualStatistics(
        market_statistics=market_statistics,
        proxy="ALGO",
    )

    model = BayesianSignalModel(
        residual_statistics=residual_statistics,
        residual_estimation_window=60,
    )

    model.warm_start(end=100)

    beliefs = model.belief_matrix()

    non_proxy = np.arange(
        market_statistics.number_of_tickers
    ) != residual_statistics.proxy_index

    assert np.allclose(
        np.sum(
            beliefs[non_proxy],
            axis=1,
        ),
        1.0,
    )

def test_signal_model_remains_valid_over_time() -> None:
    universe = TickerUniverse("prices.txt")

    market_statistics = MarketStatistics(
        universe=universe
    )

    residual_statistics = ResidualStatistics(
        market_statistics=market_statistics,
        proxy="ALGO",
    )

    model = BayesianSignalModel(
        residual_statistics=residual_statistics,
        residual_estimation_window=60,
        forgetting_rate=0.03,
    )

    model.warm_start(end=100)

    check_dates = [
        110,
        125,
        150,
    ]

    for target_end in check_dates:
        start = (
            model.last_processed_end + 1
            if model.last_processed_end
            is not None
            else 62
        )

        for current_end in range(
            start,
            target_end + 1,
        ):
            model.update(end=current_end)

        signals = model.signal_vector(
            end=target_end
        )

        beliefs = model.belief_matrix()

        diagnostics = (
            model.diagnostic_matrix()
        )

        assert signals.shape == (
            market_statistics
            .number_of_tickers,
        )

        assert beliefs.shape == (
            market_statistics
            .number_of_tickers,
            3,
        )

        assert diagnostics.shape == (
            market_statistics
            .number_of_tickers,
            5,
        )

        assert np.all(
            np.isfinite(signals)
        )

        non_proxy_mask = (
            np.arange(
                market_statistics
                .number_of_tickers
            )
            != residual_statistics.proxy_index
        )

        assert np.allclose(
            np.sum(
                beliefs[non_proxy_mask],
                axis=1,
            ),
            1.0,
        )

        assert np.all(
            np.isfinite(
                diagnostics[
                    non_proxy_mask
                ]
            )
        )

def test_signal_model_dataframe_has_expected_columns() -> None:
    universe = TickerUniverse("prices.txt")

    market_statistics = MarketStatistics(
        universe=universe
    )

    residual_statistics = ResidualStatistics(
        market_statistics=market_statistics,
        proxy="ALGO",
    )

    model = BayesianSignalModel(
        residual_statistics=residual_statistics,
        residual_estimation_window=60,
    )

    model.warm_start(end=100)

    dataframe = model.as_dataframe(
        end=100
    )

    expected_columns = {
        "Ticker Index",
        "Predictive Mean",
        "Predictive Variance",
        "Momentum Probability",
        "Mean-Reversion Probability",
        "Noise Probability",
        "Winning Hypothesis",
        "Winning Probability",
        "Posterior Entropy",
        "Negative Log Likelihood",
        "Information Gain",
        "Signal",
    }

    assert dataframe.loc[
        "ALGO",
        "Winning Hypothesis",
    ] == "proxy"

    assert np.isnan(
        dataframe.loc[
            "ALGO",
            "Winning Probability",
        ]
    )

    assert len(dataframe) == (
        market_statistics.number_of_tickers
    )

    assert expected_columns.issubset(
        dataframe.columns
    )

    assert dataframe.loc[
        "ALGO",
        "Signal",
    ] == 0.0
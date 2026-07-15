import numpy as np

from models.ticker_universe import TickerUniverse
from statistics import residual_statistics
from statistics.market_statistics import MarketStatistics
from statistics.residual_statistics import ResidualStatistics


def test_fast_residual_matrix_matches_original() -> None:
    universe = TickerUniverse("prices.txt")

    market_statistics = MarketStatistics(
        universe=universe
    )

    residual_statistics = ResidualStatistics(
        market_statistics=market_statistics,
        proxy="ALGO",
    )

    window = 60

    fast = (
        residual_statistics
        .residual_matrix_fast(
            estimation_window=window
        )
    )

    for ticker_index in range(
        residual_statistics.number_of_tickers
    ):
        if ticker_index == residual_statistics.proxy_index:
            assert np.allclose(
                fast[ticker_index],
                0.0,
            )
            continue
        
        original = (
            residual_statistics
            .residual_returns(
                ticker=ticker_index,
                estimation_window=window,
            )
        )

        assert np.allclose(
            fast[ticker_index],
            original,
            rtol=1e-9,
            atol=1e-11,
        )
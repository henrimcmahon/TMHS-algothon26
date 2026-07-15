import numpy as np

from models.ticker_universe import TickerUniverse
from statistics.market_statistics import MarketStatistics
from statistics.residual_statistics import ResidualStatistics

nInst = 51
currentPos = np.zeros(nInst)


def getMyPosition(
    prcSoFar: np.ndarray,
) -> np.ndarray:
    global currentPos

    n_instruments, n_days = prcSoFar.shape

    if n_days < 61:
        return np.zeros(n_instruments)

    stats = MarketStatistics(
        prices=np.asarray(
            prcSoFar,
            dtype=np.float64,
        )
    )

    market, features = stats.trading_state(
        end=stats.number_of_return_days,
        window=60,
    )

    mean_return = features[:, 0]
    volatility = features[:, 1]
    sharpe = features[:, 2]
    relative_strength = features[:, 8]

    # Example score only — this needs backtesting.
    score = (
        0.35 * sharpe
        + 0.40 * relative_strength
        + 0.25 * mean_return
    )

    # Reduce risk when the market is highly concentrated
    # or displaying unusually heavy tails.
    regime_multiplier = 1.0

    if market.pc1_ratio > 0.35:
        regime_multiplier *= 0.70

    if market.excess_kurtosis > 4:
        regime_multiplier *= 0.60

    inverse_volatility = 1.0 / np.maximum(
        volatility,
        1e-8,
    )

    raw_weights = (
        score
        * inverse_volatility
        * regime_multiplier
    )

    gross_exposure = np.sum(
        np.abs(raw_weights)
    )

    if gross_exposure == 0:
        return np.zeros(n_instruments)

    weights = raw_weights / gross_exposure

    capital = 5000

    positions = (
        capital
        * weights
        / prcSoFar[:, -1]
    )

    currentPos = np.asarray(
        positions,
        dtype=int,
    )

    return currentPos

if __name__ == "__main__":
    from visualisation.strategy_visualiser import StrategyVisualizer

    universe = TickerUniverse("prices.txt")

    market_stats = MarketStatistics(
        universe=universe
    )

    residual_stats = ResidualStatistics(
        market_statistics=market_stats,
        proxy="ALGO",
    )

    # all_residual_stats = (
    #     residual_stats.residual_statistics(
    #         universe
    #     )
    # )

    # features = residual_stats.feature_matrix(
    #     end=market_stats.number_of_return_days,
    #     estimation_window=60,
    #     signal_window=20,
    # )

    # all_residual_stats.to_csv("data/residual_stats.csv")
    # features.to_csv("data/features.csv")

    visualiser = StrategyVisualizer(
        statistics=market_stats
    )

    # algo_stats = market_stats.ticker_statistics(
    #     "ALGO",
    #     window=60,
    #     beta_window=30,
    # )

    # algo_stats.to_csv("ALGO_stats.csv")

    # summary, lead_lag = market_stats.analyse_market_proxy(
    #     "ALGO"
    # )

    # summary.to_csv("ALGO_proxy_summary.csv")
    # lead_lag.to_csv("ALGO_lead_lag.csv")

    visualiser.plot_interactive_dashboard(
        window=60,
        beta_window=30,
        interval=120,
        neighbour_count=3,
    )

    
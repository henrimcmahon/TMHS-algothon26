import time

import numpy as np

from models.ticker_universe import TickerUniverse
from strategies.information_graph_strategies import (
    LeaderFollowerStrategy,
    InformationCentralityMomentumStrategy,
)


def benchmark_strategy(strategy):
    universe = TickerUniverse("prices.txt")

    prices = np.asarray(
        universe.as_price_matrix(),
        dtype=np.float64,
    )

    print(f"\n{strategy.name}")

    timings = []

    for day in range(100, prices.shape[1] - 1):

        history = prices[:, : day + 1]

        start = time.perf_counter()

        strategy.get_positions(history)

        timings.append(
            time.perf_counter() - start
        )

    timings = np.asarray(timings)

    print(
        f"Mean {timings.mean():.4f}s"
    )
    print(
        f"Median {np.median(timings):.4f}s"
    )
    print(
        f"95th {np.percentile(timings,95):.4f}s"
    )
    print(
        f"Max {timings.max():.4f}s"
    )
    print(
        f"Total {timings.sum():.1f}s"
    )


def test_leader_follower_speed():

    benchmark_strategy(
        LeaderFollowerStrategy()
    )


def test_information_centrality_speed():

    benchmark_strategy(
        InformationCentralityMomentumStrategy()
    )
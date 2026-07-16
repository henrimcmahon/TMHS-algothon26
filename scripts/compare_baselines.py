from __future__ import annotations

import numpy as np

from backtesting import BaselineEvaluator
from backtesting.baselines import (
    AlgoHoldStrategy,
    AllAssetsHoldStrategy,
    NoPositionStrategy,
    PreviousReturnStrategy,
)
from models.ticker_universe import (
    TickerUniverse,
)


def main() -> None:
    universe = TickerUniverse(
        "prices.txt"
    )

    prices = np.asarray(
        universe.as_price_matrix(),
        dtype=np.float64,
    )

    strategies = [
        NoPositionStrategy(),
        AlgoHoldStrategy("long"),
        AlgoHoldStrategy("short"),
        AllAssetsHoldStrategy("long"),
        AllAssetsHoldStrategy("short"),
        PreviousReturnStrategy(
            mode="momentum",
            exposure_fraction=0.25,
        ),
        PreviousReturnStrategy(
            mode="reversal",
            exposure_fraction=0.25,
        ),
    ]

    evaluator = BaselineEvaluator()

    comparison = evaluator.compare(
        strategies=strategies,
        price_history=prices,
    )

    print()
    print(
        comparison.to_string(
            float_format=lambda value: (
                f"{value:,.2f}"
            )
        )
    )
    print()

    comparison.to_csv(
        "data/baseline_comparison.csv"
    )


if __name__ == "__main__":
    main()
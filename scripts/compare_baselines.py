from __future__ import annotations

import numpy as np

from backtesting import (
    BaselineEvaluator,
    OracleEvaluator,
)
from backtesting.leader_analysis import (
    LeaderAnalyser,
)
from visualisation import (
    LeaderVisualiser,
)
from backtesting.baselines import (
    AlgoHoldStrategy,
    AllAssetsHoldStrategy,
    BaselineStrategy,
    LeaderStrategy,
    NoPositionStrategy,
    PositiveScoreEnsemble,
    PreviousReturnStrategy,
    SoftmaxScoreEnsemble,
    TopKScoreEnsemble,
    CrossSectionalRankStrategy,
)
from backtesting.leader_diagnostics import (
    analyse_leadership,
)
from models.ticker_universe import TickerUniverse


def create_components() -> list[BaselineStrategy]:
    """
    Create fresh component strategies for a meta-strategy.

    Meta-strategies must not share component instances because each
    component keeps its own positions and virtual PnL history.
    """
    return [
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

        CrossSectionalRankStrategy(
            lookback=1,
            selection_fraction=0.20,
            mode="momentum",
            exposure_fraction=1.0,
        ),
        CrossSectionalRankStrategy(
            lookback=1,
            selection_fraction=0.20,
            mode="reversal",
            exposure_fraction=1.0,
        ),

        CrossSectionalRankStrategy(
            lookback=5,
            selection_fraction=0.20,
            mode="momentum",
            exposure_fraction=1.0,
        ),
        CrossSectionalRankStrategy(
            lookback=5,
            selection_fraction=0.20,
            mode="reversal",
            exposure_fraction=1.0,
        ),

        CrossSectionalRankStrategy(
            lookback=20,
            selection_fraction=0.20,
            mode="momentum",
            exposure_fraction=1.0,
        ),
        CrossSectionalRankStrategy(
            lookback=20,
            selection_fraction=0.20,
            mode="reversal",
            exposure_fraction=1.0,
        ),
    ]


def create_strategies() -> list[BaselineStrategy]:
    return [
        # Simple baselines
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

        # Hard strategy selection
        LeaderStrategy(
            create_components(),
            metric="score",
            minimum_metric=0.0,
        ),
        LeaderStrategy(
            create_components(),
            metric="score",
            window=50,
            minimum_metric=0.0,
        ),
        LeaderStrategy(
            create_components(),
            metric="sharpe",
        ),
        LeaderStrategy(
            create_components(),
            metric="mean",
        ),
        LeaderStrategy(
            create_components(),
            metric="lowest_volatility",
        ),

        # Strategy ensembles
        PositiveScoreEnsemble(
            create_components(),
        ),
        SoftmaxScoreEnsemble(
            create_components(),
            temperature=10.0,
        ),
        SoftmaxScoreEnsemble(
            create_components(),
            temperature=10.0,
            window=50,
        ),
        TopKScoreEnsemble(
            create_components(),
            top_k=2,
        ),
        TopKScoreEnsemble(
            create_components(),
            top_k=2,
            window=50,
        ),
        CrossSectionalRankStrategy(
            lookback=1,
            selection_fraction=0.2,
            mode="momentum",
            exposure_fraction=1.0,
        ),

        CrossSectionalRankStrategy(
            lookback=1,
            selection_fraction=0.2,
            mode="reversal",
            exposure_fraction=1.0,
        ),

        CrossSectionalRankStrategy(
            lookback=5,
            selection_fraction=0.2,
            mode="momentum",
            exposure_fraction=1.0,
        ),

        CrossSectionalRankStrategy(
            lookback=5,
            selection_fraction=0.2,
            mode="reversal",
            exposure_fraction=1.0,
        ),

        CrossSectionalRankStrategy(
            lookback=20,
            selection_fraction=0.2,
            mode="momentum",
            exposure_fraction=1.0,
        ),

        CrossSectionalRankStrategy(
            lookback=20,
            selection_fraction=0.2,
            mode="reversal",
            exposure_fraction=1.0,
        ),
    ]


def main() -> None:
    universe = TickerUniverse(
        "prices.txt"
    )

    prices = np.asarray(
        universe.as_price_matrix(),
        dtype=np.float64,
    )

    strategies = create_strategies()

    evaluator = BaselineEvaluator()

    # Evaluate once so the same daily results can be reused for both
    # the comparison table and the perfect-foresight oracle.
    results = {
        strategy.name: evaluator.evaluate(
            strategy=strategy,
            price_history=prices,
        )
        for strategy in strategies
    }

    leader_component_names = {
        "ALGO long-and-hold",
        "ALGO short-and-hold",
        "Max long all assets",
        "Max short all assets",
        "Previous-day momentum",
        "Previous-day reversal",
        (
            "1-day winners-long losers-short "
            "top/bottom 20%"
        ),
        (
            "1-day losers-long winners-short "
            "top/bottom 20%"
        ),
        (
            "5-day winners-long losers-short "
            "top/bottom 20%"
        ),
        (
            "5-day losers-long winners-short "
            "top/bottom 20%"
        ),
        (
            "20-day winners-long losers-short "
            "top/bottom 20%"
        ),
        (
            "20-day losers-long winners-short "
            "top/bottom 20%"
        ),
    }

    leader_components = {
        name: result
        for name, result in results.items()
        if name in leader_component_names
    }

    leader_analysis = LeaderAnalyser(
        annualisation_days=250,
        minimum_observations=20,
        window=None,
        minimum_score=0.0,
        switch_margin=1.0,
    ).analyse(
        leader_components
    )

    print()
    print("Leader analysis")
    print("-" * 44)
    print(
        leader_analysis.summary().to_string(
            float_format=lambda value: (
                f"{value:,.2f}"
            )
        )
    )

    print()
    print(
        leader_analysis
        .leadership_dataframe()
        .to_string(
            index=False,
            formatters={
                "Selection Fraction": (
                    lambda value: (
                        f"{value:.2%}"
                    )
                ),
            },
        )
    )

    leader_analysis.leadership_dataframe().to_csv(
        "data/leader_selection.csv",
        index=False,
    )

    LeaderVisualiser(
        result=leader_analysis,
        component_results=leader_components,
    ).plot_dashboard()

    comparison = evaluator.results_dataframe(
        results
    )

    print()
    print("Baseline and meta-strategy comparison")
    print("-" * 44)
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

    # The oracle should only choose between the underlying static
    # strategies, not between other meta-strategies.
    oracle_component_names = {
        "ALGO long-and-hold",
        "ALGO short-and-hold",
        "Max long all assets",
        "Max short all assets",
        "Previous-day momentum",
        "Previous-day reversal",
    }

    oracle_components = {
        name: result
        for name, result in results.items()
        if name in oracle_component_names
    }

    oracle_result = OracleEvaluator().evaluate(
        oracle_components
    )

    print("Perfect-foresight oracle")
    print("-" * 44)
    print(
        oracle_result.summary().to_string(
            float_format=lambda value: (
                f"{value:,.2f}"
            )
        )
    )
    print()

    diagnostics = analyse_leadership(
        oracle_result.selected_strategies
    )

    print("Oracle leadership diagnostics")
    print("-" * 44)
    print(
        diagnostics.as_series().to_string(
            float_format=lambda value: (
                f"{value:,.2f}"
            )
        )
    )
    print()

    leadership = (
        diagnostics.leadership_dataframe()
    )

    print(
        leadership.to_string(
            index=False,
            formatters={
                "Fraction Leading": (
                    lambda value: (
                        f"{value:.2%}"
                    )
                ),
            },
        )
    )
    print()

    leadership.to_csv(
        "data/oracle_leadership.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
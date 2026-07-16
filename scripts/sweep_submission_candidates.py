from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import (
    BaselineEvaluator,
)
from backtesting.baselines import (
    BaselineStrategy,
    CrossSectionalRankStrategy,
    PreviousReturnStrategy,
    WeightedBlendStrategy,
)
from models.ticker_universe import (
    TickerUniverse,
)


OUTPUT_PATH = Path(
    "data/submission_candidate_sweep.csv"
)

EVALUATION_WINDOWS = (
    100,
    150,
    250,
    500,
    None,
)

MOMENTUM_EXPOSURES = (
    0.25,
    0.50,
    0.75,
    1.00,
)

REVERSION_EXPOSURES = (
    0.25,
    0.50,
    0.75,
    1.00,
)

# Weight assigned to previous-day momentum.
# Mean-reversion receives 1 - momentum_weight.
MOMENTUM_WEIGHTS = (
    0.00,
    0.20,
    0.40,
    0.50,
    0.60,
    0.80,
    1.00,
)


@dataclass(
    frozen=True,
    slots=True,
)
class Candidate:
    name: str
    factory: Callable[
        [],
        BaselineStrategy,
    ]


def previous_day_momentum(
    exposure: float,
) -> PreviousReturnStrategy:
    return PreviousReturnStrategy(
        mode="momentum",
        exposure_fraction=exposure,
    )


def mean_reversion_20d(
    exposure: float,
) -> CrossSectionalRankStrategy:
    return CrossSectionalRankStrategy(
        lookback=20,
        selection_fraction=0.20,
        mode="reversal",
        exposure_fraction=exposure,
        include_algo=False,
    )


def create_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    # Pure previous-day momentum candidates.
    for exposure in MOMENTUM_EXPOSURES:
        candidates.append(
            Candidate(
                name=(
                    "Previous-day momentum "
                    f"exposure={exposure:.2f}"
                ),
                factory=(
                    lambda exposure=exposure:
                    previous_day_momentum(
                        exposure
                    )
                ),
            )
        )

    # Pure 20-day mean-reversion candidates.
    for exposure in REVERSION_EXPOSURES:
        candidates.append(
            Candidate(
                name=(
                    "20-day mean reversion "
                    f"exposure={exposure:.2f}"
                ),
                factory=(
                    lambda exposure=exposure:
                    mean_reversion_20d(
                        exposure
                    )
                ),
            )
        )

    # Momentum + mean-reversion blends.
    for (
        momentum_exposure,
        reversion_exposure,
        momentum_weight,
    ) in product(
        MOMENTUM_EXPOSURES,
        REVERSION_EXPOSURES,
        MOMENTUM_WEIGHTS,
    ):
        if momentum_weight in {
            0.0,
            1.0,
        }:
            # Pure candidates already exist above.
            continue

        reversion_weight = (
            1.0 - momentum_weight
        )

        name = (
            "Blend "
            f"momExp={momentum_exposure:.2f} "
            f"revExp={reversion_exposure:.2f} "
            f"momWeight={momentum_weight:.2f}"
        )

        candidates.append(
            Candidate(
                name=name,
                factory=(
                    lambda
                    momentum_exposure=momentum_exposure,
                    reversion_exposure=reversion_exposure,
                    momentum_weight=momentum_weight,
                    reversion_weight=reversion_weight,
                    name=name:
                    WeightedBlendStrategy(
                        strategies=[
                            previous_day_momentum(
                                momentum_exposure
                            ),
                            mean_reversion_20d(
                                reversion_exposure
                            ),
                        ],
                        weights=[
                            momentum_weight,
                            reversion_weight,
                        ],
                        name=name,
                    )
                ),
            )
        )

    return candidates


def evaluate_candidates(
    prices: np.ndarray,
) -> pd.DataFrame:
    evaluator = BaselineEvaluator()

    rows: list[
        dict[str, object]
    ] = []

    total_price_days = (
        prices.shape[1]
    )

    candidates = create_candidates()

    total_runs = (
        len(candidates)
        * len(EVALUATION_WINDOWS)
    )

    completed = 0

    for candidate in candidates:
        for requested_window in (
            EVALUATION_WINDOWS
        ):
            completed += 1

            effective_window = (
                total_price_days
                if requested_window is None
                else min(
                    requested_window,
                    total_price_days,
                )
            )

            strategy = (
                candidate.factory()
            )

            result = evaluator.evaluate(
                strategy=strategy,
                price_history=prices,
                num_test_days=(
                    requested_window
                ),
            )

            rows.append(
                {
                    "Candidate": (
                        candidate.name
                    ),
                    "Window Days": (
                        effective_window
                    ),
                    "Start Day": (
                        result.scoring_start_day
                    ),
                    "Score": result.score,
                    "Total PnL": (
                        result.total_pnl
                    ),
                    "Mean Daily PnL": (
                        result.mean_daily_pnl
                    ),
                    "PnL Std": (
                        result.pnl_std
                    ),
                    "Sharpe": (
                        result
                        .annualised_sharpe
                    ),
                    "Max Drawdown": (
                        result
                        .maximum_drawdown
                    ),
                    "Commissions": (
                        result
                        .total_commissions
                    ),
                }
            )

            print(
                f"[{completed:>4}/{total_runs}] "
                f"{candidate.name} | "
                f"{effective_window}d | "
                f"score={result.score:,.2f}"
            )

    return pd.DataFrame(
        rows
    )


def candidate_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        results
        .groupby(
            "Candidate",
            as_index=False,
        )
        .agg(
            Mean_Score=(
                "Score",
                "mean",
            ),
            Median_Score=(
                "Score",
                "median",
            ),
            Minimum_Score=(
                "Score",
                "min",
            ),
            Maximum_Score=(
                "Score",
                "max",
            ),
            Mean_Sharpe=(
                "Sharpe",
                "mean",
            ),
            Minimum_Sharpe=(
                "Sharpe",
                "min",
            ),
            Mean_Max_Drawdown=(
                "Max Drawdown",
                "mean",
            ),
        )
    )

    # Reward consistently strong candidates while penalising
    # candidates that collapse in one evaluation window.
    grouped[
        "Robustness Score"
    ] = (
        0.50
        * grouped["Median_Score"]
        + 0.30
        * grouped["Minimum_Score"]
        + 0.20
        * grouped["Mean_Score"]
    )

    return (
        grouped
        .sort_values(
            "Robustness Score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    universe = TickerUniverse(
        "prices.txt"
    )

    prices = np.asarray(
        universe.as_price_matrix(),
        dtype=np.float64,
    )

    results = evaluate_candidates(
        prices
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = candidate_summary(
        results
    )

    summary_path = Path(
        "data/submission_candidate_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "Top robust candidates"
    )
    print("-" * 90)

    print(
        summary.head(
            20
        ).to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:,.2f}"
            ),
        )
    )

    print()
    print(
        f"Detailed results: {OUTPUT_PATH}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
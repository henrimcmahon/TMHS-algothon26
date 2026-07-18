from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from backtesting import BaselineEvaluator
from models.ticker_universe import TickerUniverse
from strategies.information_graph_strategies import (
    InformationCentralityMomentumStrategy,
    LeaderFollowerStrategy,
    SignalVersion
)
from information.rolling_graph_cache import (
    RollingInformationGraphCache,
)

NUM_TEST_DAYS = 250

OUTPUT_PATH = Path(
    "data/information_strategy_comparison.csv"
)


def create_information_strategies(
    graph_cache: RollingInformationGraphCache,
) -> list[LeaderFollowerStrategy]:

    strategies: list[LeaderFollowerStrategy] = []

    signal_versions: list[SignalVersion] = [
        "raw",
        "tanh",
        "weighted",
        "latest_raw",
    ]

    for signal_version in signal_versions:
        strategies.append(
            LeaderFollowerStrategy(
                window=85,
                lag=2,
                top_k=7,
                percentile=97.0,
                signal_lookback=2,
                rebalance_interval=3,
                self_move_penalty=0.125,
                graph_cache=graph_cache,
                time_decay=0.6,
                forecast_threshold=0.25,
                signal_version=signal_version,
            )
        )

    return strategies


def strategy_parameters(
    strategy,
) -> dict[str, object]:
    return {
        "Strategy": strategy.name,
        "Strategy Type": type(strategy).__name__,
        "Signal Version": getattr(
            strategy,
            "signal_version",
            None,
        ),
        "Window": strategy.window,
        "Lag": strategy.lag,
        "Top K": strategy.top_k,
        "Percentile": strategy.percentile,
        "Signal Lookback": getattr(
            strategy,
            "signal_lookback",
            None,
        ),
        "Rebalance Interval": getattr(
            strategy,
            "rebalance_interval",
            None,
        ),
        "Self Move Penalty": getattr(
            strategy,
            "self_move_penalty",
            None,
        ),
        "Time Decay": getattr(
            strategy,
            "time_decay",
            None,
        ),
        "Forecast Threshold": getattr(
            strategy,
            "forecast_threshold",
            None,
        ),
        "Normalise Incoming Weights": getattr(
            strategy,
            "normalise_incoming_weights",
            None,
        ),
        "Discretisation Method": strategy.method,
        "Bias Corrected": strategy.bias_correct,
        "Gross Notional": strategy.gross_notional,
        "Max Notional Per Ticker": (
            strategy.max_notional_per_ticker
        ),
        "Minimum Days": strategy.minimum_days,
    }


def result_metrics(result) -> dict[str, float | int]:
    return {
        "Total PnL": result.total_pnl,
        "Gross PnL": result.total_gross_pnl,
        "Commissions": result.total_commissions,
        "Total Shares Traded":
            result.total_shares_traded,
        "Average Daily Shares Traded":
            result.average_daily_turnover,
        "Maximum Daily Turnover":
            result.maximum_daily_turnover,
        "Mean Daily PnL": result.mean_daily_pnl,
        "PnL Std": result.pnl_std,
        "Annualised Sharpe": result.annualised_sharpe,
        "Score": result.score,
        "Maximum Drawdown": result.maximum_drawdown,
        "Scoring Start Day": result.scoring_start_day,
        "Scoring Days": result.scoring_days,
    }


def main() -> None:
    universe = TickerUniverse("prices.txt")

    prices = np.asarray(
        universe.as_price_matrix(),
        dtype=np.float64,
    )

    evaluator = BaselineEvaluator()
    graph_cache = RollingInformationGraphCache()

    strategies = create_information_strategies(
        graph_cache
    )

    rows: list[dict[str, object]] = []

    progress_bar = tqdm(
        strategies,
        desc="Evaluating strategies",
        unit="strategy",
        dynamic_ncols=True,
    )

    for strategy in progress_bar:
        progress_bar.set_postfix_str(strategy.name)

        result = evaluator.evaluate(
            strategy=strategy,
            price_history=prices,
            num_test_days=NUM_TEST_DAYS,
        )

        rows.append(
            {
                **strategy_parameters(strategy),
                **result_metrics(result),
            }
        )

    cache_stats = graph_cache.statistics()

    print()
    print("Information graph cache")
    print("-" * 40)

    for key, value in cache_stats.items():
        if "Rate" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:,}")

    comparison = pd.DataFrame(rows)

    comparison.sort_values(
        by="Score",
        ascending=False,
        inplace=True,
    )

    comparison.reset_index(
        drop=True,
        inplace=True,
    )

    comparison.insert(
        0,
        "Rank",
        np.arange(1, len(comparison) + 1),
    )

    comparison["Commission Share of Gross Movement"] = (
        comparison["Commissions"]
        / comparison["Gross PnL"].abs().replace(0.0, np.nan)
    )

    comparison["Net vs Gross Difference"] = (
        comparison["Total PnL"]
        - comparison["Gross PnL"]
    )

    print()
    print("Information strategy comparison")
    print("-" * 120)
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:,.2f}",
        )
    )
    print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved {len(comparison)} strategy results to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from information import (
    ReturnDiscretiser,
    mutual_information_matrix,
    rolling_mutual_information,
    ticker_entropies,
)
from information.rolling_information import RollingInformationData


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse entropy and mutual information across tickers."
    )

    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("prices.txt"),
        help="Path to the price matrix.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling-window length.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Number of days between rolling calculations.",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Lead-lag offset. A value of 1 compares source_t to target_t+1.",
    )
    parser.add_argument(
        "--method",
        choices=("quantile", "volatility"),
        default="quantile",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/information"),
    )

    return parser.parse_args()


def load_prices(
    path: Path,
) -> tuple[list[str], NDArray[np.float64]]:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
    )

    names = frame.iloc[0].astype(str).tolist()

    prices = frame.iloc[1:].astype(float).to_numpy().T
    # transpose so we end up with
    # (n_tickers, n_days)

    return names, prices

def calculate_log_returns(
    prices: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.diff(np.log(prices), axis=1)

def save_matrix(
    matrix: NDArray[np.float64],
    names: list[str],
    path: Path,
) -> None:
    dataframe = pd.DataFrame(
        matrix,
        index=names,
        columns=names,
    )
    dataframe.to_csv(path)


def plot_heatmap(
    matrix: NDArray[np.float64],
    names: list[str],
    *,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 10))

    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_title(title)
    axis.set_xlabel("Target ticker")
    axis.set_ylabel("Source ticker")

    # Labels become unreadable with all 51 tickers, so display every fifth.
    tick_positions = np.arange(0, len(names), 5)
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(
        [names[index] for index in tick_positions],
        rotation=90,
    )
    axis.set_yticklabels(
        [names[index] for index in tick_positions],
    )

    figure.colorbar(image, ax=axis, label="Mutual information (bits)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def top_lagged_relationships(
    matrix: NDArray[np.float64],
    names: list[str],
    *,
    top_n: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for source_index, source_name in enumerate(names):
        for target_index, target_name in enumerate(names):
            if source_index == target_index:
                continue

            rows.append(
                {
                    "Source": source_name,
                    "Target": target_name,
                    "Lagged MI": matrix[source_index, target_index],
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("Lagged MI", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    names, prices = load_prices(arguments.prices)
    returns = calculate_log_returns(prices)

    discretiser = ReturnDiscretiser(method=arguments.method)
    states = discretiser.transform(returns)

    entropies = ticker_entropies(states)

    entropy_frame = pd.DataFrame(
        {
            "Ticker": names,
            "Entropy Bits": entropies,
        }
    ).sort_values("Entropy Bits")

    entropy_frame.to_csv(
        arguments.output_dir / "ticker_entropies.csv",
        index=False,
    )

    same_time_matrix = mutual_information_matrix(
        states,
        lag=0,
        normalised=False,
    )

    lagged_matrix = mutual_information_matrix(
        states,
        lag=arguments.lag,
        normalised=False,
    )

    save_matrix(
        same_time_matrix,
        names,
        arguments.output_dir / "same_time_mutual_information.csv",
    )
    save_matrix(
        lagged_matrix,
        names,
        arguments.output_dir / "lagged_mutual_information.csv",
    )

    top_relationships = top_lagged_relationships(
        lagged_matrix,
        names,
    )
    top_relationships.to_csv(
        arguments.output_dir / "top_lagged_relationships.csv",
        index=False,
    )

    plot_heatmap(
        same_time_matrix,
        names,
        title="Same-time mutual information",
        path=arguments.output_dir / "same_time_mutual_information.png",
    )
    plot_heatmap(
        lagged_matrix,
        names,
        title=f"Lagged mutual information: source(t) → target(t+{arguments.lag})",
        path=arguments.output_dir / "lagged_mutual_information.png",
    )

    rolling = rolling_mutual_information(
        states,
        window=arguments.window,
        lag=arguments.lag,
        step=arguments.step,
    )

    rolling_data = RollingInformationData(
        matrices=np.asarray(
            rolling.matrices,
            dtype=np.float64,
        ),
        names=names,
        window_ends=np.asarray(
            rolling.end_days,
            dtype=np.int64,
        ),
        window_size=arguments.window,
        lag=arguments.lag,
    )

    rolling_data.save(
        arguments.output_dir / "rolling_lagged_information.npz"
    )

    print("\nLowest-entropy tickers")
    print(entropy_frame.head(10).to_string(index=False))

    print("\nStrongest lagged relationships")
    print(top_relationships.head(20).to_string(index=False))

    print(
        f"\nSaved {rolling.matrices.shape[0]} rolling matrices "
        f"to {arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path

from information.rolling_information import RollingInformationData
from visualisation.information_network_simulator import (
    SimulatorConfig,
    run_information_network_simulator,
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an interactive physics simulation of "
            "rolling mutual-information networks."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/information"),
        help=(
            "Directory containing the rolling MI CSV or NPY files."
        ),
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--spring-exponent",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--spring-strength",
        type=float,
        default=18.0,
    )

    parser.add_argument(
        "--repulsion",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--damping",
        type=float,
        default=0.88,
    )

    parser.add_argument(
        "--transition",
        type=float,
        default=0.7,
        help="Seconds used to interpolate between matrices.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Seconds between rolling windows during playback.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    data = RollingInformationData.load(arguments.input)

    print(
        f"Loaded {data.matrices.shape[0]} matrices "
        f"for {len(data.names)} tickers."
    )

    config = SimulatorConfig(
        percentile=arguments.percentile,
        top_k=arguments.top_k,
        spring_exponent=arguments.spring_exponent,
        spring_strength=arguments.spring_strength,
        repulsion_strength=arguments.repulsion,
        damping=arguments.damping,
        transition_seconds=arguments.transition,
        window_interval_seconds=arguments.interval,
    )

    run_information_network_simulator(
        data=data,
        config=config,
    )


if __name__ == "__main__":
    main()
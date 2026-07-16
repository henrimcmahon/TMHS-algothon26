from __future__ import annotations

from typing import Literal

import numpy as np

from .baseline_strategy import (
    BaselineStrategy,
    FloatArray,
    IntArray,
)


class CrossSectionalRankStrategy(
    BaselineStrategy
):
    """
    Long the highest-ranked assets and short the lowest-ranked assets.

    Ranking is based on trailing log return over a configurable lookback.
    """

    def __init__(
        self,
        *,
        lookback: int = 1,
        selection_fraction: float = 0.2,
        mode: Literal[
            "momentum",
            "reversal",
        ] = "momentum",
        exposure_fraction: float = 1.0,
        include_algo: bool = False,
    ) -> None:
        super().__init__()

        if lookback < 1:
            raise ValueError(
                "lookback must be positive"
            )

        if not 0.0 < selection_fraction <= 0.5:
            raise ValueError(
                "selection_fraction must be in "
                "(0, 0.5]"
            )

        if mode not in {
            "momentum",
            "reversal",
        }:
            raise ValueError(
                "mode must be 'momentum' or "
                "'reversal'"
            )

        if not 0.0 <= exposure_fraction <= 1.0:
            raise ValueError(
                "exposure_fraction must be between "
                "0 and 1"
            )

        self.lookback = lookback
        self.selection_fraction = (
            selection_fraction
        )
        self.mode = mode
        self.exposure_fraction = (
            exposure_fraction
        )
        self.include_algo = include_algo

    @property
    def name(self) -> str:
        direction = (
            "winners-long losers-short"
            if self.mode == "momentum"
            else "losers-long winners-short"
        )

        return (
            f"{self.lookback}-day "
            f"{direction} "
            f"top/bottom "
            f"{self.selection_fraction:.0%}"
        )

    def get_positions(
        self,
        price_history: FloatArray,
    ) -> IntArray:
        prices = np.asarray(
            price_history,
            dtype=np.float64,
        )

        if prices.ndim != 2:
            raise ValueError(
                "price_history must be "
                "two-dimensional"
            )

        if prices.shape[0] != (
            self.number_of_instruments
        ):
            raise ValueError(
                "price_history must contain "
                "51 instruments"
            )

        required_days = (
            self.lookback + 1
        )

        if prices.shape[1] < required_days:
            self.current_positions.fill(
                0
            )
            return self.current_positions.copy()

        latest_prices = prices[:, -1]

        trailing_returns = np.log(
            prices[:, -1]
            / prices[
                :,
                -self.lookback - 1
            ]
        )

        eligible_indices = np.arange(
            self.number_of_instruments,
            dtype=np.int64,
        )

        if not self.include_algo:
            eligible_indices = (
                eligible_indices[1:]
            )

        eligible_returns = (
            trailing_returns[
                eligible_indices
            ]
        )

        number_selected = max(
            1,
            int(
                np.floor(
                    len(eligible_indices)
                    * self.selection_fraction
                )
            ),
        )

        ordered_local_indices = np.argsort(
            eligible_returns
        )

        loser_indices = eligible_indices[
            ordered_local_indices[
                :number_selected
            ]
        ]

        winner_indices = eligible_indices[
            ordered_local_indices[
                -number_selected:
            ]
        ]

        directions = np.zeros(
            self.number_of_instruments,
            dtype=np.float64,
        )

        if self.mode == "momentum":
            directions[winner_indices] = 1.0
            directions[loser_indices] = -1.0
        else:
            directions[winner_indices] = -1.0
            directions[loser_indices] = 1.0

        position_limits = self.position_limits()

        long_indices = np.flatnonzero(
            directions > 0
        )

        short_indices = np.flatnonzero(
            directions < 0
        )

        desired = np.zeros(
            self.number_of_instruments,
            dtype=np.int64,
        )

        if (
            len(long_indices) == 0
            or len(short_indices) == 0
        ):
            self.current_positions = desired
            return desired.copy()

        maximum_long_capacity = float(
            np.sum(
                position_limits[
                    long_indices
                ]
            )
        )

        maximum_short_capacity = float(
            np.sum(
                position_limits[
                    short_indices
                ]
            )
        )

        # The two books use the smaller available capacity so their total
        # dollar exposures are matched.
        side_target = (
            min(
                maximum_long_capacity,
                maximum_short_capacity,
            )
            * self.exposure_fraction
        )

        long_dollar_target = (
            side_target
            / len(long_indices)
        )

        short_dollar_target = (
            side_target
            / len(short_indices)
        )

        for index in long_indices:
            per_asset_target = min(
                long_dollar_target,
                position_limits[index],
            )

            desired[index] = int(
                np.floor(
                    per_asset_target
                    / latest_prices[index]
                )
            )

        for index in short_indices:
            per_asset_target = min(
                short_dollar_target,
                position_limits[index],
            )

            desired[index] = -int(
                np.floor(
                    per_asset_target
                    / latest_prices[index]
                )
            )

        self.current_positions = (
            self.clip_positions(
                desired,
                latest_prices,
            )
        )

        return self.current_positions.copy()

        self.current_positions = (
            self.clip_positions(
                desired,
                latest_prices,
            )
        )

        return self.current_positions.copy()
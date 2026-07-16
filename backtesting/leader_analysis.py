from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .baseline_evaluator import BaselineResult


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LeaderAnalysisResult:
    strategy_names: tuple[str, ...]
    days: IntArray

    selected_indices: IntArray
    selected_names: tuple[str, ...]

    contemporaneous_indices: IntArray
    contemporaneous_names: tuple[str, ...]

    oracle_indices: IntArray
    oracle_names: tuple[str, ...]

    leader_daily_pnl: FloatArray
    leader_cumulative_pnl: FloatArray

    oracle_daily_pnl: FloatArray
    oracle_cumulative_pnl: FloatArray

    best_static_daily_pnl: FloatArray
    best_static_cumulative_pnl: FloatArray

    cumulative_oracle_regret: FloatArray
    cumulative_static_regret: FloatArray

    score_matrix: FloatArray
    switch_days: IntArray

    number_of_switches: int
    mean_lead_duration: float
    median_lead_duration: float
    longest_lead_duration: int
    selection_accuracy: float

    def leadership_dataframe(
        self,
    ) -> pd.DataFrame:
        rows = []

        for strategy_index, strategy_name in enumerate(
            self.strategy_names
        ):
            days_selected = int(
                np.sum(
                    self.selected_indices
                    == strategy_index
                )
            )

            rows.append(
                {
                    "Strategy": strategy_name,
                    "Days Selected": days_selected,
                    "Selection Fraction": (
                        days_selected
                        / len(self.days)
                    ),
                }
            )

        cash_days = int(
            np.sum(
                self.selected_indices < 0
            )
        )

        if cash_days > 0:
            rows.append(
                {
                    "Strategy": "Cash",
                    "Days Selected": cash_days,
                    "Selection Fraction": (
                        cash_days
                        / len(self.days)
                    ),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values(
                "Days Selected",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "Number of switches": (
                    self.number_of_switches
                ),
                "Mean lead duration": (
                    self.mean_lead_duration
                ),
                "Median lead duration": (
                    self.median_lead_duration
                ),
                "Longest lead duration": (
                    self.longest_lead_duration
                ),
                "Leader selection accuracy": (
                    self.selection_accuracy
                ),
                "Leader total PnL": float(
                    self.leader_cumulative_pnl[-1]
                ),
                "Best static total PnL": float(
                    self.best_static_cumulative_pnl[-1]
                ),
                "Oracle total PnL": float(
                    self.oracle_cumulative_pnl[-1]
                ),
                "Final oracle regret": float(
                    self.cumulative_oracle_regret[-1]
                ),
                "Final static regret": float(
                    self.cumulative_static_regret[-1]
                ),
            },
            name="Leader analysis",
        )


class LeaderAnalyser:
    """
    Analyse score-based strategy selection without look-ahead.

    On day t, the tradeable leader is selected using component PnL
    observed only through day t - 1.
    """

    def __init__(
        self,
        *,
        annualisation_days: int = 250,
        minimum_observations: int = 20,
        window: int | None = None,
        minimum_score: float | None = None,
        switch_margin: float = 0.0,
    ) -> None:
        if annualisation_days <= 0:
            raise ValueError(
                "annualisation_days must be positive"
            )

        if minimum_observations < 2:
            raise ValueError(
                "minimum_observations must be at least 2"
            )

        if window is not None and window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        if switch_margin < 0:
            raise ValueError(
                "switch_margin cannot be negative"
            )

        self.annualisation_days = (
            annualisation_days
        )
        self.minimum_observations = (
            minimum_observations
        )
        self.window = window
        self.minimum_score = minimum_score
        self.switch_margin = switch_margin

    def _score(
        self,
        pnl: FloatArray,
    ) -> float:
        values = np.asarray(
            pnl,
            dtype=np.float64,
        )

        if self.window is not None:
            values = values[
                -self.window:
            ]

        if len(values) < (
            self.minimum_observations
        ):
            return float("nan")

        mean_pnl = float(
            np.mean(values)
        )

        pnl_std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        if (
            mean_pnl < 0
            or pnl_std < 1e-10
        ):
            return mean_pnl

        sharpe = float(
            np.sqrt(
                self.annualisation_days
            )
            * mean_pnl
            / pnl_std
        )

        sharpe_squared = sharpe**2

        return float(
            mean_pnl
            * sharpe_squared
            / (
                sharpe_squared + 1.0
            )
        )

    @staticmethod
    def _names_from_indices(
        indices: IntArray,
        strategy_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            (
                strategy_names[index]
                if index >= 0
                else "Cash"
            )
            for index in indices
        )

    @staticmethod
    def _lead_durations(
        selected_indices: IntArray,
    ) -> list[int]:
        if len(selected_indices) == 0:
            return []

        durations: list[int] = []
        previous = int(
            selected_indices[0]
        )
        duration = 1

        for raw_index in selected_indices[1:]:
            index = int(raw_index)

            if index == previous:
                duration += 1
                continue

            durations.append(
                duration
            )

            previous = index
            duration = 1

        durations.append(
            duration
        )

        return durations

    def analyse(
        self,
        results: Mapping[
            str,
            BaselineResult,
        ],
    ) -> LeaderAnalysisResult:
        if not results:
            raise ValueError(
                "at least one result is required"
            )

        strategy_names = tuple(
            results
        )

        lengths = {
            len(result.daily_pnl)
            for result in results.values()
        }

        if len(lengths) != 1:
            raise ValueError(
                "all results must have equal length"
            )

        number_of_days = next(
            iter(lengths)
        )

        pnl_matrix = np.vstack(
            [
                np.asarray(
                    results[name].daily_pnl,
                    dtype=np.float64,
                )
                for name in strategy_names
            ]
        )

        number_of_strategies = (
            len(strategy_names)
        )

        score_matrix = np.full(
            (
                number_of_strategies,
                number_of_days,
            ),
            np.nan,
            dtype=np.float64,
        )

        # score_matrix[:, day] includes PnL through that day.
        for day in range(
            number_of_days
        ):
            for strategy_index in range(
                number_of_strategies
            ):
                score_matrix[
                    strategy_index,
                    day
                ] = self._score(
                    pnl_matrix[
                        strategy_index,
                        :day + 1,
                    ]
                )

        selected_indices = np.full(
            number_of_days,
            -1,
            dtype=np.int64,
        )

        contemporaneous_indices = np.full(
            number_of_days,
            -1,
            dtype=np.int64,
        )

        current_leader = -1

        for day in range(
            number_of_days
        ):
            current_scores = score_matrix[
                :,
                day,
            ]

            current_valid = np.isfinite(
                current_scores
            )

            if np.any(current_valid):
                valid_indices = np.flatnonzero(
                    current_valid
                )

                contemporaneous_indices[day] = int(
                    valid_indices[
                        np.argmax(
                            current_scores[
                                current_valid
                            ]
                        )
                    ]
                )

            # No look-ahead: select using yesterday's scores.
            if day == 0:
                continue

            available_scores = score_matrix[
                :,
                day - 1,
            ]

            valid = np.isfinite(
                available_scores
            )

            if not np.any(valid):
                continue

            valid_indices = np.flatnonzero(
                valid
            )

            candidate = int(
                valid_indices[
                    np.argmax(
                        available_scores[
                            valid
                        ]
                    )
                ]
            )

            candidate_score = float(
                available_scores[
                    candidate
                ]
            )

            if (
                self.minimum_score is not None
                and candidate_score
                < self.minimum_score
            ):
                current_leader = -1
                continue

            if (
                current_leader >= 0
                and np.isfinite(
                    available_scores[
                        current_leader
                    ]
                )
            ):
                current_score = float(
                    available_scores[
                        current_leader
                    ]
                )

                # Do not switch for a negligible estimated advantage.
                if (
                    candidate != current_leader
                    and candidate_score
                    < current_score
                    + self.switch_margin
                ):
                    candidate = current_leader

            current_leader = candidate
            selected_indices[day] = (
                current_leader
            )

        oracle_indices = np.argmax(
            pnl_matrix,
            axis=0,
        ).astype(
            np.int64
        )

        leader_daily_pnl = np.zeros(
            number_of_days,
            dtype=np.float64,
        )

        for day, strategy_index in enumerate(
            selected_indices
        ):
            if strategy_index < 0:
                continue

            leader_daily_pnl[day] = (
                pnl_matrix[
                    strategy_index,
                    day,
                ]
            )

        day_indices = np.arange(
            number_of_days
        )

        oracle_daily_pnl = pnl_matrix[
            oracle_indices,
            day_indices,
        ]

        final_scores = score_matrix[
            :,
            -1,
        ]

        valid_final = np.isfinite(
            final_scores
        )

        if np.any(valid_final):
            valid_indices = np.flatnonzero(
                valid_final
            )

            best_static_index = int(
                valid_indices[
                    np.argmax(
                        final_scores[
                            valid_final
                        ]
                    )
                ]
            )
        else:
            best_static_index = 0

        best_static_daily_pnl = (
            pnl_matrix[
                best_static_index
            ].copy()
        )

        leader_cumulative_pnl = np.cumsum(
            leader_daily_pnl
        )

        oracle_cumulative_pnl = np.cumsum(
            oracle_daily_pnl
        )

        best_static_cumulative_pnl = (
            np.cumsum(
                best_static_daily_pnl
            )
        )

        cumulative_oracle_regret = (
            oracle_cumulative_pnl
            - leader_cumulative_pnl
        )

        cumulative_static_regret = (
            best_static_cumulative_pnl
            - leader_cumulative_pnl
        )

        switch_days = np.flatnonzero(
            selected_indices[1:]
            != selected_indices[:-1]
        ).astype(
            np.int64
        ) + 1

        durations = self._lead_durations(
            selected_indices
        )

        comparable = (
            (selected_indices >= 0)
            & (contemporaneous_indices >= 0)
        )

        if np.any(comparable):
            selection_accuracy = float(
                np.mean(
                    selected_indices[
                        comparable
                    ]
                    == contemporaneous_indices[
                        comparable
                    ]
                )
            )
        else:
            selection_accuracy = 0.0

        return LeaderAnalysisResult(
            strategy_names=strategy_names,
            days=np.arange(
                number_of_days,
                dtype=np.int64,
            ),
            selected_indices=(
                selected_indices
            ),
            selected_names=(
                self._names_from_indices(
                    selected_indices,
                    strategy_names,
                )
            ),
            contemporaneous_indices=(
                contemporaneous_indices
            ),
            contemporaneous_names=(
                self._names_from_indices(
                    contemporaneous_indices,
                    strategy_names,
                )
            ),
            oracle_indices=oracle_indices,
            oracle_names=(
                self._names_from_indices(
                    oracle_indices,
                    strategy_names,
                )
            ),
            leader_daily_pnl=(
                leader_daily_pnl
            ),
            leader_cumulative_pnl=(
                leader_cumulative_pnl
            ),
            oracle_daily_pnl=(
                oracle_daily_pnl
            ),
            oracle_cumulative_pnl=(
                oracle_cumulative_pnl
            ),
            best_static_daily_pnl=(
                best_static_daily_pnl
            ),
            best_static_cumulative_pnl=(
                best_static_cumulative_pnl
            ),
            cumulative_oracle_regret=(
                cumulative_oracle_regret
            ),
            cumulative_static_regret=(
                cumulative_static_regret
            ),
            score_matrix=score_matrix,
            switch_days=switch_days,
            number_of_switches=len(
                switch_days
            ),
            mean_lead_duration=float(
                np.mean(durations)
            ),
            median_lead_duration=float(
                np.median(durations)
            ),
            longest_lead_duration=max(
                durations
            ),
            selection_accuracy=(
                selection_accuracy
            ),
        )
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class LeaderDiagnostics:
    number_of_switches: int
    average_lead_duration: float
    median_lead_duration: float
    longest_lead_duration: int
    leadership_counts: dict[str, int]

    def as_series(self) -> pd.Series:
        return pd.Series(
            {
                "Number of switches": (
                    self.number_of_switches
                ),
                "Average lead duration": (
                    self.average_lead_duration
                ),
                "Median lead duration": (
                    self.median_lead_duration
                ),
                "Longest lead duration": (
                    self.longest_lead_duration
                ),
            }
        )

    def leadership_dataframe(
        self,
    ) -> pd.DataFrame:
        total_days = sum(
            self.leadership_counts.values()
        )

        rows = [
            {
                "Strategy": strategy,
                "Days Leading": days,
                "Fraction Leading": (
                    days / total_days
                    if total_days > 0
                    else 0.0
                ),
            }
            for strategy, days
            in self.leadership_counts.items()
        ]

        return (
            pd.DataFrame(rows)
            .sort_values(
                "Days Leading",
                ascending=False,
            )
            .reset_index(drop=True)
        )


def analyse_leadership(
    leaders: Sequence[str],
) -> LeaderDiagnostics:
    if not leaders:
        raise ValueError(
            "leaders cannot be empty"
        )

    durations: list[int] = []

    current_leader = leaders[0]
    current_duration = 1
    number_of_switches = 0

    for leader in leaders[1:]:
        if leader == current_leader:
            current_duration += 1
            continue

        durations.append(
            current_duration
        )

        number_of_switches += 1
        current_leader = leader
        current_duration = 1

    durations.append(
        current_duration
    )

    duration_series = pd.Series(
        durations,
        dtype="float64",
    )

    return LeaderDiagnostics(
        number_of_switches=(
            number_of_switches
        ),
        average_lead_duration=float(
            duration_series.mean()
        ),
        median_lead_duration=float(
            duration_series.median()
        ),
        longest_lead_duration=int(
            max(durations)
        ),
        leadership_counts=dict(
            Counter(leaders)
        ),
    )
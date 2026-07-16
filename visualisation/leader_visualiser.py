from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from backtesting.leader_analysis import (
    LeaderAnalysisResult,
)

from collections.abc import Mapping

from backtesting.baseline_evaluator import (
    BaselineResult,
)


class LeaderVisualiser:
    def __init__(
        self,
        result: LeaderAnalysisResult,
        component_results: Mapping[
            str,
            BaselineResult,
        ],
    ) -> None:
        if not component_results:
            raise ValueError(
                "component_results cannot be empty"
            )

        result_names = set(
            result.strategy_names
        )

        component_names = set(
            component_results
        )

        if result_names != component_names:
            missing = (
                result_names
                - component_names
            )

            unexpected = (
                component_names
                - result_names
            )

            raise ValueError(
                "component result names must match "
                "the analysed strategies; "
                f"missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        self.result = result
        self._component_results = dict(
            component_results
        )

        colour_cycle = (
            plt.rcParams[
                "axes.prop_cycle"
            ]
            .by_key()[
                "color"
            ]
        )

        self.strategy_colours = {
            name: colour_cycle[
                index
                % len(colour_cycle)
            ]
            for index, name in enumerate(
                result.strategy_names
            )
        }

        self.cash_colour = "0.75"

    @staticmethod
    def _currency_formatter(
        value: float,
        _: object,
    ) -> str:
        return f"${value:,.0f}"

    def _selection_colour(
        self,
        index: int,
    ) -> str:
        if index < 0:
            return self.cash_colour

        strategy_name = (
            self.result
            .strategy_names[
                index
            ]
        )

        return self.strategy_colours[
            strategy_name
        ]

    def plot_dashboard(
        self,
    ) -> None:
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(17, 9),
            sharex=False,
        )

        figure.subplots_adjust(
            left=0.08,
            right=0.90,
            top=0.90,
            bottom=0.10,
            wspace=0.20,
            hspace=0.28,
        )

        leadership_axis = axes[0, 0]
        pnl_axis = axes[0, 1]
        regret_axis = axes[1, 0]
        rank_axis = axes[1, 1]

        days = self.result.days

        strategy_names = list(
            self.result.strategy_names
        )

        number_of_strategies = len(
            strategy_names
        )

        cash_row = number_of_strategies

        def abbreviated_name(
            name: str,
        ) -> str:
            replacements = {
                "ALGO long-and-hold": (
                    "ALGO long"
                ),
                "ALGO short-and-hold": (
                    "ALGO short"
                ),
                "Max long all assets": (
                    "Max long all"
                ),
                "Max short all assets": (
                    "Max short all"
                ),
                "Previous-day momentum": (
                    "Prev-day momentum"
                ),
                "Previous-day reversal": (
                    "Prev-day reversal"
                ),
                (
                    "1-day winners-long "
                    "losers-short top/bottom 20%"
                ): (
                    "1d winners-long\n"
                    "losers-short 20%"
                ),
                (
                    "1-day losers-long "
                    "winners-short top/bottom 20%"
                ): (
                    "1d losers-long\n"
                    "winners-short 20%"
                ),
                (
                    "5-day winners-long "
                    "losers-short top/bottom 20%"
                ): (
                    "5d winners-long\n"
                    "losers-short 20%"
                ),
                (
                    "5-day losers-long "
                    "winners-short top/bottom 20%"
                ): (
                    "5d losers-long\n"
                    "winners-short 20%"
                ),
                (
                    "20-day winners-long "
                    "losers-short top/bottom 20%"
                ): (
                    "20d winners-long\n"
                    "losers-short 20%"
                ),
                (
                    "20-day losers-long "
                    "winners-short top/bottom 20%"
                ): (
                    "20d losers-long\n"
                    "winners-short 20%"
                ),
            }

            return replacements.get(
                name,
                name,
            )

        short_strategy_names = [
            abbreviated_name(name)
            for name in strategy_names
        ]

        # ==========================================================
        # Leadership lanes
        # ==========================================================

        selected_indices = np.asarray(
            self.result.selected_indices,
            dtype=np.int64,
        )

        lane_values = np.where(
            selected_indices >= 0,
            selected_indices,
            cash_row,
        )

        for row_index, strategy_name in enumerate(
            strategy_names
        ):
            active = (
                lane_values == row_index
            )

            leadership_axis.fill_between(
                days,
                row_index - 0.32,
                row_index + 0.32,
                where=active,
                step="mid",
                color=self.strategy_colours[
                    strategy_name
                ],
                alpha=0.95,
            )

        cash_active = (
            lane_values == cash_row
        )

        leadership_axis.fill_between(
            days,
            cash_row - 0.32,
            cash_row + 0.32,
            where=cash_active,
            step="mid",
            color=self.cash_colour,
            alpha=0.95,
        )

        for switch_day in (
            self.result.switch_days
        ):
            leadership_axis.axvline(
                int(switch_day),
                linewidth=0.5,
                alpha=0.18,
            )

        leadership_axis.set_yticks(
            np.arange(
                number_of_strategies + 1
            )
        )

        leadership_axis.set_yticklabels(
            short_strategy_names
            + ["Cash"],
            fontsize=7,
        )

        leadership_axis.set_ylim(
            number_of_strategies + 0.6,
            -0.6,
        )

        leadership_axis.set_title(
            "Tradeable Leader Through Time"
        )

        leadership_axis.set_ylabel(
            "Selected strategy"
        )

        leadership_axis.grid(
            axis="x",
            alpha=0.20,
        )

        leadership_axis.grid(
            axis="y",
            alpha=0.08,
        )

        diagnostics_text = (
            f"Switches: "
            f"{self.result.number_of_switches}\n"
            f"Mean duration: "
            f"{self.result.mean_lead_duration:.1f} days\n"
            f"Median duration: "
            f"{self.result.median_lead_duration:.1f} days\n"
            f"Longest duration: "
            f"{self.result.longest_lead_duration} days"
        )

        leadership_axis.text(
            0.99,
            0.02,
            diagnostics_text,
            transform=(
                leadership_axis.transAxes
            ),
            ha="right",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox={
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.85,
            },
        )

        # ==========================================================
        # Cumulative PnL
        # ==========================================================

        pnl_matrix = np.vstack(
            [
                np.asarray(
                    result.daily_pnl,
                    dtype=np.float64,
                )
                for result in (
                    self
                    ._component_results
                    .values()
                )
            ]
        )

        equal_weight_daily_pnl = np.mean(
            pnl_matrix,
            axis=0,
        )

        equal_weight_cumulative_pnl = (
            np.cumsum(
                equal_weight_daily_pnl
            )
        )

        pnl_axis.plot(
            days,
            self.result.leader_cumulative_pnl,
            linewidth=2.2,
            label="Lagged score leader",
        )

        pnl_axis.plot(
            days,
            self.result.best_static_cumulative_pnl,
            linewidth=1.8,
            linestyle="--",
            label="Best static strategy",
        )

        pnl_axis.plot(
            days,
            equal_weight_cumulative_pnl,
            linewidth=1.5,
            linestyle="-.",
            label="Equal-weight strategy basket",
        )

        pnl_axis.axhline(
            0,
            linewidth=1,
        )

        pnl_axis.set_title(
            "Cumulative Net PnL"
        )

        pnl_axis.set_ylabel(
            "Cumulative PnL"
        )

        pnl_axis.yaxis.set_major_formatter(
            FuncFormatter(
                self._currency_formatter
            )
        )

        pnl_axis.legend(
            fontsize=8,
            loc="best",
        )

        pnl_axis.grid(
            alpha=0.22,
        )

        # ==========================================================
        # Regret
        # ==========================================================

        regret_axis.plot(
            days,
            self.result.cumulative_static_regret,
            linewidth=2,
            label="Regret versus best static",
        )

        regret_axis.axhline(
            0,
            linewidth=1,
        )

        regret_axis.fill_between(
            days,
            0,
            self.result.cumulative_static_regret,
            alpha=0.12,
        )

        regret_axis.set_title(
            "Cumulative Regret Versus Best Static"
        )

        regret_axis.set_ylabel(
            "Regret"
        )

        regret_axis.yaxis.set_major_formatter(
            FuncFormatter(
                self._currency_formatter
            )
        )

        regret_axis.legend(
            fontsize=8,
            loc="best",
        )

        regret_axis.grid(
            alpha=0.22,
        )

        # ==========================================================
        # Rank through time
        # ==========================================================

        score_matrix = np.asarray(
            self.result.score_matrix,
            dtype=np.float64,
        )

        rank_matrix = np.full(
            score_matrix.shape,
            np.nan,
            dtype=np.float64,
        )

        for day_index in range(
            score_matrix.shape[1]
        ):
            scores = score_matrix[
                :,
                day_index,
            ]

            valid = np.isfinite(
                scores
            )

            if not np.any(valid):
                continue

            valid_indices = np.flatnonzero(
                valid
            )

            ordered = valid_indices[
                np.argsort(
                    scores[
                        valid_indices
                    ]
                )[::-1]
            ]

            for rank, strategy_index in enumerate(
                ordered,
                start=1,
            ):
                rank_matrix[
                    strategy_index,
                    day_index,
                ] = float(rank)

        top_rank_limit = min(
            5,
            number_of_strategies,
        )

        for strategy_index, strategy_name in enumerate(
            strategy_names
        ):
            visible_ranks = rank_matrix[
                strategy_index
            ].copy()

            visible_ranks[
                visible_ranks
                > top_rank_limit
            ] = np.nan

            rank_axis.plot(
                days,
                visible_ranks,
                linewidth=1.4,
                color=self.strategy_colours[
                    strategy_name
                ],
                label=abbreviated_name(
                    strategy_name
                ).replace(
                    "\n",
                    " ",
                ),
            )

        rank_axis.set_title(
            (
                "Strategy Rank Through Time "
                f"(Top 1–{top_rank_limit})"
            )
        )

        rank_axis.set_ylabel(
            "Rank (1 = best)"
        )

        rank_axis.set_ylim(
            top_rank_limit + 0.5,
            0.5,
        )

        rank_axis.set_yticks(
            np.arange(
                1,
                top_rank_limit + 1,
            )
        )

        rank_axis.grid(
            alpha=0.22,
        )

        rank_axis.legend(
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(
                1.01,
                0.5,
            ),
        )

        # ==========================================================
        # Shared formatting
        # ==========================================================

        for axis in (
            leadership_axis,
            pnl_axis,
            regret_axis,
            rank_axis,
        ):
            axis.set_xlabel(
                "Day"
            )

        figure.suptitle(
            "Strategy Leadership and Regret",
            fontsize=16,
        )

        plt.show()
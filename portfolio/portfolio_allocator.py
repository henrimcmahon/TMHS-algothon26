from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """
    Result of one portfolio-allocation step.
    """

    positions: IntArray

    desired_dollar_positions: FloatArray
    realised_dollar_positions: FloatArray

    shares_traded: IntArray
    dollar_volume_traded: FloatArray
    commissions: FloatArray

    total_dollar_volume: float
    total_commission: float


class PortfolioAllocator:
    """
    Convert one signal per asset into legal integer share positions.

    There is no total portfolio budget in the Algothon rules. Each
    instrument is therefore allocated independently relative to its own
    dollar position limit.

    Asset 0:
        maximum absolute dollar position = 100,000
        commission rate = 0.00002

    Assets 1-50:
        maximum absolute dollar position = 10,000
        commission rate = 0.0001
    """

    def __init__(
        self,
        position_limits: FloatArray,
        commission_rates: FloatArray,
        *,
        signal_scale: float = 1.0,
        minimum_trade_dollars: float = 0.0,
        turnover_smoothing: float = 0.0,
    ) -> None:
        """
        Args:
            position_limits:
                Maximum absolute dollar position for each asset.

            commission_rates:
                Commission charged per dollar traded for each asset.

            signal_scale:
                Controls how quickly signals saturate position limits.

                The target fraction of each position limit is:

                    tanh(signal / signal_scale)

                Smaller values make positions reach their limits faster.

            minimum_trade_dollars:
                Do not change a position when the proposed trade is smaller
                than this dollar amount, unless the existing position must
                be reduced to satisfy today's position limit.

            turnover_smoothing:
                Fraction of the previous dollar position retained when
                moving toward a new target.

                0.0:
                    move directly to the new target.

                0.8:
                    retain 80% of the previous exposure and move 20%
                    toward the new target.
        """
        limits = np.asarray(
            position_limits,
            dtype=np.float64,
        )

        rates = np.asarray(
            commission_rates,
            dtype=np.float64,
        )

        if limits.ndim != 1:
            raise ValueError(
                "position_limits must be one-dimensional"
            )

        if rates.shape != limits.shape:
            raise ValueError(
                "commission_rates must match position_limits"
            )

        if len(limits) == 0:
            raise ValueError(
                "at least one instrument is required"
            )

        if np.any(~np.isfinite(limits)):
            raise ValueError(
                "position_limits must be finite"
            )

        if np.any(limits <= 0):
            raise ValueError(
                "position_limits must be positive"
            )

        if np.any(~np.isfinite(rates)):
            raise ValueError(
                "commission_rates must be finite"
            )

        if np.any(rates < 0):
            raise ValueError(
                "commission_rates cannot be negative"
            )

        if not np.isfinite(signal_scale) or signal_scale <= 0:
            raise ValueError(
                "signal_scale must be finite and positive"
            )

        if (
            not np.isfinite(minimum_trade_dollars)
            or minimum_trade_dollars < 0
        ):
            raise ValueError(
                "minimum_trade_dollars must be finite and non-negative"
            )

        if not 0.0 <= turnover_smoothing < 1.0:
            raise ValueError(
                "turnover_smoothing must be in [0, 1)"
            )

        self.position_limits = limits
        self.commission_rates = rates

        self.signal_scale = float(signal_scale)
        self.minimum_trade_dollars = float(
            minimum_trade_dollars
        )
        self.turnover_smoothing = float(
            turnover_smoothing
        )

    @classmethod
    def algothon_defaults(
        cls,
        number_of_instruments: int = 51,
        *,
        signal_scale: float = 1.0,
        minimum_trade_dollars: float = 100.0,
        turnover_smoothing: float = 0.0,
    ) -> PortfolioAllocator:
        """
        Construct an allocator using the published Algothon limits and
        commission rates.
        """
        if number_of_instruments != 51:
            raise ValueError(
                "The current Algothon universe contains 51 instruments"
            )

        limits = np.full(
            number_of_instruments,
            10_000.0,
            dtype=np.float64,
        )

        limits[0] = 100_000.0

        commission_rates = np.full(
            number_of_instruments,
            0.0001,
            dtype=np.float64,
        )

        commission_rates[0] = 0.00002

        return cls(
            position_limits=limits,
            commission_rates=commission_rates,
            signal_scale=signal_scale,
            minimum_trade_dollars=minimum_trade_dollars,
            turnover_smoothing=turnover_smoothing,
        )

    @property
    def number_of_instruments(self) -> int:
        return len(self.position_limits)

    def maximum_shares(
        self,
        prices: FloatArray,
    ) -> IntArray:
        """
        Calculate today's maximum legal absolute share position.

        The use of floor means:

            abs(shares) * price <= dollar limit
        """
        price_values = self._validate_prices(
            prices
        )

        return np.floor(
            self.position_limits / price_values
        ).astype(np.int64)

    def allocate(
        self,
        signals: FloatArray,
        prices: FloatArray,
        current_positions: IntArray | None = None,
        exposure_fraction: float = 1.0,
    ) -> AllocationResult:
        """
        Convert signals into legal end-of-day share positions.

        Signals may be any finite real values. They are mapped into
        position-limit fractions through tanh:

            target_fraction_i = tanh(signal_i / signal_scale)

            target_dollars_i =
                target_fraction_i * position_limit_i

        Consequently:

            signal = 0      -> no position
            large positive  -> maximum long
            large negative  -> maximum short
        """
        signal_values = self._validate_signals(
            signals
        )

        price_values = self._validate_prices(
            prices
        )

        current = self._validate_positions(
            current_positions
        )

        maximum_shares = self.maximum_shares(
            price_values
        )

        # Re-clip the current position using today's prices. This models
        # the fact that a price rise can force a reduction even if the
        # strategy's desired exposure did not change.
        legal_current = np.clip(
            current,
            -maximum_shares,
            maximum_shares,
        ).astype(np.int64)

        current_dollar_positions = (
            legal_current.astype(np.float64)
            * price_values
        )

        if not 0.0 <= exposure_fraction <= 1.0:
            raise ValueError(
                "exposure_fraction must be between 0 and 1"
            )

        target_fractions = (
            np.tanh(
                signal_values / self.signal_scale
            )
            * exposure_fraction
        )

        desired_dollar_positions = (
            target_fractions
            * self.position_limits
        )

        if self.turnover_smoothing > 0:
            desired_dollar_positions = (
                self.turnover_smoothing
                * current_dollar_positions
                + (
                    1.0
                    - self.turnover_smoothing
                )
                * desired_dollar_positions
            )

        desired_dollar_positions = np.clip(
            desired_dollar_positions,
            -self.position_limits,
            self.position_limits,
        )

        target_positions = np.trunc(
            desired_dollar_positions
            / price_values
        ).astype(np.int64)

        target_positions = np.clip(
            target_positions,
            -maximum_shares,
            maximum_shares,
        ).astype(np.int64)

        proposed_share_trades = (
            target_positions
            - legal_current
        )

        proposed_dollar_volume = (
            np.abs(proposed_share_trades)
            * price_values
        )

        # Ignore uneconomic small changes. However, this must not undo a
        # mandatory limit reduction caused by today's price.
        current_was_illegal = (
            current != legal_current
        )

        suppress_trade = (
            proposed_dollar_volume
            < self.minimum_trade_dollars
        ) & (~current_was_illegal)

        final_positions = target_positions.copy()

        final_positions[suppress_trade] = (
            legal_current[suppress_trade]
        )

        # Final defensive clip using today's moving share ceilings.
        final_positions = np.clip(
            final_positions,
            -maximum_shares,
            maximum_shares,
        ).astype(np.int64)

        shares_traded = (
            final_positions - current
        ).astype(np.int64)

        realised_dollar_positions = (
            final_positions.astype(np.float64)
            * price_values
        )

        dollar_volume_traded = (
            np.abs(shares_traded)
            * price_values
        )

        commissions = (
            dollar_volume_traded
            * self.commission_rates
        )

        return AllocationResult(
            positions=final_positions,
            desired_dollar_positions=(
                desired_dollar_positions
            ),
            realised_dollar_positions=(
                realised_dollar_positions
            ),
            shares_traded=shares_traded,
            dollar_volume_traded=(
                dollar_volume_traded
            ),
            commissions=commissions,
            total_dollar_volume=float(
                np.sum(dollar_volume_traded)
            ),
            total_commission=float(
                np.sum(commissions)
            ),
        )

    def _validate_signals(
        self,
        signals: FloatArray,
    ) -> FloatArray:
        values = np.asarray(
            signals,
            dtype=np.float64,
        )

        if values.shape != (
            self.number_of_instruments,
        ):
            raise ValueError(
                "signals must contain exactly one value per instrument"
            )

        if np.any(~np.isfinite(values)):
            raise ValueError(
                "signals must be finite"
            )

        return values

    def _validate_prices(
        self,
        prices: FloatArray,
    ) -> FloatArray:
        values = np.asarray(
            prices,
            dtype=np.float64,
        )

        if values.shape != (
            self.number_of_instruments,
        ):
            raise ValueError(
                "prices must contain exactly one value per instrument"
            )

        if np.any(~np.isfinite(values)):
            raise ValueError(
                "prices must be finite"
            )

        if np.any(values <= 0):
            raise ValueError(
                "prices must be strictly positive"
            )

        return values

    def _validate_positions(
        self,
        positions: IntArray | None,
    ) -> IntArray:
        if positions is None:
            return np.zeros(
                self.number_of_instruments,
                dtype=np.int64,
            )

        values = np.asarray(positions)

        if values.shape != (
            self.number_of_instruments,
        ):
            raise ValueError(
                "current_positions must contain exactly one value "
                "per instrument"
            )

        if not np.issubdtype(
            values.dtype,
            np.integer,
        ):
            if np.any(values != np.trunc(values)):
                raise ValueError(
                    "current_positions must contain integers"
                )

        return values.astype(np.int64)
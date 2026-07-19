from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BacktestResult:
    name: str

    # Full daily histories
    daily_pnl: FloatArray
    cumulative_pnl: FloatArray
    gross_daily_pnl: FloatArray
    daily_commissions: FloatArray
    daily_turnover: FloatArray
    positions: IntArray
    trades: IntArray
    portfolio_values: FloatArray
    cash_history: FloatArray

    # P&L statistics
    total_pnl: float
    total_gross_pnl: float
    total_commissions: float
    mean_daily_pnl: float
    pnl_std: float

    # Risk-adjusted statistics
    annualised_sharpe: float
    sharpe_multiplier: float
    score: float
    maximum_drawdown: float

    # Trading statistics
    total_dollar_volume: float
    total_shares_traded: float
    average_daily_turnover: float
    maximum_daily_turnover: float
    return_on_volume: float

    # Scoring range
    scoring_start_day: int
    scoring_days: int

    @property
    def position_history(self) -> IntArray:
        return self.positions

    @property
    def trade_history(self) -> IntArray:
        return self.trades

    @property
    def commissions(self) -> FloatArray:
        return self.daily_commissions

    @property
    def sharpe(self) -> float:
        return self.annualised_sharpe

    @property
    def max_drawdown(self) -> float:
        return self.maximum_drawdown

    @property
    def net_daily_pnl(self) -> FloatArray:
        return self.daily_pnl

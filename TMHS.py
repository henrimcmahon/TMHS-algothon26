from __future__ import annotations

import numpy as np

from trading import TradingEngine


_engine = TradingEngine(
    residual_estimation_window=60,
    forgetting_rate=0.03,
    entropy_penalty=0.5,
    signal_scale=5.0,
    minimum_trade_dollars=100.0,
    turnover_smoothing=0.25,
)


def getMyPosition(
    prcSoFar: np.ndarray,
) -> np.ndarray:
    result = _engine.update(
        np.asarray(
            prcSoFar,
            dtype=np.float64,
        )
    )

    return result.positions
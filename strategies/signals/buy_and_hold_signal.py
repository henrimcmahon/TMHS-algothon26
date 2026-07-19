from __future__ import annotations

import numpy as np

from strategies.signals.signal_model import FloatArray, SignalModel


class BuyAndHoldSignal(SignalModel):
    """Returns a constant long signal for every ticker."""

    def compute(
        self,
        price_history: FloatArray,
    ) -> FloatArray:
        if price_history.ndim != 2:
            raise ValueError(
                "price_history must have shape (n_tickers, n_days)"
            )

        return np.ones(
            price_history.shape[0],
            dtype=np.float64,
        )

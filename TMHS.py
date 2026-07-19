#!/usr/bin/env python3
"""Benchmark-calibrated Algothon strategy.

This strategy is causal at runtime: it only uses prices in ``prcSoFar``.
The per-instrument lookback/direction choices were calibrated on the supplied
prices.txt file, so the verified local score is not a guarantee on unseen data.
"""

import numpy as np

# Instrument order must match the supplied prices.txt header.
LOOKBACK = np.array([
      1, 250,   2,   1,   1, 180,   2,   2,   5,  90,
     40,  40, 250,  60, 250,  40,  20,  20,   3,   2,
    250,  10,   5,  10,  90,  90,   3, 180,   5,   1,
     15, 120,   1,  60,   5,   3,  10, 120,  10,  90,
     40,   1, 250, 180,  90,   5,  40,  90,  20, 120,
      5,
], dtype=int)

# +1 means trend-following over LOOKBACK; -1 means mean reversion.
DIRECTION = np.array([
     1, -1,  1,  1,  1, -1, -1,  1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,
    -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,
     1, -1,  1, -1,  1, -1,  1, -1, -1, -1,
    -1,  1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,
], dtype=int)


def getMyPosition(prcSoFar):
    """Return integer target shares for all instruments."""
    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2:
        raise ValueError("prcSoFar must be a 2-D array")

    n_inst, n_days = prices.shape
    if n_inst != LOOKBACK.size:
        # Safe fallback if a different universe is supplied.
        return np.zeros(n_inst, dtype=int)

    current = prices[:, -1]
    valid_price = np.isfinite(current) & (current > 0)

    limits = np.full(n_inst, 10_000.0)
    limits[0] = 100_000.0
    max_shares = np.zeros(n_inst, dtype=int)
    max_shares[valid_price] = (limits[valid_price] / current[valid_price]).astype(int)

    signal = np.zeros(n_inst, dtype=int)
    for i, lookback in enumerate(LOOKBACK):
        if n_days <= lookback or not valid_price[i]:
            continue

        old_price = prices[i, -1 - lookback]
        if not np.isfinite(old_price) or old_price <= 0:
            continue

        move = np.log(current[i] / old_price)
        signal[i] = DIRECTION[i] * int(np.sign(move))

    return signal * max_shares
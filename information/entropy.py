from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def state_probabilities(
    states: ArrayLike,
    alphabet: tuple[int, ...] = (-1, 0, 1),
) -> NDArray[np.float64]:
    """Estimate the empirical probability of each state."""
    values = np.asarray(states).reshape(-1)

    if values.size == 0:
        raise ValueError("states cannot be empty.")

    counts = np.asarray(
        [np.count_nonzero(values == state) for state in alphabet],
        dtype=float,
    )

    return counts / values.size


def shannon_entropy(
    states: ArrayLike,
    *,
    base: float = 2.0,
    alphabet: tuple[int, ...] = (-1, 0, 1),
) -> float:
    """
    Empirical Shannon entropy.

    With base=2, the result is measured in bits.
    """
    if base <= 0.0 or np.isclose(base, 1.0):
        raise ValueError("base must be positive and different from 1.")

    probabilities = state_probabilities(states, alphabet)
    positive = probabilities > 0.0

    return float(
        -np.sum(
            probabilities[positive]
            * np.log(probabilities[positive])
            / np.log(base)
        )
    )


def ticker_entropies(
    state_matrix: NDArray[np.integer],
    *,
    base: float = 2.0,
) -> NDArray[np.float64]:
    """Calculate entropy independently for every ticker row."""
    states = np.asarray(state_matrix)

    if states.ndim != 2:
        raise ValueError(
            "state_matrix must have shape (n_tickers, n_observations)."
        )

    return np.asarray(
        [shannon_entropy(row, base=base) for row in states],
        dtype=float,
    )
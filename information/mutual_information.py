from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_ALPHABET = (-1, 0, 1)


def joint_probability_matrix(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alphabet: tuple[int, ...] = DEFAULT_ALPHABET,
) -> NDArray[np.float64]:
    """Estimate the empirical joint distribution P(X, Y)."""
    x_values = np.asarray(x).reshape(-1)
    y_values = np.asarray(y).reshape(-1)

    if x_values.size != y_values.size:
        raise ValueError("x and y must contain the same number of values.")

    if x_values.size == 0:
        raise ValueError("x and y cannot be empty.")

    joint = np.zeros((len(alphabet), len(alphabet)), dtype=float)

    for x_index, x_state in enumerate(alphabet):
        for y_index, y_state in enumerate(alphabet):
            joint[x_index, y_index] = np.mean(
                (x_values == x_state) & (y_values == y_state)
            )

    return joint


def mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    *,
    base: float = 2.0,
    alphabet: tuple[int, ...] = DEFAULT_ALPHABET,
) -> float:
    """
    Calculate empirical mutual information I(X; Y).

    With base=2, the result is measured in bits.
    """
    if base <= 0.0 or np.isclose(base, 1.0):
        raise ValueError("base must be positive and different from 1.")

    joint = joint_probability_matrix(x, y, alphabet=alphabet)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    independent = np.outer(p_x, p_y)

    valid = (joint > 0.0) & (independent > 0.0)

    return float(
        np.sum(
            joint[valid]
            * np.log(joint[valid] / independent[valid])
            / np.log(base)
        )
    )


def normalised_mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    *,
    base: float = 2.0,
    alphabet: tuple[int, ...] = DEFAULT_ALPHABET,
) -> float:
    """
    Symmetric normalised MI in approximately [0, 1].

    Normalisation uses:

        I(X;Y) / sqrt(H(X) H(Y))
    """
    from .entropy import shannon_entropy

    mi = mutual_information(
        x,
        y,
        base=base,
        alphabet=alphabet,
    )
    h_x = shannon_entropy(x, base=base, alphabet=alphabet)
    h_y = shannon_entropy(y, base=base, alphabet=alphabet)

    denominator = np.sqrt(h_x * h_y)

    if denominator <= 0.0:
        return 0.0

    return float(mi / denominator)


def mutual_information_matrix(
    states: NDArray[np.integer],
    *,
    lag: int = 0,
    normalised: bool = False,
    base: float = 2.0,
) -> NDArray[np.float64]:
    """
    Build a ticker-to-ticker information matrix.

    Matrix entry [source, target] is:

        lag = 0:
            I(source_t ; target_t)

        lag > 0:
            I(source_t ; target_{t + lag})

    A lagged matrix is directional because rows represent sources and
    columns represent future targets.
    """
    values = np.asarray(states)

    if values.ndim != 2:
        raise ValueError(
            "states must have shape (n_tickers, n_observations)."
        )

    if lag < 0:
        raise ValueError("lag must be non-negative.")

    if lag >= values.shape[1]:
        raise ValueError("lag must be smaller than the observation count.")

    if lag == 0:
        source_states = values
        target_states = values
    else:
        source_states = values[:, :-lag]
        target_states = values[:, lag:]

    n_tickers = values.shape[0]
    matrix = np.zeros((n_tickers, n_tickers), dtype=float)

    information_function = (
        normalised_mutual_information
        if normalised
        else mutual_information
    )

    for source_index in range(n_tickers):
        for target_index in range(n_tickers):
            matrix[source_index, target_index] = information_function(
                source_states[source_index],
                target_states[target_index],
                base=base,
            )

    return matrix


@dataclass(frozen=True)
class RollingInformationResult:
    end_days: NDArray[np.int64]
    matrices: NDArray[np.float64]


def rolling_mutual_information(
    states: NDArray[np.integer],
    *,
    window: int,
    lag: int = 0,
    step: int = 1,
    normalised: bool = False,
    base: float = 2.0,
) -> RollingInformationResult:
    """
    Calculate information matrices over rolling windows.

    Each matrix uses only observations inside its own historical window.
    """
    values = np.asarray(states)

    if values.ndim != 2:
        raise ValueError(
            "states must have shape (n_tickers, n_observations)."
        )

    if window <= lag:
        raise ValueError("window must be greater than lag.")

    if window > values.shape[1]:
        raise ValueError("window cannot exceed the observation count.")

    if step <= 0:
        raise ValueError("step must be positive.")

    matrices: list[NDArray[np.float64]] = []
    end_days: list[int] = []

    for end in range(window, values.shape[1] + 1, step):
        start = end - window
        window_states = values[:, start:end]

        matrices.append(
            mutual_information_matrix(
                window_states,
                lag=lag,
                normalised=normalised,
                base=base,
            )
        )
        end_days.append(end)

    return RollingInformationResult(
        end_days=np.asarray(end_days, dtype=np.int64),
        matrices=np.stack(matrices),
    )
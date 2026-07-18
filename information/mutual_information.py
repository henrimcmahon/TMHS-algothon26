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

    alphabet_array = np.asarray(alphabet)
    n_states = alphabet_array.size

    # Map each observed state onto an integer code:
    # alphabet[0] -> 0, alphabet[1] -> 1, ...
    x_codes = np.searchsorted(alphabet_array, x_values)
    y_codes = np.searchsorted(alphabet_array, y_values)

    x_in_bounds = x_codes < n_states
    y_in_bounds = y_codes < n_states

    x_valid = np.zeros(x_values.size, dtype=bool)
    y_valid = np.zeros(y_values.size, dtype=bool)

    x_valid[x_in_bounds] = (
        alphabet_array[x_codes[x_in_bounds]]
        == x_values[x_in_bounds]
    )
    y_valid[y_in_bounds] = (
        alphabet_array[y_codes[y_in_bounds]]
        == y_values[y_in_bounds]
    )

    valid = x_valid & y_valid

    if not np.all(valid):
        invalid_x = np.unique(x_values[~x_valid])
        invalid_y = np.unique(y_values[~y_valid])

        raise ValueError(
            "All values must belong to the supplied alphabet. "
            f"Invalid x values: {invalid_x}; "
            f"invalid y values: {invalid_y}."
        )

    # Encode each pair as one integer:
    #
    # (x_code, y_code) -> x_code * n_states + y_code
    #
    # For three states this produces codes 0 through 8.
    pair_codes = x_codes * n_states + y_codes

    counts = np.bincount(
        pair_codes,
        minlength=n_states * n_states,
    )

    return counts.reshape(n_states, n_states) / x_values.size


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
    if base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and different from 1.")

    joint = joint_probability_matrix(
        x,
        y,
        alphabet=alphabet,
    )

    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    independent = np.outer(p_x, p_y)

    valid = joint > 0.0

    log_base = np.log(base)

    return float(
        np.sum(
            joint[valid]
            * np.log(joint[valid] / independent[valid])
        )
        / log_base
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
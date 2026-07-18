import cProfile
import pstats
import time

import numpy as np

from information.mutual_information import (
    joint_probability_matrix,
    mutual_information,
    mutual_information_matrix,
    rolling_mutual_information,
)


def random_states(
    n_tickers: int = 51,
    n_days: int = 150,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.choice(
        [-1, 0, 1],
        size=(n_tickers, n_days),
    )


def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()

    profiler.enable()
    t0 = time.perf_counter()

    result = func(*args, **kwargs)

    elapsed = time.perf_counter() - t0
    profiler.disable()

    print(f"\n{'=' * 80}")
    print(f"{func.__name__}")
    print(f"Elapsed: {elapsed:.4f} s")
    print(f"{'=' * 80}")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    stats.print_stats(30)

    return result


def test_profile_joint_probability():
    states = random_states()

    profile_function(
        joint_probability_matrix,
        states[0],
        states[1],
    )


def test_profile_mutual_information():
    states = random_states()

    profile_function(
        mutual_information,
        states[0],
        states[1],
    )


def test_profile_mutual_information_matrix():
    states = random_states()

    profile_function(
        mutual_information_matrix,
        states,
        lag=1,
        normalised=False,
    )


def test_profile_rolling_mutual_information():
    states = random_states(
        n_tickers=51,
        n_days=500,
    )

    profile_function(
        rolling_mutual_information,
        states,
        window=150,
        lag=1,
    )
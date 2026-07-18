import numpy as np

from information.mutual_information import (
    joint_probability_matrix,
    mutual_information,
)


def slow_joint_probability_matrix(
    x: np.ndarray,
    y: np.ndarray,
    alphabet: tuple[int, ...] = (-1, 0, 1),
) -> np.ndarray:
    joint = np.zeros(
        (len(alphabet), len(alphabet)),
        dtype=float,
    )

    for x_index, x_state in enumerate(alphabet):
        for y_index, y_state in enumerate(alphabet):
            joint[x_index, y_index] = np.mean(
                (x == x_state) & (y == y_state)
            )

    return joint


def test_fast_joint_probability_matches_slow_version():
    rng = np.random.default_rng(42)

    x = rng.choice([-1, 0, 1], size=1_000)
    y = rng.choice([-1, 0, 1], size=1_000)

    expected = slow_joint_probability_matrix(x, y)
    actual = joint_probability_matrix(x, y)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
    )


def test_joint_probability_sums_to_one():
    x = np.array([-1, -1, 0, 1, 1])
    y = np.array([0, 1, 0, -1, 1])

    joint = joint_probability_matrix(x, y)

    assert np.isclose(joint.sum(), 1.0)


def test_identical_variables_have_positive_information():
    x = np.array(
        [-1, 0, 1, -1, 0, 1],
    )

    mi = mutual_information(x, x)

    assert mi > 0.0


def test_constant_variables_have_zero_information():
    x = np.zeros(100, dtype=int)
    y = np.zeros(100, dtype=int)

    mi = mutual_information(x, y)

    assert np.isclose(mi, 0.0)


def test_invalid_state_raises():
    x = np.array([-1, 0, 2])
    y = np.array([-1, 0, 1])

    with np.testing.assert_raises(ValueError):
        joint_probability_matrix(x, y)
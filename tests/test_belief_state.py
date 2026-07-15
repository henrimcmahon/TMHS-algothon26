import numpy as np

from bayesian import BeliefState


def test_belief_state_normalises_prior() -> None:
    beliefs = BeliefState(
        np.asarray(
            [2.0, 1.0, 1.0],
            dtype=np.float64,
        )
    )

    assert np.allclose(
        beliefs.probabilities,
        [0.5, 0.25, 0.25],
    )


def test_belief_state_rewards_likely_model() -> None:
    beliefs = BeliefState.uniform(3)

    beliefs.update(
        np.asarray(
            [-5.0, -1.0, -4.0],
            dtype=np.float64,
        )
    )

    assert beliefs.probabilities[1] > (
        beliefs.probabilities[0]
    )

    assert beliefs.probabilities[1] > (
        beliefs.probabilities[2]
    )

    assert np.isclose(
        np.sum(beliefs.probabilities),
        1.0,
    )


def test_forgetting_moves_beliefs_towards_uniform() -> None:
    beliefs = BeliefState(
        probabilities=np.asarray(
            [0.98, 0.01, 0.01],
            dtype=np.float64,
        ),
        forgetting_rate=0.10,
    )

    previous_maximum = float(
        np.max(beliefs.probabilities)
    )

    beliefs.diffuse()

    assert np.max(
        beliefs.probabilities
    ) < previous_maximum
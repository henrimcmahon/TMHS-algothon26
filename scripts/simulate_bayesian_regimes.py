from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from statistics.residual_statistics import FloatArray

from tests.test_regime_identification import (
    generate_mean_reverting_data,
    generate_momentum_data,
    generate_noise_data,
    run_inference,
)

def plot_regime(
    observations: FloatArray,
    title: str,
) -> None:
    results = run_inference(
        observations
    )

    probabilities = results[
        "probabilities"
    ]

    probability_days = np.arange(
        len(probabilities)
    )

    diagnostic_days = np.arange(
        2,
        2 + len(results["negative_log_likelihood"]),
    )

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
    )

    probability_axis = axes[0]
    entropy_axis = axes[1]
    negative_log_likelihood_axis = axes[2]
    information_axis = axes[3]

    probability_axis.plot(
        probability_days,
        probabilities[:, 0],
        label="Momentum",
    )

    probability_axis.plot(
        probability_days,
        probabilities[:, 1],
        label="Mean reversion",
    )

    probability_axis.plot(
        probability_days,
        probabilities[:, 2],
        label="Noise",
    )

    probability_axis.set_ylabel(
        "Posterior"
    )
    probability_axis.set_ylim(0, 1)
    probability_axis.legend()
    probability_axis.grid(alpha=0.25)

    entropy_axis.plot(
        diagnostic_days,
        results["posterior_entropy"],
        label="Posterior entropy",
    )

    entropy_axis.axhline(
        np.log(3),
        linestyle="--",
        linewidth=1,
        label="Maximum entropy",
    )

    entropy_axis.set_ylabel(
        "Entropy"
    )
    entropy_axis.legend()
    entropy_axis.grid(alpha=0.25)

    negative_log_likelihood_axis.plot(
        diagnostic_days,
        results["negative_log_likelihood"],
    )

    negative_log_likelihood_axis.set_ylabel(
        "negative_log_likelihood"
    )
    negative_log_likelihood_axis.grid(alpha=0.25)

    information_axis.plot(
        diagnostic_days,
        results["information_gain"],
    )

    information_axis.set_ylabel(
        "KL information gain"
    )
    information_axis.set_xlabel(
        "Update"
    )
    information_axis.grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout()

    plt.show()

plot_regime(
    generate_momentum_data(250),
    "Momentum regime",
)

plot_regime(
    generate_mean_reverting_data(250),
    "Mean-reverting regime",
)

plot_regime(
    generate_noise_data(250),
    "Noise regime",
)
# Architecture

## Overview

The project separates market analysis, systematic factor removal, probabilistic inference, and portfolio allocation into distinct layers.

```text
Price history
    |
    v
TickerUniverse
    |
    v
MarketStatistics
    |
    v
Factor / residual model
    |
    v
Residual histories
    |
    v
Predictive hypotheses
    |
    v
BayesianUpdater
    |
    v
Posterior predictive distributions
    |
    v
Signal generation
    |
    v
Portfolio allocation
    |
    v
getMyPosition()
````

Each layer has one responsibility and can be replaced or extended without rewriting the rest of the system.

---

## Data layer

### `Ticker`

Represents one instrument in the Algothon universe.

Responsibilities:

* Store the ticker symbol.
* Store its historical prices.
* Provide access to the price series.

### `TickerUniverse`

Loads and stores the complete set of instruments.

Responsibilities:

* Load the price matrix from `prices.txt`.
* Instantiate `Ticker` objects.
* Preserve the ordering required by `getMyPosition`.
* Provide access by ticker symbol or index.
* Expose the full price matrix.

---

## Market statistics

### `MarketStatistics`

Calculates observable properties of the market.

Responsibilities include:

* Log returns.
* Cross-sectional mean and median.
* Cross-sectional volatility.
* Skewness and excess kurtosis.
* Market breadth.
* Percentiles and dispersion.
* Correlation matrices.
* Hierarchical clustering.
* Principal-component analysis.
* PC1 explained variance.
* Effective market dimension.
* Rolling beta.
* Ticker-level feature snapshots.

This class contains analytical calculations only. It does not perform visualisation or make trading decisions.

---

## Market visualisation

### `Strategyvisualiser`

Displays calculations produced by `MarketStatistics`.

Responsibilities include:

* Clustered correlation heatmaps.
* Correlation dendrograms.
* Eigenvalue spectra.
* Rolling average correlation.
* Rolling PC1 dominance.
* Effective market dimension.
* Ticker comparison charts.
* Rolling beta charts.
* Slider and playback controls.

The visualiser should not duplicate calculations already available through `MarketStatistics`.

---

## Market-factor interpretation

Exploratory analysis suggests that `ALGO` behaves like a broad market factor rather than a cash asset.

Observed characteristics include:

* Market beta close to one.
* Very high correlation with the equal-weighted market.
* High regression explanatory power.
* High correlation with the first principal component.
* Low tracking error relative to the market basket.
* Little evidence of meaningful lead-lag predictive power.

This motivates using `ALGO` as a systematic market proxy and analysing other tickers relative to it.

---

## Residual modelling

### `ResidualStatistics`

Removes systematic exposure to a market proxy.

For ticker (i),

[
r_{i,t}
=======

\alpha_i
+
\beta_i r_{\mathrm{ALGO},t}
+
\varepsilon_{i,t}.
]

The residual

[
\varepsilon_{i,t}
=================

## r_{i,t}

## \alpha_i

\beta_i r_{\mathrm{ALGO},t}
]

represents the part of the ticker return not explained by the broad market factor.

Responsibilities include:

* Rolling alpha and beta estimation.
* Residual-return histories.
* Residual volatility.
* Residual Sharpe ratio.
* Residual z-score.
* Residual momentum.
* Residual skewness.
* Residual excess kurtosis.
* Residual autocorrelation.
* Strategy-ready residual feature matrices.

Residual modelling is separated from Bayesian inference. It provides the observations that the Bayesian system attempts to predict.

---

## Predictive distributions

### `PredictiveDistribution`

Defines the interface for a predictive probability distribution.

A predictive distribution exposes:

* Mean.
* Variance.
* Log probability of an observation.
* Sampling.

Initial implementations use Gaussian distributions.

Future implementations may include:

* Student-(t) distributions.
* Skew-normal distributions.
* Mixture distributions.

### `GaussianDistribution`

Represents

[
X \sim \mathcal N(\mu,\sigma^2).
]

Responsibilities include:

* Storing the predicted mean and variance.
* Evaluating Gaussian log density.
* Drawing random samples.

---

## Predictive hypotheses

### `PredictiveHypothesis`

Defines a competing model of residual-return generation.

Each hypothesis:

* Owns its internal state.
* Produces a predictive distribution.
* Updates its internal parameters after observing a new return.
* Exposes a unique name.
* Defines its minimum history requirement.

Initial hypotheses are:

### `MomentumHypothesis`

Assumes recent residual-return direction persists.

Internal state may include:

* Estimated trend.
* Predictive variance.
* Trend learning rate.
* Variance learning rate.

It predicts a Gaussian distribution centred on its current trend estimate.

### `MeanReversionHypothesis`

Assumes residual behaviour moves back toward an equilibrium.

A simple form is

[
\mathbb E[r_{t+1}]
==================

\kappa(\mu-r_t),
]

where:

* (\mu) is the estimated equilibrium.
* (\kappa) is the reversion speed.

This is a discrete approximation to an Ornstein–Uhlenbeck-style process.

### `NoiseHypothesis`

Assumes residual returns have zero conditional mean.

It predicts

[
r_{t+1}
\sim
\mathcal N(0,\sigma_t^2),
]

with variance updated through an exponentially weighted estimate.

The noise model provides a baseline that prevents momentum or mean reversion from being selected when neither has explanatory value.

---

## Belief state

### `BeliefState`

Stores posterior probabilities over competing hypotheses.

For hypotheses (H_1,\ldots,H_K), the belief state contains

[
P(H_1),\ldots,P(H_K).
]

After observing return (x_t), beliefs are updated using Bayes’ rule:

[
P(H_k\mid x_t)
==============

\frac{
p(x_t\mid H_k)P(H_k)
}{
\sum_j p(x_t\mid H_j)P(H_j)
}.
]

The implementation performs the update in log space for numerical stability.

Responsibilities include:

* Prior normalisation.
* Posterior updating.
* Belief entropy.
* Optional belief diffusion or forgetting.
* Copying and inspecting current beliefs.

### Forgetting

Financial regimes are non-stationary. Without forgetting, one hypothesis may become so dominant that the model adapts too slowly after a regime change.

Before each update, beliefs may be moved slightly toward a uniform prior:

[
P_t^{\text{diffused}}
=====================

(1-\lambda)P_t
+
\lambda U,
]

where (U) is the uniform distribution and (\lambda) is the forgetting rate.

---

## Bayesian updater

### `BayesianUpdater`

Coordinates the full inference lifecycle.

It owns:

* A collection of predictive hypotheses.
* A `BeliefState`.
* The latest predictions.
* Update diagnostics.

The daily lifecycle is:

```text
1. Each hypothesis predicts a distribution.
2. Predictions are combined using current belief probabilities.
3. The next residual return is observed.
4. Each hypothesis evaluates the observation.
5. Beliefs are updated using Bayes’ rule.
6. Each hypothesis updates its internal state.
7. Diagnostics are recorded.
```

---

## Bayesian model averaging

Each hypothesis (H_k) produces a predictive distribution with mean (\mu_k), variance (\sigma_k^2), and belief probability (w_k).

The combined predictive mean is

[
\mu
===

\sum_k w_k\mu_k.
]

The combined predictive variance is

[
\sigma^2
========

\sum_k
w_k(\sigma_k^2+\mu_k^2)
-----------------------

\mu^2.
]

This includes:

* Uncertainty within each hypothesis.
* Disagreement between hypotheses.

The second component is important. If momentum predicts a positive return while mean reversion predicts a negative return, the combined mean may be close to zero, but predictive uncertainty should remain high.

---

## Bayesian diagnostics

Each update produces diagnostics that describe prediction quality and belief change.

### Posterior entropy

[
H(P)
====

-\sum_k P(H_k)\log P(H_k).
]

Interpretation:

* High entropy: uncertainty over which hypothesis is correct.
* Low entropy: one hypothesis dominates.
* Rising entropy: confidence is breaking down.

### Negative log likelihood

For observation (x_t),

[
\mathrm{NLL}_t
==============

-\log p(x_t\mid\mathcal D_{t-1}).
]

For continuous distributions this is negative log density rather than a literal discrete-event surprise.

Interpretation:

* Low NLL: the observation was well predicted.
* High NLL: the observation was unusual under the predictive model.

Average NLL over time is an empirical estimate of predictive cross-entropy.

### Information gain

[
D_{\mathrm{KL}}
\left(
P(H\mid x_t)
\middle|
P(H)
\right).
]

Interpretation:

* Low KL divergence: the observation changed beliefs very little.
* High KL divergence: the observation strongly shifted the posterior.

Negative log likelihood and information gain measure different things:

```text
High NLL, low KL:
    Every hypothesis predicted poorly.

Moderate NLL, high KL:
    One hypothesis predicted substantially better than the others.

High NLL, high KL:
    A surprising and belief-changing observation occurred.
```

### Predictive variance

Measures uncertainty about the next residual return.

This is distinct from posterior entropy:

* Predictive variance measures uncertainty over outcomes.
* Posterior entropy measures uncertainty over models.

---

## Synthetic validation

The Bayesian system is tested using synthetic processes.

### Stable regimes

Tests verify that the model can distinguish:

* Positive-drift momentum.
* Negatively autocorrelated mean reversion.
* Independent zero-mean noise.

### Regime switching

A synthetic sequence is generated with consecutive regimes:

```text
Momentum
    ↓
Mean reversion
    ↓
Noise
```

The updater runs continuously without resetting beliefs.

Tests verify that:

* The appropriate hypothesis dominates in each segment.
* The model adapts within a bounded delay.
* Regime changes generate increased entropy or information gain.
* Posterior probabilities remain valid and normalised.

These tests establish that the Bayesian system does more than execute successfully: it can identify and adapt to changing data-generating processes.

---

## Bayesian signal model

### `BayesianSignalModel`

This is the next integration layer.

It will maintain one `BayesianUpdater` for each non-proxy ticker.

Responsibilities will include:

* Receive residual histories from `ResidualStatistics`.
* Maintain persistent beliefs for every ticker.
* Produce posterior predictive means and variances.
* Expose belief matrices.
* Expose diagnostic histories.
* Generate machine-ready signal vectors.

Conceptually:

```text
ResidualStatistics
        |
        v
Residual history for each ticker
        |
        v
One BayesianUpdater per ticker
        |
        v
Posterior predictive mean and variance
        |
        v
Signal vector
```

---

## Signal generation

The signal layer converts posterior predictive distributions into directional scores.

A simple initial signal may be

[
s_i
===

\frac{
\mathbb E[r_{i,t+1}\mid\mathcal D_t]
}{
\operatorname{Var}(r_{i,t+1}\mid\mathcal D_t)
}.
]

Possible extensions include adjusting signals for:

* Posterior entropy.
* Information gain.
* Predictive calibration.
* Residual volatility.
* Transaction costs.
* Market regime.
* Position turnover.

The Bayesian inference layer predicts returns and uncertainty. It should not directly enforce portfolio constraints.

---

## Portfolio allocation

The portfolio layer converts signals into valid Algothon positions.

Responsibilities include:

* Dollar position limits.
* Different ALGO position limits.
* Commission costs.
* Risk scaling.
* Turnover control.
* Optional market neutrality.
* Conversion from dollar exposure to integer share positions.
* Preservation of the original ticker ordering.

The final strategy should remain small:

```python
residuals = residual_model.update(prcSoFar)

predictions = bayesian_model.update(residuals)

signals = signal_generator.generate(predictions)

positions = portfolio_allocator.allocate(
    signals=signals,
    prices=prcSoFar[:, -1],
)

return positions
```

---

## Separation of concerns

| Component                | Responsibility                            |
| ------------------------ | ----------------------------------------- |
| `TickerUniverse`         | Load and organise instruments             |
| `MarketStatistics`       | Calculate observable market features      |
| `Strategyvisualiser`     | Display market analytics                  |
| `ResidualStatistics`     | Remove systematic market exposure         |
| `PredictiveDistribution` | Represent return uncertainty              |
| `PredictiveHypothesis`   | Model one return-generating mechanism     |
| `BeliefState`            | Store posterior hypothesis probabilities  |
| `BayesianUpdater`        | Coordinate prediction and belief updating |
| `BayesianSignalModel`    | Apply inference across all tickers        |
| Signal layer             | Convert predictions into scores           |
| Portfolio layer          | Convert scores into valid positions       |

---

## Design principles

### No look-ahead bias

Every prediction must use only observations available before the return being predicted.

The correct sequence is:

```text
Predict using history through t
    ↓
Observe return at t + 1
    ↓
Update beliefs and internal model state
```

### Persistent state

Bayesian beliefs and hypothesis parameters must persist between observations. Reinitialising them each day would destroy the learning process.

### Numerical stability

Bayesian updates are performed in log space using log-sum-exp-style normalisation.

### Extensibility

New hypotheses should plug into the existing updater without modifying its core logic.

Possible future hypotheses include:

* Autoregressive models.
* Volatility-regime models.
* Cluster-relative momentum.
* Factor-neutral mean reversion.
* Jump or heavy-tail processes.
* Hidden Markov regime models.

### Testability

Each layer should be independently testable:

* Distribution probability calculations.
* Hypothesis state updates.
* Bayes-rule updates.
* Model averaging.
* Stable-regime identification.
* Regime-switching adaptation.
* Portfolio constraint enforcement.

---

## Future work

Planned extensions include:

1. Integrate residual histories with one updater per ticker.
2. Add a `BayesianSignalModel`.
3. Compare Gaussian and Student-(t) predictive distributions.
4. Measure rolling predictive cross-entropy.
5. Add posterior and diagnostic visualisation.
6. Convert predictions into risk-scaled positions.
7. Backtest against the starter strategy.
8. Analyse calibration, turnover, drawdown, and score stability.
9. Introduce multi-factor residual models.
10. Investigate more formal variational or active-inference objectives only if exact Bayesian inference becomes intractable.

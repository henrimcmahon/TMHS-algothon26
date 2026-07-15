import numpy as np

nInst=51
currentPos = np.zeros(nInst)
def getMyPosition (prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:,-1] / prcSoFar[:,-2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:,-1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos

from models.ticker_universe import TickerUniverse
from visualisation.strategy_visualiser import StrategyVisualizer
from statistics.market_statistics import MarketStatistics

from matplotlib import pyplot as plt

universe = TickerUniverse("prices.txt")

# print(universe.as_dataframe())

# visualiser = StrategyVisualizer()

# visualiser.plot_normalised_tickers(
#     universe,
#     scale="log_return",
#     show_mean=True,
#     show_sigma_bands=True,
#     show_best_fit=True,
# )

stats = MarketStatistics(universe)

# ---------- Derived statistics ----------

tail_spread = stats.percentile_95 - stats.percentile_5

rolling_window = 20

rolling_breadth = np.convolve(

    stats.positive_fraction,

    np.ones(rolling_window) / rolling_window,

    mode="valid",

)

rolling_sharpe = np.convolve(

    stats.mu,

    np.ones(rolling_window) / rolling_window,

    mode="valid",

) / np.convolve(

    stats.sigma,

    np.ones(rolling_window) / rolling_window,

    mode="valid",

)

pc1_ratio = stats.rolling_pc1_ratio(window=60)

# ---------- Plot ----------

figure, axes = plt.subplots(

    3,

    2,

    figsize=(14, 8),

)

axes = axes.flatten()

# ------------------------------------------------------------------

# 1. Mean & Median

# ------------------------------------------------------------------

axes[0].plot(stats.mu, label="Mean")

axes[0].plot(stats.median, label="Median")

axes[0].axhline(0, linestyle="--", linewidth=1)

axes[0].set_title("Cross-sectional Mean & Median")

axes[0].legend()

# ------------------------------------------------------------------

# 2. Rolling Sharpe

# ------------------------------------------------------------------

axes[1].plot(

    np.arange(rolling_window - 1, len(stats.mu)),

    rolling_sharpe,

)

axes[1].axhline(0, linestyle="--", linewidth=1)

axes[1].set_title(f"{rolling_window}-Day Rolling Sharpe")

# ------------------------------------------------------------------

# 3. Breadth

# ------------------------------------------------------------------

axes[2].plot(stats.positive_fraction, alpha=0.35, label="Daily")

axes[2].plot(

    np.arange(rolling_window - 1, len(stats.positive_fraction)),

    rolling_breadth,

    linewidth=2,

    label="20-day MA",

)

axes[2].axhline(0.5, linestyle="--", linewidth=1)

axes[2].set_ylim(0, 1)

axes[2].set_title("Market Breadth")

axes[2].legend()

# ------------------------------------------------------------------

# 4. Tail Spread

# ------------------------------------------------------------------

axes[3].plot(tail_spread)

axes[3].set_title("95th − 5th Percentile Spread")

# ------------------------------------------------------------------

# 5. Skewness & Kurtosis

# ------------------------------------------------------------------

axes[4].plot(stats.skewness, label="Skewness")

axes[4].plot(stats.kurtosis, label="Excess Kurtosis")

axes[4].axhline(0, linestyle="--", linewidth=1)

axes[4].set_title("Distribution Shape")

axes[4].legend()

# ------------------------------------------------------------------

# 6. Market Structure

# ------------------------------------------------------------------

axes[5].plot(

    np.arange(60 - 1, 60 - 1 + len(pc1_ratio)),

    pc1_ratio,

)

axes[5].set_title("Rolling PC1 Explained Variance")

# ------------------------------------------------------------------

for axis in axes:

    axis.grid(alpha=0.25)

    axis.set_xlabel("Day")

figure.tight_layout()

plt.show()

universe = TickerUniverse("prices.txt")

visualiser = StrategyVisualizer()

visualiser.plot_interactive_market_explorer(
    universe,
    window=60,
    interval=100,
    beta_window=30,
)
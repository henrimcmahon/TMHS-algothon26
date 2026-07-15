from bayesian import BayesianSignalModel
from models.ticker_universe import TickerUniverse
from statistics.market_statistics import MarketStatistics
from statistics.residual_statistics import ResidualStatistics

universe = TickerUniverse("prices.txt")

market_statistics = MarketStatistics(universe)

residual_statistics = ResidualStatistics(
    market_statistics=market_statistics,
    proxy="ALGO",
)

model = BayesianSignalModel(
    residual_statistics=residual_statistics,
    residual_estimation_window=60,
)

model.warm_start(end=100)

model.as_dataframe().to_csv("data/bayesian_signal_model.csv", index=False)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

IntArray = NDArray[np.int64]

class BaselineStrategy(ABC):
    """
    Common interface for simple benchmark strategies.

    A strategy receives all prices available through the current day and
    returns the desired end-of-day position vector.
    """
    number_of_instruments: int = 51

    def __init__(self) -> None:
        self.current_positions = np.zeros(self.number_of_instruments, dtype=np.int64)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def get_positions(self, price_history: FloatArray) -> IntArray:
        """
        Return the desired total share positions for the current day.
        """

    def reset(self) -> None:
        self.current_positions = np.zeros(self.number_of_instruments, dtype=np.int64)

    @staticmethod
    def position_limits() -> FloatArray:
        limits = np.full(51, 10000.0, dtype=np.float64)
        limits[0] = 100000.0
        return limits

    @staticmethod
    def commission_rates() -> FloatArray:
        rates = np.full(51, 0.0001, dtype=np.float64)
        rates[0] = 2e-05
        return rates

    def maximum_shares(self, prices: FloatArray) -> IntArray:
        prices = np.asarray(prices, dtype=np.float64)
        if prices.shape != (51,):
            raise ValueError('prices must contain exactly 51 instruments')
        if np.any(~np.isfinite(prices)):
            raise ValueError('prices must be finite')
        if np.any(prices <= 0):
            raise ValueError('prices must be strictly positive')
        return np.floor(self.position_limits() / prices).astype(np.int64)

    def clip_positions(self, positions: IntArray, prices: FloatArray) -> IntArray:
        maximum = self.maximum_shares(prices)
        return np.clip(positions, -maximum, maximum).astype(np.int64)

DiscretisationMethod = Literal['quantile', 'volatility']

@dataclass(frozen=True)
class ReturnDiscretiser:
    """
    Convert continuous returns into three discrete states:

        -1: down
         0: neutral
         1: up

    Quantile mode gives approximately balanced state frequencies.
    Volatility mode defines neutral moves relative to each ticker's volatility.
    """
    method: DiscretisationMethod = 'quantile'
    lower_quantile: float = 1.0 / 3.0
    upper_quantile: float = 2.0 / 3.0
    volatility_threshold: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.lower_quantile < self.upper_quantile < 1.0:
            raise ValueError('Expected 0 < lower_quantile < upper_quantile < 1.')
        if self.volatility_threshold < 0.0:
            raise ValueError('volatility_threshold must be non-negative.')

    def transform(self, returns: NDArray[np.floating]) -> NDArray[np.int8]:
        """
        Parameters
        ----------
        returns:
            Matrix shaped (n_tickers, n_observations).

        Returns
        -------
        NDArray[np.int8]
            State matrix with the same shape as returns.
        """
        values = np.asarray(returns, dtype=float)
        if values.ndim != 2:
            raise ValueError('returns must have shape (n_tickers, n_observations).')
        if not np.all(np.isfinite(values)):
            raise ValueError('returns contains non-finite values.')
        if self.method == 'quantile':
            return self._quantile_transform(values)
        if self.method == 'volatility':
            return self._volatility_transform(values)
        raise ValueError(f'Unknown discretisation method: {self.method}')

    def _quantile_transform(self, returns: NDArray[np.float64]) -> NDArray[np.int8]:
        lower = np.quantile(returns, self.lower_quantile, axis=1, keepdims=True)
        upper = np.quantile(returns, self.upper_quantile, axis=1, keepdims=True)
        states = np.zeros(returns.shape, dtype=np.int8)
        states[returns < lower] = -1
        states[returns > upper] = 1
        return states

    def _volatility_transform(self, returns: NDArray[np.float64]) -> NDArray[np.int8]:
        volatility = np.std(returns, axis=1, ddof=1, keepdims=True)
        volatility = np.where(volatility > 0.0, volatility, 1.0)
        boundary = self.volatility_threshold * volatility
        states = np.zeros(returns.shape, dtype=np.int8)
        states[returns < -boundary] = -1
        states[returns > boundary] = 1
        return states

def state_probabilities(states: ArrayLike, alphabet: tuple[int, ...]=(-1, 0, 1)) -> NDArray[np.float64]:
    """Estimate the empirical probability of each state."""
    values = np.asarray(states).reshape(-1)
    if values.size == 0:
        raise ValueError('states cannot be empty.')
    counts = np.asarray([np.count_nonzero(values == state) for state in alphabet], dtype=float)
    return counts / values.size

def shannon_entropy(states: ArrayLike, *, base: float=2.0, alphabet: tuple[int, ...]=(-1, 0, 1)) -> float:
    """
    Empirical Shannon entropy.

    With base=2, the result is measured in bits.
    """
    if base <= 0.0 or np.isclose(base, 1.0):
        raise ValueError('base must be positive and different from 1.')
    probabilities = state_probabilities(states, alphabet)
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive]) / np.log(base)))

DEFAULT_ALPHABET = (-1, 0, 1)

def joint_probability_matrix(x: ArrayLike, y: ArrayLike, *, alphabet: tuple[int, ...]=DEFAULT_ALPHABET) -> NDArray[np.float64]:
    """Estimate the empirical joint distribution P(X, Y)."""
    x_values = np.asarray(x).reshape(-1)
    y_values = np.asarray(y).reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError('x and y must contain the same number of values.')
    if x_values.size == 0:
        raise ValueError('x and y cannot be empty.')
    alphabet_array = np.asarray(alphabet)
    n_states = alphabet_array.size
    x_codes = np.searchsorted(alphabet_array, x_values)
    y_codes = np.searchsorted(alphabet_array, y_values)
    x_in_bounds = x_codes < n_states
    y_in_bounds = y_codes < n_states
    x_valid = np.zeros(x_values.size, dtype=bool)
    y_valid = np.zeros(y_values.size, dtype=bool)
    x_valid[x_in_bounds] = alphabet_array[x_codes[x_in_bounds]] == x_values[x_in_bounds]
    y_valid[y_in_bounds] = alphabet_array[y_codes[y_in_bounds]] == y_values[y_in_bounds]
    valid = x_valid & y_valid
    if not np.all(valid):
        invalid_x = np.unique(x_values[~x_valid])
        invalid_y = np.unique(y_values[~y_valid])
        raise ValueError(f'All values must belong to the supplied alphabet. Invalid x values: {invalid_x}; invalid y values: {invalid_y}.')
    pair_codes = x_codes * n_states + y_codes
    counts = np.bincount(pair_codes, minlength=n_states * n_states)
    return counts.reshape(n_states, n_states) / x_values.size

def mutual_information(x: ArrayLike, y: ArrayLike, *, base: float=2.0, alphabet: tuple[int, ...]=DEFAULT_ALPHABET) -> float:
    """
    Calculate empirical mutual information I(X; Y).

    With base=2, the result is measured in bits.
    """
    if base <= 0.0 or base == 1.0:
        raise ValueError('base must be positive and different from 1.')
    joint = joint_probability_matrix(x, y, alphabet=alphabet)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    independent = np.outer(p_x, p_y)
    valid = joint > 0.0
    log_base = np.log(base)
    return float(np.sum(joint[valid] * np.log(joint[valid] / independent[valid])) / log_base)

def normalised_mutual_information(x: ArrayLike, y: ArrayLike, *, base: float=2.0, alphabet: tuple[int, ...]=DEFAULT_ALPHABET) -> float:
    """
    Symmetric normalised MI in approximately [0, 1].

    Normalisation uses:

        I(X;Y) / sqrt(H(X) H(Y))
    """
    mi = mutual_information(x, y, base=base, alphabet=alphabet)
    h_x = shannon_entropy(x, base=base, alphabet=alphabet)
    h_y = shannon_entropy(y, base=base, alphabet=alphabet)
    denominator = np.sqrt(h_x * h_y)
    if denominator <= 0.0:
        return 0.0
    return float(mi / denominator)

def mutual_information_matrix(states: NDArray[np.integer], *, lag: int=0, normalised: bool=False, base: float=2.0) -> NDArray[np.float64]:
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
        raise ValueError('states must have shape (n_tickers, n_observations).')
    if lag < 0:
        raise ValueError('lag must be non-negative.')
    if lag >= values.shape[1]:
        raise ValueError('lag must be smaller than the observation count.')
    if lag == 0:
        source_states = values
        target_states = values
    else:
        source_states = values[:, :-lag]
        target_states = values[:, lag:]
    n_tickers = values.shape[0]
    matrix = np.zeros((n_tickers, n_tickers), dtype=float)
    information_function = normalised_mutual_information if normalised else mutual_information
    for source_index in range(n_tickers):
        for target_index in range(n_tickers):
            matrix[source_index, target_index] = information_function(source_states[source_index], target_states[target_index], base=base)
    return matrix

@dataclass(frozen=True, slots=True)
class InformationMatrixKey:
    """
    Identifies one rolling information matrix.

    day_count is the number of price observations currently available.
    """
    day_count: int
    window: int
    lag: int
    method: str
    bias_correct: bool

@dataclass(frozen=True, slots=True)
class FilteredGraphKey:
    """
    Identifies one filtered graph derived from an information matrix.
    """
    matrix_key: InformationMatrixKey
    percentile: float
    top_k: int

class RollingInformationGraphCache:
    """
    Cache rolling mutual-information matrices and filtered graphs.

    The expensive MI matrix depends on:

        current day
        rolling window
        lag
        discretisation method
        bias correction

    It does not depend on the strategy's trading interpretation, so the same
    matrix can be reused by LeaderFollowerStrategy, centrality strategies,
    diffusion strategies, and parameter variants.
    """

    def __init__(self) -> None:
        self._matrix_cache: dict[InformationMatrixKey, FloatArray] = {}
        self._graph_cache: dict[FilteredGraphKey, FloatArray] = {}
        self.matrix_hits = 0
        self.matrix_misses = 0
        self.graph_hits = 0
        self.graph_misses = 0

    def get_information_matrix(self, price_history: FloatArray, *, window: int, lag: int, method: DiscretisationMethod, bias_correct: bool) -> FloatArray:
        """
        Return the rolling lagged-MI matrix for the current day.
        """
        prices = np.asarray(price_history, dtype=np.float64)
        if prices.ndim != 2:
            raise ValueError('price_history must have shape (n_tickers, n_days).')
        if window < 2:
            raise ValueError('window must be at least 2.')
        if lag < 0:
            raise ValueError('lag cannot be negative.')
        required_price_days = window + 1
        if prices.shape[1] < required_price_days:
            raise ValueError(f'At least {required_price_days} price days are required for window={window}.')
        key = InformationMatrixKey(day_count=prices.shape[1], window=window, lag=lag, method=str(method), bias_correct=bias_correct)
        cached = self._matrix_cache.get(key)
        if cached is not None:
            self.matrix_hits += 1
            return cached
        self.matrix_misses += 1
        window_prices = prices[:, -required_price_days:]
        if np.any(window_prices <= 0.0) or not np.all(np.isfinite(window_prices)):
            raise ValueError('Prices must be finite and strictly positive.')
        window_returns = np.diff(np.log(window_prices), axis=1)
        discretiser = ReturnDiscretiser(method=method)
        states = discretiser.transform(window_returns)
        matrix = np.asarray(mutual_information_matrix(states, lag=lag, normalised=False), dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)
        matrix = np.maximum(matrix, 0.0)
        if bias_correct:
            effective_n = max(states.shape[1] - lag, 1)
            bias = 4.0 / (2.0 * effective_n * np.log(2.0))
            matrix = np.maximum(matrix - bias, 0.0)
        matrix.setflags(write=False)
        self._matrix_cache[key] = matrix
        return matrix

    def get_filtered_graph(self, price_history: FloatArray, *, window: int, lag: int, method: DiscretisationMethod, percentile: float, top_k: int, bias_correct: bool) -> FloatArray:
        """
        Return a thresholded, top-k incoming information graph.
        """
        if not 0.0 <= percentile <= 100.0:
            raise ValueError('percentile must be between 0 and 100.')
        if top_k < 1:
            raise ValueError('top_k must be at least 1.')
        matrix_key = InformationMatrixKey(day_count=np.asarray(price_history).shape[1], window=window, lag=lag, method=str(method), bias_correct=bias_correct)
        graph_key = FilteredGraphKey(matrix_key=matrix_key, percentile=round(float(percentile), 8), top_k=top_k)
        cached = self._graph_cache.get(graph_key)
        if cached is not None:
            self.graph_hits += 1
            return cached
        self.graph_misses += 1
        matrix = self.get_information_matrix(price_history, window=window, lag=lag, method=method, bias_correct=bias_correct)
        positive = matrix[matrix > 0.0]
        if positive.size == 0:
            graph = np.zeros_like(matrix)
            graph.setflags(write=False)
            self._graph_cache[graph_key] = graph
            return graph
        threshold = float(np.percentile(positive, percentile))
        graph = np.zeros_like(matrix)
        for target in range(matrix.shape[1]):
            column = matrix[:, target].copy()
            column[target] = 0.0
            candidates = np.flatnonzero(column >= threshold)
            if candidates.size > top_k:
                ordering = np.argsort(column[candidates])[::-1]
                candidates = candidates[ordering[:top_k]]
            graph[candidates, target] = column[candidates]
        graph.setflags(write=False)
        self._graph_cache[graph_key] = graph
        return graph

    def clear(self) -> None:
        self._matrix_cache.clear()
        self._graph_cache.clear()
        self.matrix_hits = 0
        self.matrix_misses = 0
        self.graph_hits = 0
        self.graph_misses = 0

    @property
    def matrix_count(self) -> int:
        return len(self._matrix_cache)

    @property
    def graph_count(self) -> int:
        return len(self._graph_cache)

    def statistics(self) -> dict[str, int | float]:
        matrix_requests = self.matrix_hits + self.matrix_misses
        graph_requests = self.graph_hits + self.graph_misses
        matrix_hit_rate = self.matrix_hits / matrix_requests if matrix_requests > 0 else 0.0
        graph_hit_rate = self.graph_hits / graph_requests if graph_requests > 0 else 0.0
        return {'Matrices Cached': self.matrix_count, 'Filtered Graphs Cached': self.graph_count, 'Matrix Hits': self.matrix_hits, 'Matrix Misses': self.matrix_misses, 'Matrix Hit Rate': matrix_hit_rate, 'Graph Hits': self.graph_hits, 'Graph Misses': self.graph_misses, 'Graph Hit Rate': graph_hit_rate}

SignalVersion: TypeAlias = Literal['raw', 'tanh', 'weighted', 'latest_raw']

def _safe_log_returns(prices: FloatArray) -> FloatArray:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError('prices must have shape (n_tickers, n_days).')
    if values.shape[1] < 2:
        return np.empty((values.shape[0], 0), dtype=np.float64)
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError('prices must be finite and strictly positive.')
    return np.diff(np.log(values), axis=1)

def _cross_sectional_zscore(values: FloatArray) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(vector))
    std = float(np.std(vector))
    if std < 1e-12:
        return np.zeros_like(vector)
    return (vector - mean) / std

def _normalise_signal(signal: FloatArray) -> FloatArray:
    vector = np.asarray(signal, dtype=np.float64)
    vector = np.where(np.isfinite(vector), vector, 0.0)
    vector -= float(np.mean(vector))
    gross = float(np.sum(np.abs(vector)))
    if gross < 1e-12:
        return np.zeros_like(vector)
    return vector / gross

def _positions_from_signal(signal: FloatArray, latest_prices: FloatArray, *, gross_notional: float, max_notional_per_ticker: float) -> IntArray:
    weights = _normalise_signal(signal)
    target_notional = weights * gross_notional
    target_notional = np.clip(target_notional, -max_notional_per_ticker, max_notional_per_ticker)
    positions = np.divide(target_notional, latest_prices, out=np.zeros_like(target_notional), where=latest_prices > 0.0)
    return np.rint(positions).astype(np.int64)

@dataclass
class InformationStrategy(BaselineStrategy, ABC):
    window: int = 150
    lag: int = 1
    method: DiscretisationMethod = 'volatility'
    percentile: float = 95.0
    top_k: int = 3
    bias_correct: bool = True
    gross_notional: float = 100000.0
    max_notional_per_ticker: float = 10000.0
    minimum_days: int = 170
    rebalance_interval: int = 1
    forecast_threshold: float = 0.5
    time_decay: float = 0.6
    graph_cache: RollingInformationGraphCache | None = field(default=None, repr=False, compare=False)
    _current_positions: IntArray | None = field(default=None, init=False, repr=False)
    _last_rebalance_day: int = field(default=-1, init=False, repr=False)

    def _get_graph_cache(self) -> RollingInformationGraphCache:
        if self.graph_cache is None:
            self.graph_cache = RollingInformationGraphCache()
        return self.graph_cache

    @abstractmethod
    def compute_signal(self, graph: FloatArray, returns: FloatArray) -> FloatArray:
        raise NotImplementedError

    def get_positions(self, price_history: FloatArray) -> IntArray:
        values = np.asarray(price_history, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError('price_history must have shape (n_tickers, n_days).')
        n_tickers, n_days = values.shape
        if self.rebalance_interval < 1:
            raise ValueError('rebalance_interval must be at least 1.')
        if self._current_positions is not None and self._last_rebalance_day >= 0 and (n_days - self._last_rebalance_day < self.rebalance_interval):
            return self._current_positions.copy()
        required_days = max(self.minimum_days, self.window + self.lag + 1, self.required_signal_days)
        if n_days < required_days:
            positions = np.zeros(n_tickers, dtype=np.int64)
            self._current_positions = positions
            return positions
        returns = _safe_log_returns(values)
        cache = self._get_graph_cache()
        graph = cache.get_filtered_graph(values, window=self.window, lag=self.lag, method=self.method, percentile=self.percentile, top_k=self.top_k, bias_correct=self.bias_correct)
        signal = np.asarray(self.compute_signal(graph, returns), dtype=np.float64)
        if signal.shape != (n_tickers,):
            raise ValueError(f'compute_signal must return shape ({n_tickers},), got {signal.shape}.')
        positions = _positions_from_signal(signal, values[:, -1], gross_notional=self.gross_notional, max_notional_per_ticker=self.max_notional_per_ticker)
        self._current_positions = positions
        self._last_rebalance_day = n_days
        return positions.copy()

    @property
    def required_signal_days(self) -> int:
        return 2

    def getMyPosition(self, prcSoFar: FloatArray) -> IntArray:
        return self.get_positions(prcSoFar)

@dataclass
class LeaderFollowerStrategy(InformationStrategy):
    signal_lookback: int = 2
    self_move_penalty: float = 0.15
    normalise_incoming_weights: bool = True
    signal_version: SignalVersion = 'raw'

    @property
    def required_signal_days(self) -> int:
        return self.signal_lookback + 1

    @property
    def name(self) -> str:
        return f'LeaderFollower_V{self.signal_version}_W{self.window}_L{self.lag}_K{self.top_k}_P{self.percentile:g}_S{self.signal_lookback}_R{self.rebalance_interval}_M{self.self_move_penalty:.3f}_D{self.time_decay:.2f}_T{self.forecast_threshold:.2f}'

    def compute_signal(self, graph: FloatArray, returns: FloatArray) -> FloatArray:
        working_graph = np.asarray(graph, dtype=np.float64)
        if working_graph.ndim != 2:
            raise ValueError('graph must have shape (n_tickers, n_tickers).')
        if returns.ndim != 2:
            raise ValueError('returns must have shape (n_tickers, n_days).')
        if returns.shape[1] < self.signal_lookback:
            return np.zeros(returns.shape[0], dtype=np.float64)
        if self.normalise_incoming_weights:
            incoming_weight = np.sum(working_graph, axis=0)
            working_graph = np.divide(working_graph, incoming_weight[np.newaxis, :], out=np.zeros_like(working_graph), where=incoming_weight[np.newaxis, :] > 1e-12)
        recent_returns = returns[:, -self.signal_lookback:]
        if self.signal_version == 'raw':
            time_weights = self.time_decay ** np.arange(self.signal_lookback - 1, -1, -1, dtype=np.float64)
            time_weights /= np.sum(time_weights)
            recent_move = np.sum(recent_returns, axis=1)
        elif self.signal_version == 'latest_raw':
            recent_move = returns[:, -1]
        elif self.signal_version == 'weighted':
            time_weights = self.time_decay ** np.arange(self.signal_lookback - 1, -1, -1, dtype=np.float64)
            time_weights /= np.sum(time_weights)
            recent_move = np.sum(recent_returns * time_weights[np.newaxis, :], axis=1)
        elif self.signal_version == 'tanh':
            recent_move = np.sum(recent_returns, axis=1)
        else:
            raise ValueError(f'Unknown signal version: {self.signal_version}')
        source_move = _cross_sectional_zscore(recent_move)
        forecast = working_graph.T @ source_move
        forecast -= self.self_move_penalty * source_move
        forecast = np.where(np.isfinite(forecast), forecast, 0.0)
        if self.signal_version == 'tanh':
            forecast = np.tanh(forecast)
        return forecast

def getMyPosition(prices: np.ndarray) -> np.ndarray:
    """
    Return the desired integer share positions for all instruments.
    """
    price_history = np.asarray(prices, dtype=np.float64)
    if price_history.ndim != 2:
        raise ValueError('prices must have shape (number_of_instruments, number_of_days)')
    positions = _STRATEGY.get_positions(price_history)
    positions = np.asarray(positions, dtype=np.int64)
    expected_shape = (price_history.shape[0],)
    if positions.shape != expected_shape:
        raise ValueError(f'strategy returned shape {positions.shape}; expected {expected_shape}')
    return positions

_STRATEGY = LeaderFollowerStrategy(window=85, lag=2, top_k=7, percentile=97.0, signal_lookback=2, rebalance_interval=3, self_move_penalty=0.125, normalise_incoming_weights=True, signal_version='raw')
import numpy as np

# ------------------------------- configuration ------------------------------
WARMUP = 150
REFIT_EVERY = 50
MAX_TRAIN_PAIRS = 500
VALIDATION_PAIRS = 50

# (rank, covariance ridge, weight on regularised CCA vs ordinary ridge)
MODEL_SPECS = (
    (3, 0.10, 0.75),
    (1, 0.30, 0.75),
)
RIDGE_PENALTY = 300.0

REVERSAL_HORIZON = 60
REVERSAL_WEIGHT = 0.20
AGGRESSION = 16.0

# Aggressive submission: ALGO is either disabled or may use its full limit.
MAX_ALGO_FRACTION = 1.0

EPS = 1.0e-12

_state = {
    "n_inst": None,
    "fit_at": -1,
    "x_mean": None,
    "x_std": None,
    "models": None,
    "algo_fraction": 0.0,
}


def _reset(n_inst):
    _state["n_inst"] = n_inst
    _state["fit_at"] = -1
    _state["x_mean"] = None
    _state["x_std"] = None
    _state["models"] = None
    _state["algo_fraction"] = 0.0


def _safe_log_returns(prices):
    """Return observations by instruments, replacing invalid values safely."""
    clean = np.asarray(prices, dtype=float)
    clean = np.where(np.isfinite(clean) & (clean > 0.0), clean, np.nan)

    # Forward-fill the rare invalid entry without pandas.
    if np.isnan(clean).any():
        clean = clean.copy()
        for j in range(clean.shape[1]):
            if j == 0:
                clean[:, j] = np.where(np.isnan(clean[:, j]), 1.0, clean[:, j])
            else:
                clean[:, j] = np.where(
                    np.isnan(clean[:, j]), clean[:, j - 1], clean[:, j]
                )

    return np.diff(np.log(clean), axis=1).T


def _matrix_inv_sqrt(matrix):
    """Symmetric inverse square root and square root of a PSD matrix."""
    eigval, eigvec = np.linalg.eigh(matrix)
    eigval = np.maximum(eigval, 1.0e-8)
    inv_sqrt = eigvec @ np.diag(eigval ** -0.5) @ eigvec.T
    sqrt = eigvec @ np.diag(eigval ** 0.5) @ eigvec.T
    return inv_sqrt, sqrt


def _fit_one_model(x, y, rank, covariance_ridge, cca_weight):
    """Fit a shrinkage blend of low-rank rCCA and ridge VAR(1)."""
    n_obs, n_inst = x.shape
    eye = np.eye(n_inst)

    cov_x = (x.T @ x) / n_obs + covariance_ridge * eye
    cov_y = (y.T @ y) / n_obs + covariance_ridge * eye
    cov_xy = (x.T @ y) / n_obs

    inv_sqrt_x, _ = _matrix_inv_sqrt(cov_x)
    inv_sqrt_y, sqrt_y = _matrix_inv_sqrt(cov_y)

    whitened = inv_sqrt_x @ cov_xy @ inv_sqrt_y
    left, singular_values, right_t = np.linalg.svd(whitened, full_matrices=False)

    effective_rank = min(rank, n_inst)
    cca_model = (
        inv_sqrt_x
        @ left[:, :effective_rank]
        @ np.diag(singular_values[:effective_rank])
        @ right_t[:effective_rank]
        @ sqrt_y
    )

    ridge_model = np.linalg.solve(
        x.T @ x + RIDGE_PENALTY * eye,
        x.T @ y,
    )

    return (1.0 - cca_weight) * ridge_model + cca_weight * cca_model


def _fit_models(raw_x, raw_y):
    x_mean = raw_x.mean(axis=0)
    x_std = np.maximum(raw_x.std(axis=0), 1.0e-5)
    y_mean = raw_y.mean(axis=0)
    y_std = np.maximum(raw_y.std(axis=0), 1.0e-5)

    x = (raw_x - x_mean) / x_std
    y = (raw_y - y_mean) / y_std

    models = [
        _fit_one_model(x, y, rank, covariance_ridge, cca_weight)
        for rank, covariance_ridge, cca_weight in MODEL_SPECS
    ]
    return x_mean, x_std, models


def _scale_vector(signal):
    """Scale by dispersion across normal instruments, not by ALGO."""
    signal = np.asarray(signal, dtype=float)
    denom = signal[1:].std()
    if not np.isfinite(denom) or denom <= EPS:
        return np.zeros_like(signal)
    return signal / denom


def _scale_rows(signals):
    """Row-wise equivalent of _scale_vector for validation predictions."""
    denom = signals[:, 1:].std(axis=1, keepdims=True)
    denom = np.maximum(denom, EPS)
    return signals / denom


def _ensemble_prediction(x_standardised, models):
    predictions = []
    for model in models:
        pred = _scale_vector(x_standardised @ model)
        pred[1:] -= pred[1:].mean()  # remove common dollar-direction bias
        predictions.append(pred)
    return _scale_vector(np.mean(predictions, axis=0))


def _validation_algo_gate(raw_x, raw_y):
    """Use only past validation forecasts to decide whether ALGO is active."""
    n_obs = raw_x.shape[0]
    validation_size = min(VALIDATION_PAIRS, max(0, n_obs - 100))
    if validation_size < 20:
        return 0.0

    split = n_obs - validation_size
    val_mean, val_std, val_models = _fit_models(raw_x[:split], raw_y[:split])
    x_val = (raw_x[split:] - val_mean) / val_std

    predictions = []
    for model in val_models:
        pred = _scale_rows(x_val @ model)
        pred[:, 1:] -= pred[:, 1:].mean(axis=1, keepdims=True)
        predictions.append(pred)

    ensemble = _scale_rows(np.mean(predictions, axis=0))
    algo_dollars = np.tanh(AGGRESSION * ensemble[:, 0]) * 100_000.0
    realised_algo_returns = np.expm1(raw_y[split:, 0])
    simulated_pnl = algo_dollars * realised_algo_returns

    pnl_std = simulated_pnl.std()
    if not np.isfinite(pnl_std) or pnl_std <= EPS:
        return 0.0

    t_stat = simulated_pnl.mean() / (pnl_std / np.sqrt(simulated_pnl.size) + EPS)
    return MAX_ALGO_FRACTION if t_stat > 0.0 else 0.0


def _refit(returns):
    raw_x = returns[:-1]
    raw_y = returns[1:]

    if raw_x.shape[0] > MAX_TRAIN_PAIRS:
        raw_x = raw_x[-MAX_TRAIN_PAIRS:]
        raw_y = raw_y[-MAX_TRAIN_PAIRS:]

    algo_fraction = _validation_algo_gate(raw_x, raw_y)
    x_mean, x_std, models = _fit_models(raw_x, raw_y)

    _state["x_mean"] = x_mean
    _state["x_std"] = x_std
    _state["models"] = models
    _state["algo_fraction"] = algo_fraction


def getMyPosition(prcSoFar):
    """Return desired integer share positions for all instruments."""
    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2:
        raise ValueError("prcSoFar must be a 2-D NumPy array")

    n_inst, n_days = prices.shape
    if _state["n_inst"] != n_inst or n_days <= _state["fit_at"]:
        _reset(n_inst)

    if n_days < WARMUP or n_days <= REVERSAL_HORIZON:
        return np.zeros(n_inst, dtype=int)

    current_prices = prices[:, -1]
    if np.any(~np.isfinite(current_prices)) or np.any(current_prices <= 0.0):
        return np.zeros(n_inst, dtype=int)

    returns = _safe_log_returns(prices)
    must_refit = (
        _state["models"] is None
        or n_days - _state["fit_at"] >= REFIT_EVERY
    )

    if must_refit:
        try:
            _refit(returns)
            _state["fit_at"] = n_days
        except np.linalg.LinAlgError:
            return np.zeros(n_inst, dtype=int)

    latest_x = (returns[-1] - _state["x_mean"]) / _state["x_std"]
    base_signal = _ensemble_prediction(latest_x, _state["models"])

    # Weak, generic cross-sectional long-horizon reversal overlay.
    horizon_return = np.log(current_prices / prices[:, -1 - REVERSAL_HORIZON])
    non_algo_mean = horizon_return[1:].mean()
    non_algo_std = max(horizon_return[1:].std(), EPS)
    reversal_signal = -(horizon_return - non_algo_mean) / non_algo_std
    reversal_signal[0] = 0.0

    final_signal = _scale_vector(base_signal + REVERSAL_WEIGHT * reversal_signal)
    strength = np.tanh(AGGRESSION * final_signal)
    strength[0] *= _state["algo_fraction"]

    dollar_limits = np.full(n_inst, 10_000.0)
    dollar_limits[0] = 100_000.0
    target_dollars = dollar_limits * strength

    # Truncation gives exact integers and remains inside the evaluator's limits.
    positions = np.trunc(target_dollars / current_prices).astype(int)
    return positions
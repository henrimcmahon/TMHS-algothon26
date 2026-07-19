import numpy as np
 
LAM        = 10.0
REFIT      = 10
SCALE      = 1e6
NAME_CAP   = 9_500.0
ALGO_CAP   = 95_000.0
VOL_WIN    = 60
BUFFER     = 0.20
WARMUP     = 80
 
_state = {"pos": None, "W": None, "fit_at": -1}
 
 
def _reset(n):
    _state["pos"] = np.zeros(n)
    _state["W"] = None
    _state["fit_at"] = -1
 
 
def getMyPosition(prcSoFar):
    prc = np.asarray(prcSoFar, dtype=float)
    nInst, nt = prc.shape
    if _state["pos"] is None or _state["pos"].shape[0] != nInst or nt < _state["fit_at"]:
        _reset(nInst)
    if nt < WARMUP:
        return _state["pos"].astype(int)
 
    rets = np.diff(np.log(prc), axis=1)          # (nInst, nt-1)
 
    # ---- refit ridge VAR(1) on all history every REFIT days ----
    if _state["W"] is None or nt - _state["fit_at"] >= REFIT:
        X = rets[:, :-1].T                        # predictors: ret day k
        Y = rets[:, 1:].T                         # targets:    ret day k+1
        _state["W"] = np.linalg.solve(X.T @ X + LAM * np.eye(nInst), X.T @ Y)
        _state["fit_at"] = nt
 
    pred = rets[:, -1] @ _state["W"]              # forecast of tomorrow's returns
 
    vol  = np.maximum(rets[:, -VOL_WIN:].std(axis=1), 1e-4)
    caps = np.full(nInst, NAME_CAP)
    caps[0] = ALGO_CAP
    dollars = np.clip(SCALE * pred / vol, -caps, caps)
 
    last   = prc[:, -1]
    shares = dollars / last
 
    # ---- turnover buffer ----
    prev = _state["pos"]
    diff = shares - prev
    tol  = BUFFER * np.maximum(np.abs(shares), 1.0)
    keep = np.abs(diff) <= tol
    shares[keep] = prev[keep]
 
    # ---- hard clip at true dollar limits (buffer can hold stale shares) ----
    lim = np.full(nInst, 10_000.0)
    lim[0] = 100_000.0
    shares = np.clip(shares, -lim / last, lim / last)
 
    _state["pos"] = shares
    return shares.astype(int)
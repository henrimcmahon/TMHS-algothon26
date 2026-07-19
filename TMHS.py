#!/usr/bin/env python3
"""Cross-asset ensemble strategy for Algothon 2026."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression

WARMUP = 150
REFIT_EVERY = 50
MAX_TRAIN_PAIRS = 650
RIDGE_ALPHA = 300.0
RRR_ALPHA = 10.0
RRR_RANK = 8
PLS_COMPONENTS = 8
W_RIDGE = 0.80
W_PLS = 0.00
W_RRR = 0.20
FACTOR_COUNT = 0
FACTOR_SHRINK = 0.0
NEUTRALISATION = 1.0
AGGRESSION = 64.0
ALGO_RISK = 0.0
EPS = 1e-12

_state = {"n": None, "fit_at": -1, "xm": None, "xs": None,
          "ridge": None, "rrr": None, "pls": None, "factors": None}

def _reset(n):
    _state.update({"n":n,"fit_at":-1,"xm":None,"xs":None,
                   "ridge":None,"rrr":None,"pls":None,"factors":None})

def _returns(prices):
    p=np.asarray(prices,float)
    p=np.where(np.isfinite(p)&(p>0),p,np.nan)
    with np.errstate(divide='ignore',invalid='ignore'):
        r=np.diff(np.log(p),axis=1).T
    return np.nan_to_num(r,nan=0.,posinf=0.,neginf=0.)

def _fit(r):
    x=r[:-1]; y=r[1:]
    if len(x)>MAX_TRAIN_PAIRS: x=x[-MAX_TRAIN_PAIRS:];y=y[-MAX_TRAIN_PAIRS:]
    xm=x.mean(0); xs=np.maximum(x.std(0),1e-5)
    ym=y.mean(0); ys=np.maximum(y.std(0),1e-5)
    xn=(x-xm)/xs; yn=(y-ym)/ys
    gram=xn.T@xn; rhs=xn.T@yn
    ridge=np.linalg.solve(gram+RIDGE_ALPHA*np.eye(xn.shape[1]),rhs)
    b=np.linalg.solve(gram+RRR_ALPHA*np.eye(xn.shape[1]),rhs)
    yh=xn@b
    try:
        _,_,vt=np.linalg.svd(yh,full_matrices=False)
        v=vt[:min(RRR_RANK,vt.shape[0])].T
        rrr=b@v@v.T
    except np.linalg.LinAlgError:
        rrr=ridge.copy()
    nc=min(PLS_COMPONENTS,xn.shape[1],yn.shape[1],len(xn)-1)
    pls=None
    if nc>=1:
        try:
            pls=PLSRegression(n_components=nc,scale=False,max_iter=500,tol=1e-6)
            pls.fit(xn,yn)
        except Exception:
            pls=None
    factors=None
    if FACTOR_COUNT>0 and FACTOR_SHRINK>0:
        c=np.cov(xn,rowvar=False)
        try:
            vals,vec=np.linalg.eigh(c)
            factors=vec[:,np.argsort(vals)[::-1][:FACTOR_COUNT]]
        except np.linalg.LinAlgError:
            factors=None
    _state.update({"xm":xm,"xs":xs,"ridge":ridge,"rrr":rrr,"pls":pls,"factors":factors})

def _scale(z):
    z=np.asarray(z,float).copy()
    sd=z[1:].std()
    if not np.isfinite(sd) or sd<EPS:return np.zeros_like(z)
    return z/sd

def getMyPosition(prcSoFar):
    p=np.asarray(prcSoFar,float)
    if p.ndim!=2: raise ValueError('prcSoFar must be 2-D')
    n,nt=p.shape
    if _state['n']!=n or nt<=_state['fit_at']:_reset(n)
    out=np.zeros(n,dtype=int)
    cur=p[:,-1];valid=np.isfinite(cur)&(cur>0)
    if nt<WARMUP:return out
    r=_returns(p)
    if len(r)<3:return out
    if _state['ridge'] is None or nt-_state['fit_at']>=REFIT_EVERY:
        _fit(r);_state['fit_at']=nt
    x=(r[-1]-_state['xm'])/_state['xs']
    a=_scale(x@_state['ridge'])
    c=_scale(x@_state['rrr'])
    if _state['pls'] is None:b=a.copy()
    else:b=_scale(np.asarray(_state['pls'].predict(x.reshape(1,-1))).reshape(-1))
    s=W_RIDGE*a+W_PLS*b+W_RRR*c
    f=_state['factors']
    if f is not None:s=s-FACTOR_SHRINK*((s@f)@f.T)
    s=_scale(s)
    s[1:]-=NEUTRALISATION*s[1:].mean()
    s[0]*=ALGO_RISK
    strength=np.tanh(AGGRESSION*s)
    lim=np.full(n,10000.);lim[0]=100000.
    target=lim*strength
    # ALGO_RISK has already been applied inside tanh; do not apply twice.
    out[valid]=np.trunc(target[valid]/cur[valid]).astype(int)
    return out
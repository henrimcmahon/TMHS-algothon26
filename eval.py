#!/usr/bin/env python3
import numpy as np
import pandas as pd
from TMHS import getMyPosition as getPosition

pricesFile = './prices.txt'
numTestDays = 500
scoreDefaultParam = 1.0
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

def loadPrices(fn):
    df = pd.read_csv(fn, sep=r'\s+', header=0, index_col=None)
    nt, nInst = df.shape
    return df.values.T, nt, nInst

def score(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10:return mu
    sr=np.sqrt(250)*mu/sigma
    return mu*sr**2/(sr**2+param**2)

prcAll,nt,nInst=loadPrices(pricesFile)
commRate=np.full(nInst,defaultCommRate);commRate[0]=inst0CommRate
dlrPosLimit=np.full(nInst,defaultDlrPosLimit);dlrPosLimit[0]=inst0DlrPosLimit
cash=0.;curPos=np.zeros(nInst);totDVolume=0.;value=0.;comm=0.;todayPLL=[]
startDay=nt-numTestDays
for t in range(startDay,nt+1):
    hist=prcAll[:,:t];curPrices=hist[:,-1]
    if t<nt:
        orig=getPosition(hist)
        lim=(dlrPosLimit/curPrices).astype(int)
        newPos=np.clip(orig,-lim,lim).astype(int)
    else:newPos=np.array(curPos)
    delta=newPos-curPos
    cash-=curPrices.dot(delta)+comm
    dvolumes=curPrices*np.abs(delta);totDVolume+=dvolumes.sum();comm=np.sum(dvolumes*commRate)
    curPos=np.array(newPos);posValue=curPos.dot(curPrices)
    todayPL=cash+posValue-value;value=cash+posValue
    if t>startDay:todayPLL.append(todayPL)
pll=np.array(todayPLL);mu=pll.mean();sd=pll.std();sr=np.sqrt(250)*mu/sd
print(f'days={len(pll)} instruments={nInst}')
print(f'mean(PL): {mu:.6f}')
print(f'StdDev(PL): {sd:.6f}')
print(f'annSharpe(PL): {sr:.6f}')
print(f'totDvolume: {totDVolume:.2f}')
print(f'Score: {score(mu,sd):.6f}')
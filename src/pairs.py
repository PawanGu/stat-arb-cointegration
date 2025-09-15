"""
Pair/basket selection: correlation screening, Engle–Granger cointegration, Johansen (optional).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict, Any

def corr_screen(returns: pd.DataFrame, min_corr: float=0.6) -> List[Tuple[str,str,float]]:
    out = []
    cols = returns.columns
    for a,b in combinations(cols, 2):
        rho = returns[a].corr(returns[b])
        if np.isfinite(rho) and abs(rho) >= min_corr:
            out.append((a,b,rho))
    return sorted(out, key=lambda x: -abs(x[2]))

def hedge_ratio(logP_a: pd.Series, logP_b: pd.Series) -> float:
    """OLS hedge ratio from log prices: logP_a ~ alpha + beta*logP_b"""
    X = np.vstack([np.ones(len(logP_b)), logP_b.values]).T
    beta = np.linalg.lstsq(X, logP_a.values, rcond=None)[0][1]
    return float(beta)

def engle_granger(logP_a: pd.Series, logP_b: pd.Series, adf_alpha: float=0.05) -> Dict[str, Any]:
    """
    Engle–Granger 2-step: OLS -> residual -> ADF unit-root test
    Returns dict with beta, adf_pvalue, stationary(bool).
    """
    import statsmodels.api as sm
    X = sm.add_constant(logP_b.values)
    res = sm.OLS(logP_a.values, X).fit()
    resid = logP_a.values - res.predict(X)
    # ADF on residuals
    from statsmodels.tsa.stattools import adfuller
    adf_stat, pval, *_ = adfuller(resid, regression='c')  # include constant
    return {"beta": float(res.params[1]), "alpha": float(res.params[0]), "adf_pvalue": float(pval),
            "stationary": pval < adf_alpha, "residual": pd.Series(resid, index=logP_a.index)}

def johansen_df(log_prices: pd.DataFrame, det_order: int=0, k_ar_diff: int=1) -> Dict[str, Any]:
    """Optional Johansen test for small baskets. Returns eigenvalues and coint rank."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    res = coint_johansen(log_prices.dropna(), det_order, k_ar_diff)
    rank = (res.lr1 > res.cvt[:, 1]).sum()  # trace test vs 5% crit
    return {"eig": res.eig, "rank": int(rank)}

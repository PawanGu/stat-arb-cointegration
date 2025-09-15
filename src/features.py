"""
Feature engineering: returns, residuals, z-scores, OU half-life.
"""
import pandas as pd
import numpy as np

def log_prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()

def rolling_stats(series: pd.Series, window: int) -> pd.DataFrame:
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std(ddof=1)
    return pd.DataFrame({"mu": mu, "sigma": sigma})

def zscore(series: pd.Series, window: int) -> pd.Series:
    rs = rolling_stats(series, window)
    z = (series - rs["mu"]) / rs["sigma"]
    return z

def ou_halflife(series: pd.Series) -> float:
    """Estimate OU half-life via AR(1) on differences: S_t = a + b S_{t-1} + e_t"""
    s = series.dropna()
    y = s[1:]
    x = s.shift(1)[1:]
    X = np.vstack([np.ones(len(x)), x.values]).T
    beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
    b = beta[1]
    if b >= 1:
        return np.inf
    hl = -np.log(2)/np.log(b) if b > 0 else np.nan
    return float(hl)

"""
Simple risk utilities: exposure caps, volatility targeting, stop rules.
"""
import pandas as pd
import numpy as np

def vol_target_weights(series: pd.Series, target_vol: float=0.1, lookback: int=20, cap: float=3.0) -> pd.Series:
    vol = series.pct_change().rolling(lookback).std() * np.sqrt(252)
    w = target_vol / vol
    return w.clip(upper=cap).fillna(0.0)

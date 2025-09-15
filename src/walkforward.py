"""
Walk-forward training/testing for cointegration parameters.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from .pairs import engle_granger
from .features import zscore

def rolling_windows(index: pd.DatetimeIndex, train_days: int, test_days: int):
    start = 0
    n = len(index)
    while True:
        train_end = start + train_days
        test_end = train_end + test_days
        if test_end >= n:
            break
        yield (index[start:train_end], index[train_end:test_end])
        start += test_days

def walkforward_backtest(prices: pd.DataFrame,
                         A: str, B: str,
                         train_days: int=252*2,
                         test_days: int=63,
                         roll_window: int=60,
                         z_entry: float=2.0, z_exit: float=0.5, z_stop: float=4.0,
                         **bt_kwargs) -> pd.DataFrame:
    logP = np.log(prices[[A,B]]).dropna()
    results = []
    for train_idx, test_idx in rolling_windows(logP.index, train_days, test_days):
        train = logP.loc[train_idx]
        test = logP.loc[test_idx]
        eg = engle_granger(train[A], train[B])
        res_train = eg["residual"]
        # Build z on train to set context; apply to test using same alpha/beta
        alpha, beta = eg["alpha"], eg["beta"]
        resid_test = test[A] - (alpha + beta*test[B])
        z_test = zscore(resid_test, roll_window)
        from .backtest import pair_backtest
        bt = pair_backtest(prices.loc[test_idx, [A,B]], (A,B), beta, resid_test, z_test,
                           z_entry=z_entry, z_exit=z_exit, z_stop=z_stop, **bt_kwargs)
        df = bt["trades"]
        df["period"] = f"{train_idx[0].date()}_{test_idx[-1].date()}"
        results.append(df)
    out = pd.concat(results).sort_index()
    return out

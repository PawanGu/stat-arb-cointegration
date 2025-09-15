"""
Evaluation metrics & plotting helpers.
"""
import pandas as pd
import numpy as np

def annualize_ret(series: pd.Series, freq: int=252):
    return series.mean()*freq

def annualize_vol(series: pd.Series, freq: int=252):
    return series.std(ddof=1)*np.sqrt(freq)

def sharpe(series: pd.Series, rf: float=0.0, freq: int=252):
    ex = series - rf/freq
    vol = annualize_vol(ex, freq)
    return np.nan if vol == 0 else annualize_ret(ex, freq) / vol

def max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    dd = equity - roll_max
    mdd = dd.min()
    return float(mdd), dd

def summarize(pnl_net: pd.Series) -> dict:
    eq = pnl_net.cumsum()
    mdd, _ = max_drawdown(eq)
    return {
        "ann_return": annualize_ret(pnl_net),
        "ann_vol": annualize_vol(pnl_net),
        "sharpe": sharpe(pnl_net),
        "max_drawdown": mdd,
        "avg_daily_pnl": pnl_net.mean(),
        "hit_rate": (pnl_net > 0).mean()
    }

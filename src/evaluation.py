"""
Evaluation metrics & plotting helpers.
"""
import pandas as pd
import numpy as np

def pnl_to_returns(pnl: pd.Series, initial_capital: float, equity: pd.Series | None = None) -> pd.Series:
    """
    Convert daily PnL ($) to daily returns.
    If equity is provided, use rolling equity for divisor; else use constant initial_capital.
    """
    if equity is not None:
        # avoid divide-by-zero
        denom = equity.shift(1).replace(0, float('nan')).fillna(method="bfill")
        return pnl / denom
    else:
        return pnl / float(initial_capital)

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

def summarize(pnl_net: pd.Series, initial_capital: float = 1_000_000, use_equity: bool = True) -> dict:
    eq = pnl_net.cumsum() + initial_capital
    rets = pnl_to_returns(pnl_net, initial_capital, equity=eq if use_equity else None)
    mdd, _ = max_drawdown(eq - initial_capital)  # drawdown in $
    return {
        "ann_return": annualize_ret(rets),
        "ann_vol": annualize_vol(rets),
        "sharpe": sharpe(rets),
        "max_drawdown": mdd,
        "avg_daily_return": rets.mean(),
        "hit_rate": (pnl_net > 0).mean()
    }

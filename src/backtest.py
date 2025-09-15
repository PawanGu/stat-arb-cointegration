"""
Vectorized backtester for pairs mean-reversion using z-score of cointegration residuals.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple

def pair_backtest(prices: pd.DataFrame,
                  pair: Tuple[str,str],
                  beta: float,
                  residual: pd.Series,
                  z: pd.Series,
                  z_entry: float=2.0,
                  z_exit: float=0.5,
                  z_stop: float=4.0,
                  per_trade_notional: float=50_000,
                  commission_per_share: float=0.0005,
                  spread_bps: float=5,
                  slippage_bps: float=2,
                  dollar_neutral: bool=True) -> Dict[str, pd.DataFrame]:
    """
    Simplified daily-close backtest.
    Long spread (z<-entry): long A, short beta*B
    Short spread (z> entry): short A, long beta*B
    """
    A, B = pair
    pxA, pxB = prices[A], prices[B]

    # Signals: +1 short spread, -1 long spread, 0 flat
    sig = pd.Series(0, index=z.index, dtype=float)
    sig[z >= z_entry] = 1.0
    sig[z <= -z_entry] = -1.0
    # exit to 0 when |z| <= z_exit
    sig[(z.abs() <= z_exit)] = 0.0
    # stop out to flat when |z| > z_stop
    sig[(z.abs() > z_stop)] = 0.0

    # Position is last non-zero signal held until exit condition triggers
    pos = sig.replace(0, np.nan).ffill().fillna(0.0)
    # Clear position when exit condition met
    pos[(z.abs() <= z_exit) | (z.abs() > z_stop)] = 0.0
    pos = pos.shift(1).fillna(0.0)  # enter next day open ~ modeled as next close

    # Notional allocation
    N = per_trade_notional
    if dollar_neutral:
        qtyA = (N / 2) / pxA
        qtyB = (N / 2) / pxB * beta
    else:
        qtyA = (N) / pxA
        qtyB = (N*beta) / pxB

    # Returns from legs
    retA = pxA.pct_change().fillna(0.0)
    retB = pxB.pct_change().fillna(0.0)

    # PnL: sign convention: pos=+1 means short spread: short A, long beta*B
    pnl = pos * ((-qtyA*pxA.shift(1)*retA) + (qtyB*pxB.shift(1)*retB))
    # Costs when position changes (turnover)
    turns = pos.diff().abs().fillna(0.0)
    # Approx cost in $
    cost_perc = (spread_bps + slippage_bps) / 1e4
    costs = turns * ( (qtyA*pxA*cost_perc) + (qtyB*pxB*cost_perc) ) + turns*(qtyA+qtyB)*commission_per_share

    pnl_net = pnl - costs
    eq = pnl_net.cumsum()
    out = pd.DataFrame({
        "pos": pos,
        "pnl_gross": pnl,
        "costs": -costs,
        "pnl_net": pnl_net,
        "equity": eq
    })
    return {"trades": out}

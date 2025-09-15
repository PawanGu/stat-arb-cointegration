"""
Vectorized backtester for pairs mean-reversion using z-score of cointegration residuals.
Improved signal logic with crossover entries and explicit exits including time stop.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def pair_backtest(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
    beta: float,
    residual: pd.Series,
    z: pd.Series,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    per_trade_notional: float = 50_000,
    commission_per_share: float = 0.0005,
    spread_bps: float = 5,
    slippage_bps: float = 2,
    dollar_neutral: bool = True,
    beta_neutral: bool = False,          # reserved for future use (market beta neutrality)
    time_stop_days: Optional[int] = 20,  # None to disable
) -> Dict[str, pd.DataFrame]:
    """
    Daily-close backtest for a single pair using z-score of cointegration residuals.

    Trading logic (spread S = A - beta*B):
      - If z crosses up above +z_entry: enter SHORT spread  (pos = +1) => short A, long beta*B
      - If z crosses down below -z_entry: enter LONG spread (pos = -1) => long A, short beta*B
      - Exit to flat when |z| <= z_exit, or |z| > z_stop, or after 'time_stop_days' bars in position.

    Sizing:
      - Dollar-neutral by default: split notional equally across legs; hedge with 'beta' on the B leg.
      - Per-leg shares are computed at each bar from current price (static notional sizing).

    Costs:
      - Per-share commission + proportional spread/slippage (bps) charged on position changes (turnover).

    Returns:
      dict with a single key "trades" -> DataFrame with columns:
        pos, pnl_gross, costs, pnl_net, equity
    """
    A, B = pair
    # Align all series to common index
    idx = prices.index.intersection(z.index)
    pxA = prices[A].reindex(idx)
    pxB = prices[B].reindex(idx)
    z = z.reindex(idx)

    # --- Build crossover-based entries and exits ---
    # Cross up into short-spread region:
    enter_short = (z.shift(1) < z_entry) & (z >= z_entry)
    # Cross down into long-spread region:
    enter_long = (z.shift(1) > -z_entry) & (z <= -z_entry)
    # Exit conditions (flat):
    exit_any = (z.abs() <= z_exit) | (z.abs() > z_stop)

    # --- Simulate position path with time stop (loop for clarity & correctness) ---
    pos_vals = []
    pos = 0.0
    days_in_pos = 0
    z_prev = z.shift(1)

    for t in idx:
        z_t = z.loc[t]
        exit_t = bool(exit_any.loc[t])

        # Evaluate entries only if currently flat
        if pos == 0.0:
            if bool(enter_short.loc[t]):
                pos = 1.0
                days_in_pos = 0
            elif bool(enter_long.loc[t]):
                pos = -1.0
                days_in_pos = 0
            else:
                pos = 0.0
                days_in_pos = 0
        else:
            # We are in a position: check exit/stop/time-stop
            trigger_time_stop = (time_stop_days is not None) and (days_in_pos + 1 >= time_stop_days)
            if exit_t or trigger_time_stop:
                pos = 0.0
                days_in_pos = 0
            else:
                # Continue holding
                days_in_pos += 1

        pos_vals.append(pos)

    pos_series = pd.Series(pos_vals, index=idx, dtype=float)
    # Positions become active on the next bar (can't trade on the same close you measured on)
    pos_series = pos_series.shift(1).fillna(0.0)

    # --- Notional allocation (static notional per trade) ---
    N = float(per_trade_notional)
    if dollar_neutral:
        # split notional; hedge ratio applied on B leg
        qtyA = (N / 2.0) / pxA
        qtyB = (N / 2.0) / pxB * beta
    else:
        # directional scaling (rarely used in pairs; kept for compatibility)
        qtyA = (N) / pxA
        qtyB = (N * beta) / pxB

    # --- Leg returns ---
    retA = pxA.pct_change().fillna(0.0)
    retB = pxB.pct_change().fillna(0.0)

    # PnL: pos=+1 (short spread): short A, long beta*B
    # leg notionals at t-1 applied to returns into t
    pnl_legA = (-qtyA * pxA.shift(1) * retA) * pos_series
    pnl_legB = (+qtyB * pxB.shift(1) * retB) * pos_series
    pnl = (pnl_legA + pnl_legB).fillna(0.0)

    # --- Transaction costs on position changes (turnover) ---
    turns = pos_series.diff().abs().fillna(0.0)  # 0->1 or 0->-1 -> 1; 1->0 -> 1; 1->-1 -> 2, etc.
    cost_perc = (float(spread_bps) + float(slippage_bps)) / 1e4
    # Approximate one-shot cost using current prices
    leg_cost_A = (qtyA * pxA * cost_perc)
    leg_cost_B = (qtyB * pxB * cost_perc)
    comm_cost = (qtyA.abs() + qtyB.abs()) * float(commission_per_share)

    costs = (turns * (leg_cost_A + leg_cost_B) + turns * comm_cost).fillna(0.0)
    # Costs reduce PnL
    pnl_net = pnl - costs

    equity = pnl_net.cumsum()

    out = pd.DataFrame(
        {
            "pos": pos_series,
            "pnl_gross": pnl,
            "costs": -costs,  # negative cash flow
            "pnl_net": pnl_net,
            "equity": equity,
        }
    )
    return {"trades": out}

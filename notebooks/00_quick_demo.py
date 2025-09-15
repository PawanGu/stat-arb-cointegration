"""
Demo: quick run from CLI (for illustration). Replace tickers/universe as needed.
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import yaml, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.data import load_prices
from src.features import zscore
from src.pairs import engle_granger, corr_screen
from src.backtest import pair_backtest
from src.evaluation import summarize
from src.walkforward import walkforward_backtest
from src.plotting import plot_equity, plot_drawdown

cfg = yaml.safe_load(Path("config.yaml").read_text())

tickers = cfg["universe"]["tickers"]
prices = load_prices(
    tickers,
    cfg["data"]["start"],
    cfg["data"]["end"],
    price_field=cfg["data"]["price_field"],
)

returns = np.log(prices).diff().dropna()
pairs = corr_screen(returns, min_corr=cfg["screening"]["min_corr"])
print("Top screened pairs:", pairs[:5])

if pairs:
    A, B, _ = pairs[0]
    logP = np.log(prices[[A, B]]).dropna()
    eg = engle_granger(logP[A], logP[B], cfg["cointegration"]["adf_alpha"])
    resid = eg["residual"]
    z = zscore(resid, cfg["signal"]["rolling_window"])

    bt = pair_backtest(
        prices[[A, B]], (A, B), eg["beta"], resid, z,
        z_entry=cfg["signal"]["z_entry"],
        z_exit=cfg["signal"]["z_exit"],
        z_stop=cfg["signal"]["z_stop"],
        per_trade_notional=cfg["backtest"]["per_trade_notional"],
        commission_per_share=cfg["costs"]["commission_per_share"],
        spread_bps=cfg["costs"]["spread_bps"],
        slippage_bps=cfg["costs"]["slippage_bps"],
        dollar_neutral=cfg["backtest"]["dollar_neutral"]
    )

    summ = summarize(
        bt["trades"]["pnl_net"].fillna(0.0),
        initial_capital=cfg["backtest"]["initial_capital"]
    )
    print("Summary:", summ)

    # Save plots
    figs_dir = Path("paper/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)

    eq = bt["trades"]["pnl_net"].cumsum()

    fig1 = plot_equity(eq, title=f"Equity {A}-{B}")
    fig1.savefig(figs_dir / "equity_curve.png", dpi=160)
    plt.close(fig1)

    fig2 = plot_drawdown(eq, title=f"Drawdown {A}-{B}")
    fig2.savefig(figs_dir / "drawdown.png", dpi=160)
    plt.close(fig2)

    print(f"Saved plots to {figs_dir}")

    # Walk-forward example (short)
    wf = walkforward_backtest(
        prices[[A, B]], A, B,
        train_days=252, test_days=63,
        roll_window=cfg["signal"]["rolling_window"],
        z_entry=cfg["signal"]["z_entry"],
        z_exit=cfg["signal"]["z_exit"],
        z_stop=cfg["signal"]["z_stop"],
        per_trade_notional=cfg["backtest"]["per_trade_notional"],
        commission_per_share=cfg["costs"]["commission_per_share"],
        spread_bps=cfg["costs"]["spread_bps"],
        slippage_bps=cfg["costs"]["slippage_bps"],
        dollar_neutral=cfg["backtest"]["dollar_neutral"]
    )
    print("WF trades shape:", wf.shape)

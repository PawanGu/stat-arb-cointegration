"""
01_pair_analysis.py

Select candidate pairs, run Engle–Granger + mean-reversion backtests,
and save equity/drawdown plots + results CSV into paper/ for the report.
Also runs walk-forward validation on the best pair.
"""

import os, sys, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Ensure project root on path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_prices
from src.features import zscore
from src.pairs import corr_screen, engle_granger
from src.backtest import pair_backtest
from src.evaluation import summarize as eval_summarize
from src.plotting import plot_equity, plot_drawdown
from src.walkforward import walkforward_backtest


def safe_summarize(pnl_net: pd.Series, initial_capital: float) -> dict:
    """Use project evaluation.summarize; fallback to inline metrics if needed."""
    try:
        return eval_summarize(pnl_net.fillna(0.0), initial_capital=initial_capital)
    except TypeError:
        eq = pnl_net.fillna(0.0).cumsum() + initial_capital
        rets = pnl_net.fillna(0.0) / eq.shift(1).replace(0, np.nan).bfill()
        ann_ret = rets.mean() * 252
        ann_vol = rets.std(ddof=1) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        dd = (eq - eq.cummax()).min()
        return {
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(dd),
            "avg_daily_return": float(rets.mean()),
            "hit_rate": float((pnl_net > 0).mean()),
        }


def main():
    # --- Load config ---
    CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
    FIGDIR = ROOT / "paper" / "figs"
    FIGDIR.mkdir(parents=True, exist_ok=True)

    tickers = CFG["universe"]["tickers"]
    start = CFG["data"]["start"]
    end = CFG["data"]["end"]
    pfield = CFG["data"]["price_field"]

    min_corr = CFG["screening"]["min_corr"]
    lookback = CFG["screening"]["lookback_days"]

    adf_alpha = CFG["cointegration"]["adf_alpha"]
    roll_win = CFG["signal"]["rolling_window"]
    z_entry = CFG["signal"]["z_entry"]
    z_exit = CFG["signal"]["z_exit"]
    z_stop = CFG["signal"]["z_stop"]

    init_cap = CFG["backtest"]["initial_capital"]
    per_notional = CFG["backtest"]["per_trade_notional"]
    dollar_neutral = CFG["backtest"]["dollar_neutral"]

    comm = CFG["costs"]["commission_per_share"]
    spread_bps = CFG["costs"]["spread_bps"]
    slip_bps = CFG["costs"]["slippage_bps"]

    time_stop = CFG["risk"]["time_stop_days"]

    # --- Load Prices ---
    print(f"Loading prices for {tickers} ...")
    prices = load_prices(tickers, start, end, price_field=pfield)

    # --- Screen pairs ---
    logret = np.log(prices).diff().dropna()
    pairs = corr_screen(logret.tail(lookback), min_corr=min_corr)
    print(f"Top screened pairs: {pairs[:10]}")

    if not pairs:
        print("No pairs passed the correlation screen.")
        return

    # --- Backtest top pairs ---
    results = []
    TOP_N = min(10, len(pairs))   # evaluate top 10 pairs

    for (A, B, rho) in pairs[:TOP_N]:
        logP = np.log(prices[[A, B]]).dropna()
        eg = engle_granger(logP[A], logP[B], adf_alpha=adf_alpha)

        # Skip if residual not stationary
        if not eg.get("stationary", False):
            print(f"Skip {A}-{B}: residual not stationary (ADF p={eg['adf_pvalue']:.3g})")
            continue

        resid = eg["residual"]
        z = zscore(resid, roll_win)

        bt = pair_backtest(
            prices[[A, B]],
            (A, B),
            eg["beta"],
            resid,
            z,
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=z_stop,
            per_trade_notional=per_notional,
            commission_per_share=comm,
            spread_bps=spread_bps,
            slippage_bps=slip_bps,
            dollar_neutral=dollar_neutral,
            time_stop_days=time_stop,
        )
        trades = bt["trades"]
        summ = safe_summarize(trades["pnl_net"], initial_capital=init_cap)
        summ.update({"pair": f"{A}-{B}", "rho": float(rho)})
        results.append(summ)

        # Save per-pair plots
        eq = trades["pnl_net"].cumsum()
        fig1 = plot_equity(eq, title=f"Equity {A}-{B}")
        fig1.savefig(FIGDIR / f"equity_{A}_{B}.png", dpi=160)
        plt.close(fig1)

        fig2 = plot_drawdown(eq, title=f"Drawdown {A}-{B}")
        fig2.savefig(FIGDIR / f"drawdown_{A}_{B}.png", dpi=160)
        plt.close(fig2)

    if not results:
        print("No cointegrated pairs produced results.")
        return

    dfres = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # Round for cleaner LaTeX table
    dfres = dfres.round({
        "rho": 3,
        "ann_return": 4,
        "ann_vol": 4,
        "sharpe": 2,
        "max_drawdown": 2,
        "hit_rate": 2
    })

    print("Results:")
    print(dfres)

    # --- Save results CSV for LaTeX ---
    out_csv = ROOT / "paper" / "results_pair_analysis.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dfres.to_csv(out_csv, index=False)
    print(f"Saved summary CSV -> {out_csv}")

    # --- Walk-forward backtest on best Sharpe pair ---
    best_pair = dfres.iloc[0]["pair"]
    A, B = best_pair.split("-")
    print(f"Running walk-forward on {best_pair} ...")

    wf_trades = walkforward_backtest(
        prices[[A, B]],
        A, B,
        train_days=CFG["walkforward"]["train_days"],
        test_days=CFG["walkforward"]["test_days"],
        roll_window=CFG["signal"]["rolling_window"],
        z_entry=CFG["signal"]["z_entry"],
        z_exit=CFG["signal"]["z_exit"],
        z_stop=CFG["signal"]["z_stop"],
        per_trade_notional=CFG["backtest"]["per_trade_notional"],
        commission_per_share=CFG["costs"]["commission_per_share"],
        spread_bps=CFG["costs"]["spread_bps"],
        slippage_bps=CFG["costs"]["slippage_bps"],
        dollar_neutral=CFG["backtest"]["dollar_neutral"],
    )

    wf_eq = wf_trades["pnl_net"].cumsum()

    fig_wf1 = plot_equity(wf_eq, title=f"Walk-forward Equity {best_pair}")
    fig_wf1.savefig(FIGDIR / "equity_walkforward.png", dpi=160)
    plt.close(fig_wf1)

    fig_wf2 = plot_drawdown(wf_eq, title=f"Walk-forward Drawdown {best_pair}")
    fig_wf2.savefig(FIGDIR / "drawdown_walkforward.png", dpi=160)
    plt.close(fig_wf2)

    print("Saved walk-forward plots.")


if __name__ == "__main__":
    main()

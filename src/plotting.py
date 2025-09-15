"""
Plotting helpers using matplotlib.
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_equity(equity: pd.Series, title: str="Equity Curve"):
    plt.figure(figsize=(9,4))
    equity.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL ($)")
    plt.tight_layout()
    return plt.gcf()

def plot_drawdown(equity: pd.Series, title: str="Drawdown"):
    peak = equity.cummax()
    dd = equity - peak
    plt.figure(figsize=(9,3))
    dd.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown ($)")
    plt.tight_layout()
    return plt.gcf()

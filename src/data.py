"""
Data utilities: download (yfinance) or read local CSVs; basic cleaning & alignment.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

def load_prices(tickers: List[str],
                start: str,
                end: str,
                source: str = "yfinance",
                price_field: str = "Adj Close",
                csv_dir: str = "data") -> pd.DataFrame:
    """
    Returns a price DataFrame indexed by date with columns as tickers.
    """
    if source == "csv":
        frames = []
        for t in tickers:
            fp = os.path.join(csv_dir, f"{t}.csv")
            df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date")
            frames.append(df[[price_field]].rename(columns={price_field: t}))
        prices = pd.concat(frames, axis=1).sort_index()
    else:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("Install yfinance or set source='csv'") from e
        data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[price_field]
        else:
            prices = data.to_frame(name=tickers[0])
        prices = prices.dropna(how="all")
    # Forward-fill small gaps; drop rows still NA
    prices = prices.ffill(limit=5).dropna(how="any")
    return prices

def load_ohlcv(tickers: List[str], start: str, end: str, source: str="yfinance", csv_dir: str="data") -> Dict[str, pd.DataFrame]:
    """Return dict ticker->OHLCV DataFrame."""
    if source == "csv":
        out = {}
        for t in tickers:
            df = pd.read_csv(os.path.join(csv_dir, f"{t}.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
            out[t] = df
        return out
    else:
        import yfinance as yf
        out = {}
        for t in tickers:
            df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
            df.index.name = "Date"
            out[t] = df
        return out

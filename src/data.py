"""
Data utilities: Alpha Vantage downloader with adjusted->daily fallback and CSV cache.
"""

from __future__ import annotations
import os, time, json
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path


def load_prices(
    tickers: List[str],
    start: str,
    end: str,
    price_field: str = "Adj Close",       # "Adj Close" or "Close"
    api_key_env: str = "ALPHAVANTAGE_API_KEY",
    sleep_seconds: float = 13.0,          # free tier ~5 req/min -> ~12s; use 13s for safety
    max_retries: int = 2,
    cache_dir: str = "data/alphavantage", # per-ticker CSV cache
    use_adjusted: bool = True,            # try TIME_SERIES_DAILY_ADJUSTED first
    allow_partial: bool = True,           # don't crash if some tickers fail
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by Date with columns=tickers.
    For each ticker:
      1) Try TIME_SERIES_DAILY_ADJUSTED (if use_adjusted=True)
      2) On failure, fallback to TIME_SERIES_DAILY (unadjusted Close)
    Then filter [start, end], forward-fill small gaps, drop remaining NA rows.
    """
    try:
        from alpha_vantage.timeseries import TimeSeries  # type: ignore
    except ImportError as e:
        raise ImportError("Please install alpha_vantage: pip install alpha_vantage") from e

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set env var {api_key_env}.")

    ts = TimeSeries(key=api_key, output_format="pandas")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    series_list: List[pd.Series] = []
    loaded: List[str] = []
    failed: List[str] = []

    def _from_cache(t: str) -> Optional[pd.Series]:
        fp = cache_path / f"{t}.csv"
        meta = cache_path / f"{t}.meta.json"
        if fp.exists():
            try:
                df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
                if price_field in df.columns:
                    return df[price_field].rename(t)
                # allow fallback to Close if user asks Adj Close but cache lacks it
                if price_field == "Adj Close" and "Close" in df.columns:
                    return df["Close"].rename(t)
            except Exception:
                pass
        return None

    def _to_cache(t: str, df: pd.DataFrame, source: str):
        try:
            out = df.copy()
            out.index.name = "Date"
            out = out.rename(
                columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. adjusted close": "Adj Close",
                    "6. volume": "Volume",
                }
            )
            out.to_csv(cache_path / f"{t}.csv")
            (cache_path / f"{t}.meta.json").write_text(json.dumps({"source": source}))
        except Exception:
            pass

    for i, t in enumerate(tickers, 1):
        # 0) try cache first
        cached = _from_cache(t)
        if cached is not None:
            series_list.append(cached)
            loaded.append(t)
            # still sleep a tiny bit to be polite if we plan more API calls
            if i < len(tickers):
                time.sleep(0.1)
            continue

        last_err: Optional[Exception] = None

        # 1) Try ADJUSTED (if requested)
        s_out: Optional[pd.Series] = None
        if use_adjusted:
            for attempt in range(1, max_retries + 1):
                try:
                    df, _ = ts.get_daily_adjusted(symbol=t, outputsize="full")
                    df = df.sort_index()
                    # normalize columns
                    s = df["5. adjusted close"].rename(t)
                    s_out = s
                    _to_cache(t, df, source="adjusted")
                    break
                except Exception as ex:
                    last_err = ex
                    if attempt >= max_retries:
                        print(f"[AlphaVantage] Adjusted failed for {t}: {ex}")
                    time.sleep(1.0)

        # 2) Fallback to DAILY (unadjusted close)
        if s_out is None:
            for attempt in range(1, max_retries + 1):
                try:
                    df, _ = ts.get_daily(symbol=t, outputsize="full")
                    df = df.sort_index()
                    s = df["4. close"].rename(t)
                    s_out = s
                    _to_cache(t, df, source="daily")
                    break
                except Exception as ex:
                    last_err = ex
                    if attempt >= max_retries:
                        print(f"[AlphaVantage] Daily failed for {t}: {ex}")
                    time.sleep(1.0)

        if s_out is not None:
            series_list.append(s_out)
            loaded.append(t)
        else:
            failed.append(t)

        # Rate-limit pause between symbols
        if i < len(tickers):
            time.sleep(sleep_seconds)

    if not series_list:
        raise RuntimeError("No price data downloaded (rate limit? endpoint restrictions? symbols?).")

    prices = pd.concat(series_list, axis=1).sort_index()
    # choose requested field if possible
    if price_field == "Adj Close":
        # If some came from daily fallback, they won't have Adj Close; use Close for those
        have_adj = [c for c in prices.columns if c in loaded]  # columns already set to tickers
        # nothing special needed: we already built series_list with either adjusted or close
        pass

    # filter date range
    prices = prices.loc[(prices.index >= pd.to_datetime(start)) & (prices.index <= pd.to_datetime(end))]
    # clean
    prices = prices.dropna(how="all").ffill(limit=5)

    # drop rows with remaining NA across any kept columns
    non_na = prices.dropna(how="any")

    print(f"[AlphaVantage] Loaded {len(loaded)}/{len(tickers)} symbols.")
    if failed:
        print(f"[AlphaVantage] Missing: {', '.join(failed)}")

    if non_na.empty and not allow_partial:
        raise RuntimeError("All loaded series contain gaps after cleaning; set allow_partial=True or adjust tickers/dates.")
    return non_na

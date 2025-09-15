# Statistical Arbitrage: Pairs Trading with Cointegration

This project implements and analyzes a cointegration-based statistical arbitrage strategy.

## Features
- Data loading from Alpha Vantage
- Engle–Granger cointegration tests
- Z-score signal generation
- Vectorized backtester with transaction costs
- Walk-forward validation
- LaTeX report auto-generating tables & figures

## Repo Structure
```
stat-arb/
├─ README.md
├─ config.yaml
├─ paper/
│  ├─ stat_arb_mini_paper.md
│  └─ figs/
├─ src/
│  ├─ data.py
│  ├─ features.py
│  ├─ pairs.py
│  ├─ backtest.py
│  ├─ risk.py
│  ├─ evaluation.py
│  ├─ walkforward.py
│  └─ plotting.py
├─ notebooks/
│  ├─ 00_quick_demo.py
│  ├─ 01_quick_demo.py
└─ data/
   └─ (put raw/processed data here)
```

## Quickstart
1. **Install** (Python 3.10+ recommended)
```bash
pip install -r requirements.txt
```
2. **Configure** parameters in `config.yaml` (tickers, dates, costs, thresholds).
3. **Fetch data** (e.g., via `src/data.py` using yfinance or your own CSVs).
4. **Select pairs** with `src/pairs.py` (correlation screen + Engle–Granger/Johansen).
5. **Run backtest** with `src/backtest.py` and evaluate via `src/evaluation.py`.
6. **Walk-forward** train/test splits using `src/walkforward.py`.

```bash
python notebooks/00_quick_demo.py
python notebooks/01_pair_analysis.py
```

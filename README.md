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
│  ├─ 01_explore_data.ipynb
│  ├─ 02_pair_selection.ipynb
│  └─ 03_backtest_results.ipynb
└─ data/
   └─ (put raw/processed data here)
```

## Quick Demo
```bash
python notebooks/00_quick_demo.py

python notebooks/01_pair_analysis.py

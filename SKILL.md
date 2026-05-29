---
name: quant-trading
description: A-share quantitative trading system with volume-price signal engine (8 signals), backtesting with win-rate/sharpe/max-drawdown, and four-layer stock screening (non-ST, ROE above 10%, debt below 60%, price above MA20 above MA60, 20d gain above 10%, turnover above 3%, volume expansion). Use when building quantitative trading strategies for Chinese A-stocks, running backtests on volume-price signals, screening stocks by financial/trend/capital-flow multi-layer filters, or starting a local Flask web dashboard at localhost:5000 for interactive analysis.
---

# Quant Trading - A鑲￠噺浠烽噺鍖栦氦鏄撶郴缁?
A鑲￠噺浠峰叧绯婚噺鍖栦氦鏄撶郴缁燂細鍏ぇ淇″彿寮曟搸 + 鍥炴祴 + 鍥涘眰閫夎偂 + Web 鐪嬫澘銆?
## Quickstart

```bash
cd scripts/
pip install -r requirements.txt
python run.py
# Open http://localhost:5000
```

## Architecture

```
scripts/
鈹溾攢鈹€ run.py                  # Entry: starts Flask on :5000
鈹溾攢鈹€ core/
鈹?  鈹溾攢鈹€ signal_engine.py    # 8 volume-price signals (S1-S8)
鈹?  鈹溾攢鈹€ backtest.py         # Backtest engine with stop-loss/take-profit
鈹?  鈹溾攢鈹€ data_fetcher.py     # Daily + realtime data (akshare + demo fallback)
鈹?  鈹溾攢鈹€ screener.py         # 4-layer screening + composite scoring
鈹?  鈹斺攢鈹€ stock_screener.py   # Screening data: Sina direct API + THS financials
鈹斺攢鈹€ web/
    鈹溾攢鈹€ app.py              # Flask API routes (scan, backtest, monitor, screen)
    鈹斺攢鈹€ templates/
        鈹斺攢鈹€ index.html      # Dashboard UI: 2 tabs (Signals + Screening)
```

## Web API

| Endpoint | Method | Description |
|---|---|---|
| `/api/scan?code=600519` | GET | Scan all historical signals |
| `/api/backtest?code=600519&start=2023-01-01` | GET | Run backtest |
| `/api/monitor?code=600519` | GET | Real-time latest signals |
| `/api/history?code=600519` | GET | OHLCV + volume ratio history |
| `/api/screen/run` | GET | Run 4-layer screening (3-5 min) |
| `/api/screen/stock/<code>` | GET | Single stock screening detail |
| `/api/signal_definitions` | GET | All 8 signal definitions |

## Four-Layer Screening

| Layer | Filter | Threshold |
|---|---|---|
| L1 Garbage | Non-ST, ROE, Debt ratio | ROE above 10%, debt below 60% |
| L2 Trend | MA alignment, 20d momentum | price above MA20 above MA60, 20d gain above 10% |
| L3 Capital | Turnover, volume expansion | turnover above 3%, Vol5 > Vol20 |
| L4 Score | Weighted composite | Trend 40% + Volume 25% + Financial 20% + Valuation 15% |

## Signal Engine (S1-S8)

S1-S8 are volume-price confirmation/anomaly signals. See `/api/signal_definitions` for full details.

## Configuration

Key thresholds in `core/screener.py` and `core/signal_engine.py`:
- `FILTER_ROE_MIN`, `FILTER_DEBT_MAX`, `FILTER_GAIN_20D_MIN`, `FILTER_TURNOVER_MIN`
- `WEIGHT_TREND`, `WEIGHT_VOLUME`, `WEIGHT_FINANCIAL`, `WEIGHT_VALUATION`
- `VOL_HIGH_THRESHOLD`, `VOL_LOW_THRESHOLD`, `DEEP_DROP_THRESHOLD`

## Notes

- Requires akshare for live data; falls back to simulated demo data if unavailable
- Sina direct API used for spot snapshots (bypasses akshare rate limits)
- THS financial data for ROE/EPS/BVPS
- First run downloads ~4500 stock list; screening takes 3-5 minutes

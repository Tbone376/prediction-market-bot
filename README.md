# Stock Prediction Dashboard

**Institutional-grade Monte Carlo + Particle Filter + Edge Detection for SPX, SPY, AAPL, NVDA, TSLA**

⚡️ Live prediction dashboard with auto-refreshing data, GBM simulations, RSI/MACD/Bollinger Band signals, and rolling Brier score calibration tracking.

---

## Features

- **Live Data Pipeline**: yfinance-powered real-time price, vol, RSI, MACD, Bollinger Bands for 5 tickers
- **Monte Carlo Prediction Engine**: GBM path simulation with 1d/5d/30d probability targets
- **Particle Filter**: Bayesian fair-value estimator that smooths market noise
- **Multi-Factor Edge Detection**: Composite BUY/SELL/HOLD signals from 6 weighted factors
- **Brier Score Tracker**: SQLite-backed rolling calibration scoring — tracks model accuracy over time
- **FastAPI Backend**: Auto-refreshes every 5 minutes, serves JSON endpoints + HTML dashboard
- **Zero JavaScript Frameworks**: Pure HTML + Fetch API — runs in any browser

---

## Installation

```bash
git clone https://github.com/Tbone376/prediction-market-bot.git
cd prediction-market-bot
pip install -r requirements.txt
python app.py
```

Dashboard launches at: **http://localhost:8000**

---

## Architecture

```
modules/
  monte_carlo.py       ← GBM binary contract engine + importance sampling
  data_fetcher.py      ← yfinance live data + technical indicators
  predictor.py         ← GBM path simulator + particle filter
  edge_detector.py     ← multi-factor composite scoring (RSI/MACD/BB/GBM/PF)
  brier_tracker.py     ← SQLite prediction log + rolling calibration scores

app.py                 ← FastAPI backend with /api/* endpoints
index.html             ← Live dashboard UI
data/brier_scores.db   ← Auto-generated SQLite database
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serve dashboard HTML |
| `GET /api/snapshot` | Current price + vol + technicals for all tickers |
| `GET /api/predictions` | GBM + particle filter predictions (1d/5d/30d) |
| `GET /api/edges` | BUY/SELL/HOLD signals with composite scores |
| `GET /api/full` | Combined snapshot + predictions + edges |
| `GET /api/brier` | Rolling Brier scores for all tickers + horizons |
| `GET /api/history/{ticker}` | Price series for charting (last 90 days) |
| `GET /api/status` | Health check + refresh schedule |
| `POST /api/refresh` | Manual data refresh trigger |

---

## Edge Detection Signal Weights

| Signal | Weight | Description |
|---|---|---|
| GBM Probability | 25% | 1-day P(up) from Monte Carlo simulation |
| Particle Filter | 20% | Fair-value divergence from market price |
| RSI (14) | 20% | Mean-reversion signal (oversold/overbought) |
| MACD Histogram | 20% | Momentum + crossover detection |
| Bollinger Bands | 10% | %B position (band squeeze/expansion) |
| Day Momentum | 5% | Intraday % change |

**Composite Score Thresholds:**
- `>= +0.15`: **BUY** signal
- `<= -0.15`: **SELL** signal
- Between: **HOLD**

---

## Brier Score Calibration

The Brier score measures prediction accuracy:

**Brier = mean((predicted_prob - actual_outcome)^2)**

- **< 0.10**: EXCELLENT
- **0.10 – 0.20**: GOOD
- **0.20 – 0.25**: FAIR
- **> 0.25**: POOR

Best forecasters (538, Economist) achieve 0.06–0.12 on presidential races. Your model's rolling Brier score is displayed per ticker and horizon.

---

## Tech Stack

- **Backend**: FastAPI + APScheduler (auto-refresh)
- **Data**: yfinance (free, no API key needed)
- **Simulation**: NumPy + SciPy (Monte Carlo + particle filter)
- **Storage**: SQLite (Brier score history)
- **Frontend**: Pure HTML5 + Fetch API

---

## Usage

1. Launch: `python app.py`
2. Open browser: `http://localhost:8000`
3. Dashboard auto-refreshes every 30 seconds from cached backend data
4. Backend refreshes from yfinance every 5 minutes
5. Click any ticker to expand prediction details

---

## Notes

- **yfinance limitations**: Free tier, ~15min delayed data during market hours
- **SPX (^GSPC)**: S&P 500 index, no direct trading instrument
- **Brier tracking**: Requires 3+ resolved predictions before score is valid
- **Particle filter warm-up**: Initialized with ±0.5% noise around current price

---

## License

MIT

---

## Author

Built by **Tbone376** — Mechanical Engineer + Quant Trader

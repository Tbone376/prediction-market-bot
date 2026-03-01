# modules/data_fetcher.py
# Live + historical data fetcher for SPX, SPY, AAPL, NVDA, TSLA via yfinance
# Computes rolling volatility, drift, RSI, MACD, Bollinger Bands per ticker

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

TICKERS = {
    "SPX": "^GSPC",
    "SPY": "SPY",
    "AAPL": "AAPL",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
}


def fetch_history(ticker_symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV history. ticker_symbol is the yfinance symbol (e.g. '^GSPC')."""
    df = yf.download(ticker_symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators: returns, rolling vol, RSI(14), MACD, Bollinger Bands."""
    df = df.copy()
    close = df["Close"].squeeze()

    # Log returns + rolling annualised vol
    df["log_return"] = np.log(close / close.shift(1))
    df["rolling_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)
    df["rolling_vol_60"] = df["log_return"].rolling(60).std() * np.sqrt(252)

    # Annualised drift (mu)
    df["rolling_drift_20"] = df["log_return"].rolling(20).mean() * 252

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    return df


def get_ticker_snapshot(name: str) -> dict:
    """
    Full snapshot for one ticker: current price, vol, drift, technicals.
    Returns dict ready for JSON serialisation.
    """
    symbol = TICKERS[name]
    df = fetch_history(symbol, period="6mo", interval="1d")
    df = compute_indicators(df)
    latest = df.iloc[-1]
    close = df["Close"].squeeze()

    # Live quote (most recent close - yfinance free tier)
    current_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    day_change_pct = ((current_price - prev_close) / prev_close) * 100

    sigma = float(latest["rolling_vol_20"]) if not np.isnan(latest["rolling_vol_20"]) else 0.25
    mu = float(latest["rolling_drift_20"]) if not np.isnan(latest["rolling_drift_20"]) else 0.08

    return {
        "ticker": name,
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "prev_close": round(prev_close, 2),
        "day_change_pct": round(day_change_pct, 3),
        "sigma_20d": round(sigma, 4),
        "sigma_60d": round(float(latest["rolling_vol_60"]) if not np.isnan(latest["rolling_vol_60"]) else sigma, 4),
        "mu_annualised": round(mu, 4),
        "rsi_14": round(float(latest["rsi_14"]), 2) if not np.isnan(latest["rsi_14"]) else None,
        "macd": round(float(latest["macd"]), 4) if not np.isnan(latest["macd"]) else None,
        "macd_signal": round(float(latest["macd_signal"]), 4) if not np.isnan(latest["macd_signal"]) else None,
        "macd_hist": round(float(latest["macd_hist"]), 4) if not np.isnan(latest["macd_hist"]) else None,
        "bb_upper": round(float(latest["bb_upper"]), 2) if not np.isnan(latest["bb_upper"]) else None,
        "bb_lower": round(float(latest["bb_lower"]), 2) if not np.isnan(latest["bb_lower"]) else None,
        "bb_pct": round(float(latest["bb_pct"]), 4) if not np.isnan(latest["bb_pct"]) else None,
        "last_updated": datetime.utcnow().isoformat() + "Z",
    }


def get_all_snapshots() -> list:
    """Fetch snapshots for all 5 tickers. Returns list of dicts."""
    results = []
    for name in TICKERS:
        try:
            snap = get_ticker_snapshot(name)
            results.append(snap)
        except Exception as e:
            results.append({"ticker": name, "error": str(e)})
    return results


def get_price_series(name: str, days: int = 90) -> list:
    """Return last N days of close prices as list of {date, close} for charting."""
    symbol = TICKERS[name]
    df = fetch_history(symbol, period="6mo", interval="1d")
    close = df["Close"].squeeze().tail(days)
    return [
        {"date": str(idx.date()), "close": round(float(val), 2)}
        for idx, val in close.items()
    ]

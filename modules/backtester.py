# modules/backtester.py
# Historical backtest engine — replays edge detection signals over 6 months
# of daily data and simulates BUY/SELL round-trip trades per ticker.
# Returns trade log, equity curve, and summary statistics.

import numpy as np
import pandas as pd
from datetime import datetime
from modules.data_fetcher import fetch_history, compute_indicators, TICKERS


# ---------------------------------------------------------------------------
# SIGNAL REPLAY — mirrors edge_detector.py logic on historical bars
# ---------------------------------------------------------------------------

def _rsi_score(rsi):
    if rsi is None or np.isnan(rsi):
        return 0.0
    if rsi < 30:
        return min((30 - rsi) / 30, 1.0)
    elif rsi > 70:
        return max(-(rsi - 70) / 30, -1.0)
    return 0.0


def _macd_score(macd_hist, macd):
    if macd_hist is None or np.isnan(macd_hist):
        return 0.0
    scale = abs(macd) if macd and abs(macd) > 1e-6 else 1.0
    return float(np.clip(macd_hist / scale, -1.0, 1.0))


def _bb_score(bb_pct):
    if bb_pct is None or np.isnan(bb_pct):
        return 0.0
    if bb_pct < 0.2:
        return min((0.2 - bb_pct) / 0.2, 1.0)
    elif bb_pct > 0.8:
        return max(-(bb_pct - 0.8) / 0.2, -1.0)
    return 0.0


def _momentum_score(day_change_pct):
    return float(np.clip(day_change_pct / 5.0, -1.0, 1.0))


def _gbm_score(p_up):
    return float(np.clip((p_up - 0.5) * 4.0, -1.0, 1.0))


# Weights must match edge_detector.py
WEIGHTS = {
    "gbm": 0.25,
    "rsi": 0.20,
    "macd": 0.20,
    "bollinger": 0.10,
    "momentum": 0.05,
    # PF weight (0.20) is redistributed to GBM for backtest since we
    # can't meaningfully run the particle filter on historical bars.
    "gbm_pf_extra": 0.20,
}

EDGE_THRESHOLD_BUY = 0.15
EDGE_THRESHOLD_SELL = -0.15


def _composite_score(row):
    """Compute composite edge score from a single historical bar's indicators."""
    rsi = _rsi_score(row.get("rsi_14"))
    macd = _macd_score(row.get("macd_hist"), row.get("macd"))
    bb = _bb_score(row.get("bb_pct"))
    mom = _momentum_score(row.get("day_change_pct", 0))

    # Quick GBM 1-day P(up) from drift/vol
    mu = row.get("rolling_drift_20", 0.08)
    sigma = row.get("rolling_vol_20", 0.25)
    if sigma == 0 or np.isnan(sigma):
        sigma = 0.25
    if np.isnan(mu):
        mu = 0.0
    dt = 1 / 252
    # P(S_T > S_0) = P(Z > -(mu - 0.5*sigma^2)*sqrt(dt)/sigma)
    from scipy.stats import norm
    d = (mu - 0.5 * sigma ** 2) * np.sqrt(dt) / sigma
    p_up = float(norm.cdf(d))
    gbm = _gbm_score(p_up)

    composite = (
        (WEIGHTS["gbm"] + WEIGHTS["gbm_pf_extra"]) * gbm
        + WEIGHTS["rsi"] * rsi
        + WEIGHTS["macd"] * macd
        + WEIGHTS["bollinger"] * bb
        + WEIGHTS["momentum"] * mom
    )
    return composite, p_up


# ---------------------------------------------------------------------------
# BACKTEST RUNNER
# ---------------------------------------------------------------------------

def run_backtest(ticker_name: str, period: str = "1y") -> dict:
    """
    Run a historical backtest for one ticker.

    Returns:
        {
            "ticker": str,
            "trades": [ {entry_date, entry_price, exit_date, exit_price, pnl, pnl_pct, hold_days, signal_at_entry} ],
            "equity_curve": [ {date, equity} ],
            "summary": { total_trades, winners, losers, win_rate, avg_win, avg_loss,
                         profit_factor, total_pnl, max_drawdown, sharpe }
        }
    """
    symbol = TICKERS[ticker_name]
    df = fetch_history(symbol, period=period, interval="1d")
    df = compute_indicators(df)

    # Flatten multi-level columns from yfinance
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = [col[0] if col[1] == '' or col[1] == symbol else col[0] for col in df.columns]

    df.dropna(subset=["rolling_vol_20"], inplace=True)

    close = df["Close"]
    if hasattr(close, 'squeeze'):
        close = close.squeeze()
    prev_close = close.shift(1)
    df["day_change_pct"] = ((close - prev_close) / prev_close * 100).fillna(0)

    # Convert to list of row dicts for iteration
    records = []
    for idx, row in df.iterrows():
        r = {}
        for col in df.columns:
            val = row[col]
            try:
                r[col] = float(val)
            except (TypeError, ValueError):
                r[col] = val
        r["date"] = str(idx.date()) if hasattr(idx, 'date') else str(idx)
        r["close"] = float(close.loc[idx])
        records.append(r)

    # --- Simulate trades ---
    trades = []
    equity_curve = []
    position = None  # None or {entry_date, entry_price, score}
    starting_equity = 10000.0
    equity = starting_equity

    for rec in records:
        score, p_up = _composite_score(rec)
        price = rec["close"]
        date = rec["date"]

        if score >= EDGE_THRESHOLD_BUY:
            signal = "BUY"
        elif score <= EDGE_THRESHOLD_SELL:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Position management
        if signal == "BUY" and position is None:
            position = {"entry_date": date, "entry_price": price, "score": score}
        elif signal == "SELL" and position is not None:
            pnl = price - position["entry_price"]
            pnl_pct = (pnl / position["entry_price"]) * 100
            entry_dt = datetime.strptime(position["entry_date"], "%Y-%m-%d")
            exit_dt = datetime.strptime(date, "%Y-%m-%d")
            hold_days = (exit_dt - entry_dt).days

            equity += pnl  # 1-share simplification
            trades.append({
                "entry_date": position["entry_date"],
                "entry_price": round(position["entry_price"], 2),
                "exit_date": date,
                "exit_price": round(price, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "hold_days": hold_days,
                "edge_at_entry": round(position["score"], 4),
            })
            position = None

        equity_curve.append({"date": date, "equity": round(equity, 2)})

    # Close any remaining open position at last price
    if position is not None and len(records) > 0:
        last = records[-1]
        pnl = last["close"] - position["entry_price"]
        pnl_pct = (pnl / position["entry_price"]) * 100
        entry_dt = datetime.strptime(position["entry_date"], "%Y-%m-%d")
        exit_dt = datetime.strptime(last["date"], "%Y-%m-%d")
        hold_days = (exit_dt - entry_dt).days
        equity += pnl
        trades.append({
            "entry_date": position["entry_date"],
            "entry_price": round(position["entry_price"], 2),
            "exit_date": last["date"] + " (open)",
            "exit_price": round(last["close"], 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "hold_days": hold_days,
            "edge_at_entry": round(position["score"], 4),
        })
        equity_curve.append({"date": last["date"], "equity": round(equity, 2)})

    # --- Summary stats ---
    if trades:
        pnls = [t["pnl"] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        win_rate = len(winners) / len(pnls) * 100

        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0
        profit_factor = (sum(winners) / abs(sum(losers))) if losers and sum(losers) != 0 else float("inf")

        # Max drawdown from equity curve
        eq_values = [e["equity"] for e in equity_curve]
        peak = eq_values[0]
        max_dd = 0
        for v in eq_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe (annualised from daily trade P&L)
        daily_returns = np.array(pnls) / starting_equity
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0

        summary = {
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "Inf",
            "total_pnl": round(sum(pnls), 2),
            "total_pnl_pct": round((equity - starting_equity) / starting_equity * 100, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe": round(float(sharpe), 2),
            "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
        }
    else:
        summary = {
            "total_trades": 0, "winners": 0, "losers": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "total_pnl": 0,
            "total_pnl_pct": 0, "max_drawdown_pct": 0, "sharpe": 0, "avg_hold_days": 0,
        }

    return {
        "ticker": ticker_name,
        "period": period,
        "total_bars": len(records),
        "trades": trades,
        "equity_curve": equity_curve,
        "summary": summary,
    }


def run_all_backtests(period: str = "1y") -> list:
    """Run backtests for all tracked tickers."""
    results = []
    for name in TICKERS:
        try:
            results.append(run_backtest(name, period))
        except Exception as e:
            results.append({"ticker": name, "error": str(e)})
    return results

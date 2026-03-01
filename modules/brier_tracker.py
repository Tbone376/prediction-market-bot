# modules/brier_tracker.py
# Rolling Brier Score calibration tracker with SQLite persistence
# Tracks model predictions vs actual price direction outcomes per ticker

import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("data/brier_scores.db")


def init_db():
    """Initialise SQLite database for storing predictions and outcomes."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            horizon     TEXT    NOT NULL,
            p_up        REAL    NOT NULL,
            price_at_pred REAL  NOT NULL,
            target_price  REAL,
            outcome     INTEGER,
            predicted_at TEXT   NOT NULL,
            resolved_at  TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS brier_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            horizon     TEXT    NOT NULL,
            brier_score REAL    NOT NULL,
            n_samples   INTEGER NOT NULL,
            computed_at TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(ticker: str, horizon: str, p_up: float, current_price: float):
    """
    Store a new prediction. outcome is None until resolved.
    horizon: '1d', '5d', or '30d'
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute("""
        INSERT INTO predictions (ticker, horizon, p_up, price_at_pred, predicted_at)
        VALUES (?, ?, ?, ?, ?)
    """, (ticker, horizon, p_up, current_price, now))
    conn.commit()
    conn.close()


def resolve_predictions(ticker: str, current_price: float):
    """
    Check unresolved predictions whose horizon has elapsed and mark outcome.
    outcome=1 if price went up, 0 if went down.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow()

    horizon_days = {"1d": 1, "5d": 5, "30d": 30}

    for horizon, days in horizon_days.items():
        cutoff = (now - timedelta(days=days)).isoformat()
        rows = c.execute("""
            SELECT id, price_at_pred FROM predictions
            WHERE ticker=? AND horizon=? AND outcome IS NULL AND predicted_at <= ?
        """, (ticker, horizon, cutoff)).fetchall()

        for row_id, price_at_pred in rows:
            outcome = 1 if current_price > price_at_pred else 0
            c.execute("""
                UPDATE predictions
                SET outcome=?, target_price=?, resolved_at=?
                WHERE id=?
            """, (outcome, current_price, now.isoformat(), row_id))

    conn.commit()
    conn.close()


def compute_brier_score(ticker: str, horizon: str, window_days: int = 90) -> dict:
    """
    Compute rolling Brier score for a ticker+horizon over the last N days.
    Brier = mean((p_predicted - outcome)^2). Lower is better.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = (datetime.utcnow() - timedelta(days=window_days)).isoformat()

    rows = c.execute("""
        SELECT p_up, outcome FROM predictions
        WHERE ticker=? AND horizon=? AND outcome IS NOT NULL AND predicted_at >= ?
    """, (ticker, horizon, cutoff)).fetchall()
    conn.close()

    if len(rows) < 3:
        return {
            "ticker": ticker, "horizon": horizon,
            "brier_score": None, "n_samples": len(rows),
            "grade": "INSUFFICIENT DATA"
        }

    preds = np.array([r[0] for r in rows])
    outcomes = np.array([r[1] for r in rows])
    brier = float(np.mean((preds - outcomes) ** 2))

    if brier < 0.10:
        grade = "EXCELLENT"
    elif brier < 0.20:
        grade = "GOOD"
    elif brier < 0.25:
        grade = "FAIR"
    else:
        grade = "POOR"

    return {
        "ticker": ticker,
        "horizon": horizon,
        "brier_score": round(brier, 4),
        "n_samples": len(rows),
        "grade": grade,
        "computed_at": datetime.utcnow().isoformat() + "Z"
    }


def get_all_brier_scores() -> list:
    """Return Brier scores for all tickers and all horizons."""
    tickers = ["SPX", "SPY", "AAPL", "NVDA", "TSLA"]
    horizons = ["1d", "5d", "30d"]
    results = []
    for ticker in tickers:
        for horizon in horizons:
            results.append(compute_brier_score(ticker, horizon))
    return results


def get_prediction_history(ticker: str, limit: int = 50) -> list:
    """Return recent prediction log for a ticker."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute("""
        SELECT ticker, horizon, p_up, price_at_pred, outcome, predicted_at, resolved_at
        FROM predictions
        WHERE ticker=?
        ORDER BY predicted_at DESC
        LIMIT ?
    """, (ticker, limit)).fetchall()
    conn.close()
    keys = ["ticker", "horizon", "p_up", "price_at_pred", "outcome", "predicted_at", "resolved_at"]
    return [dict(zip(keys, row)) for row in rows]

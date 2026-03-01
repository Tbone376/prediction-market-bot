# modules/position_tracker.py
# Live P&L tracker for BUY/HOLD/SELL signals
# Auto-opens positions on BUY, holds through HOLD, closes on SELL
# Tracks unrealized + realized P&L with full trade history

import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/positions.db")


def init_db():
    """Initialize SQLite database for position tracking."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            exit_price REAL,
            exit_time TEXT,
            quantity REAL DEFAULT 1.0,
            status TEXT DEFAULT 'OPEN',
            pnl REAL,
            pnl_pct REAL,
            signal_strength TEXT,
            entry_edge_score REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS pnl_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            signal TEXT NOT NULL,
            price REAL NOT NULL,
            unrealized_pnl REAL,
            total_realized_pnl REAL
        )
    """)
    conn.commit()
    conn.close()


def get_open_position(ticker: str):
    """Return current open position for ticker, or None."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute("""
        SELECT id, entry_price, entry_time, quantity, signal_strength, entry_edge_score
        FROM positions
        WHERE ticker=? AND status='OPEN'
        ORDER BY entry_time DESC LIMIT 1
    """, (ticker,)).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0], "entry_price": row[1], "entry_time": row[2],
        "quantity": row[3], "signal_strength": row[4], "entry_edge_score": row[5]
    }


def open_position(ticker: str, entry_price: float, signal_strength: str, edge_score: float, quantity: float = 1.0):
    """Open a new position on BUY signal."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute("""
        INSERT INTO positions (ticker, entry_price, entry_time, quantity, status, signal_strength, entry_edge_score)
        VALUES (?, ?, ?, ?, 'OPEN', ?, ?)
    """, (ticker, entry_price, now, quantity, signal_strength, edge_score))
    conn.commit()
    conn.close()
    print(f"[POSITION] Opened {ticker} @ ${entry_price:.2f} | {signal_strength} | Edge: {edge_score:.4f}")


def close_position(ticker: str, exit_price: float):
    """Close current open position on SELL signal."""
    position = get_open_position(ticker)
    if not position:
        return None
    
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    
    pnl = (exit_price - position["entry_price"]) * position["quantity"]
    pnl_pct = ((exit_price - position["entry_price"]) / position["entry_price"]) * 100
    
    c.execute("""
        UPDATE positions
        SET exit_price=?, exit_time=?, status='CLOSED', pnl=?, pnl_pct=?
        WHERE id=?
    """, (exit_price, now, pnl, pnl_pct, position["id"]))
    conn.commit()
    conn.close()
    
    print(f"[POSITION] Closed {ticker} @ ${exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    return {"pnl": pnl, "pnl_pct": pnl_pct}


def update_positions(edges: list):
    """
    Called on every data refresh. Opens/closes positions based on signal changes.
    - BUY signal + no open position => OPEN
    - SELL signal + open position => CLOSE
    - HOLD => do nothing
    """
    for edge in edges:
        if "error" in edge:
            continue
        
        ticker = edge["ticker"]
        signal = edge["signal"]
        current_price = edge["current_price"]
        signal_strength = edge["signal_strength"]
        edge_score = edge["composite_score"]
        
        position = get_open_position(ticker)
        
        if signal == "BUY" and not position:
            open_position(ticker, current_price, signal_strength, edge_score)
        elif signal == "SELL" and position:
            close_position(ticker, current_price)
        elif signal == "HOLD" and position:
            # Update unrealized P&L in history log
            unrealized_pnl = (current_price - position["entry_price"]) * position["quantity"]
            log_pnl_snapshot(ticker, signal, current_price, unrealized_pnl)


def log_pnl_snapshot(ticker: str, signal: str, price: float, unrealized_pnl: float):
    """Log current P&L snapshot to history."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    total_realized = get_total_realized_pnl(ticker)
    c.execute("""
        INSERT INTO pnl_history (ticker, timestamp, signal, price, unrealized_pnl, total_realized_pnl)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ticker, now, signal, price, unrealized_pnl, total_realized))
    conn.commit()
    conn.close()


def get_total_realized_pnl(ticker: str) -> float:
    """Sum of all closed position P&L for a ticker."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    result = c.execute("""
        SELECT COALESCE(SUM(pnl), 0) FROM positions
        WHERE ticker=? AND status='CLOSED'
    """, (ticker,)).fetchone()[0]
    conn.close()
    return float(result)


def get_all_positions_summary() -> dict:
    """Return current positions + unrealized P&L + realized P&L summary."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Open positions
    open_rows = c.execute("""
        SELECT ticker, entry_price, entry_time, quantity, signal_strength, entry_edge_score
        FROM positions WHERE status='OPEN'
    """).fetchall()
    
    # Closed positions (last 10)
    closed_rows = c.execute("""
        SELECT ticker, entry_price, exit_price, entry_time, exit_time, pnl, pnl_pct
        FROM positions WHERE status='CLOSED'
        ORDER BY exit_time DESC LIMIT 10
    """).fetchall()
    
    conn.close()
    
    open_positions = []
    for row in open_rows:
        open_positions.append({
            "ticker": row[0], "entry_price": row[1], "entry_time": row[2],
            "quantity": row[3], "signal_strength": row[4], "entry_edge_score": row[5]
        })
    
    closed_trades = []
    for row in closed_rows:
        closed_trades.append({
            "ticker": row[0], "entry_price": row[1], "exit_price": row[2],
            "entry_time": row[3], "exit_time": row[4], "pnl": row[5], "pnl_pct": row[6]
        })
    
    # Total realized P&L across all tickers
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total_realized = c.execute("""
        SELECT COALESCE(SUM(pnl), 0) FROM positions WHERE status='CLOSED'
    """).fetchone()[0]
    conn.close()
    
    return {
        "open_positions": open_positions,
        "closed_trades": closed_trades,
        "total_realized_pnl": float(total_realized)
    }


def get_current_unrealized_pnl(ticker: str, current_price: float) -> dict:
    """Calculate unrealized P&L for currently open position."""
    position = get_open_position(ticker)
    if not position:
        return {"unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0, "has_position": False}
    
    unrealized_pnl = (current_price - position["entry_price"]) * position["quantity"]
    unrealized_pnl_pct = ((current_price - position["entry_price"]) / position["entry_price"]) * 100
    
    return {
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
        "has_position": True,
        "entry_price": position["entry_price"],
        "entry_time": position["entry_time"],
        "signal_strength": position["signal_strength"]
    }

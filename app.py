# app.py
import sys
import os
import time
from pathlib import Path

# Absolute path to current directory
BASE_DIR = Path(__file__).resolve().parent

# CRITICAL STARTUP WRAPPER
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from apscheduler.schedulers.background import BackgroundScheduler
    import uvicorn
    import logging
    from datetime import datetime

    # Module Imports
    from modules.data_fetcher import get_all_snapshots, TICKERS
    from modules.predictor import generate_prediction
    from modules.edge_detector import run_all_edge_detection
    from modules.position_tracker import update_positions, get_all_positions_summary, init_db as init_pos_db
    from modules.brier_tracker import log_prediction, resolve_predictions, get_all_brier_scores, init_db as init_brier_db

except Exception as e:
    print("--- STARTUP ERROR ---")
    print(str(e))
    time.sleep(60)
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

# Init Databases
os.makedirs(BASE_DIR / "data", exist_ok=True)
init_pos_db()
init_brier_db()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {
    "snapshots": [],
    "predictions": [],
    "edges": [],
    "brier_scores": [],
    "last_updated": None,
    "status": "initializing"
}

def refresh_data():
    global _cache
    try:
        snapshots = get_all_snapshots()
        predictions = [generate_prediction(s, N_paths=10000) for s in snapshots if "error" not in s]
        edges = run_all_edge_detection(snapshots, predictions)
        
        # Track and Resolve Predictions (for Brier Scores)
        for snap in snapshots:
            if "error" in snap: continue
            resolve_predictions(snap["ticker"], snap["current_price"])
            
            # Find the matching prediction
            pred = next((p for p in predictions if p["ticker"] == snap["ticker"]), None)
            if pred:
                # Log 1d, 5d, 30d predictions for calibration tracking
                log_prediction(snap["ticker"], "1d", pred["predictions"]["1d"]["p_up"], snap["current_price"])
                log_prediction(snap["ticker"], "5d", pred["predictions"]["5d"]["p_up"], snap["current_price"])
                log_prediction(snap["ticker"], "30d", pred["predictions"]["30d"]["p_up"], snap["current_price"])

        # Update Live Positions (PNL tracking)
        update_positions(edges)
        
        # Update Cache
        _cache.update({
            "snapshots": snapshots,
            "predictions": predictions,
            "edges": edges,
            "brier_scores": get_all_brier_scores(),
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "status": "ok"
        })
        logger.info(f"Data refreshed at {datetime.utcnow()}")
    except Exception as e:
        logger.error(f"Scan failed: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(refresh_data, "interval", minutes=1)

@app.on_event("startup")
def startup_event():
    refresh_data()
    scheduler.start()

@app.get("/api/full")
def api_full():
    portfolio = get_all_positions_summary()
    return JSONResponse({
        "snapshots": _cache["snapshots"],
        "predictions": _cache["predictions"],
        "edges": _cache["edges"],
        "brier_scores": _cache["brier_scores"],
        "portfolio": portfolio,
        "positions": portfolio["open_positions"],
        "status": _cache["status"],
        "last_updated": _cache["last_updated"]
    })

@app.get("/api/price_history/{ticker}")
def api_price_history(ticker: str, days: int = 90):
    """Return last N days of close prices for charting."""
    from modules.data_fetcher import get_price_series
    try:
        series = get_price_series(ticker, days)
        return JSONResponse(series)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/backtest")
def api_backtest(period: str = "1y"):
    """Run backtest across all tickers. Cached for 10 minutes."""
    from modules.backtester import run_all_backtests
    cache_key = f"backtest_{period}"
    # Simple cache check
    if cache_key in _cache and _cache.get(f"{cache_key}_ts"):
        from datetime import timedelta
        age = datetime.utcnow() - datetime.fromisoformat(_cache[f"{cache_key}_ts"])
        if age < timedelta(minutes=10):
            return JSONResponse(_cache[cache_key])
    results = run_all_backtests(period)
    _cache[cache_key] = results
    _cache[f"{cache_key}_ts"] = datetime.utcnow().isoformat()
    return JSONResponse(results)

@app.get("/api/backtest/{ticker}")
def api_backtest_ticker(ticker: str, period: str = "1y"):
    """Run backtest for a single ticker."""
    from modules.backtester import run_backtest
    try:
        result = run_backtest(ticker, period)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/closed_trades")
def api_closed_trades():
    """Return closed trade history with P&L."""
    summary = get_all_positions_summary()
    return JSONResponse(summary["closed_trades"])

@app.get("/")
def serve_dashboard():
    return FileResponse(str(BASE_DIR / "index.html"))

if __name__ == "__main__":
    try:
        print("--- SERVER STARTING ---")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print("--- SERVER CRASH ---")
        print(str(e))
        time.sleep(60)

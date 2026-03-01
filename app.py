# app.py
# FastAPI backend for the Stock Prediction Dashboard
# Serves live data for SPX, SPY, AAPL, NVDA, TSLA
# Auto-cached, refreshes every 5 minutes, serves index.html

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import uvicorn
import logging
import json
from pathlib import Path

from modules.data_fetcher import get_all_snapshots, get_price_series, TICKERS
from modules.predictor import generate_prediction
from modules.edge_detector import run_all_edge_detection
from modules.brier_tracker import (
    log_prediction, resolve_predictions,
    get_all_brier_scores, get_prediction_history
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Prediction Dashboard",
    description="Live GBM + Particle Filter + Edge Detection for SPX, SPY, AAPL, NVDA, TSLA",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# IN-MEMORY CACHE
# ---------------------------------------------------------------------------

_cache = {
    "snapshots": [],
    "predictions": [],
    "edges": [],
    "last_updated": None,
    "status": "initializing"
}


def refresh_data():
    """Full data refresh: fetch -> predict -> edge detect -> log Brier."""
    global _cache
    logger.info("Refreshing data...")
    try:
        snapshots = get_all_snapshots()
        predictions = []
        for snap in snapshots:
            if "error" not in snap:
                pred = generate_prediction(snap, N_paths=15_000)
                predictions.append(pred)
                # Log predictions to Brier tracker
                for horizon in ["1d", "5d", "30d"]:
                    p_up = pred["predictions"][horizon]["p_up"]
                    log_prediction(snap["ticker"], horizon, p_up, snap["current_price"])
                # Resolve any matured predictions
                resolve_predictions(snap["ticker"], snap["current_price"])

        edges = run_all_edge_detection(snapshots, predictions)

        _cache["snapshots"] = snapshots
        _cache["predictions"] = predictions
        _cache["edges"] = edges
        _cache["last_updated"] = datetime.utcnow().isoformat() + "Z"
        _cache["status"] = "ok"
        logger.info(f"Data refreshed at {_cache['last_updated']}")
    except Exception as e:
        _cache["status"] = f"error: {str(e)}"
        logger.error(f"Refresh failed: {e}")


# ---------------------------------------------------------------------------
# SCHEDULER  (refresh every 5 minutes)
# ---------------------------------------------------------------------------

scheduler = BackgroundScheduler()
scheduler.add_job(refresh_data, "interval", minutes=5, id="data_refresh")


@app.on_event("startup")
def startup_event():
    refresh_data()  # initial load
    scheduler.start()
    logger.info("Scheduler started")


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()


# ---------------------------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------------------------

@app.get("/api/snapshot")
def api_snapshot():
    """All ticker snapshots: price, vol, RSI, MACD, BB."""
    if not _cache["snapshots"]:
        raise HTTPException(503, "Data not yet loaded")
    return JSONResponse({
        "data": _cache["snapshots"],
        "last_updated": _cache["last_updated"],
        "status": _cache["status"]
    })


@app.get("/api/predictions")
def api_predictions():
    """GBM + Particle Filter predictions for all tickers."""
    return JSONResponse({
        "data": _cache["predictions"],
        "last_updated": _cache["last_updated"]
    })


@app.get("/api/edges")
def api_edges():
    """BUY/SELL/HOLD edge signals with composite scores."""
    return JSONResponse({
        "data": _cache["edges"],
        "last_updated": _cache["last_updated"]
    })


@app.get("/api/full")
def api_full():
    """Full combined payload: snapshots + predictions + edges in one call."""
    return JSONResponse({
        "snapshots": _cache["snapshots"],
        "predictions": _cache["predictions"],
        "edges": _cache["edges"],
        "last_updated": _cache["last_updated"],
        "status": _cache["status"]
    })


@app.get("/api/brier")
def api_brier():
    """Rolling Brier scores for all tickers and horizons."""
    return JSONResponse({"data": get_all_brier_scores()})


@app.get("/api/history/{ticker}")
def api_history(ticker: str, days: int = 90):
    """Price series for charting (last N days)."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not found")
    series = get_price_series(ticker, days)
    return JSONResponse({"ticker": ticker, "series": series})


@app.get("/api/predictions_log/{ticker}")
def api_predictions_log(ticker: str):
    """Brier prediction log for a specific ticker."""
    return JSONResponse({"data": get_prediction_history(ticker.upper())})


@app.post("/api/refresh")
def api_manual_refresh():
    """Manually trigger a data refresh."""
    refresh_data()
    return JSONResponse({"status": "refreshed", "at": _cache["last_updated"]})


@app.get("/api/status")
def api_status():
    """Health check."""
    return JSONResponse({
        "status": _cache["status"],
        "last_updated": _cache["last_updated"],
        "tickers": list(TICKERS.keys()),
        "next_refresh": "every 5 minutes"
    })


# ---------------------------------------------------------------------------
# SERVE DASHBOARD
# ---------------------------------------------------------------------------

@app.get("/")
def serve_dashboard():
    return FileResponse("index.html")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

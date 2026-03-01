# app.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import uvicorn
import logging
import json
import os
from pathlib import Path
from modules.data_fetcher import get_all_snapshots, get_price_series, TICKERS
from modules.predictor import generate_prediction
from modules.edge_detector import run_all_edge_detection
from modules.position_tracker import update_all_positions, get_all_positions, get_pnl_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Absolute path to current directory
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.html"

app = FastAPI(title="Quant-Stock Live Dashboard")

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
    "last_updated": None,
    "status": "initializing"
}

def refresh_data():
    global _cache
    logger.info("Scanning market for edges...")
    try:
        snapshots = get_all_snapshots()
        predictions = []
        for snap in snapshots:
            if "error" not in snap:
                pred = generate_prediction(snap, N_paths=15_000)
                predictions.append(pred)
        
        edges = run_all_edge_detection(snapshots, predictions)
        update_all_positions(snapshots, edges)
        
        _cache.update({
            "snapshots": snapshots,
            "predictions": predictions,
            "edges": edges,
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "status": "ok"
        })
        logger.info("Market scan complete.")
    except Exception as e:
        _cache["status"] = f"error: {str(e)}"
        logger.error(f"Scan failed: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(refresh_data, "interval", minutes=1, id="market_scanner")

@app.on_event("startup")
def startup_event():
    refresh_data()
    scheduler.start()

@app.get("/api/full")
def api_full():
    summary = get_pnl_summary()
    return JSONResponse({
        "snapshots": _cache["snapshots"],
        "predictions": _cache["predictions"],
        "edges": _cache["edges"],
        "portfolio": summary,
        "positions": get_all_positions(),
        "last_updated": _cache["last_updated"],
        "status": _cache["status"]
    })

@app.get("/")
def serve_dashboard():
    if not INDEX_PATH.exists():
        return JSONResponse({"error": f"index.html not found at {INDEX_PATH}. Current directory: {os.getcwd()}"}, status_code=404)
    return FileResponse(str(INDEX_PATH))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

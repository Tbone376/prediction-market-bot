# app.py
import sys
import os
import time
from pathlib import Path

# Absolute path to current directory
BASE_DIR = Path(__file__).resolve().parent

# SAFETY WRAPPER: This will print errors and WAIT so you can read them
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
    from modules.position_tracker import update_all_positions, get_all_positions, get_pnl_summary
    
except Exception as e:
    print("
" + "!"*60)
    print("CRITICAL STARTUP ERROR")
    print("!"*60)
    print(f"
ERROR DETAILS:
{e}")
    print("
FIX STEPS:")
    print("1. Make sure you are in the correct folder")
    print("2. Run: pip install -r requirements.txt")
    print("
This window will stay open for 2 minutes so you can read the error.")
    time.sleep(120)
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

# Create data folder automatically
os.makedirs(BASE_DIR / "data", exist_ok=True)

app = FastAPI(title="Quant-Stock Live Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache = {"snapshots": [], "predictions": [], "edges": [], "last_updated": None, "status": "initializing"}

def refresh_data():
    global _cache
    logger.info("Market scan starting...")
    try:
        snapshots = get_all_snapshots()
        predictions = [generate_prediction(s, N_paths=10000) for s in snapshots if "error" not in s]
        edges = run_all_edge_detection(snapshots, predictions)
        update_all_positions(snapshots, edges)
        
        _cache.update({
            "snapshots": snapshots, "predictions": predictions, "edges": edges,
            "last_updated": datetime.utcnow().isoformat() + "Z", "status": "ok"
        })
        logger.info("Market scan complete.")
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
    return JSONResponse({
        "snapshots": _cache["snapshots"], "edges": _cache["edges"],
        "portfolio": get_pnl_summary(), "positions": get_all_positions(),
        "last_updated": _cache["last_updated"], "status": _cache["status"]
    })

@app.get("/")
def serve_dashboard():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"error": f"index.html missing at {index_file}"}, status_code=404)
    return FileResponse(str(index_file))

if __name__ == "__main__":
    try:
        print("
--- STARTING LIVE DASHBOARD SERVER ---")
        print(f"Working Directory: {BASE_DIR}")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"
SERVER CRASHED: {e}")
        print("
Window staying open for 2 minutes...")
        time.sleep(120)

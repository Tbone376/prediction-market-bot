# app.py
import sys
import os
import time
from pathlib import Path

# Absolute path to current directory
BASE_DIR = Path(__file__).resolve().parent

# SAFETY WRAPPER: This will print errors and WAIT so you can read them
try:
    from fastapi import FastAPI# app.pyctrl+# app.py
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
    from modules.position_tracker import update_all_positions, get_all_positions, get_pnl_summary, init_db as init_pos_db
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
        update_all_positions(snapshots, edges)
        
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
    return JSONResponse({
        "snapshots": _cache["snapshots"],
        "predictions": _cache["predictions"],
        "edges": _cache["edges"],
        "brier_scores": _cache["brier_scores"],
        "portfolio": get_pnl_summary(),
        "positions": get_all_positions(),
        "status": _cache["status"],
        "last_updated": _cache["last_updated"]
    })

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
    from modules.position_tracker import update_all_positions, get_all_positions, get_pnl_summary, init_db as init_pos_db
    from modules.brier_tracker import log_prediction, resolve_predictions, get_all_brier_scores, init_db as init_brier_db

except Exception as e:
    print("--- STARTUP ERROR ---")
    print(str(e))
    print("Run: pip install -r requirements.txt")
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
        for i, snap in enumerate(snapshots):
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
        update_all_positions(snapshots, edges)
        
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
    return JSONResponse({
        "snapshots": _cache["snapshots"],
        "predictions": _cache["predictions"],
        "edges": _cache["edges"],
        "brier_scores": _cache["brier_scores"],
        "portfolio": get_pnl_summary(),
        "positions": get_all_positions(),
        "status": _cache["status"],
        "last_updated": _cache["last_updated"]
    })

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
ctrl+, HTTPException
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

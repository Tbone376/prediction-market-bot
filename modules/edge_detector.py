# modules/edge_detector.py
# Multi-factor edge detection engine for SPX, SPY, AAPL, NVDA, TSLA
# Combines GBM probability, RSI, MACD, Bollinger Band, and Particle Filter signals
# Outputs BUY / SELL / HOLD with composite edge score per ticker

import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# INDIVIDUAL SIGNAL SCORERS  (each returns -1 to +1)
# ---------------------------------------------------------------------------

def rsi_signal(rsi: float) -> tuple:
    """RSI mean-reversion signal. Oversold=bullish, overbought=bearish."""
    if rsi is None:
        return 0.0, "RSI N/A"
    if rsi < 30:
        score = (30 - rsi) / 30          # 0 to 1 as RSI drops below 30
        return min(score, 1.0), f"RSI oversold ({rsi:.1f})"
    elif rsi > 70:
        score = -(rsi - 70) / 30
        return max(score, -1.0), f"RSI overbought ({rsi:.1f})"
    else:
        return 0.0, f"RSI neutral ({rsi:.1f})"


def macd_signal(macd_hist: float, macd: float, macd_sig: float) -> tuple:
    """MACD histogram momentum signal."""
    if macd_hist is None:
        return 0.0, "MACD N/A"
    # Normalise by absolute MACD value to get -1..1
    scale = abs(macd) if macd and abs(macd) > 1e-6 else 1.0
    score = np.clip(macd_hist / scale, -1.0, 1.0)
    direction = "bullish" if macd_hist > 0 else "bearish"
    crossover = ""
    if macd is not None and macd_sig is not None:
        if macd > macd_sig and macd_hist > 0:
            crossover = " (golden cross)"
        elif macd < macd_sig and macd_hist < 0:
            crossover = " (death cross)"
    return float(score), f"MACD {direction}{crossover} (hist={macd_hist:.3f})"


def bollinger_signal(bb_pct: float) -> tuple:
    """Bollinger Band %B mean-reversion signal."""
    if bb_pct is None:
        return 0.0, "BB N/A"
    # bb_pct=0 => at lower band (oversold), bb_pct=1 => at upper band (overbought)
    if bb_pct < 0.2:
        score = (0.2 - bb_pct) / 0.2
        return min(float(score), 1.0), f"BB oversold (%B={bb_pct:.2f})"
    elif bb_pct > 0.8:
        score = -(bb_pct - 0.8) / 0.2
        return max(float(score), -1.0), f"BB overbought (%B={bb_pct:.2f})"
    else:
        return 0.0, f"BB neutral (%B={bb_pct:.2f})"


def momentum_signal(day_change_pct: float) -> tuple:
    """Short-term price momentum signal."""
    score = np.clip(day_change_pct / 5.0, -1.0, 1.0)
    direction = "up" if day_change_pct > 0 else "down"
    return float(score), f"Day momentum {direction} ({day_change_pct:+.2f}%)"


def gbm_signal(p_up: float) -> tuple:
    """Convert 1-day GBM P(up) to a -1..+1 score."""
    # p_up=0.5 is neutral; scale so 0.6 => +0.2, 0.7 => +0.4, etc.
    score = np.clip((p_up - 0.5) * 4.0, -1.0, 1.0)
    return float(score), f"GBM P(up 1d)={p_up:.3f}"


def pf_signal(divergence: float, current_price: float) -> tuple:
    """Particle filter fair-value divergence signal."""
    if current_price == 0:
        return 0.0, "PF N/A"
    pct_divergence = divergence / current_price
    score = np.clip(pct_divergence * 20.0, -1.0, 1.0)  # ±5% => ±1
    direction = "above" if divergence > 0 else "below"
    return float(score), f"PF fair value {direction} market ({divergence:+.2f})"


# ---------------------------------------------------------------------------
# COMPOSITE EDGE SCORER
# ---------------------------------------------------------------------------

# Signal weights (must sum to 1.0)
WEIGHTS = {
    "gbm":      0.25,
    "pf":       0.20,
    "rsi":      0.20,
    "macd":     0.20,
    "bollinger": 0.10,
    "momentum": 0.05,
}

EDGE_THRESHOLD_BUY  = 0.15
EDGE_THRESHOLD_SELL = -0.15


def compute_edge(snapshot: dict, prediction: dict) -> dict:
    """
    Compute composite edge score and BUY/SELL/HOLD signal for one ticker.

    snapshot: output of data_fetcher.get_ticker_snapshot()
    prediction: output of predictor.generate_prediction()
    """
    ticker = snapshot["ticker"]
    current_price = snapshot["current_price"]

    # --- Individual signals ---
    s_rsi,  r_rsi  = rsi_signal(snapshot.get("rsi_14"))
    s_macd, r_macd = macd_signal(
        snapshot.get("macd_hist"),
        snapshot.get("macd"),
        snapshot.get("macd_signal")
    )
    s_bb,   r_bb   = bollinger_signal(snapshot.get("bb_pct"))
    s_mom,  r_mom  = momentum_signal(snapshot.get("day_change_pct", 0))

    p_up_1d = prediction["predictions"]["1d"]["p_up"]
    s_gbm,  r_gbm  = gbm_signal(p_up_1d)

    pf_div = prediction["particle_filter"]["divergence_from_market"]
    s_pf,   r_pf   = pf_signal(pf_div, current_price)

    # --- Composite score ---
    composite = (
        WEIGHTS["gbm"]       * s_gbm  +
        WEIGHTS["pf"]        * s_pf   +
        WEIGHTS["rsi"]       * s_rsi  +
        WEIGHTS["macd"]      * s_macd +
        WEIGHTS["bollinger"] * s_bb   +
        WEIGHTS["momentum"]  * s_mom
    )

    # --- Signal classification ---
    if composite >= EDGE_THRESHOLD_BUY:
        signal = "BUY"
        signal_strength = "STRONG" if composite >= 0.35 else "MODERATE"
    elif composite <= EDGE_THRESHOLD_SELL:
        signal = "SELL"
        signal_strength = "STRONG" if composite <= -0.35 else "MODERATE"
    else:
        signal = "HOLD"
        signal_strength = "NEUTRAL"

    return {
        "ticker": ticker,
        "current_price": current_price,
        "signal": signal,
        "signal_strength": signal_strength,
        "composite_score": round(composite, 4),
        "signals": {
            "gbm":       {"score": round(s_gbm, 3),  "reason": r_gbm},
            "particle_filter": {"score": round(s_pf, 3), "reason": r_pf},
            "rsi":       {"score": round(s_rsi, 3),  "reason": r_rsi},
            "macd":      {"score": round(s_macd, 3), "reason": r_macd},
            "bollinger": {"score": round(s_bb, 3),   "reason": r_bb},
            "momentum":  {"score": round(s_mom, 3),  "reason": r_mom},
        },
        "prediction_1d": prediction["predictions"]["1d"],
        "prediction_5d": prediction["predictions"]["5d"],
        "prediction_30d": prediction["predictions"]["30d"],
        "rsi_14": snapshot.get("rsi_14"),
        "macd_hist": snapshot.get("macd_hist"),
        "bb_pct": snapshot.get("bb_pct"),
        "sigma_20d": snapshot.get("sigma_20d"),
        "day_change_pct": snapshot.get("day_change_pct"),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }


def run_all_edge_detection(snapshots: list, predictions: list) -> list:
    """Run edge detection across all tickers. Returns list of edge dicts."""
    snap_map = {s["ticker"]: s for s in snapshots if "error" not in s}
    pred_map = {p["ticker"]: p for p in predictions if "error" not in p}
    results = []
    for ticker in snap_map:
        if ticker in pred_map:
            try:
                edge = compute_edge(snap_map[ticker], pred_map[ticker])
                results.append(edge)
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})
    return results

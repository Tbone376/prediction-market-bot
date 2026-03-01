# modules/polymarket.py
# Fetches live Polymarket prediction markets related to our tracked tickers,
# extracts implied probabilities from contract prices, and compares them to
# model-generated probabilities to surface mispriced contracts.

import requests
import json
import re
from datetime import datetime, timezone

GAMMA_API = "https://gamma-api.polymarket.com"

# Search queries that reliably surface stock-related markets for each ticker
TICKER_SEARCH_QUERIES = {
    "SPX": ["S&P 500 SPX close", "SPX above", "SPX week"],
    "SPY": ["SPY close", "SPY above", "SPY week"],
    "AAPL": ["AAPL close", "AAPL above", "AAPL week"],
    "NVDA": ["NVDA close", "NVDA above", "NVDA week"],
    "TSLA": ["TSLA close", "TSLA above", "TSLA week"],
}

# Map ticker names to patterns in Polymarket titles
TICKER_PATTERNS = {
    "SPX": [r"S&P\s*500", r"SPX"],
    "SPY": [r"\bSPY\b"],
    "AAPL": [r"\bAAPL\b", r"\bApple\b"],
    "NVDA": [r"\bNVDA\b", r"\bNVIDIA\b"],
    "TSLA": [r"\bTSLA\b", r"\bTesla\b"],
}


def _match_ticker(title: str) -> str | None:
    """Identify which tracked ticker a market title refers to."""
    for ticker, patterns in TICKER_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, title, re.IGNORECASE):
                return ticker
    return None


def _parse_price_level(question: str) -> dict:
    """
    Extract the price level and direction from a Polymarket question.
    Examples:
      "Will NVIDIA (NVDA) close above $180 on February 23?" -> {level: 180, direction: 'above'}
      "Will AAPL close at $240-$245 ...?" -> {level: 242.5, direction: 'range', low: 240, high: 245}
      "Will AAPL close at <$240 ...?" -> {level: 240, direction: 'below'}
    """
    # "close above $X" or "finish week above $X" pattern
    above = re.search(r"(?:close|finish)\s+(?:[\w\s]+?)?above\s+\$?([\d,]+(?:\.\d+)?)", question, re.IGNORECASE)
    if above:
        level = float(above.group(1).replace(",", ""))
        return {"level": level, "direction": "above"}

    # "close over $X" pattern
    over = re.search(r"close\s+over\s+\$?([\d,]+(?:\.\d+)?)", question, re.IGNORECASE)
    if over:
        level = float(over.group(1).replace(",", ""))
        return {"level": level, "direction": "above"}

    # "close at <$X" pattern
    below = re.search(r"close\s+at\s+<\s*\$?([\d,]+(?:\.\d+)?)", question, re.IGNORECASE)
    if below:
        level = float(below.group(1).replace(",", ""))
        return {"level": level, "direction": "below"}

    # "close at >$X" pattern
    above_gt = re.search(r"close\s+at\s+>\s*\$?([\d,]+(?:\.\d+)?)", question, re.IGNORECASE)
    if above_gt:
        level = float(above_gt.group(1).replace(",", ""))
        return {"level": level, "direction": "above"}

    # "close at $X-$Y" range pattern
    rng = re.search(r"close\s+at\s+\$?([\d,]+(?:\.\d+)?)\s*[-\u2013]\s*\$?([\d,]+(?:\.\d+)?)", question, re.IGNORECASE)
    if rng:
        low = float(rng.group(1).replace(",", ""))
        high = float(rng.group(2).replace(",", ""))
        return {"level": (low + high) / 2, "direction": "range", "low": low, "high": high}

    return {"level": None, "direction": "unknown"}


def fetch_polymarket_events(ticker: str) -> list:
    """
    Search Polymarket for active, open events related to a ticker.
    Returns list of event dicts from the Gamma API.
    """
    queries = TICKER_SEARCH_QUERIES.get(ticker, [ticker])
    seen_ids = set()
    events = []

    for q in queries:
        try:
            resp = requests.get(
                f"{GAMMA_API}/public-search",
                params={"q": q, "limit": 10},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for ev in data.get("events", []):
                eid = ev.get("id")
                if eid in seen_ids:
                    continue
                # Only include active, non-closed events
                if ev.get("active") and not ev.get("closed"):
                    seen_ids.add(eid)
                    events.append(ev)
        except Exception:
            continue

    return events


def extract_contracts(events: list, ticker: str) -> list:
    """
    From a list of Polymarket events, extract individual market contracts
    with their implied probabilities and price levels.

    Returns list of contract dicts ready for comparison with our model.
    """
    contracts = []
    now = datetime.now(timezone.utc)

    for ev in events:
        event_title = ev.get("title", "")
        event_slug = ev.get("slug", "")
        end_date = ev.get("endDate", "")

        for mkt in ev.get("markets", []):
            # Skip closed markets
            if mkt.get("closed"):
                continue
            if not mkt.get("enableOrderBook"):
                continue

            question = mkt.get("question", "")
            matched_ticker = _match_ticker(question) or _match_ticker(event_title)

            # Must match our ticker
            if matched_ticker != ticker:
                continue

            # Parse outcomes and prices
            try:
                outcomes = json.loads(mkt.get("outcomes", "[]"))
                prices = json.loads(mkt.get("outcomePrices", "[]"))
            except (json.JSONDecodeError, TypeError):
                continue

            if len(outcomes) < 2 or len(prices) < 2:
                continue

            # "Yes" price = implied probability of the event happening
            yes_price = float(prices[0])
            no_price = float(prices[1])

            # Parse the price level from the question
            price_info = _parse_price_level(question)

            # Parse end date
            mkt_end = mkt.get("endDate", end_date)
            try:
                end_dt = datetime.fromisoformat(mkt_end.replace("Z", "+00:00"))
                days_to_expiry = max((end_dt - now).days, 0)
            except (ValueError, TypeError):
                days_to_expiry = None

            contracts.append({
                "ticker": ticker,
                "event_title": event_title,
                "event_slug": event_slug,
                "question": question,
                "market_slug": mkt.get("slug", ""),
                "market_id": mkt.get("id", ""),
                "condition_id": mkt.get("conditionId", ""),
                "implied_prob_yes": round(yes_price, 4),
                "implied_prob_no": round(no_price, 4),
                "price_level": price_info.get("level"),
                "direction": price_info.get("direction", "unknown"),
                "range_low": price_info.get("low"),
                "range_high": price_info.get("high"),
                "volume": float(mkt.get("volume", 0) or 0),
                "volume_24hr": float(mkt.get("volume24hr", 0) or 0),
                "end_date": mkt_end,
                "days_to_expiry": days_to_expiry,
                "polymarket_url": f"https://polymarket.com/event/{event_slug}",
                "last_trade_price": float(mkt.get("lastTradePrice", 0) or 0),
            })

    # Sort by volume (most liquid first)
    contracts.sort(key=lambda c: c["volume_24hr"], reverse=True)
    return contracts


def compute_model_vs_market(
    contracts: list,
    current_price: float,
    model_p_up_1d: float,
    model_p_up_5d: float,
    sigma_20d: float,
    mu_annualised: float,
) -> list:
    """
    For each contract, estimate our model's implied probability and
    compare it to the market's. Returns contracts enriched with:
      - model_prob: our estimated probability
      - edge: model_prob - market_prob (positive = we think YES is underpriced)
      - edge_pct: edge as percentage
    """
    import numpy as np
    from scipy.stats import norm

    enriched = []
    for c in contracts:
        level = c.get("price_level")
        direction = c.get("direction")

        if level is None or current_price == 0:
            c["model_prob"] = None
            c["edge"] = None
            c["edge_pct"] = None
            enriched.append(c)
            continue

        # Use GBM to estimate P(close > level) at expiry
        days = c.get("days_to_expiry")
        if days is None or days <= 0:
            days = 1

        # Cap at reasonable horizons
        T = min(days, 365) / 252.0  # annualised time
        sigma = sigma_20d if sigma_20d > 0 else 0.25
        mu = mu_annualised

        # GBM: log(S_T/S_0) ~ N((mu - sigma^2/2)*T, sigma^2*T)
        drift = (mu - 0.5 * sigma ** 2) * T
        vol = sigma * np.sqrt(T)

        if direction == "above":
            # P(S_T > level) = P(Z > (ln(level/S_0) - drift) / vol)
            if level > 0:
                d = (np.log(level / current_price) - drift) / vol if vol > 0 else 0
                model_prob = float(1 - norm.cdf(d))
            else:
                model_prob = 1.0

        elif direction == "below":
            # P(S_T < level)
            if level > 0:
                d = (np.log(level / current_price) - drift) / vol if vol > 0 else 0
                model_prob = float(norm.cdf(d))
            else:
                model_prob = 0.0

        elif direction == "range":
            low = c.get("range_low", level)
            high = c.get("range_high", level)
            if low and high and low > 0 and high > 0:
                d_low = (np.log(low / current_price) - drift) / vol if vol > 0 else 0
                d_high = (np.log(high / current_price) - drift) / vol if vol > 0 else 0
                model_prob = float(norm.cdf(d_high) - norm.cdf(d_low))
            else:
                model_prob = None
        else:
            model_prob = None

        if model_prob is not None:
            market_prob = c["implied_prob_yes"]
            edge = round(model_prob - market_prob, 4)
            edge_pct = round(edge * 100, 2)
            c["model_prob"] = round(model_prob, 4)
            c["edge"] = edge
            c["edge_pct"] = edge_pct
        else:
            c["model_prob"] = None
            c["edge"] = None
            c["edge_pct"] = None

        enriched.append(c)

    # Sort by absolute edge (biggest mispricings first)
    enriched.sort(key=lambda c: abs(c.get("edge") or 0), reverse=True)
    return enriched


def get_all_polymarket_signals(snapshots: list, predictions: list) -> list:
    """
    Master function: for each tracked ticker, fetch Polymarket contracts
    and compute model vs market edge.

    Args:
        snapshots: list of ticker snapshot dicts from data_fetcher
        predictions: list of prediction dicts from predictor

    Returns:
        list of enriched contract dicts with edge calculations
    """
    all_contracts = []

    for snap in snapshots:
        if "error" in snap:
            continue

        ticker = snap["ticker"]
        current_price = snap["current_price"]
        sigma = snap.get("sigma_20d", 0.25)
        mu = snap.get("mu_annualised", 0.08)

        # Get model probabilities
        pred = next((p for p in predictions if p["ticker"] == ticker), None)
        p_up_1d = pred["predictions"]["1d"]["p_up"] if pred else 0.5
        p_up_5d = pred["predictions"]["5d"]["p_up"] if pred else 0.5

        # Fetch and process Polymarket contracts
        try:
            events = fetch_polymarket_events(ticker)
            contracts = extract_contracts(events, ticker)
            enriched = compute_model_vs_market(
                contracts, current_price, p_up_1d, p_up_5d, sigma, mu
            )
            all_contracts.extend(enriched)
        except Exception as e:
            all_contracts.append({
                "ticker": ticker,
                "error": str(e),
            })

    return all_contracts

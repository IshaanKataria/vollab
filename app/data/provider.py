"""
Options chain data provider.

Fetches live data from yfinance with fallback to pre-cached JSON snapshots.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

from .cache import TTLCache

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
_cache = TTLCache()
_LIVE_TTL = 900  # 15 minutes


def get_cached_snapshot(ticker: str) -> dict | None:
    path = SNAPSHOTS_DIR / f"{ticker.upper()}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fetch_options_chain(ticker: str) -> dict[str, Any]:
    """
    Fetch full options chain for a ticker.
    Returns dict with keys: ticker, underlying_price, expiries, chains, fetched_at, source.
    """
    ticker_upper = ticker.upper()

    cached = _cache.get(f"chain:{ticker_upper}")
    if cached is not None:
        return cached

    try:
        data = _fetch_live(ticker_upper)
        _cache.set(f"chain:{ticker_upper}", data, _LIVE_TTL)
        return data
    except Exception:
        snapshot = get_cached_snapshot(ticker_upper)
        if snapshot is not None:
            snapshot["source"] = "cached"
            return snapshot
        raise ValueError(f"Ticker not found: {ticker_upper}")


def _fetch_live(ticker: str) -> dict[str, Any]:
    tk = yf.Ticker(ticker)
    info = tk.info
    underlying_price = info.get("regularMarketPrice") or info.get("previousClose")
    if underlying_price is None:
        raise ValueError(f"No price data for {ticker}")

    expiries = list(tk.options)
    if not expiries:
        raise ValueError(f"No options data for {ticker}")

    chains = {}
    for exp in expiries:
        opt = tk.option_chain(exp)
        chains[exp] = {
            "calls": _df_to_records(opt.calls),
            "puts": _df_to_records(opt.puts),
        }

    return {
        "ticker": ticker,
        "underlying_price": float(underlying_price),
        "expiries": expiries,
        "chains": chains,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "live",
    }


def _df_to_records(df) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        records.append({
            "strike": float(row.get("strike", 0)),
            "lastPrice": float(row.get("lastPrice", 0)),
            "bid": float(row.get("bid", 0)),
            "ask": float(row.get("ask", 0)),
            "volume": int(row.get("volume", 0) or 0),
            "openInterest": int(row.get("openInterest", 0) or 0),
            "impliedVolatility": float(row.get("impliedVolatility", 0)),
            "inTheMoney": bool(row.get("inTheMoney", False)),
        })
    return records

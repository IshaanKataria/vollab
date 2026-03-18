"""
CLI script to refresh pre-cached options chain snapshots.

Usage: uv run refresh-snapshots
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
DEFAULT_TICKERS = ["SPY", "AAPL", "TSLA", "NVDA", "MSFT", "QQQ"]


def fetch_snapshot(ticker: str) -> dict:
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
        "source": "snapshot",
    }


def _df_to_records(df) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        records.append({
            "strike": _safe_float(row.get("strike", 0)),
            "lastPrice": _safe_float(row.get("lastPrice", 0)),
            "bid": _safe_float(row.get("bid", 0)),
            "ask": _safe_float(row.get("ask", 0)),
            "volume": _safe_int(row.get("volume", 0)),
            "openInterest": _safe_int(row.get("openInterest", 0)),
            "impliedVolatility": _safe_float(row.get("impliedVolatility", 0)),
            "inTheMoney": bool(row.get("inTheMoney", False)),
        })
    return records


def _safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v) -> int:
    try:
        f = float(v)
        return 0 if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0


def main():
    parser = argparse.ArgumentParser(description="Refresh pre-cached options chain snapshots")
    parser.add_argument(
        "tickers", nargs="*", default=DEFAULT_TICKERS,
        help=f"Tickers to refresh (default: {' '.join(DEFAULT_TICKERS)})",
    )
    args = parser.parse_args()

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in args.tickers:
        ticker = ticker.upper()
        print(f"Fetching {ticker}...", end=" ", flush=True)
        try:
            data = fetch_snapshot(ticker)
            path = SNAPSHOTS_DIR / f"{ticker}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            n_expiries = len(data["expiries"])
            n_calls = sum(len(c["calls"]) for c in data["chains"].values())
            n_puts = sum(len(c["puts"]) for c in data["chains"].values())
            print(f"OK — {n_expiries} expiries, {n_calls} calls, {n_puts} puts")
        except Exception as e:
            print(f"FAILED — {e}", file=sys.stderr)

    print(f"\nSnapshots saved to {SNAPSHOTS_DIR}")


if __name__ == "__main__":
    main()

"""
IV surface extraction pipeline.

Takes raw options chain data, extracts implied volatilities via Newton-Raphson,
filters noisy data, computes log-moneyness, and prepares structured slice data
for SVI fitting and 3D plotting.
"""

import math
from datetime import datetime

import numpy as np

from . import implied_vol

DEFAULT_RATE = 0.045
MIN_VOLUME = 1
MIN_OI = 10
MAX_SPREAD_RATIO = 0.50  # exclude if spread > 50% of mid
MIN_POINTS_PER_SLICE = 5
MAX_EXPIRIES = 8  # limit to nearest N expiries for performance


def extract_iv_surface(
    chain_data: dict,
    r: float = DEFAULT_RATE,
    max_expiries: int = MAX_EXPIRIES,
) -> dict:
    """
    Extract IV surface from options chain data.

    Returns dict with:
      raw_points: list of {k, T, iv, strike, expiry} for 3D scatter
      slices: list of {expiry, T, k, market_iv} for SVI fitting
      underlying_price: float
      ticker: str
    """
    S = chain_data["underlying_price"]
    expiries = chain_data["expiries"]
    chains = chain_data["chains"]

    # Filter to future expiries, limit count
    now = datetime.now()
    future_expiries = []
    for exp in expiries:
        T = _years_to_expiry(exp, now)
        if T > 1 / 365.25:  # at least 1 day out
            future_expiries.append((exp, T))
    future_expiries.sort(key=lambda x: x[1])
    future_expiries = future_expiries[:max_expiries]

    raw_points = []
    slices = []

    for exp, T in future_expiries:
        chain = chains.get(exp, {})
        calls = chain.get("calls", [])
        puts = chain.get("puts", [])

        # Use OTM options: calls for K > S, puts for K < S (better liquidity)
        options = []
        for c in calls:
            if c["strike"] >= S:
                options.append(("call", c))
        for p in puts:
            if p["strike"] < S:
                options.append(("put", p))

        k_vals = []
        iv_vals = []

        for opt_type, opt in options:
            strike = opt["strike"]
            if strike <= 0:
                continue

            # Filter by liquidity
            if opt.get("volume", 0) < MIN_VOLUME and opt.get("openInterest", 0) < MIN_OI:
                continue

            # Mid price
            bid = opt.get("bid", 0) or 0
            ask = opt.get("ask", 0) or 0
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread = ask - bid
                if spread / mid > MAX_SPREAD_RATIO:
                    continue
            else:
                mid = opt.get("lastPrice", 0) or 0

            if mid <= 0:
                continue

            # Compute IV
            iv_result = implied_vol.solve(mid, S, strike, T, r, opt_type)
            if not iv_result.converged:
                continue
            if iv_result.sigma < 0.01 or iv_result.sigma > 3.0:
                continue

            # Forward price for log-moneyness
            F = S * math.exp(r * T)
            k = math.log(strike / F)

            k_vals.append(k)
            iv_vals.append(iv_result.sigma)
            raw_points.append({
                "k": k,
                "T": T,
                "iv": iv_result.sigma,
                "strike": strike,
                "expiry": exp,
            })

        if len(k_vals) >= MIN_POINTS_PER_SLICE:
            slices.append({
                "expiry": exp,
                "T": T,
                "k": k_vals,
                "market_iv": iv_vals,
            })

    return {
        "raw_points": raw_points,
        "slices": slices,
        "underlying_price": S,
        "ticker": chain_data["ticker"],
    }


def _years_to_expiry(expiry_str: str, now: datetime | None = None) -> float:
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        if now is None:
            now = datetime.now()
        delta = (exp_date - now).total_seconds()
        return max(delta / (365.25 * 86400), 0.0)
    except (ValueError, TypeError):
        return 0.0

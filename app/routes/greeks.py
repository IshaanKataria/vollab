import json
import math

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.data.provider import fetch_options_chain
from app.engine import black_scholes as bs
from app.engine import greeks_fd as fd
from app.engine.heston import HestonParams, price as heston_price
from app.engine.implied_vol import solve as iv_solve

router = APIRouter()

DEFAULT_RATE = 0.045
# Default Heston params (reasonable equity market values)
DEFAULT_HESTON = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


def _bs_price_fn(S, K, T, r, sigma, option_type):
    result = bs.price(S, K, T, r, sigma)
    return result.call_price if option_type == "call" else result.put_price


def _heston_price_fn(S, K, T, r, params, option_type):
    result = heston_price(S, K, T, r, params)
    return result.call_price if option_type == "call" else result.put_price


@router.get("/greeks/{ticker}", response_class=HTMLResponse)
async def greeks_page(
    request: Request,
    ticker: str,
    strike: float = Query(0),
    expiry: str = Query(""),
    sigma: float = Query(0),
):
    templates = request.app.state.templates
    ticker = ticker.strip().upper()

    try:
        chain_data = fetch_options_chain(ticker)
    except ValueError as e:
        cached = request.app.state.cached_tickers
        suggestion = f" Try one of: {', '.join(cached)}" if cached else ""
        return templates.TemplateResponse("greeks.html", {
            "request": request, "ticker": ticker,
            "error": f"{e}{suggestion}",
        })

    S = chain_data["underlying_price"]
    expiries = chain_data["expiries"]

    # Auto-select: use provided params or pick near-ATM option from first future expiry
    from datetime import datetime
    now = datetime.now()
    future_expiries = [e for e in expiries if e > now.strftime("%Y-%m-%d")]

    if not strike or not expiry:
        if not future_expiries:
            return templates.TemplateResponse("greeks.html", {
                "request": request, "ticker": ticker,
                "error": "No future expiries available.",
            })
        expiry = future_expiries[min(1, len(future_expiries) - 1)]  # 2nd nearest
        chain = chain_data["chains"].get(expiry, {})
        calls = chain.get("calls", [])
        # Find near-ATM call
        best = min(calls, key=lambda c: abs(c["strike"] - S)) if calls else None
        if best:
            strike = best["strike"]
            mid = ((best.get("bid", 0) or 0) + (best.get("ask", 0) or 0)) / 2
            if mid > 0:
                exp_date = datetime.strptime(expiry, "%Y-%m-%d")
                T = max((exp_date - now).total_seconds() / (365.25 * 86400), 1 / 365.25)
                iv_result = iv_solve(mid, S, strike, T, DEFAULT_RATE, "call")
                sigma = iv_result.sigma if iv_result.converged else 0.20
            else:
                sigma = 0.20
        else:
            strike = round(S)
            sigma = 0.20

    # Compute T
    try:
        exp_date = datetime.strptime(expiry, "%Y-%m-%d")
        T = max((exp_date - now).total_seconds() / (365.25 * 86400), 1 / 365.25)
    except (ValueError, TypeError):
        T = 0.25

    r = DEFAULT_RATE
    K = strike

    # 1. Analytical BS Greeks
    bs_greeks = bs.greeks(S, K, T, r, sigma, "call")
    bs_greeks_put = bs.greeks(S, K, T, r, sigma, "put")

    # 2. FD BS Greeks
    fd_bs_call = fd.compute(_bs_price_fn, S, K, T, r, sigma, "call")
    fd_bs_put = fd.compute(_bs_price_fn, S, K, T, r, sigma, "put")

    # 3. FD Heston Greeks
    fd_heston_call = fd.compute(_heston_price_fn, S, K, T, r, DEFAULT_HESTON, "call")
    fd_heston_put = fd.compute(_heston_price_fn, S, K, T, r, DEFAULT_HESTON, "put")

    # 4. Error analysis: |analytical - FD| vs h for Delta and Gamma
    delta_err = fd.error_vs_step_size(bs_greeks.delta, _bs_price_fn, S, K, T, r, sigma, "call", "delta")
    gamma_err = fd.error_vs_step_size(bs_greeks.gamma, _bs_price_fn, S, K, T, r, sigma, "call", "gamma")

    # 5. Sensitivity: Delta vs S
    S_range = [S * (0.8 + 0.02 * i) for i in range(21)]
    delta_vs_S = [bs.greeks(s, K, T, r, sigma, "call").delta for s in S_range]
    gamma_vs_S = [bs.greeks(s, K, T, r, sigma, "call").gamma for s in S_range]
    heston_delta_vs_S = []
    for s in S_range:
        try:
            g = fd.compute(_heston_price_fn, s, K, T, r, DEFAULT_HESTON, "call", h_S=0.01 * s)
            heston_delta_vs_S.append(g.delta)
        except Exception:
            heston_delta_vs_S.append(None)

    greeks_table = _build_greeks_table(bs_greeks, fd_bs_call, fd_heston_call, "call")
    greeks_table_put = _build_greeks_table(bs_greeks_put, fd_bs_put, fd_heston_put, "put")

    return templates.TemplateResponse("greeks.html", {
        "request": request,
        "ticker": ticker,
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "expiry": expiry,
        "error": None,
        "greeks_table": greeks_table,
        "greeks_table_put": greeks_table_put,
        "delta_err_h": json.dumps(delta_err["h_values"]),
        "delta_err_vals": json.dumps(delta_err["errors"]),
        "gamma_err_h": json.dumps(gamma_err["h_values"]),
        "gamma_err_vals": json.dumps(gamma_err["errors"]),
        "S_range": json.dumps(S_range),
        "delta_vs_S": json.dumps(delta_vs_S),
        "gamma_vs_S": json.dumps(gamma_vs_S),
        "heston_delta_vs_S": json.dumps(heston_delta_vs_S),
        "future_expiries": future_expiries[:8],
        "source": chain_data.get("source", "live"),
    })


def _build_greeks_table(analytical, fd_bs, fd_heston, option_type):
    rows = []
    names = [
        ("Delta", "Δ", "delta"),
        ("Gamma", "Γ", "gamma"),
        ("Theta", "Θ", "theta"),
        ("Vega", "ν", "vega"),
        ("Rho", "ρ", "rho"),
    ]
    for name, symbol, attr in names:
        a_val = getattr(analytical, attr)
        fd_val = getattr(fd_bs, attr)
        h_val = getattr(fd_heston, attr)
        err = abs(a_val - fd_val) if a_val != 0 else 0.0
        rel_err = abs(err / a_val) * 100 if abs(a_val) > 1e-15 else 0.0
        rows.append({
            "name": name,
            "symbol": symbol,
            "analytical": a_val,
            "fd_bs": fd_val,
            "fd_heston": h_val,
            "abs_error": err,
            "rel_error": rel_err,
        })
    return rows

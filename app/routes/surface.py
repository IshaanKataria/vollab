import json

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.data.provider import fetch_options_chain
from app.engine import svi
from app.engine.surface import extract_iv_surface

router = APIRouter()


@router.get("/surface/{ticker}", response_class=HTMLResponse)
async def surface_page(request: Request, ticker: str):
    templates = request.app.state.templates
    ticker = ticker.strip().upper()

    try:
        chain_data = fetch_options_chain(ticker)
    except ValueError as e:
        cached = request.app.state.cached_tickers
        suggestion = f" Try one of: {', '.join(cached)}" if cached else ""
        return templates.TemplateResponse("surface.html", {
            "request": request,
            "ticker": ticker,
            "error": f"{e}{suggestion}",
        })

    # Extract IV surface
    iv_data = extract_iv_surface(chain_data)

    if not iv_data["slices"]:
        return templates.TemplateResponse("surface.html", {
            "request": request,
            "ticker": ticker,
            "error": "Not enough liquid options to build a volatility surface. Try a more liquid ticker.",
        })

    # Fit SVI surface
    svi_surface = svi.fit_surface(iv_data["slices"], n_starts=8)

    # Prepare raw scatter data for Plotly
    raw_k = [p["k"] for p in iv_data["raw_points"]]
    raw_T = [p["T"] for p in iv_data["raw_points"]]
    raw_iv = [p["iv"] * 100 for p in iv_data["raw_points"]]
    raw_labels = [f"K={p['strike']:.0f} {p['expiry']}" for p in iv_data["raw_points"]]

    # Build fitted SVI mesh for 3D surface
    mesh_data = _build_svi_mesh(svi_surface)

    # Per-slice data for 2D smile charts
    slice_charts = []
    for s in svi_surface.slices:
        k_fit = np.linspace(min(s.k_values) - 0.05, max(s.k_values) + 0.05, 80)
        iv_fit = svi.svi_implied_vol(k_fit, s.T, s.params) * 100
        slice_charts.append({
            "expiry": s.expiry,
            "T": f"{s.T:.4f}",
            "market_k": s.k_values,
            "market_iv": [v / s.T * 100 for v in s.market_var],  # back to IV %
            "fitted_k": k_fit.tolist(),
            "fitted_iv": iv_fit.tolist(),
            "rmse": f"{s.rmse:.6f}",
            "params": {
                "a": f"{s.params.a:.6f}",
                "b": f"{s.params.b:.6f}",
                "rho": f"{s.params.rho:.4f}",
                "m": f"{s.params.m:.6f}",
                "sigma": f"{s.params.sigma:.6f}",
            },
            "butterfly_violations": s.butterfly_violations,
            "n_points": s.n_points,
        })

    # Convert market_var back to IV% for slice display
    for sc in slice_charts:
        s_match = [s for s in svi_surface.slices if s.expiry == sc["expiry"]][0]
        sc["market_iv"] = [np.sqrt(w / s_match.T) * 100 for w in s_match.market_var]

    return templates.TemplateResponse("surface.html", {
        "request": request,
        "ticker": ticker,
        "S": chain_data["underlying_price"],
        "source": chain_data.get("source", "live"),
        "fetched_at": chain_data.get("fetched_at", ""),
        "error": None,
        "raw_k": json.dumps(raw_k),
        "raw_T": json.dumps(raw_T),
        "raw_iv": json.dumps(raw_iv),
        "raw_labels": json.dumps(raw_labels),
        "mesh_k": json.dumps(mesh_data["k"]),
        "mesh_T": json.dumps(mesh_data["T"]),
        "mesh_iv": json.dumps(mesh_data["iv"]),
        "slice_charts": json.dumps(slice_charts),
        "total_rmse": f"{svi_surface.total_rmse:.6f}",
        "calendar_violations": svi_surface.calendar_violations,
        "n_slices": len(svi_surface.slices),
        "total_butterfly": sum(s.butterfly_violations for s in svi_surface.slices),
    })


def _build_svi_mesh(surface: svi.SVISurface, k_points: int = 50) -> dict:
    """Build a 2D mesh grid of fitted SVI implied vols for Plotly surface plot."""
    if not surface.slices:
        return {"k": [], "T": [], "iv": []}

    all_k = []
    for s in surface.slices:
        all_k.extend(s.k_values)
    k_min, k_max = min(all_k) - 0.05, max(all_k) + 0.05

    k_grid = np.linspace(k_min, k_max, k_points)
    T_vals = sorted([s.T for s in surface.slices])

    mesh_iv = []
    for T in T_vals:
        s = [sl for sl in surface.slices if sl.T == T][0]
        row = svi.svi_implied_vol(k_grid, T, s.params) * 100
        mesh_iv.append(row.tolist())

    return {
        "k": k_grid.tolist(),
        "T": T_vals,
        "iv": mesh_iv,
    }

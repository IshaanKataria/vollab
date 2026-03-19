import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.data.provider import fetch_options_chain
from app.engine.calibrate import calibrate, prepare_calibration_options
from app.engine.surface import extract_iv_surface

router = APIRouter()

DEFAULT_RATE = 0.045


@router.get("/calibration/{ticker}", response_class=HTMLResponse)
async def calibration_page(request: Request, ticker: str):
    templates = request.app.state.templates
    ticker = ticker.strip().upper()

    try:
        chain_data = fetch_options_chain(ticker)
    except ValueError as e:
        cached = request.app.state.cached_tickers
        suggestion = f" Try one of: {', '.join(cached)}" if cached else ""
        return templates.TemplateResponse("calibration.html", {
            "request": request, "ticker": ticker,
            "error": f"{e}{suggestion}",
        })

    iv_data = extract_iv_surface(chain_data)

    if not iv_data["slices"]:
        return templates.TemplateResponse("calibration.html", {
            "request": request, "ticker": ticker,
            "error": "Not enough liquid options for calibration.",
        })

    S = chain_data["underlying_price"]
    options = prepare_calibration_options(iv_data, S, DEFAULT_RATE, max_options=15)

    if len(options) < 3:
        return templates.TemplateResponse("calibration.html", {
            "request": request, "ticker": ticker,
            "error": "Not enough valid options after filtering. Try a more liquid ticker.",
        })

    result = calibrate(options, S, DEFAULT_RATE, popsize=6, max_de_iter=20)

    # Prepare chart data
    per_opt = result.per_option
    scatter_market = []
    scatter_heston = []
    scatter_bs = []
    residual_k = []
    residual_heston = []
    residual_bs = []
    residual_T = []
    residual_heston_T = []

    for row in per_opt:
        mkt_iv_pct = row["market_iv"] * 100
        k = row["log_moneyness"]
        T = row["T"]

        if row.get("heston_iv") is not None:
            h_iv_pct = row["heston_iv"] * 100
            scatter_market.append(mkt_iv_pct)
            scatter_heston.append(h_iv_pct)
            scatter_bs.append(row["bs_iv"] * 100)
            residual_k.append(k)
            residual_heston.append(row["heston_error"] * 100)
            residual_bs.append(row["bs_error"] * 100)
            residual_T.append(T)
            residual_heston_T.append(row["heston_error"] * 100)

    return templates.TemplateResponse("calibration.html", {
        "request": request,
        "ticker": ticker,
        "S": S,
        "source": chain_data.get("source", "live"),
        "error": None,
        "result": result,
        "params": result.params,
        "scatter_market": json.dumps(scatter_market),
        "scatter_heston": json.dumps(scatter_heston),
        "scatter_bs": json.dumps(scatter_bs),
        "residual_k": json.dumps(residual_k),
        "residual_heston": json.dumps(residual_heston),
        "residual_bs": json.dumps(residual_bs),
        "residual_T": json.dumps(residual_T),
        "residual_heston_T": json.dumps(residual_heston_T),
        "per_option": per_opt,
    })

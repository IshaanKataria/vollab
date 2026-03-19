from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from app.engine import black_scholes as bs
from app.engine.heston import HestonParams, price as heston_price

router = APIRouter(prefix="/pricer")


@router.get("", response_class=HTMLResponse)
async def pricer_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("pricer.html", {"request": request})


@router.post("/compute", response_class=HTMLResponse)
async def compute(
    request: Request,
    S: float = Form(...),
    K: float = Form(...),
    T: float = Form(...),
    r: float = Form(...),
    sigma: float = Form(...),
    option_type: str = Form("call"),
    v0: float = Form(0.04),
    kappa: float = Form(2.0),
    theta: float = Form(0.04),
    xi: float = Form(0.5),
    rho: float = Form(-0.7),
):
    templates = request.app.state.templates
    try:
        bs_result = bs.price(S, K, T, r, sigma)
        greeks = bs.greeks(S, K, T, r, sigma, option_type)

        heston_params = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        heston_result = heston_price(S, K, T, r, heston_params)

        return templates.TemplateResponse("partials/pricer_result.html", {
            "request": request,
            "bs": bs_result,
            "heston": heston_result,
            "greeks": greeks,
            "option_type": option_type,
            "heston_params": heston_params,
            "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
        })
    except ValueError as e:
        return HTMLResponse(
            f'<div class="text-loss text-sm p-4 bg-surface-light border border-red-900 rounded-lg">{e}</div>'
        )

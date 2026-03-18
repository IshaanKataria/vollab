from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from app.engine import black_scholes as bs

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
):
    templates = request.app.state.templates
    try:
        result = bs.price(S, K, T, r, sigma)
        greeks = bs.greeks(S, K, T, r, sigma, option_type)
        return templates.TemplateResponse("partials/pricer_result.html", {
            "request": request,
            "result": result,
            "greeks": greeks,
            "option_type": option_type,
        })
    except ValueError as e:
        return HTMLResponse(
            f'<div class="text-loss text-sm p-4 bg-surface-light border border-red-900 rounded-lg">{e}</div>'
        )

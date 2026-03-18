from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("home.html", {
        "request": request,
        "cached_tickers": request.app.state.cached_tickers,
    })

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes import about, calibration, chain, greeks, home, pricer, surface

SNAPSHOTS_DIR = Path(__file__).parent / "app" / "data" / "snapshots"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cached = {}
    for f in SNAPSHOTS_DIR.glob("*.json"):
        with open(f) as fh:
            cached[f.stem.upper()] = json.load(fh)
    app.state.cached_snapshots = cached
    app.state.cached_tickers = sorted(cached.keys())
    yield


app = FastAPI(title="VolLab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "app" / "static"), name="static")

templates = Jinja2Templates(directory=Path(__file__).parent / "app" / "templates")
app.state.templates = templates

app.include_router(home.router)
app.include_router(pricer.router)
app.include_router(chain.router)
app.include_router(surface.router)
app.include_router(calibration.router)
app.include_router(greeks.router)
app.include_router(about.router)


@app.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "cached_tickers": request.app.state.cached_tickers,
    }

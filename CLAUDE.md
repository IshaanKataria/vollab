VolLab — Options Pricing & Calibration Workbench

Calibrate stochastic volatility models to real market data and show where
Black-Scholes breaks down. Every algorithm implemented from scratch.

Tech Stack
  Single Python service: FastAPI + Jinja2 + HTMX
  Math: NumPy, SciPy (optimize/integrate only — all pricing from scratch)
  Charts: Plotly.js (CDN)
  Styling: Tailwind CSS (CDN)
  Data: yfinance (live) + pre-cached JSON snapshots
  Deploy: Railway (Procfile)

Running Locally
  cd /Users/katana/vollab
  uv sync
  uv run uvicorn main:app --reload   (port 8000)

Project Structure
  main.py                  FastAPI app, route registration, lifespan
  app/engine/              Core math (all from scratch except scipy optimize/integrate)
    black_scholes.py       BS pricing + analytical Greeks
    implied_vol.py         Newton-Raphson IV solver
    svi.py                 SVI parameterization + fitting + arbitrage checks
    heston.py              Heston char. function (corrected) + Fourier pricing
    calibrate.py           Two-stage calibration (DE + L-BFGS-B)
    greeks_fd.py           Finite difference Greeks (model-agnostic)
    surface.py             IV extraction pipeline
  app/routes/              FastAPI route handlers
  app/templates/           Jinja2 + HTMX templates
  app/data/                yfinance provider, cache, pre-cached snapshots
  tests/                   pytest (127+ tests)

Pages
  /                        Landing — ticker input, demo tickers
  /pricer                  BS vs Heston side-by-side
  /chain?ticker=SPY        Options chain with NR implied vols
  /surface/SPY             3D vol surface (raw + SVI fitted)
  /calibration/SPY         Heston calibration to market data
  /greeks/SPY              Analytical vs FD Greeks, error analysis
  /about                   Technical documentation
  /health                  JSON health check

Key Design Decisions
  - All pricing/Greeks from scratch (math.erf for CDF, not scipy.stats)
  - SciPy used only for generic optimization and numerical integration
  - Heston uses corrected "little trap" formulation (Albrecher 2006)
  - SVI chosen over SABR (industry standard for equity options)
  - Pre-cached snapshots ensure demo reliability (yfinance can break)
  - Single service, no separate frontend — math is the star
  - 365 trading days for crypto-style annualization

Tests
  uv run pytest tests/ -q                           (fast, ~0.5s)
  uv run pytest tests/ -q --ignore=tests/test_calibrate.py  (skip slow calibration)
  uv run pytest tests/                              (full, ~5min with calibration)

Refresh Snapshots
  uv run python3 -m app.data.snapshot_refresh SPY AAPL TSLA NVDA MSFT QQQ

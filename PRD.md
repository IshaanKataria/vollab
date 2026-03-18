VolLab — Options Pricing & Calibration Workbench
March 2026 | Personal Project

One-liner: Calibrate stochastic volatility models to real market data and show where Black-Scholes breaks down — every algorithm implemented from scratch.

Interview Narrative
  "I built a calibration workbench that fits stochastic volatility models to real options market data.
  I extract implied volatilities using Newton-Raphson, fit an arbitrage-free SVI surface, then
  calibrate a Heston model via least-squares on the characteristic function. The dashboard shows
  exactly where Black-Scholes assumptions break down — the skew it can't capture, the fat tails
  it misses. Every algorithm is my own implementation. I can walk you through the math."

Tech Stack
  Backend: FastAPI + Jinja2 + HTMX (single Python service, server-rendered)
  Math: NumPy, SciPy (only for optimisation/integration — all pricing/Greeks from scratch)
  Charts: Plotly.js (3D vol surface, sensitivity plots, convergence charts)
  Styling: Tailwind CSS (CDN)
  Data: yfinance (live options chains) + pre-cached snapshots (demo reliability)
  Deployment: Railway (single service)

Architecture Decision Records

  Why single service (FastAPI + Jinja2 + HTMX) over Next.js + FastAPI?
    The interesting part of this project is the math, not the infrastructure. A separate frontend
    adds CORS config, two deploy targets, API client boilerplate, and TypeScript type duplication —
    none of which demonstrates quant skills. HTMX gives interactivity (partial page updates, form
    submissions) without JavaScript framework overhead. Plotly.js handles the charts. One process,
    one deploy, maximum signal-to-noise ratio.

  Why implement pricing from scratch instead of using QuantLib/py_vollib?
    The entire point. Any student can pip install a library. Implementing Black-Scholes, Newton-Raphson
    IV extraction, Heston characteristic function pricing, and SVI fitting from NumPy primitives
    demonstrates that you understand the mathematics — not just the API. SciPy is used only for
    generic optimisation (minimize) and numerical integration (quad) where reimplementing would be
    pointless — the domain-specific math is all hand-rolled.

  Why SVI over SABR for the vol surface?
    Gatheral's SVI (Stochastic Volatility Inspired) parameterization is the industry standard for
    equity index options. It has 5 parameters per slice, is analytically tractable, and has known
    no-arbitrage conditions. SABR is more common in rates/FX. For equity options with real market
    data from Yahoo Finance, SVI is the right choice. Mentioning this tradeoff in an interview
    shows awareness of when each model applies.

  Why Heston over other stochastic vol models?
    Heston is the canonical stochastic volatility model. It has a known characteristic function
    (semi-closed-form via Fourier inversion), which means you can price options without Monte Carlo.
    It captures volatility smile/skew that Black-Scholes cannot. More complex models (SABR, rough
    volatility, local-stochastic vol) exist but Heston is the right level for demonstrating
    understanding without scope explosion. If an interviewer asks "what would you try next?",
    you can discuss these extensions.

  Why pre-cached data alongside live yfinance?
    yfinance is an unofficial Yahoo Finance scraper that breaks periodically. If the demo fails
    during an interview, that's worse than not having the project. Pre-caching full options chain
    snapshots for 5-6 liquid tickers (SPY, AAPL, TSLA, NVDA, MSFT, QQQ) means the demo always
    works. Live data is a bonus, not a dependency. Cached snapshots include timestamp so the UI
    shows "Market data as of [date]" transparently.

  Why Plotly.js over other charting libraries?
    The vol surface is a 3D object. Plotly.js has the best browser-native 3D surface rendering
    with rotation, zoom, and hover tooltips. For 2D charts (Greeks sensitivity, convergence plots),
    Plotly also works fine — no need for a second library. It loads from CDN, no build step needed.

Overview

VolLab is a web-based options analytics workbench focused on model calibration and the gap between
theory and market reality. The core workflow mirrors what quantitative analysts actually do:

  1. Observe market option prices (real options chain data)
  2. Extract implied volatilities (Newton-Raphson on Black-Scholes)
  3. Fit a volatility surface (SVI parameterization with no-arbitrage constraints)
  4. Calibrate a stochastic volatility model (Heston) to the surface
  5. Compare model prices vs market prices — show where each model breaks down
  6. Analyse risk sensitivities (Greeks) using both analytical and numerical methods

Every page maps to a talking point in the interview narrative. The project tells a story
about understanding model assumptions, limitations, and calibration — not just "I can price
an option."

Design Direction

  Clean, technical, data-focused:
  - Dark background (#0f0f0f), light toggle available
  - Monospace font for numbers and formulas (JetBrains Mono)
  - Sans-serif for UI text (Inter)
  - Muted colour palette: white text, subtle borders, accent blue (#3b82f6) for interactive elements
  - Math rendered inline where helpful (formatted as code blocks, not LaTeX — keeps it readable)
  - Dense but not cluttered — every element serves the narrative
  - Mobile: functional but desktop is the primary target (this is a demo tool, not a mobile app)
  - No animations, no gradients, no design fluff. Let the math speak.

Pages and Routes

  /                          Landing — ticker input, project description, pre-cached demo tickers
  /chain/{ticker}            Raw options chain data (strikes, expiries, bid/ask, volume, OI, IV)
  /surface/{ticker}          3D implied volatility surface (raw + SVI fitted)
  /pricer                    Interactive option pricer (BS + Heston, side by side)
  /greeks/{ticker}           Greeks dashboard (analytical vs finite difference comparison)
  /calibration/{ticker}      Heston calibration results (fitted params, error surface, model comparison)
  /about                     Technical documentation — formulas, references, implementation notes
  /health                    JSON health check (for Railway uptime monitoring, not in nav)

Data Strategy

  Pre-cached snapshots (always available):
  - Tickers: SPY, AAPL, TSLA, NVDA, MSFT, QQQ
  - Data per ticker: full options chain (all expiries, all strikes), underlying price, risk-free rate
  - Stored as JSON files in /data/snapshots/
  - Each snapshot timestamped: "Market data as of 2026-03-14 16:00 ET"
  - Refreshed manually (run a script to update snapshots periodically)

  Live data (best-effort via yfinance):
  - User can enter any US equity ticker
  - yfinance fetches current options chain (~15min delay)
  - If yfinance fails: graceful fallback message, suggest using a pre-cached ticker
  - Live data cached in memory for 15 minutes to reduce Yahoo Finance load

  Risk-free rate:
  - Use US 10Y Treasury yield, fetched from FRED with fallback to hardcoded 4.5%
  - Displayed in the UI so the user knows the assumption
  - User-overridable: editable field pre-filled with the fetched/default value

Core Modules (1-6) — The A-Grade Core

Module 1: Black-Scholes Pricer

  Implementation: from scratch using NumPy. No scipy.stats.norm — implement the CDF using
  the error function (math.erf) or a polynomial approximation.

  Inputs:
  - S: underlying price
  - K: strike price
  - T: time to expiry (years)
  - r: risk-free rate
  - sigma: volatility
  - option_type: call or put

  Outputs:
  - Option price (call and put)
  - d1, d2 intermediate values (displayed for transparency)
  - Put-call parity verification

  Features:
  - Interactive form: adjust any input, see price update via HTMX
  - Show the formula alongside the computation (not LaTeX, just clean code-formatted math)
  - Put-call parity check: C - P = S - K*e^(-rT), show the residual
  - Comparison table: your implementation vs py_vollib (validation, not dependency)

  Numerical details:
  - Standard normal CDF via: 0.5 * (1 + erf(x / sqrt(2)))
  - Handle edge cases: T=0 (intrinsic value), sigma=0 (deterministic payoff), deep ITM/OTM

Module 2: Implied Volatility Solver

  Method: Newton-Raphson iteration on Black-Scholes.
  The equation: find sigma such that BS(S, K, T, r, sigma) = market_price.
  Newton step: sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)

  Implementation details:
  - Starting guess: Brenner-Subrahmanyam approximation (sigma_0 ≈ sqrt(2*pi/T) * price/S)
  - Convergence criterion: |BS(sigma_n) - market_price| < 1e-8
  - Max iterations: 100 (should converge in 5-10 for reasonable inputs)
  - Safeguards: clamp sigma to [0.001, 5.0] to prevent numerical blowup
  - Handle no-solution cases: if market price < intrinsic value (arbitrage), return NaN

  UI:
  - Show iteration table: step, sigma estimate, BS price, error, vega
  - Convergence plot (error vs iteration number)
  - Batch mode: extract IV for entire options chain at once (used by surface module)

  Why Newton-Raphson over Brent's method:
  - Newton-Raphson uses vega (the derivative), which you already compute for Greeks
  - Quadratic convergence vs linear — much faster
  - Interview talking point: "I chose Newton-Raphson because I already have the analytical
    derivative (vega), so I get quadratic convergence for free"

Module 3: Volatility Surface (SVI Fitting)

  Step 1 — Raw implied volatility surface:
  - For each (strike, expiry) pair in the options chain, extract IV using Module 2
  - Filter: remove options with zero volume, zero open interest, or bid=0
  - Filter: remove deep OTM options with spread > 50% of mid price
  - Result: a grid of (log-moneyness, time-to-expiry, implied_vol) points

  Step 2 — SVI parameterization (per expiry slice):
  Gatheral's raw SVI formula for total implied variance w(k) where k = log(K/F):
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

  5 parameters per slice: a, b, rho, m, sigma
  - a: overall variance level
  - b: slope of the wings (must be >= 0)
  - rho: skew/asymmetry (-1 < rho < 1)
  - m: horizontal shift
  - sigma: curvature at the money (must be > 0)

  Fitting: scipy.optimize.minimize (L-BFGS-B) on sum of squared errors between
  SVI(k) and market total variance w_market = sigma_iv^2 * T

  Step 3 — No-arbitrage constraints:
  - Calendar spread arbitrage: total variance must be non-decreasing in T at every strike
  - Butterfly arbitrage: the density implied by the surface must be non-negative
    Condition: 1 - (k * w'(k)) / (2 * w(k)) + (w'(k)^2 / 4) * (-1/4 - 1/w(k) + k^2/w(k)^2) + w''(k)/2 >= 0
  - Enforce via constrained optimisation or post-fit validation with warnings
  - Display arbitrage violations in the UI if they exist (shows awareness of the problem)

  UI:
  - 3D Plotly surface: X = log-moneyness, Y = time-to-expiry, Z = implied vol
  - Toggle between raw market IV points (scatter) and fitted SVI surface (mesh)
  - Overlay both: scatter points on top of fitted surface to visualise fit quality
  - Per-slice 2D view: select an expiry, see the smile curve with SVI fit and raw points
  - Fit statistics: RMSE per slice, parameter values, arbitrage violation count

  Technical note (for implementation):
  - Fitting SVI is non-convex. Use multiple random starting points and take the best fit.
  - The no-butterfly-arbitrage condition is the hard part. Even mentioning it in the README
    and showing violations in the UI demonstrates depth beyond naive fitting.

Module 4: Heston Stochastic Volatility Model

  The model:
    dS = r*S*dt + sqrt(v)*S*dW_1
    dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_2
    corr(dW_1, dW_2) = rho

  5 parameters: v0 (initial variance), kappa (mean reversion speed), theta (long-run variance),
  xi (vol of vol), rho (correlation between spot and vol processes)

  Pricing via characteristic function (Fourier inversion):
  - Heston's characteristic function phi(u) has a semi-closed form
  - Option price = discounted integral involving phi(u)
  - Use the "little Heston trap" formulation (Albrecher, Mayer, Schoutens 2006):
    The original Heston formulation has a branch cut discontinuity in the complex logarithm.
    The corrected formulation (rotation of the complex square root) eliminates this.
    This is critical for numerical stability.

  Implementation:
  - Characteristic function: implement both the original and corrected (trap-free) versions
  - Numerical integration: scipy.integrate.quad on the real part of the integrand
  - Gauss-Laguerre quadrature as an alternative (faster, interview talking point)
  - Validate against known test cases (Heston's original 1993 paper parameters)

  UI:
  - Side-by-side: BS price vs Heston price for same option
  - Show the implied vol smile that Heston generates (by inverting Heston prices through BS)
  - Parameter sliders: adjust v0, kappa, theta, xi, rho and see the smile shape change
  - Overlay Heston-implied smile on market smile to visually assess fit
  - Convergence diagnostic: show integration accuracy

  The key insight to demonstrate:
  - Black-Scholes assumes constant volatility → flat implied vol smile
  - Market shows a skew/smile → BS is wrong
  - Heston captures skew via correlated stochastic vol → better fit to market
  - But Heston still can't capture everything (short-expiry smile steepness, for example)

Module 5: Model Calibration

  Goal: find the Heston parameters (v0, kappa, theta, xi, rho) that best fit observed
  market implied volatilities.

  Objective function:
    min_{params} sum_i w_i * (sigma_model(K_i, T_i; params) - sigma_market(K_i, T_i))^2
  where w_i are weights (e.g., by vega or by liquidity)

  Method:
  - scipy.optimize.differential_evolution (global) for initial parameter search
  - scipy.optimize.minimize (L-BFGS-B) for local refinement
  - Parameter bounds:
    v0: [0.001, 1.0]
    kappa: [0.1, 10.0]
    theta: [0.001, 1.0]
    xi (vol of vol): [0.01, 2.0]
    rho: [-0.99, 0.99]
  - Feller condition: 2*kappa*theta > xi^2 (ensures variance stays positive)
    Enforce as constraint or soft penalty in objective

  Calibration workflow:
  1. Fetch options chain for ticker
  2. Extract market IVs (Module 2)
  3. Filter: use liquid, near-the-money options (avoid noisy deep OTM)
  4. Run optimisation
  5. Show results

  UI:
  - Calibrated parameter values with confidence interpretation
  - Feller condition check (pass/fail with explanation)
  - Calibration error surface: 3D plot of (strike, expiry, price_error)
  - Model comparison table: for each option in the chain, show market_price, BS_price,
    Heston_price, and the errors. Colour-code by accuracy.
  - Scatter plot: model price vs market price (should lie on 45-degree line)
  - Residual plot: error vs moneyness, error vs expiry (shows systematic bias)
  - Total RMSE: BS vs Heston (quantifies the improvement)

  The interview story:
  "The calibration shows Heston reduces RMSE by [X]% compared to Black-Scholes,
  with the biggest improvement in OTM puts where the skew matters most. But you can
  see in the residual plot that Heston still struggles with very short-dated options —
  that's where you'd need a local vol or rough vol model."

Module 6: Greeks — Analytical vs Finite Differences

  Greeks computed:
  - Delta: dV/dS
  - Gamma: d2V/dS2
  - Theta: dV/dT
  - Vega: dV/dsigma
  - Rho: dV/dr

  Method 1 — Analytical (Black-Scholes closed-form):
  - All Greeks have known formulas derived from differentiating the BS formula
  - Implement each from scratch

  Method 2 — Finite differences:
  - Central difference: (V(x+h) - V(x-h)) / (2h)
  - For Gamma: (V(x+h) - 2*V(x) + V(x-h)) / h^2
  - Step size h: experiment with different values to show error characteristics
  - Apply to both BS and Heston (Heston has no closed-form Greeks, so FD is the only option)

  Error analysis:
  - For BS: compare analytical Greeks to FD Greeks across a range of step sizes h
  - Plot: |analytical - FD| vs h (should see V-shaped curve — too large h = truncation error,
    too small h = floating point error)
  - Optimal h ≈ sqrt(machine_epsilon) * S for first derivatives
  - This analysis demonstrates understanding of numerical methods, not just finance

  UI:
  - Greeks table for selected option: all 5 Greeks, both methods, absolute and relative error
  - Sensitivity charts: how does Delta change as S varies? (Delta vs S curve, with Gamma as slope)
  - Step size analysis plot: error vs h for each Greek
  - Heston Greeks via FD (since no closed-form exists): show how stochastic vol changes Greeks
  - Side-by-side: BS Greeks vs Heston Greeks for the same option

Testing Strategy

  Correctness is the entire value proposition. If the math is wrong, nothing else matters.

  Known-value tests (pytest, run in CI):
  - BS pricer: validate against Hull's textbook examples and py_vollib to 6 decimal places
  - IV solver: round-trip test — price an option with BS, extract IV, confirm it matches input sigma
  - IV solver: convergence in < 10 iterations for ATM options with reasonable vol (0.1–1.0)
  - SVI: fit to synthetic smile data with known parameters, recover parameters within tolerance
  - Heston: validate against Heston (1993) Table 2 values (the canonical test case)
  - Greeks: analytical vs finite difference agreement within O(h^2) for central differences
  - Put-call parity: residual < 1e-10 for all BS-priced pairs

  Boundary and edge-case tests:
  - T → 0: BS price converges to max(S-K, 0) for calls
  - sigma → 0: BS price converges to discounted intrinsic value
  - Deep ITM/OTM: no NaN, no overflow, no negative prices
  - IV solver with no solution: market price below intrinsic returns NaN, not an infinite loop
  - SVI with < 5 data points per slice: fitting should refuse or warn, not produce garbage

  Integration tests:
  - Full pipeline: cached snapshot → IV extraction → surface fit → calibration completes without error
  - Calibration determinism: same input data produces parameters within 1% across runs
  - yfinance fallback: mock a yfinance failure, confirm cached data loads transparently

  Not tested:
  - UI layout, chart rendering, CSS — visual review only, not worth automating

Error Handling and Edge Cases

  Every page that touches data or computation needs to handle failure gracefully.
  No raw tracebacks. No silent failures. No spinning forever.

  Data errors:
  - Invalid ticker: "Ticker not found. Try one of: SPY, AAPL, TSLA, NVDA, MSFT, QQQ"
  - yfinance down: automatic fallback to cached snapshot with banner "Using cached data from [date]"
  - Ticker with < 5 liquid options (e.g., illiquid small-cap): "Not enough liquid options to fit
    a volatility surface. Try a more liquid ticker."

  Computation errors:
  - IV solver non-convergence: mark that (K, T) cell as "N/C" in the chain table, exclude from surface
  - SVI fit failure (non-convergence): show raw IV scatter without the fitted surface, display warning
  - SVI arbitrage violations: show them explicitly in the UI with yellow warning markers, not silently
  - Heston calibration failure: show last-best parameters with RMSE, flag "did not converge"
  - Feller condition violation: show the violation with explanation, still display results
    (many real calibrations violate Feller — hiding this would be dishonest)

  Loading states:
  - IV extraction (< 2s): inline spinner next to the ticker input
  - Surface fitting (2-5s): skeleton chart placeholder with "Fitting SVI surface..."
  - Heston calibration (10-30s): progress bar with stage labels:
    Stage 1/2: Global search (differential evolution)...
    Stage 2/2: Local refinement (L-BFGS-B)...
  - Use HTMX hx-indicator for lightweight spinners, SSE (Server-Sent Events) for calibration progress

Performance Targets

  These are demo targets — the project needs to feel responsive in a live interview.

  Page loads (cached data path):
  - Landing, chain, about: < 500ms
  - Surface (raw IV extraction + 3D render): < 3s
  - Pricer (single option, BS + Heston): < 200ms
  - Greeks table (5 Greeks, 2 methods): < 500ms

  Heavy computation:
  - IV extraction for full chain (~200 options): < 2s
  - SVI fitting (6-8 expiry slices): < 5s
  - Heston calibration (full pipeline): < 30s hard timeout, < 15s target
  - If calibration exceeds 30s: abort, return best-so-far parameters with warning

  Memory:
  - Differential evolution is memory-hungry with large populations
  - Cap population size at 50, max iterations at 200
  - Total app memory budget: < 512MB (Railway free tier constraint)

Extension Modules (7-8) — Nice-to-Have

Module 7: American Options (Binomial Tree)

  - CRR binomial tree implementation from scratch
  - Early exercise boundary: at each time step, find the stock price where exercise becomes optimal
  - Visualise the tree (first N steps) and the exercise boundary as a function of time
  - Compare American put price to European put price — the early exercise premium
  - Convergence: price vs number of steps (should converge to BS for European options)

Module 8: Monte Carlo for Exotics

  - Geometric Brownian Motion path simulation from scratch
  - Price path-dependent options: Asian (arithmetic average), barrier (knock-in/out)
  - Variance reduction: antithetic variates, control variates (use BS European as control)
  - Convergence plot: price estimate vs number of paths, with confidence interval bands
  - Heston Monte Carlo: simulate correlated (S, v) paths using Euler-Maruyama or QE scheme

Project Structure

  vollab/
    main.py                       FastAPI app, routes, lifespan
    pyproject.toml

    app/
      __init__.py
      routes/
        home.py                   Landing page, ticker input
        chain.py                  Options chain display
        surface.py                Vol surface (raw + SVI)
        pricer.py                 Interactive BS + Heston pricer
        greeks.py                 Greeks dashboard
        calibration.py            Heston calibration page
        about.py                  Technical documentation
      engine/
        black_scholes.py          BS pricing + analytical Greeks (from scratch)
        implied_vol.py            Newton-Raphson IV solver (from scratch)
        svi.py                    SVI parameterization + fitting + arbitrage checks
        heston.py                 Heston characteristic function + pricing (from scratch)
        calibrate.py              Heston calibration (differential evolution + L-BFGS-B)
        greeks_fd.py              Finite difference Greeks (for any pricing model)
        binomial.py               CRR binomial tree (Module 7)
        monte_carlo.py            MC simulation + variance reduction (Module 8)
      data/
        provider.py               yfinance wrapper + fallback to cached data
        cache.py                  In-memory cache with TTL
        snapshots/                Pre-cached JSON files per ticker
          SPY.json
          AAPL.json
          TSLA.json
          NVDA.json
          MSFT.json
          QQQ.json
        snapshot_refresh.py       CLI to update cached snapshots (uv run refresh-snapshots)
      templates/
        base.html                 Base layout (dark theme, nav, Tailwind CDN, HTMX, Plotly)
        home.html
        chain.html
        surface.html
        pricer.html
        greeks.html
        calibration.html
        about.html
        partials/                 HTMX partial templates
          pricer_result.html
          greeks_table.html
          calibration_result.html
      static/
        style.css                 Minimal custom CSS (mostly Tailwind)

Dependencies

  Runtime:
    fastapi
    uvicorn
    jinja2
    python-multipart (form handling)
    numpy
    scipy (optimize, integrate only)
    yfinance
    httpx (for FRED risk-free rate fetch)

  Dev:
    pytest
    ruff

  Frontend (CDN, no npm):
    Tailwind CSS
    HTMX
    Plotly.js
    JetBrains Mono font

Environment Variables

  FRED_API_KEY (optional, for risk-free rate — can hardcode 4.5% as fallback)

  No other API keys needed. yfinance is keyless. Everything else is computed locally.

Milestones

  Phase 1 — Foundation + Black-Scholes (Module 1)
    - FastAPI project setup with Jinja2 + HTMX + Tailwind CDN + Plotly CDN
    - Base template: dark theme, navigation, responsive layout
    - Landing page with ticker input (pre-cached tickers as quick buttons)
    - Black-Scholes pricer: from-scratch implementation, interactive form
    - Put-call parity verification
    - Unit tests for BS pricer against known values
    - Pre-cache options chain snapshots for demo tickers (snapshot refresh script)

  Phase 2 — IV Solver + Options Chain Display (Module 2)
    - Newton-Raphson IV solver from scratch
    - Options chain page: fetch and display full chain (strikes, expiries, bid/ask, volume, OI)
    - Batch IV extraction for entire chain
    - Convergence visualisation (iteration table + plot)
    - Data provider: yfinance with fallback to cached snapshots

  Phase 3 — Volatility Surface (Module 3)
    - Raw IV surface: extract IV grid from chain data
    - 3D Plotly surface (raw market IV points)
    - SVI parameterization: implement fitting per expiry slice
    - No-arbitrage constraint checking (butterfly, calendar)
    - Fitted SVI surface overlay on raw points
    - Per-slice 2D smile view with fit statistics
    - RMSE and parameter display

  Phase 4 — Heston Model (Module 4)
    - Heston characteristic function (corrected formulation, avoid the trap)
    - Numerical integration for option pricing
    - Validation against known test cases
    - Side-by-side BS vs Heston pricing
    - Parameter sliders showing smile shape changes
    - Heston-implied vol smile overlay on market data

  Phase 5 — Calibration (Module 5)
    - Calibration pipeline: market IV -> objective function -> optimisation
    - Differential evolution for global search + L-BFGS-B refinement
    - Feller condition checking
    - Calibration error surface and residual plots
    - Model comparison table (market vs BS vs Heston per option)
    - RMSE comparison: BS vs calibrated Heston

  Phase 6 — Greeks Comparison (Module 6)
    - Analytical BS Greeks from scratch
    - Finite difference Greeks (central difference)
    - Error analysis: FD error vs step size
    - Heston Greeks via finite differences
    - Sensitivity charts (Delta vs S, etc.)
    - Side-by-side BS Greeks vs Heston Greeks

  Phase 7 — Polish + Extensions
    - About page (see About Page Structure below)
    - Mobile responsiveness pass
    - Performance: cache calibration results, lazy-load heavy charts
    - (Optional) American options binomial tree (Module 7)
    - (Optional) Monte Carlo for exotics (Module 8)
    - Deploy to Railway (see Deployment below)

About Page Structure

  The about page is the technical backbone of the interview. It's where someone who wants
  to dig deeper can see the math, understand the tradeoffs, and verify you know your stuff.

  Structure — one section per module, each containing:

  1. The math: core formula(s) written cleanly (code-formatted, not LaTeX)
  2. Implementation note: one paragraph on the key decision or numerical subtlety
     (e.g., "why the little Heston trap matters", "why Brenner-Subrahmanyam for the initial guess")
  3. Limitations: what this model/method gets wrong and what you'd try next
  4. Reference: the paper or textbook section where this comes from

  Additional sections:
  - "What I built from scratch vs what I used libraries for" — one clear table
  - "What I'd do differently" — honest reflection (e.g., "SVI fitting is sensitive to
    initial conditions — I'd explore quasi-random Sobol sequences for starting points")
  - "Extensions I'd explore" — rough vol, local-stochastic vol, jump-diffusion,
    framed as awareness not promises

  The about page should read like a well-organized set of technical notes, not a blog post.

Deployment

  Platform: Railway (single service)

  Configuration:
  - Procfile or railway.toml: uvicorn main:app --host 0.0.0.0 --port $PORT
  - Python version: 3.11+ (specify in runtime.txt or pyproject.toml)
  - Memory: 512MB (Railway free tier) — sufficient if calibration population is capped

  Health check:
  - /health returns JSON: {"status": "ok", "cached_tickers": ["SPY", ...], "timestamp": "..."}
  - Railway pings this endpoint for uptime monitoring
  - Include in Railway service config: healthcheckPath = "/health"

  Cold start:
  - Pre-load cached snapshots into memory on app startup (FastAPI lifespan event)
  - First request after cold start should not hit yfinance — cached data is always ready

  Snapshot refresh:
  - CLI command in pyproject.toml: [project.scripts] refresh-snapshots = "app.data.snapshot_refresh:main"
  - Run locally: uv run refresh-snapshots
  - Updates all 6 ticker snapshots with current market data and timestamps

Key References

  - Black, F. and Scholes, M. (1973) "The Pricing of Options and Corporate Liabilities"
  - Heston, S. (1993) "A Closed-Form Solution for Options with Stochastic Volatility"
  - Gatheral, J. (2004) "A parsimonious arbitrage-free implied volatility parameterization" (SVI)
  - Albrecher, H., Mayer, P., Schoutens, W. (2006) "The Little Heston Trap" (numerical stability fix)
  - Gatheral, J. and Jacquier, A. (2014) "Arbitrage-free SVI volatility surfaces"
  - Hull, J. "Options, Futures, and Other Derivatives" (general reference)

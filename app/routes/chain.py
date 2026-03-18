import math
from datetime import datetime

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.data.provider import fetch_options_chain
from app.engine import implied_vol

router = APIRouter()

DEFAULT_RATE = 0.045


@router.get("/chain", response_class=HTMLResponse)
async def chain_page(request: Request, ticker: str = Query("")):
    templates = request.app.state.templates

    ticker = ticker.strip().upper()
    if not ticker:
        return templates.TemplateResponse("chain.html", {
            "request": request,
            "error": "Enter a ticker symbol.",
            "ticker": "",
            "data": None,
        })

    try:
        data = fetch_options_chain(ticker)
    except ValueError as e:
        cached = request.app.state.cached_tickers
        suggestion = f" Try one of: {', '.join(cached)}" if cached else ""
        return templates.TemplateResponse("chain.html", {
            "request": request,
            "error": f"{e}{suggestion}",
            "ticker": ticker,
            "data": None,
        })

    expiries = data["expiries"]
    first_expiry = expiries[0] if expiries else None

    return templates.TemplateResponse("chain.html", {
        "request": request,
        "ticker": ticker,
        "data": data,
        "expiries": expiries,
        "selected_expiry": first_expiry,
        "table_html": _build_table(data, first_expiry) if first_expiry else "",
        "error": None,
    })


@router.get("/chain/table", response_class=HTMLResponse)
async def chain_table(request: Request, ticker: str = Query(""), expiry: str = Query("")):
    """HTMX partial: returns just the options table for a given expiry."""
    templates = request.app.state.templates
    ticker = ticker.strip().upper()

    try:
        data = fetch_options_chain(ticker)
    except ValueError:
        return HTMLResponse('<div class="text-loss text-sm">Failed to load data.</div>')

    html = _build_table(data, expiry)
    return HTMLResponse(html)


@router.get("/chain/convergence", response_class=HTMLResponse)
async def convergence_detail(
    request: Request,
    ticker: str = Query(""),
    strike: float = Query(0),
    expiry: str = Query(""),
    option_type: str = Query("call"),
):
    """HTMX partial: Newton-Raphson convergence detail for a single option."""
    templates = request.app.state.templates
    ticker = ticker.strip().upper()

    try:
        data = fetch_options_chain(ticker)
    except ValueError:
        return HTMLResponse('<div class="text-loss text-sm">Failed to load data.</div>')

    S = data["underlying_price"]
    T = _years_to_expiry(expiry)
    chain = data["chains"].get(expiry, {})
    options = chain.get("calls" if option_type == "call" else "puts", [])

    opt = None
    for o in options:
        if abs(o["strike"] - strike) < 0.01:
            opt = o
            break

    if opt is None:
        return HTMLResponse('<div class="text-gray-500 text-sm">Option not found.</div>')

    mid = _mid_price(opt)
    if mid <= 0:
        return HTMLResponse('<div class="text-gray-500 text-sm">No valid price for this option.</div>')

    result = implied_vol.solve(mid, S, strike, T, DEFAULT_RATE, option_type, track_steps=True)

    return templates.TemplateResponse("partials/convergence.html", {
        "request": request,
        "result": result,
        "strike": strike,
        "expiry": expiry,
        "option_type": option_type,
        "market_price": mid,
        "S": S,
    })


def _build_table(data: dict, expiry: str) -> str:
    S = data["underlying_price"]
    T = _years_to_expiry(expiry)
    chain = data["chains"].get(expiry, {})
    calls = chain.get("calls", [])
    puts = chain.get("puts", [])
    ticker = data["ticker"]

    rows_html = []
    strikes = sorted({c["strike"] for c in calls} | {p["strike"] for p in puts})

    call_map = {c["strike"]: c for c in calls}
    put_map = {p["strike"]: p for p in puts}

    for strike in strikes:
        c = call_map.get(strike, {})
        p = put_map.get(strike, {})

        c_mid = _mid_price(c) if c else 0
        p_mid = _mid_price(p) if p else 0

        c_iv = _compute_iv(c_mid, S, strike, T, "call")
        p_iv = _compute_iv(p_mid, S, strike, T, "put")

        c_yahoo_iv = c.get("impliedVolatility", 0) if c else 0
        p_yahoo_iv = p.get("impliedVolatility", 0) if p else 0

        itm_call = "bg-surface-lighter" if strike < S else ""
        itm_put = "bg-surface-lighter" if strike > S else ""

        rows_html.append(f"""<tr class="border-b border-gray-800 hover:bg-surface-light text-xs">
  <td class="px-2 py-1.5 {itm_call} text-right cursor-pointer text-accent hover:underline"
      hx-get="/chain/convergence?ticker={ticker}&strike={strike}&expiry={expiry}&option_type=call"
      hx-target="#convergence-panel" hx-swap="innerHTML"
      >{_fmt_price(c.get('bid', 0))}</td>
  <td class="px-2 py-1.5 {itm_call} text-right">{_fmt_price(c.get('ask', 0))}</td>
  <td class="px-2 py-1.5 {itm_call} text-right">{_fmt_vol(c_iv)}</td>
  <td class="px-2 py-1.5 {itm_call} text-right text-gray-500">{_fmt_vol_pct(c_yahoo_iv)}</td>
  <td class="px-2 py-1.5 {itm_call} text-right text-gray-500">{_fmt_int(c.get('volume', 0))}</td>
  <td class="px-2 py-1.5 {itm_call} text-right text-gray-500">{_fmt_int(c.get('openInterest', 0))}</td>
  <td class="px-2 py-1.5 text-center font-mono font-semibold text-white">{strike:.0f}</td>
  <td class="px-2 py-1.5 {itm_put} text-right cursor-pointer text-accent hover:underline"
      hx-get="/chain/convergence?ticker={ticker}&strike={strike}&expiry={expiry}&option_type=put"
      hx-target="#convergence-panel" hx-swap="innerHTML"
      >{_fmt_price(p.get('bid', 0))}</td>
  <td class="px-2 py-1.5 {itm_put} text-right">{_fmt_price(p.get('ask', 0))}</td>
  <td class="px-2 py-1.5 {itm_put} text-right">{_fmt_vol(p_iv)}</td>
  <td class="px-2 py-1.5 {itm_put} text-right text-gray-500">{_fmt_vol_pct(p_yahoo_iv)}</td>
  <td class="px-2 py-1.5 {itm_put} text-right text-gray-500">{_fmt_int(p.get('volume', 0))}</td>
  <td class="px-2 py-1.5 {itm_put} text-right text-gray-500">{_fmt_int(p.get('openInterest', 0))}</td>
</tr>""")

    header = """<table class="w-full text-sm font-mono">
<thead>
<tr class="border-b border-gray-700 text-xs text-gray-500 uppercase">
  <th colspan="6" class="py-2 text-center text-profit">Calls</th>
  <th class="py-2">Strike</th>
  <th colspan="6" class="py-2 text-center text-loss">Puts</th>
</tr>
<tr class="border-b border-gray-700 text-xs text-gray-500">
  <th class="px-2 py-1 text-right">Bid</th>
  <th class="px-2 py-1 text-right">Ask</th>
  <th class="px-2 py-1 text-right">IV (NR)</th>
  <th class="px-2 py-1 text-right">IV (Yho)</th>
  <th class="px-2 py-1 text-right">Vol</th>
  <th class="px-2 py-1 text-right">OI</th>
  <th class="px-2 py-1 text-center">K</th>
  <th class="px-2 py-1 text-right">Bid</th>
  <th class="px-2 py-1 text-right">Ask</th>
  <th class="px-2 py-1 text-right">IV (NR)</th>
  <th class="px-2 py-1 text-right">IV (Yho)</th>
  <th class="px-2 py-1 text-right">Vol</th>
  <th class="px-2 py-1 text-right">OI</th>
</tr>
</thead>
<tbody>"""

    return header + "\n".join(rows_html) + "</tbody></table>"


def _years_to_expiry(expiry_str: str) -> float:
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        now = datetime.now()
        delta = (exp_date - now).total_seconds()
        return max(delta / (365.25 * 86400), 1 / 365.25)  # floor at 1 day
    except (ValueError, TypeError):
        return 0.25


def _mid_price(opt: dict) -> float:
    bid = opt.get("bid", 0) or 0
    ask = opt.get("ask", 0) or 0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return opt.get("lastPrice", 0) or 0


def _compute_iv(price: float, S: float, K: float, T: float, option_type: str) -> float | None:
    if price <= 0 or T <= 0:
        return None
    result = implied_vol.solve(price, S, K, T, DEFAULT_RATE, option_type)
    return result.sigma if result.converged else None


def _fmt_price(v) -> str:
    if not v:
        return '<span class="text-gray-700">-</span>'
    return f"{float(v):.2f}"


def _fmt_vol(v) -> str:
    if v is None:
        return '<span class="text-gray-700">N/C</span>'
    return f"{v * 100:.1f}%"


def _fmt_vol_pct(v) -> str:
    if not v:
        return '<span class="text-gray-700">-</span>'
    return f"{float(v) * 100:.1f}%"


def _fmt_int(v) -> str:
    if not v:
        return '<span class="text-gray-700">-</span>'
    return f"{int(v):,}"

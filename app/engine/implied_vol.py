"""
Implied volatility extraction via Newton-Raphson — implemented from scratch.

Finds sigma such that BS(S, K, T, r, sigma) = market_price by iterating:
  sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)

Starting guess: Brenner-Subrahmanyam approximation.
"""

import math
from dataclasses import dataclass

from .black_scholes import price as bs_price, vega_raw

_SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class IVStep:
    iteration: int
    sigma: float
    bs_price: float
    error: float
    vega: float


@dataclass(frozen=True)
class IVResult:
    sigma: float
    converged: bool
    iterations: int
    steps: list[IVStep]
    final_error: float


def brenner_subrahmanyam(market_price: float, S: float, T: float) -> float:
    """Brenner-Subrahmanyam initial guess: sigma_0 ~ sqrt(2*pi/T) * price / S"""
    if T <= 0 or S <= 0:
        return 0.3
    guess = _SQRT_2PI / math.sqrt(T) * market_price / S
    return max(0.01, min(guess, 3.0))


def solve(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
    track_steps: bool = False,
) -> IVResult:
    """
    Extract implied volatility from a market option price using Newton-Raphson.

    Parameters
    ----------
    market_price : Observed market price of the option
    S, K, T, r : BS parameters (underlying, strike, time, rate)
    option_type : "call" or "put"
    tol : Convergence tolerance on |BS(sigma) - market_price|
    max_iter : Maximum iterations
    track_steps : If True, record each iteration step (for convergence plots)

    Returns
    -------
    IVResult with extracted sigma, convergence status, and optional step history.
    """
    if market_price <= 0:
        return IVResult(sigma=float("nan"), converged=False, iterations=0, steps=[], final_error=float("inf"))

    # Check if market price is below intrinsic (no valid IV)
    df = math.exp(-r * T) if T > 0 else 1.0
    if option_type.lower() == "call":
        intrinsic = max(S - K * df, 0.0)
    else:
        intrinsic = max(K * df - S, 0.0)

    if market_price < intrinsic - 1e-10:
        return IVResult(sigma=float("nan"), converged=False, iterations=0, steps=[], final_error=float("inf"))

    # Try multiple starting guesses: Brenner-Subrahmanyam first, then fallbacks.
    # OTM options can have very small prices that make B-S guess too low,
    # leading to near-zero vega and NR getting stuck.
    guesses = [
        brenner_subrahmanyam(market_price, S, T),
        0.50,
        0.20,
        1.00,
    ]

    best_result = None
    best_error = float("inf")

    for guess in guesses:
        result = _newton_raphson(
            market_price, S, K, T, r, option_type, guess, tol, max_iter, track_steps,
        )
        if result.converged:
            return result
        if result.final_error < best_error:
            best_error = result.final_error
            best_result = result

    return best_result


def _newton_raphson(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    sigma: float,
    tol: float,
    max_iter: int,
    track_steps: bool,
) -> IVResult:
    """Single Newton-Raphson run from a given starting sigma."""
    steps = []

    for i in range(1, max_iter + 1):
        result = bs_price(S, K, T, r, sigma)
        model_price = result.call_price if option_type.lower() == "call" else result.put_price
        error = model_price - market_price
        v = vega_raw(S, K, T, r, sigma)

        if track_steps:
            steps.append(IVStep(
                iteration=i,
                sigma=sigma,
                bs_price=model_price,
                error=error,
                vega=v,
            ))

        if abs(error) < tol:
            return IVResult(sigma=sigma, converged=True, iterations=i, steps=steps, final_error=abs(error))

        if v < 1e-20:
            break

        sigma = sigma - error / v
        sigma = max(0.001, min(sigma, 5.0))

    result = bs_price(S, K, T, r, sigma)
    model_price = result.call_price if option_type.lower() == "call" else result.put_price
    final_error = abs(model_price - market_price)
    return IVResult(sigma=sigma, converged=False, iterations=max_iter, steps=steps, final_error=final_error)


def solve_chain(
    options: list[dict],
    S: float,
    r: float,
    option_type: str = "call",
) -> list[dict]:
    """
    Batch IV extraction for an options chain.

    Each option dict must have: strike, lastPrice (or mid from bid/ask), expiry_years.
    Returns list of dicts with original data + computed 'iv' and 'iv_converged'.
    """
    results = []
    for opt in options:
        strike = opt["strike"]
        T = opt["expiry_years"]
        price = opt.get("mid") or opt.get("lastPrice", 0)

        if price <= 0 or T <= 0 or strike <= 0:
            results.append({**opt, "iv": None, "iv_converged": False})
            continue

        iv_result = solve(price, S, strike, T, r, option_type)

        results.append({
            **opt,
            "iv": iv_result.sigma if iv_result.converged else None,
            "iv_converged": iv_result.converged,
        })

    return results

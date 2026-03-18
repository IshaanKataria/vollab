"""
Black-Scholes option pricing and analytical Greeks — implemented from scratch.

No scipy.stats, no py_vollib. The standard normal CDF uses math.erf,
and all Greeks are derived analytically from the BS formula.
"""

import math
from dataclasses import dataclass

# Constants
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / _SQRT_2PI


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function via the error function."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


@dataclass(frozen=True)
class BSResult:
    call_price: float
    put_price: float
    d1: float
    d2: float
    parity_lhs: float  # C - P
    parity_rhs: float  # S - K * e^(-rT)
    parity_residual: float


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float  # per calendar day (divide annual by 365)
    vega: float  # per 1% move in vol (multiply by 0.01)
    rho: float  # per 1% move in rate (multiply by 0.01)


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> BSResult:
    """
    Black-Scholes European option pricing.

    Parameters
    ----------
    S : Underlying price
    K : Strike price
    T : Time to expiry in years
    r : Risk-free rate (annualised, e.g. 0.045 for 4.5%)
    sigma : Volatility (annualised, e.g. 0.20 for 20%)

    Returns
    -------
    BSResult with call price, put price, d1, d2, and put-call parity check.
    """
    if T < 0:
        raise ValueError("Time to expiry must be non-negative")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if sigma < 0:
        raise ValueError("Volatility must be non-negative")

    # Edge case: at expiry
    if T == 0.0:
        call = max(S - K, 0.0)
        put = max(K - S, 0.0)
        return BSResult(
            call_price=call,
            put_price=put,
            d1=float("inf") if S > K else float("-inf") if S < K else 0.0,
            d2=float("inf") if S > K else float("-inf") if S < K else 0.0,
            parity_lhs=call - put,
            parity_rhs=S - K,
            parity_residual=abs((call - put) - (S - K)),
        )

    # Edge case: zero volatility — deterministic payoff
    if sigma == 0.0:
        df = math.exp(-r * T)
        forward = S / df  # S * e^(rT)
        call = max(S - K * df, 0.0)
        put = max(K * df - S, 0.0)
        d1 = float("inf") if forward > K else float("-inf") if forward < K else 0.0
        return BSResult(
            call_price=call,
            put_price=put,
            d1=d1,
            d2=d1,
            parity_lhs=call - put,
            parity_rhs=S - K * df,
            parity_residual=abs((call - put) - (S - K * df)),
        )

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    df = math.exp(-r * T)
    call = S * norm_cdf(d1) - K * df * norm_cdf(d2)
    put = K * df * norm_cdf(-d2) - S * norm_cdf(-d1)

    # Clamp tiny negatives from floating point
    call = max(call, 0.0)
    put = max(put, 0.0)

    parity_lhs = call - put
    parity_rhs = S - K * df

    return BSResult(
        call_price=call,
        put_price=put,
        d1=d1,
        d2=d2,
        parity_lhs=parity_lhs,
        parity_rhs=parity_rhs,
        parity_residual=abs(parity_lhs - parity_rhs),
    )


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Greeks:
    """
    Analytical Black-Scholes Greeks.

    Theta is per calendar day (annual / 365).
    Vega is per 1 percentage point move in vol.
    Rho is per 1 percentage point move in rate.
    """
    if T <= 0 or sigma <= 0:
        return Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = math.exp(-r * T)
    n_d1 = norm_pdf(d1)

    # Gamma and Vega are the same for calls and puts
    gamma = n_d1 / (S * sigma * sqrt_T)
    vega = S * n_d1 * sqrt_T * 0.01  # per 1% vol

    is_call = option_type.lower() == "call"

    if is_call:
        delta = norm_cdf(d1)
        theta_annual = (
            -(S * n_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * df * norm_cdf(d2)
        )
        rho = K * T * df * norm_cdf(d2) * 0.01  # per 1% rate
    else:
        delta = norm_cdf(d1) - 1.0
        theta_annual = (
            -(S * n_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * df * norm_cdf(-d2)
        )
        rho = -K * T * df * norm_cdf(-d2) * 0.01  # per 1% rate

    theta = theta_annual / 365.0

    return Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
    )


def vega_raw(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Raw vega (dV/dsigma, not scaled) — used by the IV solver for Newton-Raphson."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return S * norm_pdf(d1) * sqrt_T

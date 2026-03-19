"""
Finite difference Greeks — model-agnostic.

Computes Greeks via central differences for any pricing function.
Works with both Black-Scholes and Heston (Heston has no closed-form Greeks,
so FD is the only option).

Central difference: df/dx ≈ (f(x+h) - f(x-h)) / (2h)     — O(h^2) error
Second derivative:  d2f/dx2 ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FDGreeks:
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1% vol move
    rho: float    # per 1% rate move


def compute(
    price_fn: Callable,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma_or_params,
    option_type: str = "call",
    h_S: float | None = None,
    h_T: float = 1 / 365.25,
    h_sigma: float = 0.001,
    h_r: float = 0.0001,
) -> FDGreeks:
    """
    Compute Greeks via central finite differences.

    Parameters
    ----------
    price_fn : Callable that returns a price given (S, K, T, r, sigma_or_params).
               Must return a float (the option price for the given type).
    S, K, T, r : market parameters
    sigma_or_params : volatility (float) for BS, or HestonParams for Heston
    option_type : "call" or "put"
    h_S : step size for S (default: 0.01 * S)
    h_T : step size for T in years (default: 1 day)
    h_sigma : step size for sigma (only for BS)
    h_r : step size for r
    """
    if h_S is None:
        h_S = 0.001 * S

    def P(s=S, k=K, t=T, rate=r, vol=sigma_or_params):
        return price_fn(s, k, t, rate, vol, option_type)

    base = P()

    # Delta: dV/dS
    delta = (P(s=S + h_S) - P(s=S - h_S)) / (2 * h_S)

    # Gamma: d2V/dS2
    gamma = (P(s=S + h_S) - 2 * base + P(s=S - h_S)) / (h_S * h_S)

    # Theta: dV/dT (per calendar day)
    if T > h_T:
        theta_annual = (P(t=T - h_T) - P(t=T + h_T)) / (2 * h_T)
    else:
        theta_annual = (P(t=T) - P(t=T + h_T)) / h_T
    theta = theta_annual / 365.25

    # Vega: dV/dsigma (per 1%)
    # Only meaningful for BS (scalar sigma). For Heston, we bump v0.
    if isinstance(sigma_or_params, (int, float)):
        vega = (P(vol=sigma_or_params + h_sigma) - P(vol=sigma_or_params - h_sigma)) / (2 * h_sigma) * 0.01
    else:
        # For Heston: bump v0 (initial variance)
        from .heston import HestonParams
        p = sigma_or_params
        h_v = 0.001
        p_up = HestonParams(v0=p.v0 + h_v, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho)
        p_dn = HestonParams(v0=max(p.v0 - h_v, 0.001), kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho)
        vega = (P(vol=p_up) - P(vol=p_dn)) / (2 * h_v) * 0.01

    # Rho: dV/dr (per 1%)
    rho_greek = (P(rate=r + h_r) - P(rate=r - h_r)) / (2 * h_r) * 0.01

    return FDGreeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho_greek,
    )


def error_vs_step_size(
    analytical_greek: float,
    price_fn: Callable,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    greek_name: str = "delta",
    n_points: int = 30,
) -> dict:
    """
    Compute |analytical - FD| across a range of step sizes h.

    Shows the V-shaped error curve: too large h = truncation error,
    too small h = floating point error. Optimal h ≈ sqrt(eps) * S.

    Returns dict with: h_values, errors (for plotting).
    """
    h_values = np.logspace(-10, -1, n_points) * S
    errors = []

    for h in h_values:
        def P(s=S, k=K, t=T, rate=r, vol=sigma):
            return price_fn(s, k, t, rate, vol, option_type)

        if greek_name == "delta":
            fd = (P(s=S + h) - P(s=S - h)) / (2 * h)
        elif greek_name == "gamma":
            fd = (P(s=S + h) - 2 * P() + P(s=S - h)) / (h * h)
        else:
            fd = analytical_greek  # fallback

        errors.append(abs(fd - analytical_greek))

    return {
        "h_values": h_values.tolist(),
        "errors": errors,
    }

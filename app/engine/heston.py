"""
Heston stochastic volatility model — implemented from scratch.

The model:
  dS = r*S*dt + sqrt(v)*S*dW1
  dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW2
  corr(dW1, dW2) = rho

Pricing uses Fourier inversion of the characteristic function.
Uses the corrected "little Heston trap" formulation (Albrecher, Mayer, Schoutens 2006)
to avoid branch cut discontinuities in the complex logarithm.

5 parameters: v0, kappa, theta, xi, rho
  v0    : initial variance
  kappa : mean reversion speed
  theta : long-run variance
  xi    : vol of vol
  rho   : correlation between spot and vol processes
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class HestonParams:
    v0: float     # initial variance
    kappa: float  # mean reversion speed
    theta: float  # long-run variance
    xi: float     # vol of vol
    rho: float    # spot-vol correlation

    @property
    def feller_satisfied(self) -> bool:
        """Feller condition: 2*kappa*theta > xi^2 ensures variance stays positive."""
        return 2 * self.kappa * self.theta > self.xi * self.xi

    @property
    def feller_ratio(self) -> float:
        """2*kappa*theta / xi^2. Must be > 1 for Feller condition."""
        if self.xi == 0:
            return float("inf")
        return 2 * self.kappa * self.theta / (self.xi * self.xi)


@dataclass(frozen=True)
class HestonResult:
    call_price: float
    put_price: float
    integration_abserr: float


def characteristic_function(
    phi: complex,
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    j: int,
) -> complex:
    """
    Heston characteristic function fj(phi) for j=1,2.

    Uses the corrected formulation (Albrecher et al. 2006) to avoid
    the "little Heston trap" — a branch cut discontinuity in the
    original Heston (1993) formulation.

    j=1: used for P1 (delta probability)
    j=2: used for P2 (exercise probability)
    """
    x = math.log(S)
    a = params.kappa * params.theta
    v0 = params.v0
    xi = params.xi
    rho = params.rho
    kappa = params.kappa

    # Parameters differ for j=1 and j=2
    if j == 1:
        u = 0.5
        b = kappa - rho * xi
    else:
        u = -0.5
        b = kappa

    # Complex discriminant
    d = np.sqrt(
        (rho * xi * 1j * phi - b) ** 2
        - xi * xi * (2 * u * 1j * phi - phi * phi)
    )

    # Corrected formulation: g uses (b - rho*xi*i*phi - d) in numerator
    g = (b - rho * xi * 1j * phi - d) / (b - rho * xi * 1j * phi + d)

    # Avoid numerical issues when g*exp(-dT) ≈ 1
    exp_neg_dT = np.exp(-d * T)

    C = (
        r * 1j * phi * T
        + (a / (xi * xi))
        * ((b - rho * xi * 1j * phi - d) * T - 2 * np.log((1 - g * exp_neg_dT) / (1 - g)))
    )

    D = (
        (b - rho * xi * 1j * phi - d) / (xi * xi)
        * (1 - exp_neg_dT) / (1 - g * exp_neg_dT)
    )

    return np.exp(C + D * v0 + 1j * phi * x)


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
) -> HestonResult:
    """
    Price a European option under the Heston model via Fourier inversion.

    Call = S*P1 - K*e^(-rT)*P2
    Pj = 1/2 + 1/pi * integral_0^inf Re[e^(-i*phi*ln(K)) * fj(phi) / (i*phi)] dphi
    """
    if T <= 0:
        call = max(S - K, 0.0)
        put = max(K - S, 0.0)
        return HestonResult(call_price=call, put_price=put, integration_abserr=0.0)

    log_K = math.log(K)
    total_abserr = 0.0

    probabilities = []
    for j in [1, 2]:
        def integrand(phi, _j=j):
            if phi == 0:
                return 0.0
            cf = characteristic_function(phi, S, K, T, r, params, _j)
            val = np.exp(-1j * phi * log_K) * cf / (1j * phi)
            return float(val.real)

        integral, abserr = quad(integrand, 0, 200, limit=200, epsabs=1e-10, epsrel=1e-10)
        total_abserr += abserr
        P = 0.5 + (1.0 / math.pi) * integral
        P = max(0.0, min(1.0, P))  # clamp
        probabilities.append(P)

    P1, P2 = probabilities
    df = math.exp(-r * T)
    call = max(S * P1 - K * df * P2, 0.0)
    put = max(call - S + K * df, 0.0)  # put-call parity

    return HestonResult(
        call_price=call,
        put_price=put,
        integration_abserr=total_abserr,
    )


def heston_smile(
    S: float,
    T: float,
    r: float,
    params: HestonParams,
    strikes: np.ndarray | list[float] | None = None,
    n_strikes: int = 30,
) -> dict:
    """
    Generate the Heston-implied volatility smile.

    Prices options at various strikes under Heston, then inverts each
    price through Black-Scholes to get the implied volatility.

    Returns dict with: strikes, implied_vols, log_moneyness, prices
    """
    from .implied_vol import solve
    from .black_scholes import price as bs_price

    if strikes is None:
        F = S * math.exp(r * T)
        strikes = np.linspace(0.7 * F, 1.3 * F, n_strikes)
    else:
        strikes = np.array(strikes)

    F = S * math.exp(r * T)
    ivs = []
    prices = []
    k_vals = []

    for K_val in strikes:
        K_val = float(K_val)
        if K_val <= 0:
            continue

        result = price(S, K_val, T, r, params)
        opt_type = "put" if K_val < S else "call"
        opt_price = result.put_price if opt_type == "put" else result.call_price

        if opt_price <= 0:
            continue

        iv_result = solve(opt_price, S, K_val, T, r, opt_type)
        if iv_result.converged:
            ivs.append(iv_result.sigma)
            prices.append(opt_price)
            k_vals.append(math.log(K_val / F))

    return {
        "log_moneyness": k_vals,
        "implied_vols": ivs,
        "prices": prices,
    }

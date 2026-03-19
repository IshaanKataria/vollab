"""
SVI (Stochastic Volatility Inspired) volatility surface parameterization.

Gatheral's raw SVI formula for total implied variance w(k):
  w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where k = log(K/F) is log-moneyness.

Fitting uses scipy.optimize.minimize with multiple random starting points
(SVI is non-convex). No-arbitrage constraints are checked post-fit.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SVIParams:
    a: float      # overall variance level
    b: float      # slope of the wings (>= 0)
    rho: float    # skew/asymmetry (-1 < rho < 1)
    m: float      # horizontal shift
    sigma: float  # curvature at the money (> 0)


@dataclass
class SVISliceFit:
    expiry: str
    T: float
    params: SVIParams
    rmse: float
    n_points: int
    butterfly_violations: int
    k_values: list[float] = field(default_factory=list)
    market_var: list[float] = field(default_factory=list)
    fitted_var: list[float] = field(default_factory=list)


@dataclass
class SVISurface:
    slices: list[SVISliceFit]
    calendar_violations: int
    total_rmse: float


def svi_total_variance(k: float | np.ndarray, params: SVIParams) -> float | np.ndarray:
    """Compute total implied variance w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    km = k - params.m
    return params.a + params.b * (params.rho * km + np.sqrt(km * km + params.sigma * params.sigma))


def svi_implied_vol(k: float | np.ndarray, T: float, params: SVIParams) -> float | np.ndarray:
    """Convert SVI total variance to implied volatility: sigma_iv = sqrt(w(k) / T)"""
    w = svi_total_variance(k, params)
    w = np.maximum(w, 1e-10)  # clamp to avoid sqrt of negative
    return np.sqrt(w / T)


def fit_slice(
    k: np.ndarray,
    market_iv: np.ndarray,
    T: float,
    n_starts: int = 10,
) -> SVISliceFit:
    """
    Fit SVI parameters to a single expiry slice of market implied volatilities.

    Parameters
    ----------
    k : log-moneyness values, k = log(K/F)
    market_iv : market implied volatilities for each k
    T : time to expiry in years
    n_starts : number of random starting points for optimization

    Returns
    -------
    SVISliceFit with best-fit parameters, RMSE, and arbitrage violation count.
    """
    market_w = market_iv ** 2 * T  # total variance
    atm_var = float(np.median(market_w))

    # Parameter bounds: a, b, rho, m, sigma
    bounds = [
        (-0.5 * atm_var, 2.0 * atm_var),  # a
        (1e-4, 5.0),                        # b
        (-0.99, 0.99),                       # rho
        (float(k.min()) - 0.5, float(k.max()) + 0.5),  # m
        (1e-4, 2.0),                         # sigma
    ]

    def objective(x):
        p = SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
        fitted_w = svi_total_variance(k, p)
        residuals = fitted_w - market_w
        return float(np.sum(residuals ** 2))

    best_result = None
    best_cost = float("inf")
    rng = np.random.default_rng(42)

    for i in range(n_starts):
        if i == 0:
            x0 = [atm_var * 0.5, 0.1, -0.3, 0.0, 0.1]
        else:
            x0 = [
                rng.uniform(*bounds[0]),
                rng.uniform(*bounds[1]),
                rng.uniform(*bounds[2]),
                rng.uniform(*bounds[3]),
                rng.uniform(*bounds[4]),
            ]

        try:
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 500, "ftol": 1e-12})
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        # Fallback: return flat surface at median variance
        p = SVIParams(a=atm_var, b=0.0, rho=0.0, m=0.0, sigma=0.1)
        fitted_w = svi_total_variance(k, p)
        rmse = float(np.sqrt(np.mean((fitted_w - market_w) ** 2)))
        return SVISliceFit(
            expiry="", T=T, params=p, rmse=rmse, n_points=len(k),
            butterfly_violations=0,
            k_values=k.tolist(), market_var=market_w.tolist(), fitted_var=fitted_w.tolist(),
        )

    x = best_result.x
    params = SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
    fitted_w = svi_total_variance(k, params)
    rmse = float(np.sqrt(np.mean((fitted_w - market_w) ** 2)))
    n_butterfly = count_butterfly_violations(k, params)

    return SVISliceFit(
        expiry="", T=T, params=params, rmse=rmse, n_points=len(k),
        butterfly_violations=n_butterfly,
        k_values=k.tolist(), market_var=market_w.tolist(), fitted_var=fitted_w.tolist(),
    )


def count_butterfly_violations(k: np.ndarray, params: SVIParams, n_test: int = 200) -> int:
    """
    Check butterfly arbitrage: the implied density must be non-negative.

    The condition (Gatheral & Jacquier 2014):
      g(k) = (1 - k*w'/(2w))^2 - w'/4 * (1/4 + 1/w) + w''/2 >= 0

    where w = w(k), w' = dw/dk, w'' = d2w/dk2.
    """
    k_test = np.linspace(float(k.min()) - 0.1, float(k.max()) + 0.1, n_test)
    violations = 0

    for ki in k_test:
        w = svi_total_variance(ki, params)
        if w <= 0:
            violations += 1
            continue

        w_prime = _svi_dw_dk(ki, params)
        w_double_prime = _svi_d2w_dk2(ki, params)

        term1 = (1 - ki * w_prime / (2 * w)) ** 2
        term2 = w_prime ** 2 / 4 * (1 / 4 + 1 / w)
        term3 = w_double_prime / 2

        g = term1 - term2 + term3
        if g < -1e-10:
            violations += 1

    return violations


def check_calendar_arbitrage(slices: list[SVISliceFit], n_test: int = 100) -> int:
    """
    Calendar spread arbitrage: total variance must be non-decreasing in T.
    Check at a grid of k values across consecutive expiry slices.
    """
    if len(slices) < 2:
        return 0

    sorted_slices = sorted(slices, key=lambda s: s.T)
    violations = 0

    k_min = min(min(s.k_values) for s in sorted_slices if s.k_values)
    k_max = max(max(s.k_values) for s in sorted_slices if s.k_values)
    k_test = np.linspace(k_min, k_max, n_test)

    for i in range(len(sorted_slices) - 1):
        s1 = sorted_slices[i]
        s2 = sorted_slices[i + 1]
        for ki in k_test:
            w1 = svi_total_variance(ki, s1.params)
            w2 = svi_total_variance(ki, s2.params)
            if w2 < w1 - 1e-10:
                violations += 1

    return violations


def fit_surface(
    slices_data: list[dict],
    n_starts: int = 10,
) -> SVISurface:
    """
    Fit SVI to multiple expiry slices.

    Each entry in slices_data should have:
      expiry: str, T: float, k: np.ndarray, market_iv: np.ndarray
    """
    fits = []
    for sd in slices_data:
        k = np.array(sd["k"])
        iv = np.array(sd["market_iv"])
        if len(k) < 5:
            continue
        fit = fit_slice(k, iv, sd["T"], n_starts)
        fit.expiry = sd["expiry"]
        fits.append(fit)

    calendar_viols = check_calendar_arbitrage(fits)

    total_mse = 0.0
    total_n = 0
    for f in fits:
        total_mse += f.rmse ** 2 * f.n_points
        total_n += f.n_points
    total_rmse = math.sqrt(total_mse / total_n) if total_n > 0 else 0.0

    return SVISurface(slices=fits, calendar_violations=calendar_viols, total_rmse=total_rmse)


def _svi_dw_dk(k: float, params: SVIParams) -> float:
    """First derivative of SVI total variance w.r.t. k."""
    km = k - params.m
    denom = math.sqrt(km * km + params.sigma * params.sigma)
    return params.b * (params.rho + km / denom)


def _svi_d2w_dk2(k: float, params: SVIParams) -> float:
    """Second derivative of SVI total variance w.r.t. k."""
    km = k - params.m
    s2 = params.sigma * params.sigma
    denom = (km * km + s2) ** 1.5
    return params.b * s2 / denom

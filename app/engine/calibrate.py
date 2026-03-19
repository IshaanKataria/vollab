"""
Heston model calibration to market implied volatilities.

Two-stage optimization:
  1. Differential evolution (global search) — finds the basin
  2. L-BFGS-B (local refinement) — polishes the result

Objective: minimize weighted sum of squared IV errors between
Heston-implied vols and market-observed vols.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .black_scholes import price as bs_price
from .heston import HestonParams, price as heston_price
from .implied_vol import solve as iv_solve


@dataclass(frozen=True)
class CalibrationOption:
    strike: float
    T: float
    market_iv: float
    market_price: float
    option_type: str  # "call" or "put"


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_heston: float
    rmse_bs: float         # RMSE using flat BS at mean IV (baseline)
    improvement_pct: float # (rmse_bs - rmse_heston) / rmse_bs * 100
    n_options: int
    elapsed_ms: float
    converged: bool
    per_option: list[dict] = field(default_factory=list)


# Parameter bounds: v0, kappa, theta, xi, rho
BOUNDS = [
    (0.001, 1.0),   # v0
    (0.1, 10.0),    # kappa
    (0.001, 1.0),   # theta
    (0.01, 2.0),    # xi
    (-0.99, 0.99),  # rho
]


def calibrate(
    options: list[CalibrationOption],
    S: float,
    r: float,
    popsize: int = 15,
    max_de_iter: int = 80,
    seed: int = 42,
) -> CalibrationResult:
    """
    Calibrate Heston parameters to a set of market options.

    Parameters
    ----------
    options : list of CalibrationOption (strike, T, market_iv, etc.)
    S : underlying price
    r : risk-free rate
    popsize : differential evolution population size
    max_de_iter : max DE iterations
    seed : random seed for reproducibility
    """
    t0 = time.monotonic()

    if len(options) < 3:
        raise ValueError("Need at least 3 options for calibration")

    # Precompute weights (vega-based: ATM options weighted more heavily)
    market_ivs = np.array([o.market_iv for o in options])
    mean_iv = float(np.mean(market_ivs))

    def objective(x):
        params = HestonParams(v0=x[0], kappa=x[1], theta=x[2], xi=x[3], rho=x[4])
        total_sq_err = 0.0

        for opt in options:
            try:
                h_result = heston_price(S, opt.strike, opt.T, r, params, fast=True)
                h_price = h_result.call_price if opt.option_type == "call" else h_result.put_price

                if h_price <= 0:
                    total_sq_err += 1.0  # penalty
                    continue

                h_iv_result = iv_solve(h_price, S, opt.strike, opt.T, r, opt.option_type)
                if h_iv_result.converged:
                    err = h_iv_result.sigma - opt.market_iv
                else:
                    err = 0.5  # large penalty for non-convergence
            except Exception:
                err = 0.5

            total_sq_err += err * err

        # Soft Feller penalty: encourage 2*kappa*theta > xi^2
        feller = 2 * x[1] * x[2] - x[3] * x[3]
        if feller < 0:
            total_sq_err += 0.1 * feller * feller

        return total_sq_err / len(options)

    # Stage 1: Differential evolution (global)
    de_result = differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=seed,
        popsize=popsize,
        maxiter=max_de_iter,
        tol=1e-8,
        polish=False,
    )

    # Stage 2: L-BFGS-B refinement (local)
    local_result = minimize(
        objective,
        de_result.x,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"maxiter": 200, "ftol": 1e-12},
    )

    best_x = local_result.x if local_result.fun < de_result.fun else de_result.x
    best_params = HestonParams(
        v0=best_x[0], kappa=best_x[1], theta=best_x[2],
        xi=best_x[3], rho=best_x[4],
    )

    # Compute per-option results and RMSE
    per_option = []
    heston_sq_errors = []
    bs_sq_errors = []

    for opt in options:
        row = {
            "strike": opt.strike,
            "T": opt.T,
            "type": opt.option_type,
            "market_iv": opt.market_iv,
            "market_price": opt.market_price,
        }

        # BS price at flat mean IV
        bs_result = bs_price(S, opt.strike, opt.T, r, mean_iv)
        bs_opt_price = bs_result.call_price if opt.option_type == "call" else bs_result.put_price
        bs_iv_result = iv_solve(bs_opt_price, S, opt.strike, opt.T, r, opt.option_type)
        bs_iv = bs_iv_result.sigma if bs_iv_result.converged else mean_iv
        bs_err = bs_iv - opt.market_iv
        bs_sq_errors.append(bs_err * bs_err)
        row["bs_iv"] = bs_iv
        row["bs_error"] = bs_err

        # Heston price
        try:
            h_result = heston_price(S, opt.strike, opt.T, r, best_params)
            h_price = h_result.call_price if opt.option_type == "call" else h_result.put_price
            h_iv_result = iv_solve(h_price, S, opt.strike, opt.T, r, opt.option_type)
            h_iv = h_iv_result.sigma if h_iv_result.converged else None
            h_err = (h_iv - opt.market_iv) if h_iv is not None else None
            if h_err is not None:
                heston_sq_errors.append(h_err * h_err)
            row["heston_price"] = h_price
            row["heston_iv"] = h_iv
            row["heston_error"] = h_err
        except Exception:
            row["heston_price"] = None
            row["heston_iv"] = None
            row["heston_error"] = None

        # Log-moneyness for residual plots
        F = S * math.exp(r * opt.T)
        row["log_moneyness"] = math.log(opt.strike / F)

        per_option.append(row)

    rmse_heston = math.sqrt(np.mean(heston_sq_errors)) if heston_sq_errors else float("inf")
    rmse_bs = math.sqrt(np.mean(bs_sq_errors)) if bs_sq_errors else float("inf")
    improvement = (rmse_bs - rmse_heston) / rmse_bs * 100 if rmse_bs > 0 else 0.0

    elapsed = (time.monotonic() - t0) * 1000

    return CalibrationResult(
        params=best_params,
        rmse_heston=rmse_heston,
        rmse_bs=rmse_bs,
        improvement_pct=improvement,
        n_options=len(options),
        elapsed_ms=elapsed,
        converged=local_result.success or de_result.success,
        per_option=per_option,
    )


def prepare_calibration_options(
    iv_surface_data: dict,
    S: float,
    r: float,
    max_options: int = 60,
) -> list[CalibrationOption]:
    """
    Select liquid, near-the-money options from IV surface data for calibration.

    Filters to options with converged IVs, reasonable moneyness, and
    subsamples if there are too many (for performance).
    """
    options = []
    F_base = S  # approximate

    for point in iv_surface_data["raw_points"]:
        k = point["k"]
        # Filter to reasonable moneyness range
        if abs(k) > 0.4:
            continue

        T = point["T"]
        iv = point["iv"]
        strike = point["strike"]

        if iv < 0.01 or iv > 3.0 or T <= 0:
            continue

        opt_type = "put" if strike < S else "call"
        bs_result = bs_price(S, strike, T, r, iv)
        mkt_price = bs_result.call_price if opt_type == "call" else bs_result.put_price

        if mkt_price <= 0:
            continue

        options.append(CalibrationOption(
            strike=strike, T=T, market_iv=iv,
            market_price=mkt_price, option_type=opt_type,
        ))

    # Subsample if too many (stratified by T)
    if len(options) > max_options:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(options), max_options, replace=False)
        options = [options[i] for i in sorted(indices)]

    return options

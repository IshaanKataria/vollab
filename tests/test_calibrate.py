"""
Heston calibration tests.

Core validation: calibrate to synthetic Heston smile, recover params.
"""

import math

import numpy as np
import pytest

from app.engine.black_scholes import price as bs_price
from app.engine.calibrate import CalibrationOption, CalibrationResult, calibrate
from app.engine.heston import HestonParams, heston_smile, price as heston_price
from app.engine.implied_vol import solve as iv_solve


def _make_synthetic_options(
    true_params: HestonParams,
    S: float = 100,
    r: float = 0.05,
    T_values: list[float] | None = None,
    n_strikes: int = 8,
) -> list[CalibrationOption]:
    """Generate synthetic market options from known Heston parameters."""
    if T_values is None:
        T_values = [0.25, 0.5, 1.0]

    options = []
    for T in T_values:
        F = S * math.exp(r * T)
        strikes = np.linspace(0.85 * F, 1.15 * F, n_strikes)
        for K in strikes:
            K = float(K)
            opt_type = "put" if K < S else "call"
            h_result = heston_price(S, K, T, r, true_params)
            h_price = h_result.call_price if opt_type == "call" else h_result.put_price
            if h_price <= 0:
                continue

            iv_result = iv_solve(h_price, S, K, T, r, opt_type)
            if not iv_result.converged:
                continue

            options.append(CalibrationOption(
                strike=K, T=T, market_iv=iv_result.sigma,
                market_price=h_price, option_type=opt_type,
            ))
    return options


class TestSyntheticCalibration:
    def test_recovers_params_approximately(self):
        true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        options = _make_synthetic_options(true_params)
        assert len(options) >= 10

        result = calibrate(options, S=100, r=0.05, popsize=10, max_de_iter=40)
        assert result.converged
        assert result.rmse_heston < 0.005  # < 0.5% IV RMSE

    def test_heston_beats_flat_bs(self):
        true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        options = _make_synthetic_options(true_params)

        result = calibrate(options, S=100, r=0.05, popsize=10, max_de_iter=40)
        assert result.rmse_heston < result.rmse_bs
        assert result.improvement_pct > 0

    def test_per_option_populated(self):
        true_params = HestonParams(v0=0.04, kappa=1.5, theta=0.05, xi=0.4, rho=-0.5)
        options = _make_synthetic_options(true_params, n_strikes=6)

        result = calibrate(options, S=100, r=0.05, popsize=8, max_de_iter=30)
        assert len(result.per_option) == len(options)
        for row in result.per_option:
            assert "strike" in row
            assert "heston_iv" in row
            assert "log_moneyness" in row


class TestEdgeCases:
    def test_too_few_options(self):
        options = [
            CalibrationOption(strike=100, T=0.5, market_iv=0.2, market_price=5.0, option_type="call"),
        ]
        with pytest.raises(ValueError):
            calibrate(options, S=100, r=0.05)

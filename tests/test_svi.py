"""
SVI fitting tests.

Core validation: fit to synthetic smile with known parameters, recover them.
"""

import numpy as np
import pytest

from app.engine.svi import (
    SVIParams,
    SVISurface,
    count_butterfly_violations,
    check_calendar_arbitrage,
    fit_slice,
    fit_surface,
    svi_implied_vol,
    svi_total_variance,
)


class TestSVIFormula:
    def test_atm_value(self):
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        w = svi_total_variance(0.0, p)
        expected = 0.04 + 0.1 * (0.0 + np.sqrt(0.0 + 0.01))
        assert w == pytest.approx(expected, rel=1e-10)

    def test_symmetry_when_rho_zero(self):
        p = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
        assert svi_total_variance(0.1, p) == pytest.approx(svi_total_variance(-0.1, p), rel=1e-10)

    def test_skew_when_rho_negative(self):
        p = SVIParams(a=0.04, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
        # Negative rho => left wing (k<0) has higher variance
        assert svi_total_variance(-0.2, p) > svi_total_variance(0.2, p)

    def test_vectorized(self):
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        w = svi_total_variance(k, p)
        assert w.shape == (5,)
        assert np.all(w > 0)

    def test_implied_vol(self):
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        T = 0.5
        iv = svi_implied_vol(0.0, T, p)
        w = svi_total_variance(0.0, p)
        assert iv == pytest.approx(np.sqrt(w / T), rel=1e-10)


class TestSVIFitting:
    def test_recover_known_params(self):
        # Generate synthetic data from known SVI params
        true_params = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.01, sigma=0.1)
        T = 0.5
        k = np.linspace(-0.3, 0.3, 30)
        market_iv = svi_implied_vol(k, T, true_params)

        fit = fit_slice(k, market_iv, T, n_starts=10)

        assert fit.rmse < 1e-6
        assert fit.params.a == pytest.approx(true_params.a, abs=0.01)
        assert fit.params.b == pytest.approx(true_params.b, abs=0.01)
        assert fit.params.rho == pytest.approx(true_params.rho, abs=0.05)

    def test_fit_with_noise(self):
        true_params = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.0, sigma=0.1)
        T = 0.5
        k = np.linspace(-0.3, 0.3, 30)
        rng = np.random.default_rng(123)
        market_iv = svi_implied_vol(k, T, true_params) + rng.normal(0, 0.002, len(k))

        fit = fit_slice(k, market_iv, T, n_starts=10)
        assert fit.rmse < 0.01
        assert fit.n_points == 30

    def test_too_few_points(self):
        slices = fit_surface([{
            "expiry": "2026-04-01",
            "T": 0.1,
            "k": [0.0, 0.1],
            "market_iv": [0.2, 0.22],
        }])
        assert len(slices.slices) == 0  # skipped due to < 5 points


class TestArbitrageChecks:
    def test_no_butterfly_for_reasonable_params(self):
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.15)
        k = np.linspace(-0.5, 0.5, 20)
        violations = count_butterfly_violations(k, p)
        assert violations == 0

    def test_calendar_no_violation(self):
        p1 = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        p2 = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        s1 = _make_slice_fit(p1, T=0.25, expiry="2026-04-01")
        s2 = _make_slice_fit(p2, T=0.50, expiry="2026-07-01")
        violations = check_calendar_arbitrage([s1, s2])
        assert violations == 0


class TestFitSurface:
    def test_multi_slice(self):
        slices_data = []
        for i, T in enumerate([0.25, 0.5, 1.0]):
            p = SVIParams(a=0.02 + 0.02 * i, b=0.12, rho=-0.25, m=0.0, sigma=0.1)
            k = np.linspace(-0.3, 0.3, 20)
            iv = svi_implied_vol(k, T, p)
            slices_data.append({
                "expiry": f"2026-0{i+4}-01",
                "T": T,
                "k": k.tolist(),
                "market_iv": iv.tolist(),
            })
        surface = fit_surface(slices_data, n_starts=5)
        assert len(surface.slices) == 3
        assert surface.total_rmse < 1e-4


def _make_slice_fit(params, T, expiry):
    from app.engine.svi import SVISliceFit
    k = np.linspace(-0.3, 0.3, 20)
    return SVISliceFit(
        expiry=expiry, T=T, params=params, rmse=0.0, n_points=20,
        butterfly_violations=0, k_values=k.tolist(), market_var=[], fitted_var=[],
    )

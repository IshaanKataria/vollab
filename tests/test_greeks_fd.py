"""
Finite difference Greeks tests.

Core: FD Greeks agree with BS analytical Greeks within O(h^2).
"""

import pytest

from app.engine import black_scholes as bs
from app.engine.greeks_fd import compute, error_vs_step_size, FDGreeks
from app.engine.heston import HestonParams


def _bs_price_fn(S, K, T, r, sigma, option_type):
    result = bs.price(S, K, T, r, sigma)
    return result.call_price if option_type == "call" else result.put_price


def _heston_price_fn(S, K, T, r, params, option_type):
    from app.engine.heston import price as heston_price
    result = heston_price(S, K, T, r, params)
    return result.call_price if option_type == "call" else result.put_price


class TestFDvsAnalytical:
    """FD Greeks should match BS analytical Greeks closely."""

    def test_delta_call(self):
        analytical = bs.greeks(100, 100, 0.5, 0.05, 0.20, "call")
        fd_greeks = compute(_bs_price_fn, 100, 100, 0.5, 0.05, 0.20, "call")
        assert fd_greeks.delta == pytest.approx(analytical.delta, abs=1e-4)

    def test_delta_put(self):
        analytical = bs.greeks(100, 100, 0.5, 0.05, 0.20, "put")
        fd_greeks = compute(_bs_price_fn, 100, 100, 0.5, 0.05, 0.20, "put")
        assert fd_greeks.delta == pytest.approx(analytical.delta, abs=1e-4)

    def test_gamma(self):
        analytical = bs.greeks(100, 100, 0.5, 0.05, 0.20, "call")
        fd_greeks = compute(_bs_price_fn, 100, 100, 0.5, 0.05, 0.20, "call")
        assert fd_greeks.gamma == pytest.approx(analytical.gamma, abs=1e-4)

    def test_theta(self):
        analytical = bs.greeks(100, 100, 0.5, 0.05, 0.20, "call")
        fd_greeks = compute(_bs_price_fn, 100, 100, 0.5, 0.05, 0.20, "call")
        assert fd_greeks.theta == pytest.approx(analytical.theta, rel=0.02)

    def test_vega(self):
        analytical = bs.greeks(100, 100, 0.5, 0.05, 0.20, "call")
        fd_greeks = compute(_bs_price_fn, 100, 100, 0.5, 0.05, 0.20, "call")
        assert fd_greeks.vega == pytest.approx(analytical.vega, rel=0.01)

    def test_rho(self):
        analytical = bs.greeks(100, 100, 0.5, 0.05, 0.20, "call")
        fd_greeks = compute(_bs_price_fn, 100, 100, 0.5, 0.05, 0.20, "call")
        assert fd_greeks.rho == pytest.approx(analytical.rho, rel=0.01)

    @pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
    def test_delta_various_strikes(self, K):
        analytical = bs.greeks(100, K, 0.5, 0.05, 0.25, "call")
        fd_greeks = compute(_bs_price_fn, 100, K, 0.5, 0.05, 0.25, "call")
        assert fd_greeks.delta == pytest.approx(analytical.delta, abs=1e-4)


class TestHestonFD:
    """Heston FD Greeks should be reasonable (no analytical to compare against)."""

    def test_heston_delta_positive_for_call(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        fd_greeks = compute(_heston_price_fn, 100, 100, 0.5, 0.05, params, "call")
        assert 0 < fd_greeks.delta < 1

    def test_heston_delta_negative_for_put(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        fd_greeks = compute(_heston_price_fn, 100, 100, 0.5, 0.05, params, "put")
        assert -1 < fd_greeks.delta < 0

    def test_heston_gamma_positive(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        fd_greeks = compute(_heston_price_fn, 100, 100, 0.5, 0.05, params, "call")
        assert fd_greeks.gamma > 0

    def test_heston_vega_positive(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        fd_greeks = compute(_heston_price_fn, 100, 100, 0.5, 0.05, params, "call")
        assert fd_greeks.vega > 0


class TestErrorAnalysis:
    def test_v_shape_delta(self):
        """Error vs h should have a minimum (V-shape on log-log)."""
        analytical = bs.greeks(100, 100, 0.5, 0.05, 0.20, "call")
        result = error_vs_step_size(
            analytical.delta, _bs_price_fn, 100, 100, 0.5, 0.05, 0.20, "call", "delta",
        )
        errors = result["errors"]
        # The minimum error should be much smaller than the endpoints
        min_err = min(errors)
        assert min_err < errors[0]   # better than smallest h
        assert min_err < errors[-1]  # better than largest h

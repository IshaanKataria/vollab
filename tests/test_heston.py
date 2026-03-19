"""
Heston model tests.

Validates pricing against known properties and convergence to Black-Scholes.
"""

import math

import numpy as np
import pytest

from app.engine.black_scholes import price as bs_price
from app.engine.heston import HestonParams, HestonResult, heston_smile, price


class TestHestonBasics:
    def test_call_positive(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        result = price(100, 100, 1.0, 0.05, params)
        assert result.call_price > 0

    def test_put_positive(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        result = price(100, 110, 1.0, 0.05, params)
        assert result.put_price > 0

    def test_put_call_parity(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        result = price(100, 100, 1.0, 0.05, params)
        parity = result.call_price - result.put_price - (100 - 100 * math.exp(-0.05))
        assert abs(parity) < 0.01

    def test_itm_call_above_intrinsic(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5)
        result = price(120, 100, 0.5, 0.05, params)
        intrinsic = 120 - 100 * math.exp(-0.05 * 0.5)
        assert result.call_price >= intrinsic * 0.99

    def test_at_expiry(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        result = price(110, 100, 0.0, 0.05, params)
        assert result.call_price == pytest.approx(10.0)
        assert result.put_price == pytest.approx(0.0)


class TestConvergenceToBS:
    """When xi → 0 (no vol-of-vol), Heston should converge to Black-Scholes."""

    def test_low_xi_matches_bs(self):
        sigma = 0.20
        v0 = sigma * sigma  # 0.04
        params = HestonParams(v0=v0, kappa=2.0, theta=v0, xi=0.001, rho=0.0)
        heston_result = price(100, 100, 0.5, 0.05, params)
        bs_result = bs_price(100, 100, 0.5, 0.05, sigma)
        assert heston_result.call_price == pytest.approx(bs_result.call_price, rel=0.01)

    def test_low_xi_put_matches_bs(self):
        sigma = 0.25
        v0 = sigma * sigma
        params = HestonParams(v0=v0, kappa=2.0, theta=v0, xi=0.001, rho=0.0)
        heston_result = price(100, 110, 1.0, 0.03, params)
        bs_result = bs_price(100, 110, 1.0, 0.03, sigma)
        assert heston_result.put_price == pytest.approx(bs_result.put_price, rel=0.01)

    @pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
    def test_multiple_strikes(self, K):
        sigma = 0.20
        v0 = sigma * sigma
        params = HestonParams(v0=v0, kappa=3.0, theta=v0, xi=0.001, rho=0.0)
        h = price(100, K, 0.5, 0.05, params)
        b = bs_price(100, K, 0.5, 0.05, sigma)
        assert h.call_price == pytest.approx(b.call_price, rel=0.02)


class TestHestonSmile:
    def test_negative_rho_produces_skew(self):
        """Negative rho should produce higher IV for OTM puts (left skew)."""
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        smile = heston_smile(100, 1.0, 0.05, params, n_strikes=20)
        ivs = smile["implied_vols"]
        ks = smile["log_moneyness"]

        # Find IVs for negative k (OTM puts) and positive k (OTM calls)
        left_ivs = [iv for k, iv in zip(ks, ivs) if k < -0.1]
        right_ivs = [iv for k, iv in zip(ks, ivs) if k > 0.1]

        if left_ivs and right_ivs:
            assert np.mean(left_ivs) > np.mean(right_ivs)

    def test_zero_rho_symmetric(self):
        """Zero rho should produce approximately symmetric smile."""
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=0.0)
        smile = heston_smile(100, 1.0, 0.0, params, n_strikes=20)
        ivs = smile["implied_vols"]
        ks = smile["log_moneyness"]

        if len(ivs) > 5:
            left_ivs = [iv for k, iv in zip(ks, ivs) if -0.2 < k < -0.05]
            right_ivs = [iv for k, iv in zip(ks, ivs) if 0.05 < k < 0.2]
            if left_ivs and right_ivs:
                assert abs(np.mean(left_ivs) - np.mean(right_ivs)) < 0.02

    def test_smile_returns_data(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.5)
        smile = heston_smile(100, 0.5, 0.05, params, n_strikes=15)
        assert len(smile["implied_vols"]) > 5
        assert len(smile["log_moneyness"]) == len(smile["implied_vols"])


class TestFellerCondition:
    def test_feller_satisfied(self):
        p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5)
        assert p.feller_satisfied
        assert p.feller_ratio > 1.0

    def test_feller_violated(self):
        p = HestonParams(v0=0.04, kappa=0.5, theta=0.01, xi=1.0, rho=-0.5)
        assert not p.feller_satisfied
        assert p.feller_ratio < 1.0

    def test_prices_still_work_when_feller_violated(self):
        """Many real calibrations violate Feller. Pricing should still work."""
        params = HestonParams(v0=0.04, kappa=0.5, theta=0.01, xi=1.0, rho=-0.7)
        result = price(100, 100, 1.0, 0.05, params)
        assert result.call_price > 0
        assert result.put_price > 0


class TestIntegrationQuality:
    def test_low_abserr(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        result = price(100, 100, 1.0, 0.05, params)
        assert result.integration_abserr < 1e-6

"""
Implied volatility solver tests.

Core validation: round-trip test (price with BS at known sigma, extract IV, recover sigma).
"""

import math

import pytest

from app.engine.black_scholes import price as bs_price
from app.engine.implied_vol import IVResult, brenner_subrahmanyam, solve, solve_chain


class TestBrennerSubrahmanyam:
    def test_reasonable_guess(self):
        guess = brenner_subrahmanyam(5.0, 100.0, 0.25)
        assert 0.01 <= guess <= 3.0

    def test_zero_time(self):
        guess = brenner_subrahmanyam(5.0, 100.0, 0.0)
        assert guess == 0.3  # fallback

    def test_clamped_high(self):
        guess = brenner_subrahmanyam(500.0, 10.0, 0.01)
        assert guess <= 3.0


class TestRoundTrip:
    """Price with BS at known sigma, then recover sigma via IV solver."""

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50, 0.80, 1.00])
    def test_call_round_trip(self, sigma):
        S, K, T, r = 100, 100, 0.25, 0.05
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.call_price, S, K, T, r, "call")
        assert iv.converged
        assert iv.sigma == pytest.approx(sigma, abs=1e-6)

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50, 0.80, 1.00])
    def test_put_round_trip(self, sigma):
        S, K, T, r = 100, 100, 0.25, 0.05
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.put_price, S, K, T, r, "put")
        assert iv.converged
        assert iv.sigma == pytest.approx(sigma, abs=1e-6)

    @pytest.mark.parametrize("moneyness", [0.8, 0.9, 1.0, 1.1, 1.2])
    def test_various_moneyness(self, moneyness):
        S, K, T, r, sigma = 100, 100 / moneyness, 0.5, 0.05, 0.25
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.call_price, S, K, T, r, "call")
        assert iv.converged
        assert iv.sigma == pytest.approx(sigma, abs=1e-5)

    @pytest.mark.parametrize("T", [0.01, 0.1, 0.25, 0.5, 1.0, 2.0])
    def test_various_maturities(self, T):
        S, K, r, sigma = 100, 100, 0.05, 0.20
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.call_price, S, K, T, r, "call")
        assert iv.converged
        assert iv.sigma == pytest.approx(sigma, abs=1e-5)


class TestConvergenceSpeed:
    def test_atm_converges_fast(self):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.call_price, S, K, T, r, "call", track_steps=True)
        assert iv.converged
        assert iv.iterations <= 10

    def test_otm_converges(self):
        S, K, T, r, sigma = 100, 130, 0.25, 0.05, 0.30
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.call_price, S, K, T, r, "call")
        assert iv.converged
        assert iv.sigma == pytest.approx(sigma, abs=1e-5)

    def test_track_steps(self):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.call_price, S, K, T, r, "call", track_steps=True)
        assert len(iv.steps) == iv.iterations
        assert iv.steps[-1].error == pytest.approx(0.0, abs=1e-7)


class TestEdgeCases:
    def test_zero_price(self):
        iv = solve(0.0, 100, 100, 0.25, 0.05, "call")
        assert not iv.converged
        assert math.isnan(iv.sigma)

    def test_negative_price(self):
        iv = solve(-1.0, 100, 100, 0.25, 0.05, "call")
        assert not iv.converged

    def test_price_below_intrinsic(self):
        # Call with price below intrinsic S - K*e^(-rT)
        iv = solve(0.5, 200, 100, 0.25, 0.05, "call")
        assert not iv.converged

    def test_very_high_vol(self):
        # Price implying very high vol
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 2.0
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.call_price, S, K, T, r, "call")
        assert iv.converged
        assert iv.sigma == pytest.approx(sigma, abs=1e-3)

    def test_very_low_vol(self):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.02
        result = bs_price(S, K, T, r, sigma)
        iv = solve(result.call_price, S, K, T, r, "call")
        assert iv.converged
        assert iv.sigma == pytest.approx(sigma, abs=1e-4)


class TestSolveChain:
    def test_batch(self):
        options = [
            {"strike": 95, "lastPrice": 8.0, "expiry_years": 0.25},
            {"strike": 100, "lastPrice": 4.6, "expiry_years": 0.25},
            {"strike": 105, "lastPrice": 2.0, "expiry_years": 0.25},
        ]
        results = solve_chain(options, S=100, r=0.05, option_type="call")
        assert len(results) == 3
        for r in results:
            assert "iv" in r
            assert "iv_converged" in r

    def test_batch_skips_zero_price(self):
        options = [
            {"strike": 100, "lastPrice": 0.0, "expiry_years": 0.25},
        ]
        results = solve_chain(options, S=100, r=0.05)
        assert results[0]["iv"] is None
        assert results[0]["iv_converged"] is False

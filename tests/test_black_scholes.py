"""
Black-Scholes pricer tests.

Validated against Hull's textbook examples and known analytical values.
"""

import math

import pytest

from app.engine.black_scholes import Greeks, greeks, norm_cdf, norm_pdf, price, vega_raw


class TestNormCDF:
    def test_symmetry(self):
        assert norm_cdf(0.0) == pytest.approx(0.5)

    def test_positive(self):
        assert norm_cdf(1.0) == pytest.approx(0.8413447, rel=1e-6)

    def test_negative(self):
        assert norm_cdf(-1.0) == pytest.approx(0.1586553, rel=1e-6)

    def test_extreme_positive(self):
        assert norm_cdf(10.0) == pytest.approx(1.0, abs=1e-15)

    def test_extreme_negative(self):
        assert norm_cdf(-10.0) == pytest.approx(0.0, abs=1e-15)

    def test_complementarity(self):
        for x in [-2.0, -1.0, 0.0, 0.5, 1.5, 3.0]:
            assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-14)


class TestNormPDF:
    def test_at_zero(self):
        assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), rel=1e-10)

    def test_symmetry(self):
        assert norm_pdf(1.5) == pytest.approx(norm_pdf(-1.5), rel=1e-14)


class TestBSPricing:
    """Hull-style test cases for European option pricing."""

    def test_atm_call(self):
        # S=100, K=100, T=0.25, r=5%, sigma=20%
        result = price(100, 100, 0.25, 0.05, 0.20)
        assert result.call_price == pytest.approx(4.6150, rel=1e-3)

    def test_atm_put(self):
        result = price(100, 100, 0.25, 0.05, 0.20)
        assert result.put_price == pytest.approx(3.3728, rel=1e-3)

    def test_itm_call(self):
        # S=110, K=100
        result = price(110, 100, 0.25, 0.05, 0.20)
        assert result.call_price > 10.0  # at least intrinsic

    def test_otm_put(self):
        # S=110, K=100
        result = price(110, 100, 0.25, 0.05, 0.20)
        assert result.put_price < 1.0  # small value

    def test_deep_itm_call(self):
        result = price(200, 100, 1.0, 0.05, 0.20)
        intrinsic = 200 - 100 * math.exp(-0.05)
        assert result.call_price >= intrinsic * 0.99

    def test_deep_otm_call(self):
        result = price(50, 100, 0.25, 0.05, 0.20)
        assert result.call_price < 0.01
        assert result.call_price >= 0.0

    def test_hull_example(self):
        # Hull, Options Futures and Other Derivatives
        # S=42, K=40, T=0.5, r=10%, sigma=20%
        result = price(42, 40, 0.5, 0.10, 0.20)
        assert result.call_price == pytest.approx(4.7594, rel=1e-3)

    def test_no_negative_prices(self):
        for S in [50, 100, 200]:
            for K in [50, 100, 200]:
                result = price(S, K, 0.25, 0.05, 0.30)
                assert result.call_price >= 0.0
                assert result.put_price >= 0.0


class TestPutCallParity:
    """C - P = S - K*e^(-rT)"""

    def test_parity_atm(self):
        result = price(100, 100, 0.25, 0.05, 0.20)
        assert result.parity_residual < 1e-10

    def test_parity_itm(self):
        result = price(120, 100, 1.0, 0.05, 0.30)
        assert result.parity_residual < 1e-10

    def test_parity_otm(self):
        result = price(80, 100, 0.5, 0.08, 0.25)
        assert result.parity_residual < 1e-10

    def test_parity_many(self):
        for S in [80, 100, 120]:
            for K in [90, 100, 110]:
                for T in [0.1, 0.5, 1.0, 2.0]:
                    for sigma in [0.10, 0.30, 0.60]:
                        result = price(S, K, T, 0.05, sigma)
                        assert result.parity_residual < 1e-10, (
                            f"Parity failed for S={S}, K={K}, T={T}, sigma={sigma}: "
                            f"residual={result.parity_residual}"
                        )


class TestEdgeCases:
    def test_t_zero_itm_call(self):
        result = price(110, 100, 0.0, 0.05, 0.20)
        assert result.call_price == pytest.approx(10.0)
        assert result.put_price == pytest.approx(0.0)

    def test_t_zero_otm_call(self):
        result = price(90, 100, 0.0, 0.05, 0.20)
        assert result.call_price == pytest.approx(0.0)
        assert result.put_price == pytest.approx(10.0)

    def test_t_zero_atm(self):
        result = price(100, 100, 0.0, 0.05, 0.20)
        assert result.call_price == pytest.approx(0.0)
        assert result.put_price == pytest.approx(0.0)

    def test_sigma_zero_itm(self):
        result = price(110, 100, 1.0, 0.05, 0.0)
        df = math.exp(-0.05)
        assert result.call_price == pytest.approx(110 - 100 * df, abs=1e-10)
        assert result.put_price == pytest.approx(0.0)

    def test_sigma_zero_otm(self):
        result = price(90, 100, 1.0, 0.05, 0.0)
        df = math.exp(-0.05)
        assert result.call_price == pytest.approx(0.0)
        assert result.put_price == pytest.approx(100 * df - 90, abs=1e-10)

    def test_invalid_S(self):
        with pytest.raises(ValueError):
            price(0, 100, 0.25, 0.05, 0.20)

    def test_invalid_K(self):
        with pytest.raises(ValueError):
            price(100, -1, 0.25, 0.05, 0.20)

    def test_negative_T(self):
        with pytest.raises(ValueError):
            price(100, 100, -0.1, 0.05, 0.20)

    def test_negative_sigma(self):
        with pytest.raises(ValueError):
            price(100, 100, 0.25, 0.05, -0.1)


class TestGreeks:
    def test_call_delta_range(self):
        g = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        assert 0.0 < g.delta < 1.0

    def test_put_delta_range(self):
        g = greeks(100, 100, 0.25, 0.05, 0.20, "put")
        assert -1.0 < g.delta < 0.0

    def test_call_put_delta_relation(self):
        gc = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        gp = greeks(100, 100, 0.25, 0.05, 0.20, "put")
        assert gc.delta - gp.delta == pytest.approx(1.0, abs=1e-10)

    def test_gamma_positive(self):
        g = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        assert g.gamma > 0.0

    def test_gamma_same_for_call_put(self):
        gc = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        gp = greeks(100, 100, 0.25, 0.05, 0.20, "put")
        assert gc.gamma == pytest.approx(gp.gamma, abs=1e-14)

    def test_theta_negative_for_call(self):
        g = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        assert g.theta < 0.0  # time decay

    def test_vega_positive(self):
        g = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        assert g.vega > 0.0

    def test_vega_same_for_call_put(self):
        gc = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        gp = greeks(100, 100, 0.25, 0.05, 0.20, "put")
        assert gc.vega == pytest.approx(gp.vega, abs=1e-14)

    def test_deep_itm_call_delta_near_one(self):
        g = greeks(200, 100, 0.25, 0.05, 0.20, "call")
        assert g.delta > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        g = greeks(50, 100, 0.25, 0.05, 0.20, "call")
        assert g.delta < 0.01

    def test_atm_call_delta_near_half(self):
        # ATM forward delta is ~0.5, spot delta slightly above for positive rates
        g = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        assert 0.45 < g.delta < 0.65

    def test_greeks_at_expiry(self):
        g = greeks(100, 100, 0.0, 0.05, 0.20, "call")
        assert g.delta == 0.0
        assert g.gamma == 0.0
        assert g.vega == 0.0

    def test_hull_delta(self):
        # Hull example: S=42, K=40, T=0.5, r=10%, sigma=20%, d1=0.7693
        g = greeks(42, 40, 0.5, 0.10, 0.20, "call")
        assert g.delta == pytest.approx(0.7791, rel=1e-3)


class TestVegaRaw:
    def test_positive(self):
        v = vega_raw(100, 100, 0.25, 0.05, 0.20)
        assert v > 0.0

    def test_zero_at_expiry(self):
        v = vega_raw(100, 100, 0.0, 0.05, 0.20)
        assert v == 0.0

    def test_consistency_with_greeks(self):
        v_raw = vega_raw(100, 100, 0.25, 0.05, 0.20)
        g = greeks(100, 100, 0.25, 0.05, 0.20, "call")
        # greeks.vega is per 1% (multiplied by 0.01), raw is per unit
        assert g.vega == pytest.approx(v_raw * 0.01, rel=1e-10)

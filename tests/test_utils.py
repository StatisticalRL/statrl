import math

from statrl.settings.utils import klBern, klGauss, klPoisson, klExp


def test_klBern_zero_and_clipping():
    assert klBern(0.5, 0.5) == 0.0
    # eps-clamping keeps the endpoints finite instead of log(0)
    assert klBern(0.0, 0.0) == 0.0
    assert klBern(1.0, 1.0) == 0.0
    assert math.isclose(klBern(0.0, 1.0), 34.53957599234081, rel_tol=1e-9)
    assert math.isclose(klBern(0.3, 0.6), 0.18378689738681217, rel_tol=1e-9)


def test_klGauss_known_values():
    assert klGauss(0.0, 0.0) == 0.0
    assert math.isclose(klGauss(1.0, 0.0), 0.5, rel_tol=1e-9)              # (1-0)^2 / (2*1)
    assert math.isclose(klGauss(2.0, 0.0, sig2=2.0), 1.0, rel_tol=1e-9)    # (2-0)^2 / (2*2)
    assert math.isclose(klGauss(1.0, 3.0), klGauss(3.0, 1.0), rel_tol=1e-9)  # symmetric in x, y


def test_klPoisson_and_klExp_zero_at_equality():
    assert klPoisson(3.0, 3.0) == 0.0
    assert klExp(2.0, 2.0) == 0.0
    assert math.isclose(klPoisson(1.0, 2.0), 0.3068528194400547, rel_tol=1e-9)
    assert math.isclose(klExp(1.0, 2.0), 0.1931471805599453, rel_tol=1e-9)
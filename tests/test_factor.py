import numpy.random as rand
from xynergy.factor import _nmf, _svd, _rpca, _pmf
from xynergy.lnmf import _lnmf
from xynergy.test import calc_explained_var
import numpy as np
import pytest


def mats():
    rng = rand.default_rng(seed=1)
    mats = []
    for i in range(10):
        mats.append(rng.normal(size=25).reshape(5, 5) ** 2)
    return mats


# Slightly more realistic data that range from 0-100
def beta_mats():
    rng = rand.default_rng(seed=2)
    mats = []
    for i in range(10):
        mats.append((rng.beta(1, 3, size=25).reshape(5, 5) ** 2) * 100)
    return mats


@pytest.mark.slow
class TestNMF:
    def test_close(self):
        approx = [_nmf(x) for x in mats()]
        avg_var_explained = np.mean(
            [calc_explained_var(x, y) for x, y in zip(approx, mats())]
        )
        assert avg_var_explained > 0.9


@pytest.mark.slow
class TestSVD:
    def test_close(self):
        approx = [_svd(x) for x in mats()]
        avg_var_explained = np.mean(
            [calc_explained_var(x, y) for x, y in zip(approx, mats())]
        )
        assert avg_var_explained > 0.9


@pytest.mark.slow
class TestRPCA:
    def test_close(self):
        approx = [_rpca(x) for x in mats()]
        avg_var_explained = np.mean(
            [calc_explained_var(x, y) for x, y in zip(approx, mats())]
        )
        assert avg_var_explained > 0.9


@pytest.mark.slow
class TestPMF:
    def test_close(self):
        approx = [_pmf(x) for x in mats()]
        avg_var_explained = np.mean(
            [calc_explained_var(x, y) for x, y in zip(approx, mats())]
        )
        assert avg_var_explained > 0.9


@pytest.mark.slow
class TestLNMF:
    def test_close(self):
        approx = [_lnmf(x, 1) for x in beta_mats()]
        avg_var_explained = np.mean(
            [calc_explained_var(x, y) for x, y in zip(approx, mats())]
        )
        assert avg_var_explained > 0.9

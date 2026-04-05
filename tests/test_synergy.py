import numpy as np
import xynergy as xyn
import pytest
import polars as pl


@pytest.fixture
def doses():
    return [0.001, 0.01, 0.1, 1, 10, 100, 1000]


@pytest.fixture
def inhibitions(doses):
    return (1 / (1 + 10.0 ** -np.log10(doses))) * 100


@pytest.fixture
def bliss_indep(doses, inhibitions):
    resp = 1 - inhibitions / 100
    resp = [1.0] + resp.tolist()
    doses = np.append(0, doses)
    doses = [(x, y) for x in doses for y in doses]
    bliss = [100 * (1 - x * y) for x in resp for y in resp]
    df = pl.DataFrame(doses, orient="row")
    df = df.with_columns(pl.Series(name="resp", values=bliss))
    df = df.rename({"column_0": "a", "column_1": "b"})
    return df


@pytest.fixture
def only_a_works():
    return pl.DataFrame(
        {
            "a": [0, 1, 10, 0, 1, 10, 0, 1, 10],
            "b": [0, 0, 0, 1, 1, 1, 10, 10, 10],
            "resp": [0, 50, 100, 0, 50, 100, 0, 50, 100],
        }
    )


# For a lot of the tests previously implemented here, it makes more sense to
# test the reference rather than the actual synergy (since the synergy is just
# taking the difference from the response)


class TestBliss:
    def test_bliss_indep_is_near_0(self, bliss_indep):
        bliss = xyn.add_synergy(bliss_indep, ["a", "b"], "resp", None, method="bliss")
        mse = bliss["bliss_syn"].pow(2).mean()
        assert mse < 1e-10


class TestHSA:
    def test_only_one_drug_works_hsa_syn_zero(self, only_a_works):
        with_hsa = xyn.add_synergy(only_a_works, ["a", "b"], "resp", None, method="hsa")
        assert all(with_hsa["hsa_syn"] == 0)


class TestZIP:
    def test_bliss_indep_is_near_0(self, bliss_indep):
        zip = xyn.add_synergy(bliss_indep, ["a", "b"], "resp", None, method="zip")
        mse = zip["zip_syn"].pow(2).mean()
        assert mse < 1e-10


class TestLoewe:
    def test_sham_same_as_loewe(self, bliss_indep):
        # In the case of a sham experiment, where instead of there being 'drug
        # A' and 'drug B', there's just 'drug A' and 'drug A'. Loewe would call
        # this 'no interaction', and thus a Loewe synergy for this experiment
        # should be 0

        # This is admittedly more a test of reference than of synergy
        out = xyn.add_synergy(bliss_indep, ["a", "b"], "resp", None, method="loewe")
        truth = bliss_indep["resp"] - xyn.ll4(
            bliss_indep["a"] + bliss_indep["b"], 1, 0, 100, 1
        )
        mse = (out["loewe_syn"] - truth).pow(2).mean()
        assert mse < 1e-10

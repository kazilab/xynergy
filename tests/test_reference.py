import numpy as np
import pytest
import polars as pl
import xynergy as xyn


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


class TestBliss:
    def test_no_responses(self):
        df = pl.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [0, 0, 1, 1],
                "resp": [0, 0, 0, 0],
            }
        )
        truth = df["resp"]
        out = xyn.add_reference(df, ["a", "b"], "resp", method="bliss")["bliss_ref"]
        assert np.allclose(truth, out)

    def test_max_responses(self):
        df = pl.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [0, 0, 1, 1],
                "resp": [100, 100, 100, 100],
            }
        )
        truth = df["resp"]
        out = xyn.add_reference(df, ["a", "b"], "resp", method="bliss")["bliss_ref"]
        assert np.allclose(truth, out)

    def test_only_one_drug_works(self):
        df = pl.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [0, 0, 1, 1],
                "resp": [0, 100, 0, 100],
            }
        )
        truth = df["resp"]
        out = xyn.add_reference(df, ["a", "b"], "resp", method="bliss")["bliss_ref"]
        assert np.allclose(truth, out)

    def test_both_drugs_half_max(self):
        df = pl.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [0, 0, 1, 1],
                "resp": [0, 50, 50, 75],
            }
        )
        truth = df["resp"]
        out = xyn.add_reference(df, ["a", "b"], "resp", method="bliss")["bliss_ref"]
        assert np.allclose(truth, out)


class TestHSA:
    def test_no_responses(self):
        df = pl.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [0, 0, 1, 1],
                "resp": [0, 0, 0, 0],
            }
        )
        truth = df["resp"]
        out = xyn.add_reference(df, ["a", "b"], "resp", method="hsa")["hsa_ref"]
        assert np.allclose(truth, out)

    def test_max_responses(self):
        df = pl.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [0, 0, 1, 1],
                "resp": [100, 100, 100, 100],
            }
        )
        truth = df["resp"]
        out = xyn.add_reference(df, ["a", "b"], "resp", method="hsa")["hsa_ref"]
        assert np.allclose(truth, out)

    def test_only_one_drug_works(self):
        df = pl.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [0, 0, 1, 1],
                "resp": [0, 100, 0, 100],
            }
        )
        truth = df["resp"]
        out = xyn.add_reference(df, ["a", "b"], "resp", method="hsa")["hsa_ref"]
        assert np.allclose(truth, out)

    def test_both_drugs_half_max(self):
        df = pl.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [0, 0, 1, 1],
                "resp": [0, 50, 50, 50],
            }
        )
        truth = df["resp"]
        out = xyn.add_reference(df, ["a", "b"], "resp", method="hsa")["hsa_ref"]
        assert np.allclose(truth, out)


class TestZIP:
    # Testing is largely covered in test_synergy
    pass


class TestLoewe:
    # Testing is largely covered in test_synergy
    pass

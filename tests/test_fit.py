import numpy as np
import xynergy.fit as fit
import pytest
import polars as pl
from polars import col as c


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
    df = df.with_columns(
        pl.Series(name="response", values=bliss), pl.lit(1).alias("experiment_id")
    )
    df = df.rename({"column_0": "dose_a", "column_1": "dose_b"})
    return df


class TestLL4:
    def test_pos_inf_returns_Emax(self):
        Emax = 100
        out = fit.ll4(float("inf"), 1, 0, Emax, 1)
        assert out == Emax

    def test_0_returns_E0(self):
        E0 = 0
        out = fit.ll4(0, 1, E0, 100, 1)
        assert out == E0

    def test_many_doses_return_many_respones(self):
        out = fit.ll4(np.logspace(-2, 2, 5), 1, 0, 100, 1)
        assert len(out) == 5

    def test_integers_ok(self):
        # Raising integers to negative powers is not ok in numpy's book, so we
        # need to make sure this is dealt with properly
        doses = np.logspace(-2, 2, 5)
        out = fit.ll4(doses, 1, 0, 100, 1)
        expected = np.array([100 / 101, 100 / 11, 100 / 2, 100 / 1.1, 100 / 1.01])
        assert all(out == expected)


class TestLL3:
    def test_set_E0_is_neg_inf(self):
        E0 = 10
        _ll3 = fit._make_ll3(E0)
        out = _ll3(float("-inf"), slope=1, max=100, log10_ic50=0)
        assert out == E0


class TestCurveFits:
    def test_returns_min_inhibition(self, doses, inhibitions):
        out = fit.fit_curve(doses, inhibitions)
        assert pytest.approx(out["min"], abs=0.0001) == 0

    def test_returns_max_inhibition(self, doses, inhibitions):
        out = fit.fit_curve(doses, inhibitions)
        assert pytest.approx(out["max"], abs=0.0001) == 100

    def test_hill_correct(self, doses, inhibitions):
        out = fit.fit_curve(doses, inhibitions)
        assert pytest.approx(out["slope"], abs=0.0001) == 1

    def test_ic50_correct(self, doses, inhibitions):
        out = fit.fit_curve(doses, inhibitions)
        assert pytest.approx(out["ic50"], abs=0.0001) == 1


class TestFitIndividualDrugs:
    def test_custom_names_ok(self, bliss_indep):
        df = bliss_indep.rename({"dose_a": "drug_a", "dose_b": "drug_b"})
        out = fit.fit_individual_drugs(df, ["drug_a", "drug_b"])
        assert all(out["drug"] == ["drug_a", "drug_b"])

    def test_warn_about_non_zero_mins(self, bliss_indep):
        df = bliss_indep.filter(pl.col("dose_b") != 0)
        with pytest.warns(
            UserWarning,
            match="The following experiments had non-zero minimum concentrations",
        ):
            fit.fit_individual_drugs(df)


class TestAddSingleDrugResponses:
    def test_catch_malformed_columns(self, bliss_indep):
        with pytest.raises(ValueError) as excinfo:
            fit.add_uncombined_drug_responses(bliss_indep, dose_cols="just_one")
        assert "Length of dose_cols must be exactly 2" in str(excinfo.value)

    def test_calculated_resp_correct(self, bliss_indep, inhibitions):
        inhibitions = np.append(0, inhibitions)
        out = fit.add_uncombined_drug_responses(bliss_indep)
        responses = (
            pl.DataFrame({"dose_b_resp": inhibitions})
            .join(pl.DataFrame({"dose_b_resp": inhibitions}), how="cross")
            .rename({"dose_b_resp": "dose_a_resp_right"})
        )
        truth = pl.concat([bliss_indep, responses], how="horizontal")
        both = out.join(truth, on=["dose_a", "dose_b", "experiment_id", "response"])
        diffs = both.select(
            c.dose_a_resp - c.dose_a_resp_right, c.dose_b_resp - c.dose_b_resp_right
        ).unpivot()
        all_within_tolerance = diffs.select(c.value.abs().lt(0.00001).all()).item()
        assert all_within_tolerance

    def test_fit_false_returns_observed_responses(self):
        mdf = (
            pl.DataFrame({"drug_a": [[1, 2, 3]], "drug_b": [[1, 2, 3]]})
            .explode("drug_a")
            .explode("drug_b")
            .with_columns(response=np.array(range(1, 10)))
        )

        out = fit.add_uncombined_drug_responses(
            mdf,
            ["drug_a", "drug_b"],
            "response",
            None,
            fit=False,
            log="none",
        )

        a = pl.DataFrame({"drug_a": [1, 2, 3], "drug_a_resp": [1, 4, 7]})
        b = pl.DataFrame({"drug_b": [1, 2, 3], "drug_b_resp": [1, 2, 3]})

        mdf = mdf.join(a, on="drug_a", how="left")
        mdf = mdf.join(b, on="drug_b", how="left")

        assert mdf.equals(out)


class TestGetMinConcentrations:
    def test_warn_about_non_zero_mins(self, bliss_indep):
        df = bliss_indep.filter(pl.col("dose_b") != 0)
        with pytest.warns(
            UserWarning,
            match="The following experiments had non-zero minimum concentrations",
        ):
            fit._get_min_concentrations(
                df, ["dose_a", "dose_b"], ["experiment_id"], "all"
            )

    def test_returns_correct_minimums(self, bliss_indep):
        df = bliss_indep.filter(pl.col("dose_b") != 0)
        out = fit._get_min_concentrations(
            df, ["dose_a", "dose_b"], ["experiment_id"], "none"
        )
        assert out["dose_a"][0] == 0.0 and out["dose_b"][0] == 0.001

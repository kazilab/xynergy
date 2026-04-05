import xynergy.util as util
import numpy as np
import polars as pl
import pytest


class TestEnforceList:
    def test_list_stays_list(self):
        out = util.make_list_if_str_or_none([2])
        truth = [2]
        assert out == truth

    def test_number_becomes_list(self):
        out: list = util.make_list_if_str_or_none(2)
        truth = [2]
        assert out == truth

    def test_none_becomes_empty_list(self):
        out = util.make_list_if_str_or_none(None)
        truth = []
        assert out == truth

    def test_numpy_array_float_becomes_array_list(self):
        out: list = util.make_list_if_str_or_none(np.array(2))
        truth = np.array([2])
        assert out == truth

    def test_numpy_array_list_stays_array_list(self):
        out: list = util.make_list_if_str_or_none(np.array([2]))
        truth = np.array([2])
        assert out == truth

    def test_polars_series_stays_polars_series(self):
        out: list = util.make_list_if_str_or_none(pl.Series([2]))
        truth = pl.Series([2])
        assert out[0] == truth[0]


class TestVenter:
    @pytest.mark.parametrize("input", [-10, -10.0, 0, 1.0, 0.1, 1, 100])
    def test_single_digit_returns_self(self, input):
        assert util.venter([input]) == input

    @pytest.mark.parametrize(
        "input,output",
        [
            ([1, 1, 1, 2], 1),
            ([1.1, 1.1, 1.1, 1], 1.1),
            ([1, 1, 1000], 1),
            ([-1, -1, -1, 1], -1),
            ([-1, -1.0, -1, 1, 1], -1),
        ],
    )
    def test_obvious_mode_returned(self, input, output):
        assert util.venter(input) == output

    @pytest.mark.parametrize(
        "input,output",
        [
            ([2, 2.1, 10], 2.05),
            ([2, 2.1, 100], 2.05),
            ([2.1, 2.2, 2.4], 2.15),
        ],
    )
    def test_continous_mode(self, input, output):
        assert pytest.approx(util.venter(input)) == output

    @pytest.mark.parametrize(
        "input,k,output",
        [
            ([2, 2.1, 10], 1, 2.05),
            ([2, 2.1, 10], 2, 6),
            ([2.81, 2.9, 3, 3.1, 3.2, 1000], 1, 2.855),
            ([2.81, 2.9, 3, 3.1, 3.2, 1000], 2, 2.905),
            ([2.81, 2.9, 3, 3.1, 3.2, 1000], 3, 2.955),
            ([2.81, 2.9, 3, 3.1, 3.2, 1000], 4, 3.005),
            ([2.81, 2.9, 3, 3.1, 3.2, 1000], 5, 501.405),
        ],
    )
    def test_adjust_k_mode(self, input, k, output):
        assert pytest.approx(util.venter(input, k)) == output

    @pytest.mark.parametrize(
        "input,k",
        [
            ([0, 0, 0], -1),
            ([0, 0, 0], 3),
        ],
    )
    def test_catch_bad_k(self, input, k):
        with pytest.raises(ValueError) as excinfo:
            util.venter(input, k)
        assert "k must be in the range [0, 3)" in str(excinfo.value)


class TestBinnedMode:
    @pytest.mark.parametrize(
        "input,bins,output",
        [
            ([1, 1, 1, 5], 2, 2),
            ([1, 1, 1, 10], 2, 3.25),
        ],
    )
    def test_binned_mode_increases_when_range_increases(self, input, bins, output):
        assert util.binned_mode(input, bins) == output

    @pytest.mark.parametrize(
        "input,bins,output",
        [
            ([1, 1, 2, 2], 2, 1.25),
            ([2, 2, 1, 1], 2, 1.25),
        ],
    )
    def test_binned_mode_chooses_lowest_when_tied(self, input, bins, output):
        assert util.binned_mode(input, bins) == output

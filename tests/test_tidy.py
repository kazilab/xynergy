from xynergy import tidy
import pytest
import polars as pl


@pytest.fixture
def data():
    return pl.DataFrame(
        {
            "drug_1": [0, 0, 1, 1],
            "drug_2": [0, 1, 0, 1],
            "group": [0, 0, 0, 0],
            "resp": [0, 0, 0, 100],
            "dose_a": [0, 0, 0, 0],
        }
    )


@pytest.fixture
def data_ratio_resp(data):
    data = data.with_columns(pl.col("resp") / 100)
    return data


@pytest.fixture
def data_surv_resp(data):
    data = data.with_columns((100 - pl.col("resp")).alias("resp"))
    return data


@pytest.fixture
def data_with_groups():
    return pl.DataFrame(
        {
            "drug_1": [0, 0, 1, 1, 0, 0, 1, 1],
            "drug_2": [0, 1, 0, 1, 0, 1, 0, 1],
            "resp": [0, 0, 0, 100, 0, 0, 0, 100],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "dummy": [":)", ":)", ":)", ":)", ":)", ":)", ":)", ":)"],
        }
    )


@pytest.fixture
def data_missing_off_axis():
    return pl.DataFrame(
        {
            "drug_1": [0, 0, 0, 1, 1, 10, 10],
            "drug_2": [0, 1, 10, 0, 1, 0, 10],
            "resp": [0, 0, 0, 0, 0, 0, 100],
        }
    )


@pytest.fixture
def data_only_one_dose_drug_b():
    return pl.DataFrame(
        {
            "drug_1": [0, 1],
            "drug_2": [1, 1],
            "resp": [0, 100],
        }
    )


@pytest.fixture
def data_multiple_resp_cols():
    return pl.DataFrame(
        {
            "drug_1": [0, 0, 1, 1],
            "drug_2": [0, 1, 0, 1],
            "resp_1": [0, 0, 0, 100],
            "resp_2": [0, 0, 0, 100],
        }
    )


class TestTidy:
    def test_too_many_drug_cols(self, data):
        with pytest.raises(ValueError) as excinfo:
            tidy(data, ["drug_1", "drug_2", "drug_1"], ["resp"])
        assert "Length of dose_cols must be exactly 2" in str(excinfo.value)

    def test_too_few_drug_cols(self, data):
        with pytest.raises(ValueError) as excinfo:
            tidy(data, ["drug_1"], ["resp"])
        assert "Length of dose_cols must be exactly 2" in str(excinfo.value)

    def test_catch_reused_col_name(self, data):
        with pytest.raises(ValueError) as excinfo:
            tidy(data, ["drug_1", "drug_2"], ["resp"], ["resp"])
        assert "Columns cannot be used in multiple arguments" in str(excinfo.value)

    def test_catch_reserved_col_name_used(self, data):
        with pytest.warns(UserWarning, match="contain reserved names"):
            tidy(data, ["drug_1", "drug_2"], ["resp"], ["dose_a"])

    def test_catch_column_name_not_present_in_data(self, data):
        with pytest.raises(ValueError) as excinfo:
            tidy(data, ["bing", "bong"], ["bang"], ["boom"])
        assert "Column(s) do not exist in dataset" in str(excinfo.value)

    def test_ensure_grouping_works(self, data_with_groups):
        out = tidy(data_with_groups, ["drug_1", "drug_2"], ["resp"], ["group", "dummy"])
        truth = pl.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "dummy": [":)", ":)", ":)", ":)", ":)", ":)", ":)", ":)"],
                "experiment_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "dose_a": [0, 0, 1, 1, 0, 0, 1, 1],
                "dose_b": [0, 1, 0, 1, 0, 1, 0, 1],
                "response": [0, 0, 0, 100, 0, 0, 0, 100],
            }
        )
        assert out.equals(truth)

    def test_drops_unused_cols(self, data_with_groups):
        out = tidy(data_with_groups, ["drug_1", "drug_2"], ["resp"])
        assert out.columns == ["experiment_id", "dose_a", "dose_b", "response"]
        # Test supplying string instead of list fine

    def test_adds_missing_concentrations(self, data_missing_off_axis):
        doses_1 = data_missing_off_axis["drug_1"].unique()
        doses_2 = data_missing_off_axis["drug_2"].unique()
        all_dose_combos = {(x, y) for x in doses_1 for y in doses_2}

        out = tidy(data_missing_off_axis, ["drug_1", "drug_2"], ["resp"])
        out_combos = set(zip(out["dose_a"], out["dose_b"]))
        assert all_dose_combos == out_combos

    def test_normalizes_non_percent_responses(self, data_ratio_resp):
        out = tidy(
            data_ratio_resp, ["drug_1", "drug_2"], ["resp"], response_is_percent=False
        )
        truth = pl.DataFrame(
            {
                "experiment_id": [1, 1, 1, 1],
                "dose_a": [0, 0, 1, 1],
                "dose_b": [0, 1, 0, 1],
                "response": [0, 0, 0, 100],
            }
        )
        assert out.equals(truth)

    def test_normalizes_survival_responses(self, data_surv_resp):
        out = tidy(
            data_surv_resp, ["drug_1", "drug_2"], ["resp"], complete_response_is_0=True
        )
        truth = pl.DataFrame(
            {
                "experiment_id": [1, 1, 1, 1],
                "dose_a": [0, 0, 1, 1],
                "dose_b": [0, 1, 0, 1],
                "response": [0, 0, 0, 100],
            }
        )
        assert out.equals(truth)

    def test_string_args_for_resp_and_experiment_col_ok(self, data):
        out = tidy(data, ["drug_1", "drug_2"], "resp", "group")
        truth = pl.DataFrame(
            {
                "group": [0, 0, 0, 0],
                "experiment_id": [1, 1, 1, 1],
                "dose_a": [0, 0, 1, 1],
                "dose_b": [0, 1, 0, 1],
                "response": [0, 0, 0, 100],
            }
        )
        assert out.equals(truth)

    def test_catch_only_one_dose_drug_b(self, data_only_one_dose_drug_b):
        with pytest.raises(ValueError) as excinfo:
            tidy(data_only_one_dose_drug_b, ["drug_1", "drug_2"], ["resp"])
        assert "Experimental group contains drug with only 1 dose" in str(excinfo.value)

    def test_pivot_multiple_responses(self, data_multiple_resp_cols):
        out = tidy(data_multiple_resp_cols, ["drug_1", "drug_2"], ["resp_1", "resp_2"])
        truth = pl.DataFrame(
            {
                "experiment_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "dose_a": [0, 0, 0, 0, 1, 1, 1, 1],
                "dose_b": [0, 0, 1, 1, 0, 0, 1, 1],
                "replicate": [
                    "resp_1",
                    "resp_2",
                    "resp_1",
                    "resp_2",
                    "resp_1",
                    "resp_2",
                    "resp_1",
                    "resp_2",
                ],
                "response": [0, 0, 0, 0, 0, 0, 100, 100],
            }
        )
        assert out.equals(truth)

    # test replicate col ok if no mulitple response cols

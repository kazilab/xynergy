import polars as pl

from xynergy.example import get_example_xynergy_kwargs, load_example_data


class TestBundledExampleData:
    def test_load_example_data_returns_canonical_columns(self):
        df = load_example_data()

        assert df.columns == [
            "experiment_source_id",
            "line",
            "pair_index",
            "drug_a",
            "drug_b",
            "dose_a",
            "dose_b",
            "response",
        ]
        assert df.height == 16
        assert df["response"].dtype == pl.Float64
        assert df["response"].min() == 0.0
        assert df["response"].max() == 99.0

    def test_load_example_data_matches_axis_and_diagonal_pattern(self):
        df = load_example_data()
        pairs = set(zip(df["dose_a"].to_list(), df["dose_b"].to_list(), strict=False))
        doses = sorted(set(df["dose_a"].to_list()).union(df["dose_b"].to_list()))
        axis = {(0.0, dose) for dose in doses} | {(dose, 0.0) for dose in doses}
        diagonal = {(dose, dose) for dose in doses if dose != 0.0}

        assert pairs == axis | diagonal

    def test_get_example_xynergy_kwargs_match_loaded_columns(self):
        df = load_example_data()
        kwargs = get_example_xynergy_kwargs()

        assert kwargs["dose_cols"] == ["dose_a", "dose_b"]
        assert kwargs["response_col"] == "response"
        assert kwargs["response_is_percent"] is True
        assert kwargs["complete_response_is_0"] is False
        assert set(kwargs["experiment_cols"]).issubset(df.columns)

    def test_raw_example_data_preserves_workbook_headers(self):
        df = load_example_data(raw=True, convert_to_inhibition=False)

        assert df.columns == [
            "Experiment_ID",
            "Bio_specimen",
            "PairIndex",
            "Drug1",
            "Drug2",
            "Conc1",
            "Conc2",
            "Response",
        ]
        assert df["Response"].min() == 1.0
        assert df["Response"].max() == 100.0
